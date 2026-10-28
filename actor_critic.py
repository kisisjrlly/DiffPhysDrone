"""Small recurrent-policy value head and short-horizon training objectives."""

import hashlib
import io
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def load_critic_checkpoint(critic, checkpoint, *, map_location="cpu"):
    """Strictly restore value weights and identify the exact bytes loaded."""
    source = Path(checkpoint).resolve()
    payload = source.read_bytes()
    critic.load_state_dict(torch.load(io.BytesIO(payload), map_location=map_location,
                                     weights_only=True), strict=True)
    return {"path": str(source), "sha256": hashlib.sha256(payload).hexdigest()}


class LatentCostCritic(nn.Module):
    """Estimate future task cost from the actor's own recurrent memory."""

    def __init__(self, latent_dim=128, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Do not send an arbitrary initial value slope into the actor.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latent):
        return self.net(latent).squeeze(-1)

    def actor_value(self, latent):
        """Differentiate through latent state without updating critic weights."""
        x = latent
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                x = F.linear(x, layer.weight.detach(),
                             None if layer.bias is None else layer.bias.detach())
            else:
                x = layer(x)
        return x.squeeze(-1)


def task_step_costs(rollout, env, args, *, include_camera_regularizer=True):
    """Decompose the existing masked objective; costs.mean() matches its value.

    Mask/denominator values are discrete, as in the original loss. At a segment
    boundary, values from earlier segments stay numerically identical but do
    not carry gradients into the next segment's velocity/jerk regularizers.
    """
    actions = torch.stack(rollout["act_history"])
    velocities = torch.stack(rollout["v_history"])
    target_velocities = torch.stack(rollout["target_v_history"])
    pre_vec = torch.stack(rollout["vec_history"])
    post_vec = torch.stack(rollout["post_vec_history"])
    post_p = torch.stack(rollout["post_p_history"])
    valid = torch.stack(rollout["valid_history"])
    camera = torch.stack(rollout["cam_history"])
    T, B = valid.shape
    costs = velocities.new_zeros((T, B))
    boundaries = set(rollout.get("segment_boundaries", ()))

    def weighted(value, mask):
        weights = mask.to(value.dtype)
        return value * weights * (T * B / weights.sum().clamp_min(1.0))

    win = int(args.loss_v_window)
    if T > win:
        windows = []
        segment_start = 0
        for t in range(T):
            if t in boundaries:
                segment_start = t
            if t < win:
                continue
            window = torch.stack([
                velocities[i].detach() if i < segment_start else velocities[i]
                for i in range(t - win + 1, t + 1)
            ])
            windows.append(window.mean(0))
        mean_v = torch.stack(windows)
        delta = torch.norm(mean_v - target_velocities[win:], dim=-1)
        velocity_cost = F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none")
        window_valid = valid[win:].clone()
        window_valid[:-1] &= valid[:T - win - 1]
        costs[win:] += float(args.coef_v) * weighted(velocity_cost, window_valid)

    costs += float(args.coef_d_acc) * weighted(actions.square().sum(-1), valid)

    previous_action = torch.stack([
        rollout["act_buffer"][1] if t == 0 else
        (actions[t - 1].detach() if t in boundaries else actions[t - 1])
        for t in range(T)
    ])
    jerk = (actions - previous_action) * 15.0
    jerk_valid = valid.clone()
    jerk_valid[1:] &= valid[:-1]
    costs += float(args.coef_d_jerk) * weighted(jerk.square().sum(-1), jerk_valid)

    margin = env.margin
    dist_pre = torch.norm(pre_vec + 1e-6, dim=-1) - margin
    if rollout.get("post_clearance_history"):
        dist_post = torch.stack(rollout["post_clearance_history"]) - margin
    else:
        dist_post = torch.norm(post_vec + 1e-6, dim=-1) - margin
    approach = ((dist_pre - dist_post) * 135.0).clamp_min(1.0)
    if args.joint_backward != "consistent":
        approach = approach.detach()
    avoid = approach * (1.0 - dist_post).relu().clamp(max=5.0).square()
    collision = F.softplus(dist_post.clamp(min=-3.0) * -32.0) * approach
    costs += float(args.coef_obj_avoidance) * weighted(avoid, valid)
    costs += float(args.coef_collide) * weighted(collision, valid)

    previous = rollout["camera_initial"].to(camera)
    smooth_costs = []
    for t in range(T):
        if t in boundaries and t > 0:
            previous = camera[t - 1].detach()
        smooth_costs.append((camera[t] - previous).square().mean(-1))
        previous = camera[t]
    if include_camera_regularizer:
        costs += float(args.coef_cam_smooth) * weighted(torch.stack(smooth_costs), valid)

    if float(getattr(args, "coef_terminal_goal", 0.0)):
        terminal_dist = torch.norm(rollout["terminal_position"] - env.p_target, dim=-1)
        timeout_dist = torch.norm(env.p_target - post_p[-1], dim=-1)
        terminal_dist = torch.where(rollout["terminal_step"] >= 0, terminal_dist, timeout_dist)
        terminal_index = torch.where(
            rollout["terminal_step"] >= 0,
            rollout["terminal_step"],
            torch.full_like(rollout["terminal_step"], T - 1),
        )
        batch = torch.arange(B, device=costs.device)
        costs[terminal_index, batch] += float(args.coef_terminal_goal) * terminal_dist * T

    if rollout.get("guidance_history") and float(args.live_flight_guidance_weight) > 0:
        targets = torch.stack(rollout["guidance_history"])
        guidance_cost = F.smooth_l1_loss(actions, targets.detach(), reduction="none").mean(-1)
        supervised = valid.clone()
        supervised[:args.live_flight_guidance_start_step] = False
        costs += float(args.live_flight_guidance_weight) * weighted(guidance_cost, supervised)
    return costs


def discounted_cost_to_go(costs, gamma):
    returns = torch.zeros_like(costs)
    running = torch.zeros_like(costs[0])
    for t in range(costs.shape[0] - 1, -1, -1):
        running = costs[t] + float(gamma) * running
        returns[t] = running
    return returns


def short_horizon_actor_objective(rollout, costs, critic, args):
    """Use differentiable costs for H steps and critic value beyond H."""
    T, B = costs.shape
    horizon = int(args.policy_gradient_horizon)
    gamma = float(args.policy_return_gamma)
    boundaries = list(range(0, T, horizon)) + [T]
    hidden = rollout["hidden_history"]
    valid = torch.stack(rollout["valid_history"])
    objective = costs.new_zeros(())
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        discounts = costs.new_tensor(gamma).pow(torch.arange(end - start, device=costs.device))
        segment_cost = (costs[start:end] * discounts[:, None]).sum(0)
        if end < T:
            bootstrap = critic.actor_value(hidden[end]) * T * valid[end] * gamma ** (end - start)
            segment_cost = segment_cost + bootstrap
        objective = objective + segment_cost.mean()
    return objective / T


def short_horizon_critic_loss(rollout, costs, critic, args):
    # Critic predicts episode-normalized cost so its scale matches task loss.
    returns = discounted_cost_to_go(costs.detach(), args.policy_return_gamma) / costs.shape[0]
    hidden = torch.stack(rollout["hidden_history"])
    values = critic(hidden.detach())
    valid = torch.stack(rollout["valid_history"])
    weights = valid.to(values.dtype)
    error = F.smooth_l1_loss(values, returns, reduction="none")
    loss = (error * weights).sum() / weights.sum().clamp_min(1.0)
    return loss, returns, values.detach()


def camera_score_function_loss(rollout, returns, baseline, args):
    log_prob = torch.stack(rollout["camera_log_prob_history"])
    valid = torch.stack(rollout["valid_history"])
    advantage = returns - baseline
    return (log_prob * advantage.detach() * valid).sum(0).mean()

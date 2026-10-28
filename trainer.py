"""Full-BPTT trainer for grayscale task-driven camera control."""

from random import normalvariate
import json
from pathlib import Path
import time

import torch
from torch.cuda.amp import autocast
from torch.distributions import Normal
from tqdm import tqdm

from controllers.auto_exposure import (
    gradient_auto_exposure_target,
    mean_auto_exposure_target,
)
from losses import compute_camera_losses, compute_physics_losses, aggregate_loss
from rollout_ops import (
    render_gray_sensor,
    select_policy_gray_obs,
    build_local_frame,
    build_state_vector,
    compute_target_velocity,
    decode_action_direct,
    init_camera_params,
    update_camera_params,
    compute_camera_param_stats,
    policy_forward,
)
from train_utils import MetricSmoother, periodic_tail_ops
from render.scene_types import DIFFERENTIABLE_SCENES
from actor_critic import (
    camera_score_function_loss,
    discounted_cost_to_go,
    short_horizon_actor_objective,
    short_horizon_critic_loss,
    task_step_costs,
)


def _stack_or_none(values):
    if not values:
        return None
    return torch.stack(values)


def _nearest_vec_per_batch(vec, batch_size):
    """Reduce renderer collision vectors to one nearest vector per batch item."""
    if vec.ndim == 2 and vec.shape[0] == batch_size:
        return vec
    if vec.ndim == 2 and vec.shape[1] == batch_size:
        return vec.transpose(0, 1)
    if vec.ndim < 3:
        raise ValueError(f"unexpected collision-vector shape {tuple(vec.shape)}")
    candidates = vec.reshape(-1, batch_size, vec.shape[-1])
    norms = candidates.norm(2, -1)
    idx = norms.argmin(0)
    batch = torch.arange(batch_size, device=vec.device)
    return candidates[idx, batch]


def _compute_success_collision(p_history, clearance, env, args, rollout=None):
    if rollout is not None and "terminal_collision" in rollout:
        collision = rollout["terminal_collision"]
        reached = rollout["terminal_reached"]
        terminal_dist = torch.norm(env.p_target - rollout["terminal_position"], dim=-1)
        final_dist = torch.where(
            rollout["terminal_step"] >= 0,
            terminal_dist,
            torch.norm(env.p_target - p_history[-1], dim=-1),
        )
    else:
        collision = torch.any(clearance <= float(args.collision_clearance), dim=0)
        final_dist = torch.norm(env.p_target - p_history[-1], dim=-1)
        reached = final_dist < 0.35
    success = reached & (~collision)
    return success.float().mean(), collision.float().mean(), final_dist.mean()


def _build_loss_contrib_metrics(loss_terms, args):
    specs = [
        ("v", "loss_v", args.coef_v),
        ("obj_avoidance", "loss_obj_avoidance", args.coef_obj_avoidance),
        ("collide", "loss_collide", args.coef_collide),
        ("d_acc", "loss_d_acc", args.coef_d_acc),
        ("d_jerk", "loss_d_jerk", args.coef_d_jerk),
        ("cam_smooth", "loss_cam_smooth", args.coef_cam_smooth),
        ("terminal_goal", "loss_terminal_goal", getattr(args, "coef_terminal_goal", 0.0)),
        ("flight_replay", "loss_flight_replay", getattr(args, "flight_replay_weight", 0.0)),
        ("live_flight_guidance", "loss_live_flight_guidance", getattr(args, "live_flight_guidance_weight", 0.0)),
    ]
    contrib = {}
    for name, key, coef in specs:
        coef = float(coef)
        if abs(coef) <= 1e-12:
            continue
        value = loss_terms.get(key)
        if value is not None:
            contrib[name] = coef * float(value.detach())
    if not contrib:
        return {}
    total = max(sum(abs(v) for v in contrib.values()), 1e-12)
    out = {}
    for name, value in contrib.items():
        out[f"loss_contrib/{name}"] = value
        out[f"loss_share/{name}"] = abs(value) / total
    return out


def _camera_grad_norm(model):
    total = None
    for name, param in model.named_parameters():
        if not model._is_camera_parameter(name) or param.grad is None:
            continue
        term = param.grad.detach().pow(2).sum()
        total = term if total is None else total + term
    return 0.0 if total is None else float(total.sqrt().cpu())


def _flight_grad_norm(model):
    return sum(float(p.grad.detach().square().sum())
               for name, p in model.named_parameters()
               if p.grad is not None and not model._is_camera_parameter(name)) ** 0.5


def _detach_environment_state(env):
    for name in ("p", "v", "a", "act", "R", "R_old", "p_old", "dg", "v_wind"):
        value = getattr(env, name, None)
        if isinstance(value, torch.Tensor):
            setattr(env, name, value.detach())


def _detach_camera_timeline(env):
    if isinstance(getattr(env, "_camera_requested_state", None), torch.Tensor):
        env._camera_requested_state = env._camera_requested_state.detach()
    if hasattr(env, "_camera_command_queue"):
        env._camera_command_queue = [
            value.detach() if isinstance(value, torch.Tensor) else value
            for value in env._camera_command_queue
        ]


def _rollout(env, model, args, B, device, use_amp, vis, should_vis):
    guidance = None
    guidance_history = []
    if args.live_flight_guidance_weight > 0:
        from flight_guidance import LiveFlightGuidance
        guidance = LiveFlightGuidance(env, args)
    h = None
    cam_h = None
    score_h = None
    act_buffer = [env.act] * 2
    exposure, gain = init_camera_params(env, B, device)
    camera_initial = torch.stack([exposure.detach(), gain.detach()], -1)
    prev_gray = None

    p_history, v_history, target_v_history = [], [], []
    vec_history, act_history, cam_history = [], [], []
    exposure_history, gain_history = [], []
    saturation_history, dark_history, blur_history, light_history = [], [], [], []
    motion_proxy_history, char_depth_history = [], []
    post_p_history, post_vec_history, valid_history = [], [], []
    post_clearance_history = []
    angular_speed_history = []
    alive = torch.ones(B, dtype=torch.bool, device=device)
    terminal_collision = torch.zeros(B, dtype=torch.bool, device=device)
    terminal_reached = torch.zeros(B, dtype=torch.bool, device=device)
    terminal_step = torch.full((B,), -1, dtype=torch.long, device=device)
    terminal_position = torch.zeros(B, 3, dtype=env.p.dtype, device=device)
    exploration = torch.zeros(B, 3, dtype=env.p.dtype, device=device)
    hidden_history, camera_log_prob_history = [], []
    segment_boundaries = [0]
    short_horizon = args.policy_update_mode == "short_horizon_ac"
    camera_noise_std = float(args.camera_exploration_std)

    sensor_differentiable = args.sensor_grad_mode == "full"

    for t in range(args.timesteps):
        ctl_dt = normalvariate(
            1.0 / args.base_control_freq,
            getattr(args, "control_dt_jitter", 0.1) / args.base_control_freq,
        )

        gray_frame, gray_aux = render_gray_sensor(
            env,
            exposure,
            gain,
            differentiable=sensor_differentiable,
        )
        if prev_gray is None:
            prev_gray = gray_frame
        policy_gray = torch.cat([gray_frame, prev_gray], dim=1)
        policy_gray = select_policy_gray_obs(policy_gray, args.policy_gray_mode)
        prev_gray = gray_frame
        if "angular_speed_rad_s" in gray_aux:
            angular_speed_history.append(gray_aux["angular_speed_rad_s"])

        for key, history in (
            ("saturation_fraction", saturation_history),
            ("dark_fraction", dark_history),
            ("blur_strength", blur_history),
            ("light_scale", light_history),
            ("motion_proxy", motion_proxy_history),
            ("characteristic_depth", char_depth_history),
        ):
            value = gray_aux.get(key)
            if isinstance(value, torch.Tensor):
                value = value.flatten(1).mean(1) if value.ndim > 1 else value
                history.append(value)

        vec_now = _nearest_vec_per_batch(env.find_vec_to_nearest_pt(), B)
        target_position = env.p if args.joint_backward == "consistent" else env.p.detach()
        target_v_raw = env.p_target - target_position
        R = build_local_frame(env)
        target_v = compute_target_velocity(target_v_raw, env)
        state, _, camera_state, camera_motion_state = build_state_vector(
            env,
            target_v,
            R,
            exposure,
            gain,
            args.no_odom,
            args.include_camera_state_in_obs,
        )

        policy_extra = {}
        if args.flight_exploration_std > 0:
            rho = args.flight_exploration_rho
            innovation = torch.cat([torch.randn(B, 2, device=device), torch.zeros(B, 1, device=device)], -1)
            exploration = rho * exploration + (1 - rho**2)**.5 * args.flight_exploration_std * innovation
            policy_extra["action_noise"] = exploration
        boundary = short_horizon and t > 0 and t % args.policy_gradient_horizon == 0
        camera_noise = (
            torch.randn(B, 2, device=device, dtype=state.dtype) * camera_noise_std
            if short_horizon else None
        )
        if short_horizon:
            policy_extra.update(camera_logit_noise=camera_noise, return_camera_logits=True)
        previous_h, previous_cam_h = h, cam_h
        with autocast(enabled=use_amp):
            output = policy_forward(
                model, state, h, env,
                gray_obs=policy_gray,
                add_noise=False,
                cam_hx=cam_h,
                camera_state=camera_state,
                camera_motion_state=camera_motion_state,
                **policy_extra,
            )
        if short_horizon:
            _, _, h_connected, _, _, _ = output
            if boundary:
                hidden_history.append(h_connected)
                segment_boundaries.append(t)
                _detach_camera_timeline(env)
                act_buffer = [value.detach() for value in act_buffer]
                with autocast(enabled=use_amp):
                    output = policy_forward(
                        model,
                        state.detach(),
                        None if previous_h is None else previous_h.detach(),
                        env,
                        gray_obs=policy_gray.detach(),
                        add_noise=False,
                        cam_hx=None if previous_cam_h is None else previous_cam_h.detach(),
                        camera_state=camera_state.detach(),
                        camera_motion_state=camera_motion_state.detach(),
                        **policy_extra,
                    )
            act_raw, cam_params, h, cam_h, sampled_logits, mean_logits = output
            if not boundary:
                hidden_history.append(h)
            # Score-function derivatives hold realized observations and
            # commands fixed. Recurrent policy-weight derivatives remain.
            detached_env_timeline = getattr(env, "_camera_requested_state", None)
            detached_queue = getattr(env, "_camera_command_queue", None)
            _detach_camera_timeline(env)
            if boundary and score_h is not None:
                score_h = score_h.detach()
            with autocast(enabled=use_amp):
                score_output = policy_forward(
                    model, state.detach(), score_h, env,
                    gray_obs=policy_gray.detach(), add_noise=False,
                    camera_state=camera_state.detach(),
                    camera_motion_state=camera_motion_state.detach(),
                    return_camera_logits=True,
                )
            env._camera_requested_state = detached_env_timeline
            env._camera_command_queue = detached_queue
            score_h = score_output[2]
            score_mean_logits = score_output[-1]
            log_prob = Normal(score_mean_logits, camera_noise_std).log_prob(
                sampled_logits.detach()
            ).sum(-1)
            camera_log_prob_history.append(log_prob)
        else:
            act_raw, cam_params, h, cam_h = output

        action_frame = R.detach() if boundary else R
        if boundary:
            target_v = target_v.detach()
            target_v_raw = target_v_raw.detach()
            vec_now = vec_now.detach()
            exposure, gain = exposure.detach(), gain.detach()
        act = decode_action_direct(
            act_raw.float(),
            action_frame,
            env,
            B,
            args.max_acc_cmd,
        )

        if args.camera_control_mode == "mean_ae":
            cam_target = mean_auto_exposure_target(
                gray_frame.detach(),
                exposure.detach(),
                gain.detach(),
            )
        elif args.camera_control_mode == "gradient_ae":
            cam_target = gradient_auto_exposure_target(
                gray_frame.detach(),
                exposure.detach(),
                gain.detach(),
            )
        else:
            cam_target = cam_params.float()

        render_exposure, render_gain = exposure, gain
        exposure, gain, cam_hist_entry = update_camera_params(
            cam_target,
            exposure,
            gain,
            env,
        )
        act_buffer.append(act)
        if guidance is not None:
            guidance_history.append(guidance.target(env) if t >= guidance.start_step else torch.zeros_like(act).detach())

        valid_history.append(alive.clone())
        p_history.append(env.p)
        v_history.append(env.v)
        target_v_history.append(target_v)
        vec_history.append(vec_now)
        act_history.append(act)
        cam_history.append(cam_hist_entry)
        exposure_history.append(render_exposure)
        gain_history.append(render_gain)

        if (
            should_vis
            and args.vis_student
            and t % max(args.vis_every_steps, 1) == 0
        ):
            j = int(min(max(args.vis_env_idx, 0), B - 1))
            scalars = {
                "light_scale": float(gray_aux["light_scale"][j].detach().cpu()),
                "saturation_fraction": float(gray_aux["saturation_fraction"][j].detach().cpu()),
                "dark_fraction": float(gray_aux["dark_fraction"][j].detach().cpu()),
                "blur_strength": float(gray_aux["blur_strength"][j].detach().mean().cpu()),
            }
            vis.log_step(
                phase="student",
                step_idx=t,
                pos=env.p[j].detach().cpu().numpy(),
                target=env.p_target[j].detach().cpu().numpy(),
                main_img=gray_frame[j, 0].detach().cpu().numpy(),
                main_img_mode="luma",
                cam=(float(render_exposure[j].detach().cpu()), float(render_gain[j].detach().cpu())),
                scalars=scalars,
                drone_R=env.R[j].detach().cpu().numpy(),
                cam_R=env.R_cam[j].detach().cpu().numpy(),
                main_fov_half_tan=float(env._fov_x_half_tan),
                main_hw=(int(env.height), int(env.width)),
                depth_hw=(int(env.height), int(env.width)),
            )

        if boundary:
            p_history[-1] = p_history[-1].detach()
            v_history[-1] = v_history[-1].detach()
            _detach_environment_state(env)
            if prev_gray is not None:
                prev_gray = prev_gray.detach()
        env.run(act_buffer[t], ctl_dt, target_v_raw)

        # Terminal state belongs to the action just taken. Keep it in the
        # rollout so the final collision/goal is visible to both loss and
        # metrics, while masking all subsequent samples in this episode.
        post_vec = _nearest_vec_per_batch(env.find_vec_to_nearest_pt(), B)
        post_p_history.append(env.p)
        post_vec_history.append(post_vec)
        post_clearance = torch.norm(post_vec + 1e-6, 2, -1)
        if getattr(env, "scene_layout", "single_wall_slit") in DIFFERENTIABLE_SCENES:
            post_clearance = env.transition_clearance()
            post_clearance_history.append(post_clearance)
        collided = post_clearance <= float(args.collision_clearance)
        reached = torch.norm(env.p_target - env.p, dim=-1) < 0.35
        first_collision = alive & collided
        first_reached = alive & (~collided) & reached
        first_terminal = first_collision | first_reached
        terminal_collision |= first_collision
        terminal_reached |= first_reached
        terminal_step = torch.where(
            first_terminal, torch.full_like(terminal_step, t), terminal_step
        )
        terminal_position = torch.where(first_terminal[:, None], env.p, terminal_position)
        alive = alive & ~first_terminal

    return {
        "p_history": p_history,
        "post_p_history": post_p_history,
        "post_vec_history": post_vec_history,
        "post_clearance_history": post_clearance_history,
        "angular_speed_history": angular_speed_history,
        "valid_history": valid_history,
        "terminal_collision": terminal_collision,
        "terminal_reached": terminal_reached,
        "terminal_step": terminal_step,
        "terminal_position": terminal_position,
        "v_history": v_history,
        "target_v_history": target_v_history,
        "vec_history": vec_history,
        "act_history": act_history,
        "cam_history": cam_history,
        "camera_initial": camera_initial,
        "exposure_history": exposure_history,
        "gain_history": gain_history,
        "saturation_history": saturation_history,
        "dark_history": dark_history,
        "blur_history": blur_history,
        "light_history": light_history,
        "motion_proxy_history": motion_proxy_history,
        "char_depth_history": char_depth_history,
        "act_buffer": act_buffer,
        "guidance_history": guidance_history,
        "hidden_history": hidden_history,
        "camera_log_prob_history": camera_log_prob_history,
        "segment_boundaries": segment_boundaries,
    }


def _loss_from_rollout(rollout, env, args):
    p_history = torch.stack(rollout["post_p_history"])
    v_history = torch.stack(rollout["v_history"])
    target_v_history = torch.stack(rollout["target_v_history"])
    vec_history = torch.stack(rollout["vec_history"])
    post_vec_history = torch.stack(rollout["post_vec_history"])
    act_history = torch.stack(rollout["act_history"])

    physics_losses = compute_physics_losses(
        v_history,
        target_v_history,
        act_history,
        vec_history,
        p_history,
        env.margin,
        rollout["act_buffer"][1],
        win=args.loss_v_window,
        valid_mask=torch.stack(rollout["valid_history"]),
        post_vec_chunk=post_vec_history,
        post_clearance_chunk=_stack_or_none(rollout.get("post_clearance_history")),
        differentiate_approach_weight=args.joint_backward == "consistent",
    )
    camera_losses = compute_camera_losses(
        _stack_or_none(rollout["cam_history"]),
        cam_initial=rollout["camera_initial"],
        valid_mask=torch.stack(rollout["valid_history"]),
    )
    loss, terms = aggregate_loss(physics_losses, camera_losses, args)
    if args.live_flight_guidance_weight > 0:
        from flight_guidance import live_flight_guidance_loss
        guidance_loss = live_flight_guidance_loss(
            act_history, torch.stack(rollout["guidance_history"]),
            torch.stack(rollout["valid_history"]), args.live_flight_guidance_start_step)
        loss = loss + args.live_flight_guidance_weight * guidance_loss
        terms["loss_live_flight_guidance"] = guidance_loss
    if float(getattr(args, "coef_terminal_goal", 0.0)):
        terminal = torch.where((rollout["terminal_step"] >= 0)[:, None],
                               rollout["terminal_position"], p_history[-1])
        goal_loss = (terminal - env.p_target).norm(dim=-1).mean()
        loss = loss + args.coef_terminal_goal * goal_loss
        terms["loss_terminal_goal"] = goal_loss
    clearance = torch.norm(post_vec_history + 1e-6, 2, -1)
    if rollout.get("post_clearance_history"):
        clearance = torch.stack(rollout["post_clearance_history"])
    return loss, terms, p_history, clearance


def train(args, model, env_train, env_full, optim, sched, scaler, vis, checkpoint_dir, device,
          critic=None):
    _ = env_full
    use_amp = bool(args.amp and device.type == "cuda")
    smoother = MetricSmoother(args)
    replay = None
    if args.flight_replay_dataset:
        from flight_replay import FlightReplayAnchor
        replay = FlightReplayAnchor(model, args, device, checkpoint_dir)
    pbar = tqdm(range(args.num_iters), ncols=80)

    for i in pbar:
        tic = time.time()
        env_train.reset()
        model.reset()
        B = env_train.batch_size
        should_vis = bool(
            args.vis_enable and i % max(args.vis_every_iters, 1) == 0
        )
        if should_vis:
            vis.begin_iter(i)
            j = int(min(max(args.vis_env_idx, 0), B - 1))
            vis.log_environment(
                phase="student",
                balls=env_train.get_world_balls_for_env(j),
                voxels=env_train.get_world_voxels_for_env(j),
                cyl=env_train.get_world_cyl_for_env(j),
                cyl_h=env_train.get_world_cyl_h_for_env(j),
                start=env_train.p[j].detach().cpu().numpy(),
                target=env_train.p_target[j].detach().cpu().numpy(),
                scene_name=env_train.current_scene_names[j],
                scene_effects=env_train.get_scene_effects_for_env(j),
                scene_yaw=env_train.get_scene_yaw_for_env(j),
            )

        rollout = _rollout(
            env_train,
            model,
            args,
            B,
            device,
            use_amp,
            vis,
            should_vis,
        )
        loss, loss_terms, p_history, clearance = _loss_from_rollout(
            rollout,
            env_train,
            args,
        )
        if args.policy_update_mode == "short_horizon_ac":
            if critic is None:
                raise RuntimeError("short_horizon_ac requires a critic")
            costs = task_step_costs(rollout, env_train, args)
            actor_loss = short_horizon_actor_objective(rollout, costs, critic, args)
            critic_loss, returns, baseline = short_horizon_critic_loss(
                rollout, costs, critic, args)
            if args.sensor_grad_mode != "full":
                nav_costs = task_step_costs(rollout, env_train, args, include_camera_regularizer=False)
                nav_returns = discounted_cost_to_go(nav_costs.detach(), args.policy_return_gamma) / args.timesteps
                actor_loss = actor_loss + camera_score_function_loss(
                    rollout, nav_returns, baseline, args)
            task_loss = loss.detach()
            loss = actor_loss + 0.5 * critic_loss
            loss_terms["loss_actor"] = actor_loss
            loss_terms["loss_critic"] = critic_loss
        if replay is not None:
            anchor = replay.loss(model)
            loss_terms["loss_flight_replay"] = anchor
            loss = loss + args.flight_replay_weight * anchor
            if (i + 1) % 25 == 0 or i + 1 == args.num_iters:
                replay.save()

        optim.zero_grad(set_to_none=True)
        cam_grad_norm = 0.0
        flight_grad_norm = 0.0
        if torch.isfinite(loss) and loss.requires_grad:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                cam_grad_norm = _camera_grad_norm(model)
                flight_grad_norm = _flight_grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                if critic is not None:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=5.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                cam_grad_norm = _camera_grad_norm(model)
                flight_grad_norm = _flight_grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0,
                                              error_if_nonfinite=args.policy_arch == "intent_shared")
                if critic is not None:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=5.0, error_if_nonfinite=True)
                optim.step()
            sched.step()
        elif not torch.isfinite(loss):
            print(f"[warn] non-finite loss at iter {i}; optimizer step skipped")
        if device.type == "cuda":
            torch.cuda.synchronize()

        success_rate, collision_rate, final_goal_dist = _compute_success_collision(
            p_history.detach(),
            clearance.detach(),
            env_train,
            args,
            rollout=rollout,
        )
        cam_stats = compute_camera_param_stats(
            rollout["exposure_history"],
            rollout["gain_history"],
        )
        dt = time.time() - tic
        log = {
            "loss": float(loss.detach()),
            "collision_rate": float(collision_rate),
            "success_rate": float(success_rate),
            "charts/goal_dist": float(final_goal_dist),
            "iter_per_sec": 1.0 / max(dt, 1e-6),
            "sim_fps": args.timesteps * B / max(dt, 1e-6),
            "cam/grad_norm": cam_grad_norm,
            "flight/grad_norm": flight_grad_norm,
        }
        if device.type == "cuda":
            # Keep one resource metric for both full-BPTT and short-horizon
            # runs.  The AC-specific field below is retained for compatibility
            # with earlier pilot manifests.
            log["runtime/peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
        if args.policy_update_mode == "short_horizon_ac":
            log.update({"ac/task_loss": float(task_loss),
                        "ac/actor_loss": float(loss_terms["loss_actor"].detach()),
                        "ac/critic_loss": float(loss_terms["loss_critic"].detach())})
            if device.type == "cuda":
                log["ac/peak_cuda_memory_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
        log.update(_build_loss_contrib_metrics(loss_terms, args))
        log.update({f"cam/{k}": v for k, v in cam_stats.items()})
        if rollout["angular_speed_history"]:
            log["flight/angular_speed_rad_s"] = float(torch.stack(rollout["angular_speed_history"]).detach().mean())
        for key, history in (
            ("saturation_fraction", rollout["saturation_history"]),
            ("dark_fraction", rollout["dark_history"]),
            ("blur_strength", rollout["blur_history"]),
            ("light_scale", rollout["light_history"]),
            ("motion_proxy", rollout["motion_proxy_history"]),
            ("characteristic_depth", rollout["char_depth_history"]),
        ):
            if history:
                log[f"cam/{key}"] = float(torch.stack(history).detach().mean())

        smoother.add(log)
        with (Path(checkpoint_dir) / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(iteration=i + 1, **log)) + "\n")
        if args.vis_enable:
            vis.log_train_scalars(
                {k: v for k, v in log.items() if k in {
                    "loss", "collision_rate", "success_rate", "charts/goal_dist"
                }},
                iter_idx=i,
            )
        periodic_tail_ops(i, checkpoint_dir, model, smoother, args.checkpoint_every_iters)
        pbar.set_description_str(f"loss: {float(loss.detach()):.3f}")
    smoother.flush(args.num_iters)
    torch.save(model.state_dict(), str(Path(checkpoint_dir) / "final.pth"))

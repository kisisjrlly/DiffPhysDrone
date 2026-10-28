"""Estimator, truncation and terminal semantics of the opt-in hybrid pilot."""

import copy
from types import SimpleNamespace

import torch
import pytest

from actor_critic import (LatentCostCritic, camera_score_function_loss,
                          discounted_cost_to_go, short_horizon_actor_objective,
                          short_horizon_critic_loss, task_step_costs)
from config import set_global_seed
from intent_model import build_policy
from tools.intent_common import load_intent_args
from train_utils import build_env
from trainer import _loss_from_rollout, _rollout


def args_for_test(**updates):
    values = dict(batch_size=2, timesteps=10, loss_v_window=3, gray_width=16,
                  gray_height=12, gray_nn_width=16, gray_nn_height=12,
                  gray_enable_noise=False, policy_update_mode="short_horizon_ac",
                  policy_gradient_horizon=4, flight_replay_dataset=None,
                  flight_replay_weight=0, control_dt_jitter=0)
    values.update(updates)
    return load_intent_args("configs/gray_joint_parallax_lookahead.args", **values)


def rollout_for_test(args, model=None):
    set_global_seed(5102)
    env = build_env(args.batch_size, args, torch.device("cpu"))
    env.reset("dark")
    model = build_policy(args) if model is None else model
    return env, model, _rollout(env, model, args, args.batch_size, torch.device("cpu"), False, None, False)


def test_default_bptt_still_accepts_short_rollouts():
    args_for_test(policy_update_mode="full_bptt", timesteps=1, loss_v_window=1)


def test_new_ac_config_matches_common_warm_start_scene_and_replay():
    from tools.train_pole_navigation_imitation import dataset_configuration
    configs = [load_intent_args(path) for path in (
        "configs/gray_joint_parallax_ac.args", "configs/gray_joint_parallax_consistent.args")]
    for args in configs:
        args.imitation_initial_probe_steps, args.imitation_clearance_margin = 24, .08
    assert dataset_configuration(configs[0]) == dataset_configuration(configs[1])


def test_replay_configuration_mismatch_is_caught_without_training():
    from flight_replay import FlightReplayAnchor
    args = args_for_test()
    with pytest.raises(ValueError, match="gray_texture_scale"):
        FlightReplayAnchor.validate_configuration(
            {"dataset_configuration": {"gray_texture_scale": args.gray_texture_scale + 1}}, args)


def test_live_guidance_excludes_initial_measurement_and_terminal_suffix():
    from flight_guidance import live_flight_guidance_loss
    actions = torch.ones(8, 2, 3, requires_grad=True)
    targets = torch.zeros_like(actions, requires_grad=True)
    valid = torch.ones(8, 2, dtype=torch.bool)
    valid[6:, 0] = False
    loss = live_flight_guidance_loss(actions, targets, valid, 4)
    loss.backward()
    assert not actions.grad[:4].any() and not actions.grad[6:, 0].any()
    assert actions.grad[4:6].abs().sum() > 0
    assert targets.grad is None


def test_sampled_camera_logits_preserve_inference_and_mean():
    model = build_policy(args_for_test())
    state, images = torch.randn(2, 10), torch.rand(2, 2, 12, 16)
    before = model(state, gray_obs=images)
    noise = torch.tensor([[.2, -.1], [.3, .4]])
    sampled = model(state, gray_obs=images, camera_logit_noise=noise, return_camera_logits=True)
    torch.testing.assert_close(sampled[4], sampled[5] + noise)
    torch.testing.assert_close(before[1], sampled[5].sigmoid())
    torch.testing.assert_close(before[0], sampled[0])
    assert sum(p.numel() for p in model.parameters()) == 200605


def test_step_costs_reconstruct_masked_original_loss_and_terminal_cost():
    args = args_for_test(coef_terminal_goal=1., live_flight_guidance_weight=2.,
                         live_flight_guidance_start_step=4, pole_layout="three_poles")
    env, _, rollout = rollout_for_test(args)
    # Cover nonuniform terminal masks without changing stored physical values.
    for t in range(6, args.timesteps):
        rollout["valid_history"][t][0] = False
    rollout["terminal_step"][0] = 5
    rollout["terminal_position"][0] = rollout["post_p_history"][5][0]
    loss, _, _, _ = _loss_from_rollout(rollout, env, args)
    costs = task_step_costs(rollout, env, args)
    torch.testing.assert_close(costs.mean(), loss, rtol=2e-5, atol=2e-5)
    assert not costs[6:, 0].any()


def test_critic_bootstrap_masks_terminal_suffix_and_keeps_latent_gradient():
    critic = LatentCostCritic(latent_dim=3, hidden_dim=4)
    latent = torch.ones(2, 3, requires_grad=True)
    # Make a deterministic positive latent derivative.
    for p in critic.parameters():
        torch.nn.init.constant_(p, .1)
    rollout = {"hidden_history": [latent] * 4,
               "valid_history": [torch.tensor([True, True])] * 2 + [torch.tensor([True, False])] * 2}
    args = SimpleNamespace(policy_gradient_horizon=2, policy_return_gamma=1.)
    objective = short_horizon_actor_objective(rollout, torch.zeros(4, 2), critic, args)
    gradient, = torch.autograd.grad(objective, latent)
    assert gradient[0].abs().sum() > 0
    assert not gradient[1].any()
    assert all(p.grad is None for p in critic.parameters())


def test_cost_minimizing_score_estimator_sign_and_discount():
    logits = torch.tensor([0., 0.], requires_grad=True)
    # A high-cost positive sample should lower its mean; negative is low-cost.
    sample = torch.tensor([1., -1.])
    log_prob = torch.distributions.Normal(logits, 1).log_prob(sample)
    rollout = {"camera_log_prob_history": [log_prob], "valid_history": [torch.ones(2, dtype=torch.bool)]}
    loss = camera_score_function_loss(rollout, torch.tensor([[2., 0.]]), torch.ones(1, 2), None)
    loss.backward()
    assert (logits.grad > 0).all()
    torch.testing.assert_close(discounted_cost_to_go(torch.tensor([[1.], [2.], [0.]]), .5),
                               torch.tensor([[2.], [2.], [0.]]))


def test_matched_rollouts_forward_parity_and_both_camera_estimators_learn():
    args = args_for_test(coef_cam_smooth=0.)
    set_global_seed(5101)
    template = build_policy(args)
    trajectories = []
    for mode in ("full", "detached"):
        args.sensor_grad_mode = mode
        args.observation_grad_mode = mode
        env, model, rollout = rollout_for_test(args, copy.deepcopy(template))
        trajectories.append(torch.stack(rollout["post_p_history"]).detach())
        assert len(rollout["hidden_history"]) == args.timesteps
        assert rollout["segment_boundaries"] == [0, 4, 8]
        costs = task_step_costs(rollout, env, args)
        critic = LatentCostCritic()
        actor = short_horizon_actor_objective(rollout, costs, critic, args)
        value_loss, returns, baseline = short_horizon_critic_loss(rollout, costs, critic, args)
        pathwise, = torch.autograd.grad(actor, model.fc_cam.weight, allow_unused=True, retain_graph=True)
        if mode == "full":
            assert pathwise is not None and pathwise.norm() > 0
        else:
            assert pathwise is None or not pathwise.any()
            actor = actor + camera_score_function_loss(rollout, returns, baseline, args)
        (actor + .5 * value_loss).backward()
        for head in (model.fc, model.fc_cam):
            assert torch.isfinite(head.weight.grad).all() and head.weight.grad.norm() > 0
        assert any(p.grad is not None and p.grad.norm() > 0 for p in critic.parameters())
    torch.testing.assert_close(*trajectories, rtol=0, atol=0)


def test_later_segment_stops_gradients_through_motion_camera_and_memory():
    args = args_for_test(timesteps=14)
    _, _, rollout = rollout_for_test(args)
    for later in (rollout["post_p_history"][11].square().sum(),
                  rollout["hidden_history"][10].square().sum(),
                  rollout["cam_history"][10].square().sum()):
        assert later.requires_grad
        g, = torch.autograd.grad(later, rollout["act_history"][0], allow_unused=True)
        assert g is None or not g.any()


def test_score_function_treats_realized_observations_as_constants():
    args = args_for_test()
    _, model, rollout = rollout_for_test(args)
    log_prob = rollout["camera_log_prob_history"][3].sum()
    g, = torch.autograd.grad(log_prob, rollout["act_history"][0], allow_unused=True, retain_graph=True)
    assert g is None or not g.any()
    mean_gradient, = torch.autograd.grad(log_prob, model.fc_cam.weight)
    assert mean_gradient.norm() > 0

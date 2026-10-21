"""A scene-design check, not evidence of a trained active-sensing policy."""

import torch
import pytest

from rollout_ops import build_local_frame, build_state_vector, compute_target_velocity
from tools.audit_parallax_candidate import candidate_env, feasibility_audit, image_audit
from tools.intent_common import load_intent_args
from train_utils import build_env
from rollout_ops import render_gray_sensor
from tools.evaluate_pole_coordination import frozen_view_renderer
from tools.evaluate_pole_coordination import initial_probe_override
from intent_model import build_policy
from eval import run_one_episode


@pytest.mark.parametrize("config", ["configs/gray_joint_parallax.args", "configs/gray_joint_parallax_lookahead.args"])
def test_initial_images_are_ambiguous_but_a_new_view_reveals_depth_order(config):
    stats, _ = image_audit(config)
    assert stats["initial"]["policy_image_max"] < 1e-4
    assert stats["sideways_12cm"]["policy_image_mae"] > .01
    env, args = candidate_env(config=config)
    state = build_state_vector(env, compute_target_velocity(env.p_target - env.p, env),
                               build_local_frame(env), torch.full((2,), .08),
                               torch.full((2,), .05), args.no_odom, False)[0]
    torch.testing.assert_close(state[0], state[1], atol=0, rtol=0)
    # The shared near-pole plane has no lateral position clearing both worlds.
    ys = torch.linspace(-.38 + env.drone_radius, .38 - env.drone_radius, 501)
    gap = (ys[None] - env.cyl[:, :1, 1]).abs() - env.cyl[:, :1, 2] - env.drone_radius
    assert not (gap > args.collision_clearance).all(0).any()


def test_both_depth_orders_have_executed_bounded_collision_free_routes():
    report, _ = feasibility_audit(4)
    assert report["successes"] == 4
    assert report["collisions"] == 0
    assert report["min_clearance_m"] > 0
    assert report["max_reference_duration_s"] < 8


def test_longer_lookahead_leaves_time_to_measure_then_complete_actual_route():
    report, _ = feasibility_audit(4, "configs/gray_joint_parallax_lookahead.args", 24, .08)
    assert report["successes"] == 4
    assert report["collisions"] == 0
    assert report["min_clearance_m"] > 0
    assert report["max_reference_duration_s"] + 24 / 15 < report["deadline_s"]


def test_selectable_parallax_layout_preserves_initial_optical_ambiguity():
    args = load_intent_args("configs/gray_joint_parallax.args", batch_size=2,
                            pole_pair_depth_jitter=0., gray_light_jitter=0., gray_enable_noise=False)
    env = build_env(2, args, torch.device("cpu"))
    env.reset("nominal")
    assert torch.allclose(env.cyl[:, :, 0] - args.simple_start_x, torch.tensor([[1., 1.8]]))
    torch.testing.assert_close(env.cyl[:, :, 2], torch.tensor([[.06, .108]]).expand(2, -1))
    env.cyl[1] = env.cyl[0].clone()
    env.cyl[1, :, 1] *= -1
    e, g = torch.full((2,), .08), torch.full((2,), .05)
    initial, _ = render_gray_sensor(env, e, g)
    torch.testing.assert_close(initial[0], initial[1], atol=1e-4, rtol=0)
    env.p[:, 1] += .12
    moved, _ = render_gray_sensor(env, e, g)
    assert (moved[0] - moved[1]).abs().mean() > .01


@pytest.mark.parametrize("overrides", [{"gray_texture_strength": .5}, {"cam_angle": 5},
                                        {"scenarios": ["dark_to_bright"]}, {"pole_arena_half_width": .85}])
def test_parallax_scene_rejects_broken_ambiguity_or_blind_bypass(overrides):
    with pytest.raises(ValueError, match="parallax"):
        load_intent_args("configs/gray_joint_parallax.args", **overrides)


def test_frozen_view_intervention_removes_motion_information_without_moving_vehicle():
    env, _ = candidate_env()
    e, g = torch.full((2,), .08), torch.full((2,), .05)
    renderer = frozen_view_renderer()
    a, _ = renderer(env, e, g)
    env.p[:, 1] += .12
    actual_position = env.p.clone()
    b, _ = renderer(env, e, g)
    normal, _ = render_gray_sensor(env, e, g)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert (normal - b).abs().mean() > .01
    torch.testing.assert_close(env.p, actual_position, rtol=0, atol=0)


def test_external_probe_trace_keeps_proposed_and_executed_actions_distinct():
    args = load_intent_args("configs/gray_joint_parallax.args", timesteps=4,
                            loss_v_window=1, batch_size=1, camera_control_mode="fixed")
    model = build_policy(args).eval()
    with torch.no_grad():
        model.fc.weight.zero_(); model.fc.bias.copy_(torch.tensor([.3, .4, 0.]))
        env = build_env(1, args, torch.device("cpu"))
        _, trace = run_one_episode(0, "nominal", args, model, env, None, torch.device("cpu"),
                                   collect_trace=True, episode_seed=789000,
                                   action_override=initial_probe_override(args, .12, steps=2))
    assert [r["external_action_override"] for r in trace] == [True, True, False, False]
    assert abs(trace[0]["acc_cmd_forward"] - trace[0]["policy_acc_cmd_forward"]) > .1
    assert trace[2]["acc_cmd_forward"] == trace[2]["policy_acc_cmd_forward"]

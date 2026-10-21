"""Causal/physical checks for the new method, not performance claims."""

import copy
import shlex
from pathlib import Path

import pytest
import torch

from config import build_parser, parse_scenarios, set_global_seed, validate_args
from intent_model import build_policy
from render.box_scene import box_clearance, swept_box_collision, trace_boxes
from render.intent_scene import render_exposure
from render.pose import dynamics_step, rotation_exp, update_attitude
from rollout_ops import build_local_frame, build_state_vector, compute_target_velocity, render_gray_sensor
from train_utils import build_env
from trainer import _loss_from_rollout, _rollout


def intent_args(**overrides):
    args = build_parser().parse_args(shlex.split(Path("configs/gray_joint_intent.args").read_text(), comments=True))
    args.scenarios = parse_scenarios(args.scenarios)
    values = dict(batch_size=1, gray_width=20, gray_height=16, gray_nn_width=16,
                  gray_nn_height=12, timesteps=12, loss_v_window=3, gray_enable_noise=False)
    values.update(overrides)
    for key, value in values.items():
        setattr(args, key, value)
    validate_args(args)
    return args


def test_box_renderer_depth_and_pose_derivatives():
    boxes = torch.tensor([[[2., 0., 0., .1, 5., 5.]]], dtype=torch.double)
    position = torch.tensor([[.01, .02, .03]], dtype=torch.double, requires_grad=True)
    angles = torch.tensor([[.01, .02, .03]], dtype=torch.double, requires_grad=True)
    def render(p, a):
        return trace_boxes(boxes, rotation_exp(a), p, 3, 4, .3)[0]
    assert torch.autograd.gradcheck(render, (position, angles), eps=1e-6, atol=1e-5)
    depth = trace_boxes(boxes, torch.eye(3, dtype=torch.double)[None], position, 3, 4, .3)[0]
    torch.testing.assert_close(depth, torch.full_like(depth, 1.89))


def test_sphere_clearance_and_continuous_wall_collision():
    boxes = torch.tensor([[[0., 0., 0., .1, 2., 2.]]])
    start, end = torch.tensor([[-1., 0., 0.]]), torch.tensor([[1., 0., 0.]])
    assert (box_clearance(torch.stack([start, end]), boxes, .12) > 0).all()
    assert swept_box_collision(start, end, boxes, .12).item()
    assert not swept_box_collision(start + torch.tensor([0., 3., 0.]), end + torch.tensor([0., 3., 0.]), boxes, .12).item()


def test_zero_rotation_has_finite_gradients():
    vector = torch.zeros(1, 3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(rotation_exp, (vector,))


def test_intent_changes_camera_without_changing_current_flight_action():
    args = intent_args()
    model = build_policy(args)
    state, image = torch.randn(2, 10), torch.rand(2, 2, 16, 20)
    first = model(state, gray_obs=image, intent_override=torch.zeros(2, 6))
    second = model(state, gray_obs=image, intent_override=torch.ones(2, 6))
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    assert not torch.allclose(first[1], second[1], rtol=1e-6, atol=1e-7)
    # Flight head cannot receive E/G as a hidden control message.
    third = model(state, gray_obs=image, intent_override=torch.zeros(2, 6), camera_state=torch.ones(2, 2))
    torch.testing.assert_close(first[0], third[0], rtol=0, atol=0)


def test_no_intent_ablation_has_identical_parameters_and_ignores_override():
    args = intent_args()
    model = build_policy(args)
    control = copy.deepcopy(model)
    control.intent_enabled = False
    for a, b in zip(model.parameters(), control.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    state, image = torch.zeros(1, 10), torch.rand(1, 2, 16, 20)
    a = control(state, gray_obs=image, intent_override=torch.zeros(1, 6))
    b = control(state, gray_obs=image, intent_override=torch.ones(1, 6))
    torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)


def test_training_noise_is_in_actual_flight_intent_and_not_extra_actor_capacity():
    args = intent_args()
    model = build_policy(args)
    state, image = torch.zeros(2, 10), torch.rand(2, 2, 16, 20)
    original = model(state, gray_obs=image)
    noise = torch.tensor([[.2, -.3, 0.], [-.1, .25, 0.]])
    sampled = model(state, gray_obs=image, action_noise=noise)
    torch.testing.assert_close(sampled[0], (original[0] + noise).clamp(-args.max_acc_cmd, args.max_acc_cmd))
    intent = torch.cat([sampled[0] / args.max_acc_cmd,
                        sampled[0] * args.intent_horizon_s / args.flight_max_speed], -1)
    expected = model(state, gray_obs=image, intent_override=intent)
    torch.testing.assert_close(expected[1], sampled[1], rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) == 200605
    # Ordinary inference has no sampling and is unchanged by the training flag.
    torch.testing.assert_close(model(state, gray_obs=image)[0], original[0], rtol=0, atol=0)


def test_gate_geometry_is_visual_only_and_goal_does_not_encode_openings():
    args = intent_args(batch_size=32)
    env = build_env(32, args, torch.device("cpu"))
    env.reset("dark_to_bright")
    assert (env.gate_centers_y.abs() >= .5).all()
    assert (env.gate_centers_y[:, 0].sign() != env.gate_centers_y[:, 1].sign()).any()
    target = compute_target_velocity(env.p_target - env.p, env)
    state, _, _, _ = build_state_vector(env, target, build_local_frame(env), torch.zeros(32), torch.zeros(32), False, False)
    torch.testing.assert_close(state, state[:1].expand_as(state))
    assert env.signed_clearance().min() > .2
    # Same goal, either lateral side can be required. Straight center cannot
    # bypass the randomized wall by using the privileged target direction.
    env.p = torch.tensor([[0., 0., 1.5]]).expand(32, -1)
    assert (env.signed_clearance() < 0).all()


def test_lighting_is_attached_to_world_surfaces():
    args = intent_args()
    env = build_env(1, args, torch.device("cpu"))
    env.reset("dark_to_bright")
    first = env.surface_lights.clone()
    env.p = env.p + torch.tensor([[.4, .2, 0.]])
    torch.testing.assert_close(env.surface_lights, first, rtol=0, atol=0)
    assert env.gate_lights[0, 0] == args.gray_dark_scale
    assert env.gate_lights[0, 1] == args.gray_bright_scale


@pytest.mark.parametrize("first,second", [(-.75, -.75), (-.75, .75), (.75, -.75), (.75, .75)])
def test_extreme_gate_pairs_admit_collision_free_bounded_kinematic_path(first, second):
    from render.intent_scene import build_intent_gates
    from tools.audit_intent_scene import executed_dynamics_audit, feasible_path
    args = intent_args(timesteps=120)
    env = build_env(1, args, torch.device("cpu"))
    env.reset("dark_to_bright")
    build_intent_gates(env, centers=torch.tensor([[first, second]]), half_widths=torch.full((1, 2), .36))
    _, report = feasible_path(env)
    assert report["min_spherical_clearance_m"] > 0
    assert not report["conservative_swept_collision"]
    assert report["max_acceleration_component_m_s2"] < 3.5
    assert report["duration_s"] < 8
    executed = executed_dynamics_audit(env, args)
    assert executed["success"] == [True]
    assert executed["collision"] == [False]


def test_new_scene_is_rejected_if_camera_timing_or_geometry_is_invalid():
    with pytest.raises(ValueError, match="temporal shutter"):
        intent_args(gray_temporal_samples=1)
    with pytest.raises(ValueError, match="both intent gates"):
        intent_args(intent_gate_spacing=20)
    with pytest.raises(ValueError, match="half widths"):
        intent_args(simple_slit_half_y_max=.6)


def test_small_training_batches_sample_all_lighting_cases():
    args = intent_args(batch_size=1)
    env = build_env(1, args, torch.device("cpu"))
    observed = {env._choose_scene_names(1)[0] for _ in range(100)}
    assert observed == set(args.scenarios)


def test_observation_gradient_cut_preserves_forward_pixels():
    args = intent_args()
    env = build_env(1, args, torch.device("cpu"))
    env.reset("dark_to_bright")
    env.p = torch.tensor([[-.65, .1, 1.5]], requires_grad=True)
    env.v = torch.tensor([[1., .7, .1]], requires_grad=True)
    env.R_old = rotation_exp(torch.tensor([[0., 0., -.15]]))
    exposure = torch.tensor([.5], requires_grad=True)
    gain = torch.tensor([.03], requires_grad=True)
    full, _ = render_gray_sensor(env, exposure, gain)
    grads = torch.autograd.grad(full.square().mean(), (env.p, env.v, exposure), retain_graph=True)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    env.observation_grad_mode = "detached"
    detached, _ = render_gray_sensor(env, exposure, gain)
    torch.testing.assert_close(full, detached, rtol=0, atol=0)
    grads_cut = torch.autograd.grad(detached.square().mean(), (env.p, env.v, exposure), allow_unused=True)
    assert grads_cut[0] is None and grads_cut[1] is None
    assert grads_cut[2].abs().sum() > 0


def test_temporal_integration_is_identity_for_static_camera():
    env = build_env(1, intent_args(), torch.device("cpu"))
    env.reset("dark_to_bright")
    image, aux = render_exposure(env, torch.tensor([.8]), return_aux=True)
    env.gray_temporal_samples = 1
    sharp, _ = render_exposure(env, torch.tensor([.8]), return_aux=True)
    torch.testing.assert_close(image, sharp, atol=1e-6, rtol=1e-6)
    assert aux["temporal_blur_mae"].max() < 1e-6


def test_temporal_integration_velocity_and_exposure_match_finite_difference():
    env = build_env(1, intent_args(gray_width=8, gray_height=6), torch.device("cpu"))
    env.reset("dark")
    # A single textured plane keeps visibility fixed: this checks actual
    # integration derivatives, rather than a surrogate silhouette derivative.
    env.voxels = torch.tensor([[[2., 0., 1.5, .1, 10., 10.]]], dtype=torch.double)
    env.surface_lights = torch.tensor([[.2]], dtype=torch.double)
    env.p = torch.tensor([[-.8, .1, 1.5]], dtype=torch.double)
    env.R = torch.eye(3, dtype=torch.double)[None]
    env.R_old = rotation_exp(torch.tensor([[.02, -.01, .03]], dtype=torch.double))
    env.R_cam = torch.eye(3, dtype=torch.double)[None]
    env.R_scene_T = env.R_scene_T.double()
    velocity = torch.tensor([[.2, .4, .1]], dtype=torch.double, requires_grad=True)
    exposure = torch.tensor([.6], dtype=torch.double, requires_grad=True)
    weights = torch.linspace(.1, 1, 48, dtype=torch.double).reshape(1, 1, 6, 8)
    def measure(v, e):
        env.v = v
        return (render_exposure(env, e)[0] * weights).mean()
    assert torch.autograd.gradcheck(measure, (velocity, exposure), eps=1e-5, atol=1e-6, rtol=1e-3)


def test_camera_timing_information_does_not_bypass_pixels_into_flight():
    model = build_policy(intent_args())
    state, image = torch.zeros(1, 10), torch.rand(1, 2, 16, 20)
    a = model(state, gray_obs=image, camera_timeline=torch.zeros(1, 6))
    b = model(state, gray_obs=image, camera_timeline=torch.ones(1, 6))
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    assert not torch.allclose(a[1], b[1], atol=1e-7, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reference renderer")
def test_box_renderer_matches_existing_cuda_geometry():
    import quadsim_cuda
    args = intent_args(batch_size=3)
    env = build_env(3, args, torch.device("cuda"))
    env.reset("dark_to_bright")
    rotation = env.R_scene_T @ env.R @ env.R_cam
    position = (env.R_scene_T @ env.p[:, :, None]).squeeze(-1).contiguous()
    actual_depth, actual_normals, _ = trace_boxes(env.voxels, rotation, position, env.height, env.width, env._fov_x_half_tan)
    depth, normals = torch.empty_like(actual_depth), torch.empty_like(actual_normals)
    quadsim_cuda.render_geometry(depth, normals, env.balls, env.cyl, env.cyl_h, env.voxels,
                                rotation.contiguous(), position, 1, env._fov_x_half_tan)
    torch.testing.assert_close(actual_depth, depth, atol=3e-5, rtol=2e-5)
    torch.testing.assert_close(actual_normals, normals, atol=1e-6, rtol=0)


def test_joint_rollout_updates_both_heads_from_navigation_loss():
    args = intent_args()
    set_global_seed(19)
    env = build_env(1, args, torch.device("cpu"))
    env.reset("dark_to_bright")
    model = build_policy(args)
    rollout = _rollout(env, model, args, 1, torch.device("cpu"), False, None, False)
    loss, _, _, _ = _loss_from_rollout(rollout, env, args)
    loss.backward()
    for head in (model.fc, model.fc_cam):
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.norm() > 0
    assert torch.isfinite(model.stem[0].weight.grad).all()


@pytest.mark.parametrize("steps", [1, 3])
def test_swept_last_action_collision_enters_loss_and_masks_following_steps(steps):
    args = intent_args(timesteps=steps, loss_v_window=1)
    env = build_env(1, args, torch.device("cpu"))
    env.reset("dark")
    env.p = env.p.new_tensor([[-.4, 0., 1.5]])
    env.v = env.v.new_tensor([[15., 0., 0.]])
    model = build_policy(args)
    rollout = _rollout(env, model, args, 1, torch.device("cpu"), False, None, False)
    assert rollout["terminal_collision"].item()
    assert rollout["terminal_step"].item() == 0
    assert [mask.item() for mask in rollout["valid_history"]] == [True] + [False] * (steps - 1)
    # Both endpoints are outside the wall. Only continuous collision checking
    # catches the traversal through its solid center.
    end = rollout["post_p_history"][0]
    assert box_clearance(end[None], env.voxels, env.drone_radius).item() > 0
    assert rollout["post_clearance_history"][0].item() < 0
    _, terms, _, _ = _loss_from_rollout(rollout, env, args)
    assert terms["loss_collide"].item() > 0


def test_provenance_preserves_dirty_runtime_source_and_calibration(tmp_path):
    import hashlib
    import json
    from train_utils import write_run_provenance
    args = intent_args()
    write_run_provenance(tmp_path, args)
    manifest = json.loads((tmp_path / "provenance.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        saved = tmp_path / manifest["source_snapshot"] / name
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == digest
    assert hashlib.sha256((tmp_path / "calibration_snapshot.json").read_bytes()).hexdigest() == manifest["calibration_sha256"]
    assert not manifest["calibrated"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reference kernel")
def test_torch_attitude_and_dynamics_match_original_forward():
    import quadsim_cuda
    torch.manual_seed(7)
    device = "cuda"
    batch = 8
    R = rotation_exp(torch.randn(batch, 3, device=device) * .1)
    command = torch.randn(batch, 3, device=device)
    applied = torch.randn(batch, 3, device=device) * .2
    target = torch.randn(batch, 3, device=device)
    alpha = torch.full((batch, 1), .7, device=device)
    torch.testing.assert_close(update_attitude(R, command, target, alpha),
                               quadsim_cuda.update_state_vec(R, command, target, alpha, 5), atol=2e-6, rtol=2e-6)
    z = torch.ones(batch, 1, device=device)
    drag = torch.full((batch, 2), .1, device=device)
    rate = torch.full_like(z, 12)
    state = [torch.randn(batch, 3, device=device) * .1 for _ in range(5)]
    disturbance, p, v, wind, a = state
    args = (R, disturbance, z, drag, rate, command, applied, p, v, wind, a)
    actual = dynamics_step(*args, .3, 1 / 15, .5)
    reference = quadsim_cuda.run_forward(*args, 1 / 15, .5)
    for x, y in zip(actual, reference):
        torch.testing.assert_close(x, y, atol=3e-5, rtol=2e-4)

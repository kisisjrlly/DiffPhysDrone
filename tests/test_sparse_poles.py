"""Geometry, observation and terminal semantics for the sparse-pole task."""

import pytest
import torch

from config import set_global_seed
from intent_model import build_policy
from render.box_scene import box_clearance
from render.intent_scene import render_exposure
from render.pole_scene import cylinder_clearance, swept_cylinder_clearance, trace_cylinders, surface_light_scale
from render.pose import rotation_exp
from rollout_ops import build_local_frame, build_state_vector, compute_target_velocity
from tools.intent_common import load_intent_args
from train_utils import build_env
from trainer import _rollout, _loss_from_rollout


def pole_args(**overrides):
    values = dict(batch_size=1, gray_width=20, gray_height=16, gray_nn_width=16,
                  gray_nn_height=12, timesteps=12, loss_v_window=3, gray_enable_noise=False)
    values.update(overrides)
    return load_intent_args("configs/gray_joint_poles.args", **values)


def test_cylinder_ray_geometry_and_piecewise_derivatives():
    cylinders = torch.tensor([[[2., 0., .2]]], dtype=torch.double)
    p = torch.tensor([[.01, .015, .02]], dtype=torch.double, requires_grad=True)
    a = torch.tensor([[.005, .01, .005]], dtype=torch.double, requires_grad=True)
    def render(p, a):
        return trace_cylinders(cylinders, rotation_exp(a), p, 3, 3, .03)[0]
    assert torch.autograd.gradcheck(render, (p, a), eps=1e-6, atol=1e-5)
    d, normal, _ = trace_cylinders(cylinders, torch.eye(3, dtype=torch.double)[None],
                                    torch.zeros_like(p), 1, 1, .1)
    assert d.item() == pytest.approx(1.8, abs=1e-7)
    assert normal[0, 0, 0, 0].item() == pytest.approx(-1, abs=1e-7)


def test_cylinder_missed_and_parallel_rays_have_finite_backward():
    p = torch.zeros(1, 3, dtype=torch.double, requires_grad=True)
    cylinders = torch.tensor([[[2., 3., .1]]], dtype=torch.double)
    d, n, _ = trace_cylinders(cylinders, torch.eye(3, dtype=torch.double)[None], p, 3, 3, .2)
    (d.sum() + n.sum()).backward()
    assert (d == 100).all() and torch.isfinite(p.grad).all()


def test_cylinder_continuous_clearance_catches_thin_obstacle_and_has_lateral_gradient():
    cylinders = torch.tensor([[[0., .05, .08]]])
    start, end = torch.tensor([[-1., 0., 1.5]]), torch.tensor([[1., 0., 1.5]])
    assert (cylinder_clearance(torch.stack([start, end]), cylinders, .12) > 0).all()
    assert swept_cylinder_clearance(start, end, cylinders, .12).item() == pytest.approx(-.15)
    assert swept_cylinder_clearance(start + torch.tensor([0., .5, 0.]),
                                     end + torch.tensor([0., .5, 0.]), cylinders, .12).item() > 0
    p = torch.tensor([[[-.25, 0., 1.5]]], requires_grad=True)
    cylinder_clearance(p, cylinders, .12).sum().backward()
    assert p.grad[0, 0, 1] < 0
    stationary = swept_cylinder_clearance(start, start, cylinders, .12)
    torch.testing.assert_close(stationary, cylinder_clearance(start[None], cylinders, .12)[0])


def test_reset_keeps_random_poles_and_goal_state_does_not_disclose_them():
    args = pole_args(batch_size=32)
    env = build_env(32, args, torch.device("cpu"))
    env.reset("dark_to_bright")
    assert env.cyl.shape == (32, 3, 3) and env.voxels.shape == (32, 5, 6)
    assert env.cyl[:, 0, 1].std() > .05
    assert (env.cyl[:, :, 2] >= .08).all() and (env.cyl[:, :, 2] <= .12).all()
    state = build_state_vector(env, compute_target_velocity(env.p_target - env.p, env),
                               build_local_frame(env), torch.zeros(32), torch.zeros(32), False, False)[0]
    torch.testing.assert_close(state, state[:1].expand_as(state))
    assert env.signed_clearance().min() > 0
    env.p = torch.cat([env.cyl[:, 0, :2], torch.full((32, 1), 1.5)], -1)
    assert (env.signed_clearance() < 0).all()


def test_pole_lighting_is_a_field_at_surface_points_and_pose_cut_preserves_pixels():
    args = pole_args()
    env = build_env(1, args, torch.device("cpu"))
    env.reset("dark_to_bright")
    points = torch.tensor([[[[-1., 0., 1.5], [3., 0., 1.5]]]])
    before = surface_light_scale(env, points)
    env.p = (env.p + torch.tensor([[.1, .2, 0.]])).requires_grad_(True)
    torch.testing.assert_close(before, surface_light_scale(env, points))
    a = render_exposure(env, env.p.new_tensor([.3]))[0]
    grad = torch.autograd.grad(a.sum(), env.p)[0]
    assert torch.isfinite(grad).all() and grad.norm() > 0
    env.observation_grad_mode = "detached"
    b = render_exposure(env, env.p.new_tensor([.3]))[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not b.requires_grad


def test_pole_last_action_sweep_enters_loss_and_terminal_mask():
    args = pole_args(timesteps=3, loss_v_window=1)
    env = build_env(1, args, torch.device("cpu"))
    env.reset("bright")
    env.cyl[0, 0] = torch.tensor([0., .03, .08])
    env.p = torch.tensor([[-.4, 0., 1.5]])
    env.p_old = env.p.clone()
    env.v = torch.tensor([[15., 0., 0.]])
    model = build_policy(args)
    rollout = _rollout(env, model, args, 1, torch.device("cpu"), False, None, False)
    assert rollout["terminal_collision"].item()
    assert rollout["terminal_step"].item() == 0
    assert [x.item() for x in rollout["valid_history"]] == [True, False, False]
    assert rollout["post_clearance_history"][0].item() < 0
    _, terms, _, _ = _loss_from_rollout(rollout, env, args)
    assert terms["loss_collide"].item() > 0


@pytest.mark.parametrize("overrides", [dict(pole_radius_min=0), dict(pole_count=0),
    dict(pole_spacing=.2), dict(pole_lateral_range=1.7), dict(simple_goal_x=3.)])
def test_invalid_pole_layouts_are_rejected(overrides):
    with pytest.raises(ValueError):
        pole_args(**overrides)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reference renderer")
def test_pole_renderer_matches_cuda_geometry():
    import quadsim_cuda
    from render.pole_scene import trace_pole_scene
    set_global_seed(31)
    args = pole_args(batch_size=3, gray_width=64, gray_height=48)
    env = build_env(3, args, torch.device("cuda"))
    env.reset("bright")
    p = env.p.contiguous()
    R = env.R.contiguous()
    d, n, _ = trace_pole_scene(env, p, R)
    depth, normals = torch.empty_like(d), torch.empty_like(n)
    quadsim_cuda.render_geometry(depth, normals, env.balls, env.cyl, env.cyl_h, env.voxels,
                                R, p, 1, env._fov_x_half_tan)
    torch.testing.assert_close(d, depth, atol=2e-4, rtol=3e-5)
    torch.testing.assert_close(n, normals, atol=2e-3, rtol=2e-3)

"""Physical/causal requirements for the opt-in single-slit scene."""

import copy

import pytest
import torch

from config import set_global_seed
from render.box_scene import swept_box_collision
from render.intent_scene import render_exposure, render_scene_irradiance
from render.pose import rotation_exp
from render.slit_scene import slit_surface_light
from rollout_ops import render_gray_sensor
from tools.intent_common import load_intent_args
from train_utils import build_env


def slit_args(**overrides):
    values = dict(batch_size=1, gray_width=24, gray_height=18, gray_nn_width=24,
                  gray_nn_height=18, gray_enable_noise=False)
    values.update(overrides)
    return load_intent_args("configs/gray_joint_slit.args", **values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="legacy environment needs CUDA")
def test_opt_in_scene_keeps_original_slit_and_goal_without_changing_legacy_defaults():
    args = slit_args(batch_size=8)
    original_args = copy.deepcopy(args)
    original_args.scene_layout = "single_wall_slit"
    old = build_env(8, original_args, torch.device("cuda"))
    new = build_env(8, args, torch.device("cuda"))
    set_global_seed(87)
    old.reset("bright_to_dark")
    set_global_seed(87)
    new.reset("bright_to_dark")
    torch.testing.assert_close(new.voxels[:, :3], old.voxels, rtol=0, atol=0)
    torch.testing.assert_close(new.p_target, old.p_target, rtol=0, atol=0)
    assert new.slit_center_y.std() > .1
    torch.testing.assert_close(new.p_target, new.p_target[:1].expand_as(new.p_target))


def test_same_world_point_keeps_light_when_observer_moves_and_dark_wall_is_visible_ahead():
    env = build_env(1, slit_args(), torch.device("cpu"))
    env.reset("bright_to_dark")
    points = torch.tensor([[[[-1.5, 0., 1.5], [0., 0., 1.5]]]])
    before = slit_surface_light(env, points)
    env.p = env.p + torch.tensor([[.3, .2, 0.]])
    torch.testing.assert_close(slit_surface_light(env, points), before, rtol=0, atol=0)
    assert before[0, 0, 0] > before[0, 0, 1] * 5
    image, aux = render_scene_irradiance(env, env.p, env.R @ env.R_cam)
    scales = slit_surface_light(env, aux["points"])
    # A single current frame contains surfaces on both sides of the shadow boundary.
    assert scales.max() - scales.min() > 1
    assert image.isfinite().all()


def test_closed_corridor_rejects_flying_around_wall_but_preserves_slit_passage():
    env = build_env(1, slit_args(simple_slit_center_y_min=.65, simple_slit_center_y_max=.65), torch.device("cpu"))
    env.reset("dark")
    start, end = torch.tensor([[-.4, .65, 1.5]]), torch.tensor([[.4, .65, 1.5]])
    assert not swept_box_collision(start, end, env.voxels, env.drone_radius).item()
    assert swept_box_collision(start * 0 + torch.tensor([[-.4, 0., 1.5]]),
                               torch.tensor([[.4, 0., 1.5]]), env.voxels, env.drone_radius).item()
    assert swept_box_collision(torch.tensor([[-.4, 0., 1.5]]), torch.tensor([[-.4, 2., 1.5]]),
                               env.voxels, env.drone_radius).item()


def test_shutter_responds_to_translation_and_rotation_but_still_camera_does_not_blur():
    env = build_env(1, slit_args(gray_temporal_samples=17), torch.device("cpu"))
    env.reset("dark")
    exposure = torch.tensor([.95])
    _, static = render_exposure(env, exposure, return_aux=True)
    assert static["temporal_blur_mae"].item() < 1e-6
    env.v = torch.tensor([[.3, 1., 0.]])
    _, translated = render_exposure(env, exposure, return_aux=True)
    assert translated["temporal_blur_mae"].item() > 1e-6
    env.v.zero_()
    env.R_old = env.R @ rotation_exp(torch.tensor([[0., 0., -.1]]))
    _, rotated = render_exposure(env, exposure, return_aux=True)
    assert rotated["temporal_blur_mae"].item() > 1e-6


def test_motion_boundary_cut_preserves_pixels_and_only_cuts_motion_not_exposure():
    env = build_env(1, slit_args(), torch.device("cpu"))
    env.reset("dark")
    env.p = env.p.detach().requires_grad_()
    env.v = torch.tensor([[.5, .8, 0.]], requires_grad=True)
    e, g = torch.tensor([.6], requires_grad=True), torch.tensor([.02], requires_grad=True)
    full, _ = render_gray_sensor(env, e, g)
    gradients = torch.autograd.grad(full.square().mean(), (env.p, env.v, e, g))
    assert all(torch.isfinite(v).all() and v.norm() > 0 for v in gradients)
    env.observation_grad_mode = "detached"
    cut, _ = render_gray_sensor(env, e, g)
    torch.testing.assert_close(full, cut, rtol=0, atol=0)
    gradients = torch.autograd.grad(cut.square().mean(), (env.p, env.v, e, g), allow_unused=True)
    assert gradients[0] is None and gradients[1] is None
    assert gradients[2].norm() > 0 and gradients[3].norm() > 0


def test_diagnostic_decoder_reports_chance_when_saturation_erases_the_cue():
    from tools.audit_joint_slit import decision_measure
    args = slit_args()
    env = build_env(1, args, torch.device("cpu"))
    env.reset("bright")
    env.p = torch.tensor([[-.4, .55, 1.5]])
    env.R = torch.eye(3)[None]
    env.R_old = env.R.clone()
    report, _ = decision_measure(env, args, .85, .02, samples=16)
    assert report["policy_resolution_cue_rms"] == 0
    assert report["known_state_template_accuracy"] == .5
    assert report["known_state_template_dprime"] == 0


@pytest.mark.parametrize("center", [-.75, .75])
def test_extreme_slit_has_collision_free_geometric_path(center):
    # Actual dynamics are separately audited and can FAIL the training gate;
    # this test establishes only geometric feasibility, not tracking success.
    from render.box_scene import box_clearance
    args = slit_args(simple_slit_center_y_min=center, simple_slit_center_y_max=center,
                     timesteps=90)
    set_global_seed(291031)
    env = build_env(1, args, torch.device("cpu"))
    env.reset("dark")
    points = torch.tensor([[[args.simple_start_x, 0., 1.5], [-.5, center, 1.5],
                            [.5, center, 1.5], [1.5, 0., 1.5]]])
    samples = []
    for start, end in zip(points[0, :-1], points[0, 1:]):
        assert not swept_box_collision(start[None], end[None], env.voxels, env.drone_radius).item()
        samples.append(start + torch.linspace(0, 1, 33)[:, None] * (end - start))
    assert box_clearance(torch.cat(samples)[:, None], env.voxels, env.drone_radius).min() > .02

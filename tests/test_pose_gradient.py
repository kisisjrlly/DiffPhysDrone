"""Physical sign/scale and forward invariance of the optional warp Jacobian."""

import pytest
import torch

from render.pose import rotation_exp
from render.pose_gradient import image_warp_pose_surrogate
from render.intent_scene import render_exposure
from config import set_global_seed
from tools.intent_common import load_intent_args
from train_utils import build_env


def test_plane_ramp_has_correct_translation_and_rotation_jacobian():
    height, width, depth, fov = 12, 16, 2., .8
    u = 2 * (torch.arange(width, dtype=torch.double) + .5) / width - 1
    v = 2 * (torch.arange(height, dtype=torch.double) + .5) / height - 1
    image = (.4 + .2 * u[None, :] + .1 * v[:, None])[None, None]
    p = torch.zeros(1, 3, dtype=torch.double, requires_grad=True)
    angle = torch.zeros(1, 3, dtype=torch.double, requires_grad=True)
    r = rotation_exp(angle)
    output = image_warp_pose_surrogate(image, torch.full((1, height, width), depth), p, r, fov)
    torch.testing.assert_close(output, image, rtol=0, atol=0)
    # Interior pixels avoid the deliberately constant border extension.
    value = output[0, 0, 3, 5]
    gp, ga = torch.autograd.grad(value, (p, angle))
    fy = fov * height / width
    ray = torch.tensor([1., -float(u[5]) * fov + 1e-5,
                        -float(v[3]) * fy + 1e-5], dtype=torch.double)
    projected_gradient = torch.tensor([.2 * ray[1] / fov + .1 * ray[2] / fy,
                                       -.2 / fov, -.1 / fy], dtype=torch.double)
    torch.testing.assert_close(gp[0], projected_gradient / depth, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(ga[0], torch.linalg.cross(ray, projected_gradient), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("cut", [False, True])
def test_warp_estimator_preserves_hard_forward_and_respects_pose_boundary(cut):
    args = load_intent_args("configs/gray_joint_parallax.args", batch_size=1,
                           gray_width=24, gray_height=18, gray_enable_noise=False)
    set_global_seed(91)
    env = build_env(1, args, torch.device("cpu"))
    env.reset("nominal")
    env.p = (env.p + torch.tensor([[.1, .03, 0.]])).requires_grad_()
    env.v = torch.tensor([[.3, .4, 0.]], requires_grad=True)
    exposure = torch.tensor([.5], requires_grad=True)
    baseline, _ = render_exposure(env, exposure)
    env.pose_gradient_estimator = "warp_ste"
    env.observation_grad_mode = "detached" if cut else "full"
    actual, _ = render_exposure(env, exposure)
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    grads = torch.autograd.grad(actual.square().mean(), (env.p, env.v, exposure), allow_unused=True)
    if cut:
        assert grads[0] is None and grads[1] is None
    else:
        assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)

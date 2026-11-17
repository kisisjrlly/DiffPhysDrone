import torch

from rollout_ops import render_gray_sensor
from tools.intent_common import load_intent_args
from train_utils import build_env
from render.scene_types import DIFFERENTIABLE_SCENES


def _args(**overrides):
    return load_intent_args(
        "configs/gray_single_wall_slit_diff.args",
        batch_size=1,
        gray_enable_noise=False,
        scenarios=["nominal"],
        **overrides,
    )


def test_single_wall_slit_diff_is_explicit_differentiable_scene():
    assert "single_wall_slit_diff" in DIFFERENTIABLE_SCENES
    args = _args()
    env = build_env(1, args, torch.device("cpu"), eval_mode=True)
    env.reset(scene_name="nominal")
    assert tuple(env.voxels.shape) == (1, 3, 6)
    assert env.get_scene_effects_for_env()["renderer_mode"] == "differentiable_equivalent"


def test_single_wall_slit_diff_exposure_has_task_gradient():
    args = _args()
    env = build_env(1, args, torch.device("cpu"), eval_mode=True)
    env.reset(scene_name="nominal")
    exposure = env.p.new_tensor([0.01583166], requires_grad=True)
    gain = env.p.new_tensor([0.0])
    image, aux = render_gray_sensor(env, exposure, gain, differentiable=True)
    image.mean().backward()
    assert image.shape == (1, 1, 72, 96)
    assert exposure.grad is not None
    assert torch.isfinite(exposure.grad).all()
    assert float(exposure.grad.abs().sum()) > 0.0
    assert float(aux["exposure_us"].mean()) > 0.0


def test_proof_camera_has_no_legacy_box_blur_in_temporal_path():
    args = _args()
    env = build_env(1, args, torch.device("cpu"), eval_mode=True)
    env.reset(scene_name="nominal")
    exposure = env.p.new_tensor([0.998])
    gain = env.p.new_tensor([0.0])
    image, aux = render_gray_sensor(env, exposure, gain, differentiable=False)
    assert image.shape[-2:] == (72, 96)
    assert float(aux["temporal_samples"]) == 9.0
    assert float(aux["saturation_fraction"].max()) < 0.2
    assert torch.isfinite(image).all()

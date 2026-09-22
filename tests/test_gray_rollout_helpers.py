import torch

from rollout_ops import (
    build_gray_state_vector,
    init_gray_camera_params,
    render_gray_sensor,
    update_gray_camera_params,
)
from sensors.differentiable_gray_camera import DifferentiableGrayCamera
from sensors.gray_camera_semantics import GrayCameraSemantics


class _DummyEnv:
    def __init__(self, batch=2):
        self.batch_size = batch
        self.camera_control_mode = "learned"
        self.fixed_camera_exposure = 0.3
        self.fixed_camera_gain = 0.2
        self.fixed_random_exposure_range = (0.1, 0.9)
        self.fixed_random_gain_range = (0.1, 0.8)
        self.gray_enable_noise = False
        self.v = torch.zeros(batch, 3)
        self.R = torch.eye(3).repeat(batch, 1, 1)
        self.margin = torch.ones(batch)
        self.max_speed = torch.tensor(2.0)
        self.gray_camera = DifferentiableGrayCamera(
            GrayCameraSemantics(
                exposure_us_min=100.0,
                exposure_us_max=1000.0,
                exposure_reference_us=1000.0,
                gain_factor_min=1.0,
                gain_factor_max=2.0,
            ),
            blur_scale=0.0,
        )

    def render_gray_ideal(self, return_aux=False):
        image = torch.full((self.batch_size, 1, 8, 8), 0.2)
        aux = {"hit_mask": torch.ones(self.batch_size, 8, 8, dtype=torch.bool)}
        return image, aux if return_aux else None


def test_gray_sensor_gradient_switch_only_controls_camera_params():
    env = _DummyEnv(batch=1)
    exposure = torch.tensor([0.4], requires_grad=True)
    gain = torch.tensor([0.3], requires_grad=True)

    full, _ = render_gray_sensor(env, exposure, gain, differentiable=True)
    grad_e, grad_g = torch.autograd.grad(full.mean(), (exposure, gain))
    assert grad_e.abs().item() > 0
    assert grad_g.abs().item() > 0

    detached, _ = render_gray_sensor(env, exposure, gain, differentiable=False)
    assert not detached.requires_grad


def test_gray_camera_init_and_update_shapes():
    env = _DummyEnv(batch=3)
    exposure, gain = init_gray_camera_params(env, 3, torch.device("cpu"))
    assert exposure.shape == (3,)
    assert gain.shape == (3,)

    target = torch.tensor([[0.2, 0.4], [0.3, 0.5], [0.4, 0.6]])
    e2, g2, hist = update_gray_camera_params(target, exposure, gain, env)
    assert e2.shape == (3,)
    assert g2.shape == (3,)
    assert hist.shape == (3, 2)


def test_gray_state_vector_adds_two_camera_values():
    env = _DummyEnv(batch=2)
    target_v = torch.ones(2, 3)
    local_frame = torch.eye(3).repeat(2, 1, 1)
    exposure = torch.tensor([0.25, 0.75])
    gain = torch.tensor([0.1, 0.9])

    state, local_v, camera_state, camera_motion = build_gray_state_vector(
        env,
        target_v,
        local_frame,
        exposure,
        gain,
        no_odom=False,
        include_camera_state=True,
    )
    assert state.shape == (2, 12)
    assert local_v.shape == (2, 3)
    assert camera_state.shape == (2, 2)
    assert camera_motion.shape == (2, 6)

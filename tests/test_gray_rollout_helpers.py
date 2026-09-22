import torch

from rollout_ops import (
    build_state_vector,
    init_camera_params,
    render_gray_sensor,
    update_camera_params,
)
from sensors.imx900_calibration import IMX900Calibration
from sensors.imx900_camera import IMX900DifferentiableCamera


class _DummyEnv:
    def __init__(self, batch=2):
        self.batch_size = batch
        self.camera_control_mode = "learned"
        self.camera_smoothing_alpha = 0.7
        self.fixed_camera_exposure = 0.3
        self.fixed_camera_gain = 0.2
        self.fixed_random_exposure_range = (0.1, 0.9)
        self.fixed_random_gain_range = (0.1, 0.8)
        self.gray_enable_noise = False
        self.gray_motion_depth_floor = 0.35
        self.geometry_depth = torch.full((batch, 8, 8), 2.0)
        self.v = torch.zeros(batch, 3)
        self.R = torch.eye(3).repeat(batch, 1, 1)
        self.margin = torch.ones(batch)
        self.max_speed = torch.tensor(2.0)
        self.gray_camera = IMX900DifferentiableCamera(
            IMX900Calibration(
                profile_name="rollout-test",
                calibrated=True,
                source="unit test",
                exposure_us_min=100.0,
                exposure_us_max=1000.0,
                exposure_reference_us=1000.0,
                gain_factor_min=1.0,
                gain_factor_max=2.0,
                blur_scale=0.0,
            )
        )

    def render_gray_ideal(self, return_aux=False):
        image = torch.full((self.batch_size, 1, 8, 8), 0.2)
        aux = {
            "hit_mask": torch.ones(self.batch_size, 8, 8, dtype=torch.bool),
            "light_scale": torch.ones(self.batch_size),
            "geometry_depth": self.geometry_depth,
        }
        return image, aux if return_aux else None


def test_sensor_gradient_switch_only_controls_camera_params():
    env = _DummyEnv(batch=1)
    exposure = torch.tensor([0.4], requires_grad=True)
    gain = torch.tensor([0.3], requires_grad=True)

    full, _ = render_gray_sensor(env, exposure, gain, differentiable=True)
    grad_e, grad_g = torch.autograd.grad(full.mean(), (exposure, gain))
    assert grad_e.abs().item() > 0
    assert grad_g.abs().item() > 0

    detached, _ = render_gray_sensor(env, exposure, gain, differentiable=False)
    assert not detached.requires_grad


def test_camera_init_and_update_shapes():
    env = _DummyEnv(batch=3)
    exposure, gain = init_camera_params(env, 3, torch.device("cpu"))
    assert exposure.shape == (3,)
    assert gain.shape == (3,)

    target = torch.tensor([[0.2, 0.4], [0.3, 0.5], [0.4, 0.6]])
    e2, g2, hist = update_camera_params(target, exposure, gain, env)
    assert e2.shape == (3,)
    assert g2.shape == (3,)
    assert hist.shape == (3, 2)


def test_state_vector_adds_two_camera_values():
    env = _DummyEnv(batch=2)
    target_v = torch.ones(2, 3)
    local_frame = torch.eye(3).repeat(2, 1, 1)
    exposure = torch.tensor([0.25, 0.75])
    gain = torch.tensor([0.1, 0.9])

    state, local_v, camera_state, camera_motion = build_state_vector(
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


def test_camera_smoothing_uses_configured_alpha():
    env = _DummyEnv(batch=1)
    env.camera_smoothing_alpha = 0.5
    exposure = torch.tensor([0.2])
    gain = torch.tensor([0.4])
    target = torch.tensor([[0.8, 0.6]])
    e2, g2, _ = update_camera_params(target, exposure, gain, env)
    torch.testing.assert_close(e2, torch.tensor([0.5]))
    torch.testing.assert_close(g2, torch.tensor([0.5]))


def test_motion_proxy_increases_when_scene_is_closer():
    env = _DummyEnv(batch=1)
    env.v[:] = torch.tensor([[2.0, 0.0, 0.0]])
    env.gray_camera = IMX900DifferentiableCamera(
        IMX900Calibration(
            profile_name="blur-test",
            calibrated=True,
            source="unit test",
            exposure_us_min=100.0,
            exposure_us_max=1000.0,
            exposure_reference_us=1000.0,
            gain_factor_min=1.0,
            gain_factor_max=2.0,
            blur_scale=1.0,
        )
    )
    exposure = torch.tensor([0.7])
    gain = torch.tensor([0.0])

    env.geometry_depth[:] = 4.0
    _, far_aux = render_gray_sensor(env, exposure, gain, differentiable=False)

    env.geometry_depth[:] = 0.5
    _, near_aux = render_gray_sensor(env, exposure, gain, differentiable=False)

    assert near_aux["motion_proxy"].item() > far_aux["motion_proxy"].item()
    assert near_aux["blur_strength"].mean() > far_aux["blur_strength"].mean()


def test_measured_command_delay_is_separate_from_policy_smoothing():
    env = _DummyEnv(batch=1)
    env.camera_smoothing_alpha = 0.0
    env.gray_camera = IMX900DifferentiableCamera(
        IMX900Calibration(
            profile_name="delay-test",
            calibrated=True,
            source="unit test",
            exposure_us_min=100.0,
            exposure_us_max=1000.0,
            exposure_reference_us=1000.0,
            gain_factor_min=1.0,
            gain_factor_max=2.0,
            blur_scale=0.0,
            command_delay_frames=1,
        )
    )
    exposure, gain = init_camera_params(env, 1, torch.device("cpu"))
    initial = torch.stack([exposure, gain], -1)

    first_target = torch.tensor([[0.8, 0.2]])
    e1, g1, requested1 = update_camera_params(
        first_target, exposure, gain, env
    )
    torch.testing.assert_close(torch.stack([e1, g1], -1), initial)
    torch.testing.assert_close(requested1, first_target)

    second_target = torch.tensor([[0.1, 0.9]])
    e2, g2, requested2 = update_camera_params(
        second_target, e1, g1, env
    )
    torch.testing.assert_close(
        torch.stack([e2, g2], -1),
        first_target,
    )
    torch.testing.assert_close(requested2, second_target)

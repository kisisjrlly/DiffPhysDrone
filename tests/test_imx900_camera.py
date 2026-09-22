import json
from types import SimpleNamespace

import torch

from sensors.imx900_calibration import IMX900Calibration
from sensors.imx900_camera import IMX900DifferentiableCamera, build_from_args


def _profile(**overrides):
    base = dict(
        profile_name="unit-test",
        calibrated=True,
        source="synthetic unit-test profile",
        exposure_us_min=100.0,
        exposure_us_max=1000.0,
        exposure_reference_us=1000.0,
        signal_scale=1.0,
        gain_mapping="log",
        gain_factor_min=1.0,
        gain_factor_max=2.0,
        shot_noise_alpha=0.01,
        shot_noise_beta=0.0,
        read_noise_std_base=0.002,
        read_noise_gain_exponent=1.0,
        black_level=0.0,
        saturation_level=1.0,
        quantization_bits=0,
        blur_scale=1.0,
    )
    base.update(overrides)
    return IMX900Calibration(**base)


def _camera(**kwargs):
    defaults = dict(
        calibration=_profile(),
        blur_kernel_size=5,
        saturation_mode="ste",
    )
    defaults.update(kwargs)
    return IMX900DifferentiableCamera(**defaults)


def _finite_difference(camera, image, exposure, gain, which, eps=1e-4):
    with torch.no_grad():
        if which == "exposure":
            plus = camera(image, exposure + eps, gain, enable_noise=False)[0].mean()
            minus = camera(image, exposure - eps, gain, enable_noise=False)[0].mean()
        else:
            plus = camera(image, exposure, gain + eps, enable_noise=False)[0].mean()
            minus = camera(image, exposure, gain - eps, enable_noise=False)[0].mean()
    return (plus - minus) / (2.0 * eps)


def test_exposure_gradient_matches_finite_difference():
    camera = _camera(calibration=_profile(blur_scale=0.0))
    image = torch.full((1, 1, 12, 16), 0.2)
    exposure = torch.tensor([0.35], requires_grad=True)
    gain = torch.tensor([0.25], requires_grad=True)

    out = camera(image, exposure, gain, enable_noise=False)[0].mean()
    grad = torch.autograd.grad(out, exposure, retain_graph=True)[0]
    fd = _finite_difference(camera, image, exposure.detach(), gain.detach(), "exposure")
    torch.testing.assert_close(grad.squeeze(), fd, rtol=2e-3, atol=2e-4)


def test_gain_gradient_matches_finite_difference():
    camera = _camera(calibration=_profile(blur_scale=0.0))
    image = torch.full((1, 1, 12, 16), 0.2)
    exposure = torch.tensor([0.35], requires_grad=True)
    gain = torch.tensor([0.25], requires_grad=True)

    out = camera(image, exposure, gain, enable_noise=False)[0].mean()
    grad = torch.autograd.grad(out, gain, retain_graph=True)[0]
    fd = _finite_difference(camera, image, exposure.detach(), gain.detach(), "gain")
    torch.testing.assert_close(grad.squeeze(), fd, rtol=2e-3, atol=2e-4)


def test_gain_lut_is_piecewise_differentiable():
    profile = _profile(
        gain_mapping="lut",
        gain_lut_x=[0.0, 0.5, 1.0],
        gain_lut_factor=[1.0, 2.0, 5.0],
        gain_factor_min=1.0,
        gain_factor_max=5.0,
    )
    x = torch.tensor([0.25, 0.75], requires_grad=True)
    y = profile.gain_to_factor(x)
    torch.testing.assert_close(y, torch.tensor([1.5, 3.5]))
    y.sum().backward()
    assert torch.all(x.grad > 0)


def test_fixed_noise_tensors_make_forward_deterministic():
    camera = _camera(calibration=_profile(blur_scale=0.0))
    image = torch.full((2, 1, 8, 8), 0.25)
    exposure = torch.tensor([0.2, 0.8])
    gain = torch.tensor([0.1, 0.6])
    shot = torch.randn_like(image)
    read = torch.randn_like(image)

    out1, _ = camera(
        image, exposure, gain, shot_noise=shot, read_noise=read, enable_noise=True
    )
    out2, _ = camera(
        image, exposure, gain, shot_noise=shot, read_noise=read, enable_noise=True
    )
    torch.testing.assert_close(out1, out2)


def test_long_exposure_and_motion_increase_blur():
    camera = _camera()
    image = torch.zeros((1, 1, 21, 21))
    image[:, :, :, ::2] = 1.0
    _, short = camera(
        image,
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        motion=torch.tensor([1.0]),
        enable_noise=False,
    )
    moving, long = camera(
        image,
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        motion=torch.tensor([1.0]),
        enable_noise=False,
    )
    sharp, _ = camera(
        image,
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        motion=torch.tensor([0.0]),
        enable_noise=False,
    )
    assert long["blur_strength"].mean() > short["blur_strength"].mean()
    assert moving.var() < sharp.var()


def test_soft_saturation_stays_bounded_and_gradient_decays():
    camera = _camera(
        calibration=_profile(blur_scale=0.0),
        saturation_mode="soft",
        soft_clip_beta=12.0,
    )
    image = torch.ones((1, 1, 8, 8))
    low_e = torch.tensor([0.15], requires_grad=True)
    high_e = torch.tensor([0.95], requires_grad=True)
    gain = torch.tensor([0.0])

    low = camera(image, low_e, gain, enable_noise=False)[0].mean()
    high = camera(image, high_e, gain, enable_noise=False)[0].mean()
    grad_low = torch.autograd.grad(low, low_e)[0].abs()
    grad_high = torch.autograd.grad(high, high_e)[0].abs()

    assert 0.0 <= float(low) <= 1.0
    assert 0.0 <= float(high) <= 1.0
    assert grad_high < grad_low


def test_profile_json_and_builder(tmp_path):
    data = {
        "schema_version": 1,
        "profile_name": "measured-example",
        "calibrated": True,
        "source": "unit test",
        "exposure": {
            "min_us": 50.0,
            "max_us": 5000.0,
            "reference_us": 500.0,
            "step_us": 5.0,
            "signal_scale": 0.8
        },
        "gain": {
            "mapping": "linear",
            "min_factor": 1.0,
            "max_factor": 6.0,
            "step_factor": 0.0,
            "lut_x": None,
            "lut_factor": None
        },
        "noise": {
            "shot_alpha": 0.02,
            "shot_beta": 0.001,
            "read_std_base": 0.003,
            "read_gain_exponent": 0.8
        },
        "black_level": 0.01,
        "saturation_level": 1.0,
        "quantization_bits": 10,
        "motion_blur": {"scale": 0.7},
        "actuator": {"command_delay_frames": 1}
    }
    path = tmp_path / "imx900.json"
    path.write_text(json.dumps(data))

    args = SimpleNamespace(
        imx900_calibration=str(path),
        gray_blur_kernel_size=3,
        gray_dark_threshold=0.04,
        gray_saturation_mode="ste",
        gray_soft_clip_beta=5.0,
    )
    camera = build_from_args(args)
    assert camera.calibration.calibrated
    assert camera.calibration.exposure_us_min == 50.0
    assert camera.calibration.quantization_bits == 10
    assert camera.calibration.command_delay_frames == 1
    assert camera.blur_kernel_size == 3

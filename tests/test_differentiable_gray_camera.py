import torch

from sensors.differentiable_gray_camera import DifferentiableGrayCamera, build_from_args
from sensors.gray_camera_semantics import GrayCameraSemantics


def _finite_difference(camera, image, exposure, gain, which, eps=1e-4):
    with torch.no_grad():
        if which == "exposure":
            plus = camera(image, exposure + eps, gain, enable_noise=False)[0].mean()
            minus = camera(image, exposure - eps, gain, enable_noise=False)[0].mean()
        else:
            plus = camera(image, exposure, gain + eps, enable_noise=False)[0].mean()
            minus = camera(image, exposure, gain - eps, enable_noise=False)[0].mean()
    return (plus - minus) / (2.0 * eps)


def _camera(**kwargs):
    semantics = GrayCameraSemantics(
        exposure_us_min=100.0,
        exposure_us_max=1000.0,
        exposure_reference_us=1000.0,
        gain_factor_min=1.0,
        gain_factor_max=2.0,
    )
    defaults = dict(
        shot_noise_scale=0.01,
        read_noise_std=0.002,
        blur_scale=1.0,
        saturation_mode="ste",
    )
    defaults.update(kwargs)
    return DifferentiableGrayCamera(semantics, **defaults)


def test_exposure_gradient_matches_finite_difference():
    camera = _camera(blur_scale=0.0)
    image = torch.full((1, 1, 12, 16), 0.2)
    exposure = torch.tensor([0.35], requires_grad=True)
    gain = torch.tensor([0.25], requires_grad=True)

    out = camera(image, exposure, gain, enable_noise=False)[0].mean()
    grad = torch.autograd.grad(out, exposure, retain_graph=True)[0]
    fd = _finite_difference(camera, image, exposure.detach(), gain.detach(), "exposure")

    torch.testing.assert_close(grad.squeeze(), fd, rtol=2e-3, atol=2e-4)


def test_gain_gradient_matches_finite_difference():
    camera = _camera(blur_scale=0.0)
    image = torch.full((1, 1, 12, 16), 0.2)
    exposure = torch.tensor([0.35], requires_grad=True)
    gain = torch.tensor([0.25], requires_grad=True)

    out = camera(image, exposure, gain, enable_noise=False)[0].mean()
    grad = torch.autograd.grad(out, gain, retain_graph=True)[0]
    fd = _finite_difference(camera, image, exposure.detach(), gain.detach(), "gain")

    torch.testing.assert_close(grad.squeeze(), fd, rtol=2e-3, atol=2e-4)


def test_exposure_brightens_unsaturated_image():
    camera = _camera(blur_scale=0.0)
    image = torch.full((1, 1, 8, 8), 0.15)
    gain = torch.tensor([0.0])

    low = camera(image, torch.tensor([0.1]), gain, enable_noise=False)[0].mean()
    high = camera(image, torch.tensor([0.7]), gain, enable_noise=False)[0].mean()
    assert high > low


def test_high_exposure_and_gain_saturate_forward():
    camera = _camera(blur_scale=0.0)
    image = torch.ones((1, 1, 8, 8))
    out, aux = camera(
        image,
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        enable_noise=False,
    )
    assert float(out.max()) <= 1.0
    assert float(aux["saturation_fraction"].item()) > 0.99


def test_motion_and_exposure_increase_blur():
    camera = _camera()
    image = torch.zeros((1, 1, 21, 21))
    image[:, :, :, ::2] = 1.0

    no_motion, _ = camera(
        image,
        torch.tensor([0.8]),
        torch.tensor([0.0]),
        motion=torch.tensor([0.0]),
        enable_noise=False,
    )
    moving, aux_moving = camera(
        image,
        torch.tensor([0.8]),
        torch.tensor([0.0]),
        motion=torch.tensor([1.0]),
        enable_noise=False,
    )
    _, aux_short = camera(
        image,
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        motion=torch.tensor([1.0]),
        enable_noise=False,
    )
    _, aux_long = camera(
        image,
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        motion=torch.tensor([1.0]),
        enable_noise=False,
    )

    assert aux_long["blur_strength"].mean() > aux_short["blur_strength"].mean()
    assert aux_moving["blur_strength"].mean() > 0
    assert moving.var() < no_motion.var()


def test_fixed_noise_tensors_make_forward_deterministic():
    camera = _camera(blur_scale=0.0)
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


def test_semantics_clamps_normalized_commands():
    semantics = GrayCameraSemantics(
        exposure_us_min=100.0,
        exposure_us_max=1000.0,
        exposure_reference_us=1000.0,
        gain_factor_min=1.0,
        gain_factor_max=4.0,
    )
    assert semantics.exposure_to_us(-1.0) == 100.0
    assert semantics.exposure_to_us(2.0) == 1000.0
    assert semantics.gain_to_factor(-1.0) == 1.0
    assert semantics.gain_to_factor(2.0) == 4.0


def test_build_from_args_uses_gray_config_fields():
    from types import SimpleNamespace

    args = SimpleNamespace(
        gray_exposure_us_min=50.0,
        gray_exposure_us_max=5000.0,
        gray_exposure_reference_us=500.0,
        gray_gain_factor_min=1.0,
        gray_gain_factor_max=6.0,
        gray_shot_noise_scale=0.02,
        gray_read_noise_std=0.003,
        gray_read_noise_gain_scale=0.4,
        gray_black_level=0.01,
        gray_blur_scale=0.7,
        gray_blur_kernel_size=3,
        gray_dark_threshold=0.04,
        gray_saturation_mode="ste",
        gray_soft_clip_beta=5.0,
        gray_quantization_bits=8,
    )
    camera = build_from_args(args)
    assert camera.semantics.exposure_us_min == 50.0
    assert camera.semantics.exposure_us_max == 5000.0
    assert camera.semantics.gain_factor_max == 6.0
    assert camera.blur_kernel_size == 3
    assert camera.quantization_bits == 8

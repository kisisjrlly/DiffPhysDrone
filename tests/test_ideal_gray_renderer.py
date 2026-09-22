import torch

from render.ideal_gray import render_ideal_grayscale


def _plane_inputs(height=16, width=20, depth_value=2.0):
    depth = torch.full((1, height, width), depth_value)
    R = torch.eye(3).unsqueeze(0)
    pos = torch.zeros((1, 3))
    return depth, R, pos


def test_shape_range_and_finite_values():
    depth, R, pos = _plane_inputs()
    gray, aux = render_ideal_grayscale(depth, R, pos, return_aux=True)
    assert gray.shape == (1, 1, 16, 20)
    assert torch.isfinite(gray).all()
    assert float(gray.min()) >= 0.0
    assert float(gray.max()) <= 1.0
    assert aux["normals"].shape == (1, 16, 20, 3)


def test_front_parallel_plane_has_expected_lambertian_level():
    depth, R, pos = _plane_inputs()
    gray, aux = render_ideal_grayscale(
        depth,
        R,
        pos,
        ambient=0.2,
        diffuse=0.8,
        light_direction=(-1.0, 0.0, 0.0),
        base_albedo=0.5,
        texture_strength=0.0,
        return_aux=True,
    )
    center = gray[0, 0, 4:-4, 4:-4].mean()
    torch.testing.assert_close(center, torch.tensor(0.5), atol=2e-3, rtol=2e-3)
    center_normal = aux["normals"][0, 8, 10]
    assert center_normal[0] < -0.99


def test_procedural_texture_creates_monocular_spatial_cues():
    depth, R, pos = _plane_inputs(height=24, width=32)
    plain, _ = render_ideal_grayscale(
        depth,
        R,
        pos,
        light_direction=(-1.0, 0.0, 0.0),
        texture_strength=0.0,
    )
    textured, _ = render_ideal_grayscale(
        depth,
        R,
        pos,
        light_direction=(-1.0, 0.0, 0.0),
        texture_strength=0.7,
    )
    assert textured.var() > plain.var() + 1e-5


def test_background_depth_uses_background_intensity():
    depth, R, pos = _plane_inputs()
    depth[:, 3:7, 5:10] = 100.0
    gray, _ = render_ideal_grayscale(
        depth,
        R,
        pos,
        background_intensity=0.123,
        background_depth_threshold=99.0,
    )
    patch = gray[0, 0, 3:7, 5:10]
    torch.testing.assert_close(patch, torch.full_like(patch, 0.123))


def test_geometry_is_detached_by_default():
    depth, R, pos = _plane_inputs()
    depth.requires_grad_(True)
    gray, _ = render_ideal_grayscale(depth, R, pos)
    assert not gray.requires_grad

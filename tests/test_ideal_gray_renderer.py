import torch

from render.ideal_gray import render_ideal_grayscale


def _plane_inputs(height=16, width=20, depth_value=2.0, batch=1):
    depth = torch.full((batch, height, width), depth_value)
    normals = torch.zeros((batch, height, width, 3))
    normals[..., 0] = -1.0
    R = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1)
    pos = torch.zeros((batch, 3))
    return depth, normals, R, pos


def test_shape_and_finite_nonnegative_irradiance():
    depth, normals, R, pos = _plane_inputs()
    gray, aux = render_ideal_grayscale(depth, R, pos, normals, return_aux=True)
    assert gray.shape == (1, 1, 16, 20)
    assert torch.isfinite(gray).all()
    assert float(gray.min()) >= 0.0
    assert aux["normals"].shape == (1, 16, 20, 3)


def test_front_parallel_plane_has_expected_lambertian_level():
    depth, normals, R, pos = _plane_inputs()
    gray, aux = render_ideal_grayscale(
        depth,
        R,
        pos,
        normals,
        ambient=0.2,
        diffuse=0.8,
        light_direction=(-1.0, 0.0, 0.0),
        base_albedo=0.5,
        texture_strength=0.0,
        return_aux=True,
    )
    center = gray[0, 0, 4:-4, 4:-4].mean()
    torch.testing.assert_close(center, torch.tensor(0.5), atol=2e-3, rtol=2e-3)
    assert aux["normals"][0, 8, 10, 0] < -0.99


def test_procedural_texture_creates_monocular_spatial_cues():
    depth, normals, R, pos = _plane_inputs(height=24, width=32)
    plain, _ = render_ideal_grayscale(
        depth, R, pos,
        light_direction=(-1.0, 0.0, 0.0),
        texture_strength=0.0,
    )
    textured, _ = render_ideal_grayscale(
        depth, R, pos,
        light_direction=(-1.0, 0.0, 0.0),
        texture_strength=0.7,
    )
    assert textured.var() > plain.var() + 1e-5


def test_batched_bright_illumination_can_exceed_one_before_camera():
    depth, normals, R, pos = _plane_inputs(batch=2)
    gray, _ = render_ideal_grayscale(
        depth,
        R,
        pos,
        normals,
        ambient=torch.tensor([0.1, 1.0]),
        diffuse=torch.tensor([0.2, 2.0]),
        light_direction=(-1.0, 0.0, 0.0),
        base_albedo=0.8,
        texture_strength=0.0,
    )
    assert gray[1].mean() > gray[0].mean()
    assert float(gray[1].max()) > 1.0


def test_background_depth_uses_background_intensity():
    depth, normals, R, pos = _plane_inputs()
    depth[:, 3:7, 5:10] = 100.0
    gray, _ = render_ideal_grayscale(
        depth,
        R,
        pos,
        normals,
        background_intensity=0.123,
        background_depth_threshold=99.0,
    )
    patch = gray[0, 0, 3:7, 5:10]
    torch.testing.assert_close(patch, torch.full_like(patch, 0.123))


def test_geometry_is_detached_by_default():
    depth, normals, R, pos = _plane_inputs()
    depth.requires_grad_(True)
    normals.requires_grad_(True)
    gray, _ = render_ideal_grayscale(depth, R, pos, normals)
    assert not gray.requires_grad


def test_supplied_exact_normal_controls_lambertian_shading():
    depth, normals, R, pos = _plane_inputs(height=8, width=8)
    front, _ = render_ideal_grayscale(
        depth,
        R,
        pos,
        normals,
        ambient=0.0,
        diffuse=1.0,
        light_direction=(-1.0, 0.0, 0.0),
        base_albedo=1.0,
        texture_strength=0.0,
    )
    side_normals = normals.clone()
    side_normals[..., 0] = 0.0
    side_normals[..., 1] = 1.0
    side, _ = render_ideal_grayscale(
        depth,
        R,
        pos,
        side_normals,
        ambient=0.0,
        diffuse=1.0,
        light_direction=(-1.0, 0.0, 0.0),
        base_albedo=1.0,
        texture_strength=0.0,
    )
    assert front.mean() > side.mean() + 0.5

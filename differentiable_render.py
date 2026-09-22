"""Pure PyTorch soft depth renderer for the single-wall navigation scene."""

import torch


def render_slit_depth(
    position,
    camera_rotation,
    height,
    width,
    fov_x_half_tan,
    wall_x,
    back_wall_x,
    slit_center_y,
    slit_center_z,
    slit_half_y,
    slit_half_z,
    max_range=6.0,
    temperature=0.035,
):
    """Render soft front-wall/back-wall visibility without detaching geometry."""
    device, dtype = position.device, position.dtype
    fov_y = float(fov_x_half_tan) * float(height) / float(max(width, 1))
    row = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    col = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, zz = torch.meshgrid(row, col, indexing='ij')
    rays_camera = torch.stack((
        torch.ones_like(yy), yy * fov_y, zz * float(fov_x_half_tan)
    ), -1)
    rays = torch.einsum('bij,hwj->bhwi', camera_rotation, rays_camera)
    pos = position[:, None, None, :]
    ray_x = rays[..., 0]
    ray_x_safe = ray_x.clamp_min(0.05)
    wall_x = wall_x[:, None, None]
    back_wall_x = back_wall_x[:, None, None]
    front_t = ((wall_x - pos[..., 0]) / ray_x_safe).clamp(0.05, max_range)
    back_t = ((back_wall_x - pos[..., 0]) / ray_x_safe).clamp(0.05, max_range)
    front_hit = pos + front_t[..., None] * rays

    opening_y = torch.sigmoid(
        (slit_half_y[:, None, None] - (front_hit[..., 1] - slit_center_y[:, None, None]).abs())
        / temperature
    )
    opening_z = torch.sigmoid(
        (slit_half_z[:, None, None] - (front_hit[..., 2] - slit_center_z[:, None, None]).abs())
        / temperature
    )
    opening = opening_y * opening_z
    forward = torch.sigmoid((ray_x - 0.05) / temperature)
    depth = (1.0 - opening) * front_t + opening * back_t
    range_confidence = torch.sigmoid((max_range - depth) / 0.25)
    confidence = forward * range_confidence * (0.20 + 0.80 * opening)
    return depth, confidence, opening


def differentiable_look_at(position, target, up=None):
    """Construct a differentiable world-from-body frame aimed at target."""
    if up is None:
        up = torch.zeros_like(position)
        up[..., 2] = 1.0
    forward = torch.nn.functional.normalize(target - position, dim=-1, eps=1e-5)
    right = torch.nn.functional.normalize(torch.cross(up, forward, dim=-1), dim=-1, eps=1e-5)
    true_up = torch.cross(forward, right, dim=-1)
    return torch.stack((forward, right, true_up), -1)

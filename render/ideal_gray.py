"""Lightweight ideal grayscale appearance renderer built on geometric depth.

This module deliberately keeps scene geometry non-differentiable. It consumes
an already-rendered geometric depth map, reconstructs hit points and approximate
surface normals, then applies simple Lambertian lighting plus procedural
texture. The result is an *ideal irradiance/appearance* image that is later
passed through ``DifferentiableGrayCamera``.

The function uses the same camera-ray convention as ``render_depth_kernel`` in
``src/quadsim_kernel.cu``.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _camera_rays(
    R: torch.Tensor,
    height: int,
    width: int,
    fov_x_half_tan: float,
) -> torch.Tensor:
    """Return scene-frame rays with shape [B, H, W, 3]."""
    if R.ndim != 3 or R.shape[-2:] != (3, 3):
        raise ValueError("R must have shape [B, 3, 3]")
    device, dtype = R.device, R.dtype
    fov_x = torch.as_tensor(float(fov_x_half_tan), device=device, dtype=dtype)
    fov_y = fov_x * float(height) / float(max(width, 1))

    u = torch.arange(height, device=device, dtype=dtype)
    v = torch.arange(width, device=device, dtype=dtype)
    fu = (2.0 * (u + 0.5) / float(max(height, 1)) - 1.0) * fov_y - 1e-5
    fv = (2.0 * (v + 0.5) / float(max(width, 1)) - 1.0) * fov_x - 1e-5

    fwd = R[:, :, 0][:, None, None, :]
    left = R[:, :, 1][:, None, None, :]
    up = R[:, :, 2][:, None, None, :]
    return (
        fwd
        - fu[None, :, None, None] * up
        - fv[None, None, :, None] * left
    )


def _central_difference(points: torch.Tensor, dim: int) -> torch.Tensor:
    """Central difference with replicated boundary values."""
    if dim == 1:
        p = F.pad(points.permute(0, 3, 1, 2), (0, 0, 1, 1), mode="replicate")
        d = 0.5 * (p[:, :, 2:, :] - p[:, :, :-2, :])
        return d.permute(0, 2, 3, 1)
    if dim == 2:
        p = F.pad(points.permute(0, 3, 1, 2), (1, 1, 0, 0), mode="replicate")
        d = 0.5 * (p[:, :, :, 2:] - p[:, :, :, :-2])
        return d.permute(0, 2, 3, 1)
    raise ValueError("dim must be 1 or 2")


def _neighbor_depth_jump(depth: torch.Tensor) -> torch.Tensor:
    """Approximate local depth discontinuity magnitude."""
    d = depth[:, None]
    row = F.pad(d, (0, 0, 1, 1), mode="replicate")
    col = F.pad(d, (1, 1, 0, 0), mode="replicate")
    jump_row = 0.5 * (row[:, :, 2:, :] - row[:, :, :-2, :]).abs()
    jump_col = 0.5 * (col[:, :, :, 2:] - col[:, :, :, :-2]).abs()
    return torch.maximum(jump_row[:, 0], jump_col[:, 0])


def render_ideal_grayscale(
    depth: torch.Tensor,
    R: torch.Tensor,
    pos: torch.Tensor,
    *,
    fov_x_half_tan: float = 0.82,
    ambient: float = 0.25,
    diffuse: float = 0.75,
    light_direction=(-1.0, -0.3, 0.8),
    base_albedo: float = 0.65,
    texture_strength: float = 0.30,
    texture_scale: float = 5.0,
    background_intensity: float = 0.08,
    background_depth_threshold: float = 99.0,
    normal_depth_jump: float = 0.35,
    detach_geometry: bool = True,
    return_aux: bool = False,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """Convert geometric depth into an ideal grayscale irradiance image.

    ``depth`` is the geometric ray parameter produced by the existing CUDA
    renderer. In v1, geometry is deliberately detached: the differentiable
    path of interest starts at exposure/gain in ``DifferentiableGrayCamera``.
    """
    if depth.ndim != 3:
        raise ValueError("depth must have shape [B, H, W]")
    if R.ndim != 3 or R.shape != (depth.shape[0], 3, 3):
        raise ValueError("R must have shape [B, 3, 3]")
    if pos.ndim != 2 or pos.shape != (depth.shape[0], 3):
        raise ValueError("pos must have shape [B, 3]")
    if not torch.is_floating_point(depth):
        raise TypeError("depth must be floating point")

    if detach_geometry:
        depth = depth.detach()
        R = R.detach()
        pos = pos.detach()

    _, H, W = depth.shape
    rays = _camera_rays(R.to(depth), H, W, fov_x_half_tan)
    points = pos.to(depth)[:, None, None, :] + depth[..., None] * rays

    d_row = _central_difference(points, 1)
    d_col = _central_difference(points, 2)
    normal = torch.cross(d_col, d_row, dim=-1)
    normal = F.normalize(normal, p=2, dim=-1, eps=1e-6)

    view_to_camera = F.normalize(-rays, p=2, dim=-1, eps=1e-6)
    facing = (normal * view_to_camera).sum(dim=-1, keepdim=True)
    normal = torch.where(facing < 0.0, -normal, normal)

    jump = _neighbor_depth_jump(depth)
    unreliable = jump > float(normal_depth_jump)
    normal = torch.where(unreliable[..., None], view_to_camera, normal)

    light = torch.as_tensor(light_direction, device=depth.device, dtype=depth.dtype)
    if light.numel() != 3:
        raise ValueError("light_direction must contain 3 values")
    light = F.normalize(light.view(1, 1, 1, 3), p=2, dim=-1, eps=1e-6)
    ndotl = (normal * light).sum(dim=-1).clamp_min(0.0)

    x, y, z = points.unbind(dim=-1)
    tex = 0.5 + 0.25 * torch.sin(float(texture_scale) * x) + 0.25 * torch.sin(
        float(texture_scale) * (1.31 * y + 0.47 * z)
    )
    tex = tex.clamp(0.0, 1.0)
    strength = min(max(float(texture_strength), 0.0), 1.0)
    albedo = float(base_albedo) * ((1.0 - strength) + strength * (0.45 + 0.85 * tex))
    albedo = albedo.clamp(0.02, 1.0)

    irradiance = albedo * (float(ambient) + float(diffuse) * ndotl)
    irradiance = irradiance.clamp(0.0, 1.0)

    hit_mask = depth < float(background_depth_threshold)
    background = torch.full_like(irradiance, float(background_intensity))
    gray = torch.where(hit_mask, irradiance, background)[:, None]

    if not return_aux:
        return gray, None

    aux = {
        "hit_mask": hit_mask,
        "points": points,
        "normals": normal,
        "albedo": albedo,
        "ndotl": ndotl,
        "depth_jump": jump,
    }
    return gray, aux

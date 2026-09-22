"""Ideal monochrome irradiance from generic ray-hit geometry.

The CUDA geometry stage provides both ray-hit distance and exact surface normal.
This module applies material texture and lighting only. Geometry is detached by
default; active-sensing differentiability lives in the downstream grayscale
camera model with respect to exposure and gain.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F

ScalarLike = Union[float, torch.Tensor]


def _camera_rays(R: torch.Tensor, height: int, width: int, fov_x_half_tan: float) -> torch.Tensor:
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
    return fwd - fu[None, :, None, None] * up - fv[None, None, :, None] * left


def _batch_scalar(value: ScalarLike, ref: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(value, device=ref.device, dtype=ref.dtype)
    if x.ndim == 0:
        return x.view(1, 1, 1)
    if x.ndim == 1 and x.shape[0] in {1, ref.shape[0]}:
        return x.view(-1, 1, 1)
    raise ValueError("illumination scalar must be scalar or [B]")


def render_ideal_grayscale(
    depth: torch.Tensor,
    R: torch.Tensor,
    pos: torch.Tensor,
    normals: torch.Tensor,
    *,
    fov_x_half_tan: float = 0.82,
    ambient: ScalarLike = 0.25,
    diffuse: ScalarLike = 0.75,
    light_direction=(-1.0, -0.3, 0.8),
    base_albedo: float = 0.65,
    texture_strength: float = 0.30,
    texture_scale: float = 5.0,
    background_intensity: ScalarLike = 0.08,
    background_depth_threshold: float = 99.0,
    irradiance_max: float = 4.0,
    detach_geometry: bool = True,
    return_aux: bool = False,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """Convert geometric depth into ideal scene irradiance.

    The output is not clipped to [0, 1]; values above one represent bright
    scene irradiance and are intentionally left for the camera model to
    saturate according to exposure/gain.
    """
    if depth.ndim != 3:
        raise ValueError("depth must have shape [B, H, W]")
    if R.shape != (depth.shape[0], 3, 3):
        raise ValueError("R must have shape [B, 3, 3]")
    if pos.shape != (depth.shape[0], 3):
        raise ValueError("pos must have shape [B, 3]")
    if not torch.is_floating_point(depth):
        raise TypeError("depth must be floating point")
    if normals.shape != (depth.shape[0], depth.shape[1], depth.shape[2], 3):
        raise ValueError("normals must have shape [B,H,W,3]")
    if not torch.is_floating_point(normals):
        raise TypeError("normals must be floating point")

    if detach_geometry:
        depth, R, pos, normals = (
            depth.detach(),
            R.detach(),
            pos.detach(),
            normals.detach(),
        )

    _, H, W = depth.shape
    rays = _camera_rays(R.to(depth), H, W, fov_x_half_tan)
    points = pos.to(depth)[:, None, None, :] + depth[..., None] * rays

    view_to_camera = F.normalize(-rays, p=2, dim=-1, eps=1e-6)
    normal_norm = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
    normal = F.normalize(normals, p=2, dim=-1, eps=1e-6)
    normal = torch.where(normal_norm > 1e-6, normal, view_to_camera)

    # Keep lighting two-sided with respect to the viewing surface convention:
    # orient the normal toward the camera before Lambertian evaluation.
    normal = torch.where(
        ((normal * view_to_camera).sum(dim=-1, keepdim=True) < 0.0),
        -normal,
        normal,
    )

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

    ambient_b = _batch_scalar(ambient, depth)
    diffuse_b = _batch_scalar(diffuse, depth)
    irradiance = albedo * (ambient_b + diffuse_b * ndotl)
    irradiance = irradiance.clamp(0.0, float(irradiance_max))

    hit_mask = depth < float(background_depth_threshold)
    background = _batch_scalar(background_intensity, depth).expand_as(irradiance)
    gray = torch.where(hit_mask, irradiance, background)[:, None]

    if not return_aux:
        return gray, None
    return gray, {
        "hit_mask": hit_mask,
        "points": points,
        "normals": normal,
        "albedo": albedo,
        "ndotl": ndotl,
        "irradiance_mean": gray.flatten(1).mean(1),
    }

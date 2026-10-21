"""Sparse round poles, with identical geometry for pixels and body clearance.

Poles span the arena height. Outside that height the arena floor/ceiling
terminates the episode, so an infinite vertical cylinder is sufficient.
Visibility remains hard and piecewise differentiable, as in the box scene.
"""

import torch

from .box_scene import trace_boxes
from .ideal_gray import _camera_rays


def build_sparse_poles(env):
    batch, device = env.batch_size, env.device
    count = env.pole_count
    x = torch.arange(count, device=device).float() * env.pole_spacing + env.simple_wall_x
    x = x[None] + torch.empty(batch, count, device=device).uniform_(-env.pole_x_jitter, env.pole_x_jitter)
    # Stratification prevents an almost-always-clear outer lane in a tiny
    # three-pole benchmark. Permuting longitudinal order and adding continuous
    # jitter prevents a fixed scripted slalom. It is a controlled stand-in for
    # a sparse stand, not a natural-forest distribution.
    anchors = torch.linspace(-env.pole_lateral_range, env.pole_lateral_range, count, device=device)
    if count == 1:
        anchors.zero_()
    permutation = torch.rand(batch, count, device=device).argsort(1)
    y = anchors[permutation] + torch.empty(batch, count, device=device).uniform_(
        -env.pole_lateral_jitter, env.pole_lateral_jitter)
    radii = torch.empty(batch, count, device=device).uniform_(env.pole_radius_min, env.pole_radius_max)
    env.cyl = torch.stack([x, y, radii], -1)
    front, back = env.simple_start_x - 1, env.simple_goal_x + 1
    mid, length = (front + back) / 2, (back - front) / 2
    extent, zc, half_height, thick = env.pole_arena_half_width, env.simple_slit_center_z, 1.15, .1
    boxes = [
        [mid, -extent - thick, zc, length, thick, half_height],
        [mid, extent + thick, zc, length, thick, half_height],
        [mid, 0, zc - half_height - thick, length, extent, thick],
        [mid, 0, zc + half_height + thick, length, extent, thick],
        [back, 0, zc, thick, extent, half_height],
    ]
    env.voxels = torch.tensor(boxes, device=device, dtype=torch.float32)[None].expand(batch, -1, -1).clone()


def trace_cylinders(cylinders, rotation, position, height, width, fov_x_half_tan):
    """[B,N,3] vertical cylinders (x,y,r); return depth, normal, cylinder id."""
    rays = _camera_rays(rotation, height, width, fov_x_half_tan)
    direction = rays[:, None, :, :, :2]
    offset = position[:, None, None, None, :2] - cylinders[:, :, None, None, :2]
    a = direction.square().sum(-1)
    b = (offset * direction).sum(-1)
    c = offset.square().sum(-1) - cylinders[:, :, None, None, 2].square()
    disc = b.square() - a * c
    # Invalid/tangent branches must not generate NaN/inf gradients even when
    # hard visibility selects a different object.
    root = disc.clamp_min(1e-12).sqrt()
    safe_a = a.clamp_min(1e-12)
    near, far = (-b - root) / safe_a, (-b + root) / safe_a
    distance = torch.where(near > 1e-6, near, far)
    valid = (disc > 0) & (a > 1e-12) & (distance > 1e-6) & (distance < 100)
    distance = torch.where(valid, distance, torch.full_like(distance, 100))
    depth, ids = distance.min(1)
    center = cylinders[:, :, :2].gather(1, ids.flatten(1)[..., None].expand(-1, -1, 2))
    center = center.reshape(position.shape[0], height, width, 2)
    hit_xy = position[:, None, None, :2] + depth[..., None] * rays[..., :2]
    normal_xy = torch.nn.functional.normalize(hit_xy - center, dim=-1, eps=1e-8)
    normal = torch.cat([normal_xy, torch.zeros_like(normal_xy[..., :1])], -1)
    normal = torch.where((depth < 100)[..., None], normal, torch.zeros_like(normal))
    return depth, normal, ids


def trace_pole_scene(env, position, rotation):
    box_depth, box_normal, box_id = trace_boxes(
        env.voxels, rotation, position, env.height, env.width, env._fov_x_half_tan)
    pole_depth, pole_normal, pole_id = trace_cylinders(
        env.cyl, rotation, position, env.height, env.width, env._fov_x_half_tan)
    pole_hit = pole_depth < box_depth
    return (torch.where(pole_hit, pole_depth, box_depth),
            torch.where(pole_hit[..., None], pole_normal, box_normal),
            torch.where(pole_hit, pole_id + env.voxels.shape[1], box_id))


def cylinder_clearance(points, cylinders, radius=0.0):
    """Exact sphere/vertical-cylinder signed clearance: [S,B,3] -> [S,B]."""
    distance = (points[:, :, None, :2] - cylinders[None, :, :, :2]).norm(dim=-1)
    return (distance - cylinders[None, :, :, 2] - float(radius)).amin(-1)


def swept_cylinder_clearance(start, end, cylinders, radius=0.0):
    """Exact closest segment distance in XY, including stationary segments."""
    direction = (end - start)[:, None, :2]
    offset = cylinders[:, :, :2] - start[:, None, :2]
    fraction = ((offset * direction).sum(-1) / direction.square().sum(-1).clamp_min(1e-12)).clamp(0, 1)
    closest = start[:, None, :2] + fraction[..., None] * direction
    return ((closest - cylinders[:, :, :2]).norm(dim=-1) - cylinders[:, :, 2] - float(radius)).amin(-1)


def surface_light_scale(env, points):
    """A fixed smooth light field, evaluated at surfaces, never at the camera."""
    blend = torch.sigmoid((points[..., 0] - env.gray_transition_x) / env.gray_transition_width)
    scale = torch.ones_like(blend)
    for idx, name in enumerate(env.current_scene_names):
        if name == "dark":
            scale[idx] = env.gray_dark_scale
        elif name == "bright":
            scale[idx] = env.gray_bright_scale
        elif name == "dark_to_bright":
            scale[idx] = env.gray_dark_scale + (env.gray_bright_scale - env.gray_dark_scale) * blend[idx]
        elif name == "bright_to_dark":
            scale[idx] = env.gray_bright_scale + (env.gray_dark_scale - env.gray_bright_scale) * blend[idx]
    return scale

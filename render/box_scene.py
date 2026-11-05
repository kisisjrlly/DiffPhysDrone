"""Piecewise differentiable box geometry for the intent-gates benchmark.

Ray visibility is hard, not a soft occupancy field. Gradients are valid inside
each visibility region; silhouette/disocclusion derivatives are not invented.
All outputs are simulator internals and must never be policy observations.
"""

import torch
import torch.nn.functional as F

from .ideal_gray import _camera_rays


def trace_boxes(boxes, rotation, position, height, width, fov_x_half_tan):
    """Return ray parameter, surface normal and object id for [B,N,6] boxes."""
    rays = _camera_rays(rotation, height, width, fov_x_half_tan)
    direction = rays[:, None]
    origin = position[:, None, None, None, :]
    centers = boxes[:, :, None, None, :3]
    half = boxes[:, :, None, None, 3:]
    # Parallel rays outside a slab must miss; finite denominators avoid NaN in
    # backward even for branches masked out by hard visibility.
    parallel = direction.abs() < 1e-9
    safe = torch.where(parallel, torch.ones_like(direction), direction)
    t0, t1 = (centers - half - origin) / safe, (centers + half - origin) / safe
    lo, hi = torch.minimum(t0, t1), torch.maximum(t0, t1)
    inside = (origin >= centers - half) & (origin <= centers + half)
    lo = torch.where(parallel, torch.where(inside, -1e6, 1e6), lo)
    hi = torch.where(parallel, torch.where(inside, 1e6, -1e6), hi)
    enter, axis_enter = lo.max(-1)
    leave, axis_leave = hi.min(-1)
    outside = enter >= 0
    distance = torch.where(outside, enter, leave)
    valid = (leave >= enter) & (leave > 1e-6) & (distance < 100)
    distance = torch.where(valid, distance, torch.full_like(distance, 100))
    depth, object_id = distance.min(1)
    axis = torch.where(outside, axis_enter, axis_leave)
    sign = torch.where(outside[..., None], -direction.sign(), direction.sign())
    normals = F.one_hot(axis, 3).to(rays) * sign
    index = object_id[:, None, :, :, None].expand(-1, 1, -1, -1, 3)
    normal = normals.gather(1, index).squeeze(1)
    normal = torch.where((depth < 100)[..., None], normal, torch.zeros_like(normal))
    return depth, normal, object_id


def box_clearance(points, boxes, radius=0.0):
    """Signed sphere-to-box clearance, [S,B,3] points -> [S,B]."""
    q = (points[:, :, None] - boxes[None, :, :, :3]).abs() - boxes[None, :, :, 3:]
    sdf = q.clamp_min(0).norm(dim=-1) + q.amax(-1).clamp_max(0)
    return sdf.amin(-1) - float(radius)


def expanded_box_sweep(start, end, boxes, radius):
    """Conservative continuous sweep against radius-expanded boxes.

    Uses a box enclosure of the spherical body at corners; the regular signed
    clearance remains spherical. This prevents tunnelling through a thin wall.
    """
    origin = start[:, None]
    direction = (end - start)[:, None]
    half = boxes[:, :, 3:] + float(radius)
    parallel = direction.abs() < 1e-9
    safe = torch.where(parallel, torch.ones_like(direction), direction)
    t0 = (boxes[:, :, :3] - half - origin) / safe
    t1 = (boxes[:, :, :3] + half - origin) / safe
    inside = (origin - boxes[:, :, :3]).abs() <= half
    lo = torch.where(parallel, torch.where(inside, -1e6, 1e6), torch.minimum(t0, t1))
    hi = torch.where(parallel, torch.where(inside, 1e6, -1e6), torch.maximum(t0, t1))
    enter, leave = lo.amax(-1), hi.amin(-1)
    return ((leave >= enter) & (leave >= 0) & (enter <= 1)).any(-1)


def swept_box_clearance(start, end, boxes, radius):
    """Continuous sphere clearance for linear center motion, shape [B].

    Outside a box, squared point-to-box distance is a quadratic between
    successive crossings of its six slab planes. Evaluate each interval's
    endpoints and stationary point, so even a thin wall cannot be skipped.
    Intersections have clearance <= -radius; sampled signed distances retain
    deeper interior penalties. Exterior values (including positive margins)
    are exact up to floating point rounding. No expanded-corner enclosure.
    """
    origin = start[:, None, :]
    direction = (end - start)[:, None, :]
    lower, upper = boxes[..., :3] - boxes[..., 3:], boxes[..., :3] + boxes[..., 3:]
    parallel = direction.abs() < 1e-12
    safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
    crossings = torch.cat([(lower - origin) / safe_direction,
                           (upper - origin) / safe_direction], -1)
    crossings = torch.where(parallel.expand_as(origin + lower).repeat(1, 1, 2),
                            torch.zeros_like(crossings), crossings).clamp(0, 1)
    ends = torch.cat([torch.zeros_like(crossings[..., :1]), crossings,
                      torch.ones_like(crossings[..., :1])], -1).sort(-1).values
    left, right = ends[..., :-1], ends[..., 1:]
    middle = origin[..., None, :] + ((left + right) * .5)[..., None] * direction[..., None, :]
    outside_low, outside_high = middle < lower[..., None, :], middle > upper[..., None, :]
    active = outside_low | outside_high
    boundary = torch.where(outside_low, lower[..., None, :], upper[..., None, :])
    intercept = torch.where(active, origin[..., None, :] - boundary, torch.zeros_like(middle))
    slope = torch.where(active, direction[..., None, :], torch.zeros_like(middle))
    denominator = slope.square().sum(-1)
    stationary = -(intercept * slope).sum(-1) / denominator.clamp_min(1e-24)
    stationary = torch.maximum(left, torch.minimum(right, stationary))
    candidates = torch.cat([ends, stationary], -1)
    points = origin[..., None, :] + candidates[..., None] * direction[..., None, :]
    delta = (points - boxes[..., None, :3]).abs() - boxes[..., None, 3:]
    # vector_norm has a finite zero subgradient, unlike an unguarded sqrt.
    distances = torch.linalg.vector_norm(delta.clamp_min(0), dim=-1)
    outside_minimum = distances.amin(dim=(-1, -2)) - float(radius)
    fractions = torch.linspace(0, 1, 17, device=start.device, dtype=start.dtype)
    samples = start[None] + fractions[:, None, None] * (end - start)[None]
    signed_minimum = box_clearance(samples, boxes, radius).amin(0)
    return torch.minimum(outside_minimum, signed_minimum)


def swept_box_collision(start, end, boxes, radius):
    """Sphere/box contact during a complete linear motion segment."""
    return swept_box_clearance(start, end, boxes, radius) <= 0

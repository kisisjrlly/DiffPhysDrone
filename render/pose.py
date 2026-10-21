"""Differentiable counterparts of the existing point-mass attitude model."""

import math

import torch
import torch.nn.functional as F

from utils import g_decay


def update_attitude(rotation, acceleration, target_direction, alpha, yaw_inertia=5.0):
    gravity = acceleration.new_tensor([0.0, 0.0, 9.80665])
    up = F.normalize(acceleration + gravity, dim=-1, eps=1e-5)
    previous = rotation[:, :, 0]
    forward = F.normalize(previous * yaw_inertia + target_direction, dim=-1, eps=1e-6)
    forward = (1 - alpha) * forward + alpha * previous
    uz = up[:, 2]
    safe_uz = torch.where(uz.abs() > 1e-4, uz, torch.ones_like(uz))
    fz = -(forward[:, :2] * up[:, :2]).sum(-1) / safe_uz
    fz = torch.where(uz.abs() > 1e-4, fz, forward[:, 2])
    forward = F.normalize(torch.cat([forward[:, :2], fz[:, None]], -1), dim=-1, eps=1e-6)
    left = torch.cross(up, forward, dim=-1)
    return torch.stack([forward, left, up], -1)


def rotation_exp(vector):
    """Rodrigues formula, including finite derivatives at zero rotation."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(-1, 3, 3)
    angle = vector.norm(dim=-1)
    a = torch.sinc(angle / math.pi)
    b = 0.5 * torch.sinc(angle / (2 * math.pi)).square()
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device).expand_as(skew)
    return eye + a[:, None, None] * skew + b[:, None, None] * (skew @ skew)


def body_angular_velocity(previous, current, dt):
    """Local constant-rate estimate from two attitudes (<pi radians/step)."""
    relative = previous.transpose(1, 2) @ current
    vector = 0.5 * torch.stack([
        relative[:, 2, 1] - relative[:, 1, 2],
        relative[:, 0, 2] - relative[:, 2, 0],
        relative[:, 1, 0] - relative[:, 0, 1],
    ], -1)
    sine = vector.norm(dim=-1)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    angle = torch.atan2(sine, cosine)
    scale = torch.where(sine > 1e-6, angle / sine.clamp_min(1e-6), torch.ones_like(sine))
    return vector * scale[:, None] / float(dt)


def dynamics_step(rotation, disturbance, z_drag, drag, rate, command, applied,
                  position, velocity, wind, acceleration, grad_decay, dt, airmode=0.5):
    """Same forward equations as run_forward; autograd includes attitude/drag.

    atan2(cross,dot) is used instead of acos for a finite subgradient at zero
    attitude change. Gradient decay preserves the existing training semantics.
    """
    alpha = torch.exp(-rate * dt)
    next_applied = command * (1 - alpha) + applied * alpha
    local = ((velocity - wind)[:, None] @ rotation).squeeze(1)
    axis_scale = torch.cat([torch.ones_like(z_drag).expand(-1, 2), z_drag], -1)
    drag_local = (local * local.abs() * drag[:, :1] + local * drag[:, 1:]) * axis_scale
    drag_world = (rotation @ drag_local[:, :, None]).squeeze(-1)
    gravity = applied.new_tensor([0.0, 0.0, 9.80665])
    old_thrust, new_thrust = applied + gravity, next_applied + gravity
    angle = torch.atan2(torch.cross(old_thrust, new_thrust, dim=-1).norm(dim=-1),
                        (old_thrust * new_thrust).sum(-1))
    airmode_acc = F.normalize(old_thrust, dim=-1, eps=1e-5) * (angle / dt * airmode)[:, None]
    next_acc = next_applied + disturbance - drag_world + airmode_acc
    next_position = g_decay(position, grad_decay ** dt) + velocity * dt + acceleration * (0.5 * dt ** 2)
    next_velocity = g_decay(velocity, grad_decay ** dt) + (acceleration + next_acc) * (0.5 * dt)
    return next_applied, next_position, next_velocity, next_acc

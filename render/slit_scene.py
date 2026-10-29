"""Opt-in single-slit joint-sensing diagnostic; frozen legacy scene is unchanged.

The original three boxes (one slit wall and its background) are retained.
Side walls, floor and ceiling close bypass routes. Illumination is a fixed
world-space field, visible on the approach surfaces and the slit wall at once.
It is never a function of the observing drone's position.
"""

import torch


def build_joint_slit(env):
    z, thickness = env.simple_slit_center_z, env.simple_wall_half_x
    front = env.simple_start_x - 1.0
    back = env.back_wall_x
    mid, length = (front + back) / 2, (back - front) / 2
    half_width, half_height = 1.2, 1.0
    rows = []
    for center, half_length in zip(mid, length):
        rows.append([
            [center, -half_width - thickness, z, half_length, thickness, half_height],
            [center, half_width + thickness, z, half_length, thickness, half_height],
            [center, 0., z - half_height - thickness, half_length, half_width, thickness],
            [center, 0., z + half_height + thickness, half_length, half_width, thickness],
        ])
    env.voxels = torch.cat([env.voxels, env.voxels.new_tensor(rows)], 1)


def slit_surface_light(env, points):
    """Illumination at ray-hit points in scene coordinates, shape [B,H,W]."""
    x = points[..., 0]
    blend = torch.sigmoid((x - env.gray_transition_x) / env.gray_transition_width)
    levels = []
    for index, name in enumerate(env.current_scene_names):
        if name == "dark":
            value = torch.full_like(x[index], env.gray_dark_scale)
        elif name == "bright":
            value = torch.full_like(x[index], env.gray_bright_scale)
        elif name == "dark_to_bright":
            value = env.gray_dark_scale + (env.gray_bright_scale - env.gray_dark_scale) * blend[index]
        elif name == "bright_to_dark":
            value = env.gray_bright_scale + (env.gray_dark_scale - env.gray_bright_scale) * blend[index]
        else:
            value = torch.ones_like(x[index])
        levels.append(value)
    return torch.stack(levels)

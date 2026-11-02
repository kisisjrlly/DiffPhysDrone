"""Opt-in closed corridor with a hidden left/right blockage."""
import torch


def build_occluded_route(env, blocked_side=None, divider_half_width=.55, blockage_x=1.8):
    b, dev = env.batch_size, env.device
    if blocked_side is None:
        blocked_side = torch.randint(0, 2, (b,), device=dev) * 2 - 1
    env.route_blocked_side = torch.as_tensor(blocked_side, device=dev).expand(b).clone()
    if not torch.all((env.route_blocked_side == -1) | (env.route_blocked_side == 1)):
        raise ValueError("blocked side must be -1 or +1")
    if not .12 < divider_half_width < 1.16 or not .1 < blockage_x < 2.1:
        raise ValueError("divider/blockage must leave navigable enclosed routes")
    env.route_divider_half_width = float(divider_half_width)
    env.route_blockage_x = float(blockage_x)
    rows = []
    for side in env.route_blocked_side.tolist():
        w = divider_half_width
        rows.append([
            [1.1, 0, 1.5, 1.1, w, 1.0],
            [blockage_x, side * (1.4 + w) / 2, 1.5, .10, (1.4 - w) / 2, 1.0],
            [1., -1.5, 1.5, 3., .1, 1.1],
            [1., 1.5, 1.5, 3., .1, 1.1],
            [1., 0, .4, 3., 1.6, .1],
            [1., 0, 2.6, 3., 1.6, .1],
            [4., 0, 1.5, .1, 1.6, 1.1],
            [-2., 0, 1.5, .1, 1.6, 1.1],
        ])
    env.voxels = torch.tensor(rows, device=dev, dtype=torch.float32)
    levels = [env.gray_dark_scale if s == 'dark' else env.gray_bright_scale
              for s in env.current_scene_names]
    env.surface_lights = env.voxels.new_tensor(levels)[:, None].expand(b, 8).clone()

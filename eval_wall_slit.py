"""Evaluation script for the wall-slit environment.

Loads a trained checkpoint and runs N episodes in wall_slit mode,
reporting pass-through rate, collision rate, and other statistics.

Usage:
    python eval_wall_slit.py --resume checkpoint0004.pth [--num_episodes 200] [--batch_size 64]
"""

import argparse
import math
import random
from random import normalvariate
import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

from env_cuda import Env
from model import Model


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate wall-slit environment')
    parser.add_argument('--resume', required=True, help='Path to model checkpoint')
    parser.add_argument('--num_episodes', type=int, default=200, help='Number of evaluation episodes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for parallel evaluation')
    parser.add_argument('--timesteps', type=int, default=120, help='Timesteps per episode')
    parser.add_argument('--speed_mtp', type=float, default=0.5, help='Speed multiplier')
    parser.add_argument('--fov_x_half_tan', type=float, default=0.82)
    parser.add_argument('--cam_angle', type=int, default=10)
    parser.add_argument('--grad_decay', type=float, default=0.4)
    parser.add_argument('--drone_a', type=float, default=0.15, help='Ellipsoid semi-axis XY')
    parser.add_argument('--drone_c', type=float, default=0.075, help='Ellipsoid semi-axis Z')
    parser.add_argument('--ellipsoid_collision', default=False, action='store_true')
    parser.add_argument('--no_odom', default=False, action='store_true')
    parser.add_argument('--save_gif', default=False, action='store_true', help='Save a GIF of one episode')
    return parser.parse_args()


@torch.no_grad()
def evaluate_batch(env, model, args, device):
    """Run one batch of episodes, return per-drone statistics."""
    B = args.batch_size
    env.reset()
    model.reset()

    p_history = []
    distance_history = []
    h = None

    act_lag = 1
    act_buffer = [env.act] * (act_lag + 1)
    target_v_raw = env.p_target - env.p
    depth_frames = []  # for GIF

    for t in range(args.timesteps):
        ctl_dt = 1 / 15  # fixed dt for evaluation
        depth, flow = env.render(ctl_dt)
        p_history.append(env.p.clone())

        vec_to_pt = env.find_vec_to_nearest_pt()
        dist = torch.norm(vec_to_pt, 2, -1)  # (sub_steps, B)
        distance_history.append(dist)

        if args.save_gif and t % 2 == 0:
            depth_frames.append(depth[0].cpu())

        target_v_raw = env.p_target - env.p
        env.run(act_buffer[t], ctl_dt, target_v_raw)

        R = env.R
        fwd = env.R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd = F.normalize(fwd, 2, -1)
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1)

        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / target_v_norm
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)
        state = [
            torch.squeeze(target_v[:, None] @ R, 1),
            env.R[:, 2],
            env.margin[:, None]]
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom:
            state.insert(0, local_v)
        state = torch.cat(state, -1)

        x = 3 / depth.clamp_(0.3, 24) - 0.6
        x = F.max_pool2d(x[:, None], 4, 4)
        act, _, h = model(x, state, h)

        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        act_out = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act_out)

    # Compute metrics
    p_history = torch.stack(p_history)  # (T, B, 3)
    distance_history = torch.stack(distance_history)  # (T, sub_steps, B)

    # Min distance across all timesteps and sub-steps for each drone
    min_dist_per_drone = distance_history.flatten(0, 1).min(0).values - env.margin  # (B,)
    no_collision = min_dist_per_drone > 0  # (B,)

    # Did drone cross the wall?
    final_x = p_history[-1, :, 0]
    crossed = final_x > env.wall_x  # (B,)

    # Pass = crossed AND no collision
    passed = crossed & no_collision

    # Time to first cross wall (or -1 if never crossed)
    cross_time = torch.full((B,), -1.0, device=device)
    for t_idx in range(p_history.shape[0]):
        just_crossed = (p_history[t_idx, :, 0] > env.wall_x) & (cross_time < 0)
        cross_time[just_crossed] = t_idx / 15.0  # convert to seconds

    results = {
        'no_collision': no_collision.cpu(),
        'crossed': crossed.cpu(),
        'passed': passed.cpu(),
        'min_dist': min_dist_per_drone.cpu(),
        'cross_time': cross_time.cpu(),
        'final_x': final_x.cpu(),
        'wall_x': env.wall_x,
    }

    if args.save_gif and len(depth_frames) > 0:
        results['depth_frames'] = depth_frames

    return results


def main():
    args = parse_args()
    device = torch.device('cuda')

    env = Env(args.batch_size, 64, 48, args.grad_decay, device,
              fov_x_half_tan=args.fov_x_half_tan, single=True,
              wall_slit=True, speed_mtp=args.speed_mtp,
              cam_angle=args.cam_angle,
              ellipsoid_a=args.drone_a if args.ellipsoid_collision else 0.0,
              ellipsoid_c=args.drone_c if args.ellipsoid_collision else 0.0)

    if args.no_odom:
        model = Model(7, 6)
    else:
        model = Model(7 + 3, 6)
    model = model.to(device)

    state_dict = torch.load(args.resume, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
    if missing_keys:
        print("Warning - missing_keys:", missing_keys)
    if unexpected_keys:
        print("Warning - unexpected_keys:", unexpected_keys)
    model.eval()

    num_batches = (args.num_episodes + args.batch_size - 1) // args.batch_size
    total_episodes = num_batches * args.batch_size

    all_no_collision = []
    all_crossed = []
    all_passed = []
    all_min_dist = []
    all_cross_time = []
    gif_frames = None

    print(f"\nEvaluating wall-slit with {total_episodes} episodes "
          f"({'ellipsoid' if args.ellipsoid_collision else 'point'} collision)...\n")

    for batch_i in tqdm(range(num_batches), desc='Evaluating'):
        results = evaluate_batch(env, model, args, device)
        all_no_collision.append(results['no_collision'])
        all_crossed.append(results['crossed'])
        all_passed.append(results['passed'])
        all_min_dist.append(results['min_dist'])
        all_cross_time.append(results['cross_time'])

        if batch_i == 0 and args.save_gif and 'depth_frames' in results:
            gif_frames = results['depth_frames']

    all_no_collision = torch.cat(all_no_collision)
    all_crossed = torch.cat(all_crossed)
    all_passed = torch.cat(all_passed)
    all_min_dist = torch.cat(all_min_dist)
    all_cross_time = torch.cat(all_cross_time)

    n = len(all_no_collision)
    print(f"\n{'='*60}")
    print(f"  Wall-Slit Evaluation Results ({n} episodes)")
    print(f"{'='*60}")
    print(f"  No-collision rate:  {all_no_collision.float().mean():.3f}  ({all_no_collision.sum()}/{n})")
    print(f"  Wall-crossed rate:  {all_crossed.float().mean():.3f}  ({all_crossed.sum()}/{n})")
    print(f"  Pass-through rate:  {all_passed.float().mean():.3f}  ({all_passed.sum()}/{n})")
    print(f"  Min obstacle dist:  {all_min_dist.mean():.4f} ± {all_min_dist.std():.4f}")
    valid_times = all_cross_time[all_cross_time >= 0]
    if len(valid_times) > 0:
        print(f"  Avg crossing time:  {valid_times.mean():.2f}s ± {valid_times.std():.2f}s")
    else:
        print(f"  Avg crossing time:  N/A (no successful crossings)")
    print(f"{'='*60}\n")

    # Save GIF
    if args.save_gif and gif_frames is not None:
        try:
            import imageio
            gif_path = 'gifs/wall_slit_eval.gif'
            frames_np = [(f.div(10).clamp(0, 1).numpy() * 255).astype(np.uint8) for f in gif_frames]
            imageio.mimsave(gif_path, frames_np, fps=7)
            print(f"  Saved evaluation GIF to {gif_path}")
        except Exception as e:
            print(f"  Could not save GIF: {e}")


if __name__ == '__main__':
    main()

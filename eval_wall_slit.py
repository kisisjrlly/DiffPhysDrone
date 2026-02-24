"""Evaluation script for the wall-slit environment. (墙缝环境评估脚本)

Loads a trained checkpoint and runs N episodes in wall_slit mode,
reporting pass-through rate, collision rate, and other statistics.
(加载训练好的检查点并在 wall_slit 模式下运行 N 个回合，报告通过率、碰撞率和其他统计数据。)

Usage:
    python eval_wall_slit.py --resume checkpoint0004.pth [--num_episodes 200] [--batch_size 64]
    python eval_wall_slit.py --resume checkpoint0004.pth --paper_unified_control --paper_cam_obs --ellipsoid_collision
"""

import argparse
import math
import random
from random import normalvariate
import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

from env_cuda import Env, apply_camera_effects
from model import Model


def parse_args():
    """解析命令行参数 (Parse command line arguments)"""
    parser = argparse.ArgumentParser(description='Evaluate wall-slit environment')
    parser.add_argument('--resume', required=True, help='Path to model checkpoint (模型检查点路径)')
    parser.add_argument('--num_episodes', type=int, default=200, help='Number of evaluation episodes (评估回合数)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for parallel evaluation (并行评估的批次大小)')
    parser.add_argument('--timesteps', type=int, default=120, help='Timesteps per episode (每个回合的时间步数)')
    parser.add_argument('--speed_mtp', type=float, default=0.5, help='Speed multiplier (速度乘数)')
    parser.add_argument('--fov_x_half_tan', type=float, default=0.82, help='FOV x half tan (水平视场角一半的正切值)')
    parser.add_argument('--cam_angle', type=int, default=10, help='Camera angle (相机角度)')
    parser.add_argument('--grad_decay', type=float, default=0.4, help='Gradient decay (梯度衰减)')
    parser.add_argument('--drone_a', type=float, default=0.15, help='Ellipsoid semi-axis XY (椭球体 XY 半轴)')
    parser.add_argument('--drone_c', type=float, default=0.075, help='Ellipsoid semi-axis Z (椭球体 Z 半轴)')
    parser.add_argument('--ellipsoid_collision', default=False, action='store_true', help='Use ellipsoid collision (使用椭球体碰撞检测)')
    parser.add_argument('--no_odom', default=False, action='store_true', help='No odometry (无里程计)')
    parser.add_argument('--save_gif', default=False, action='store_true', help='Save a GIF of one episode (保存一个回合的 GIF)')
    # Paper.md model architecture flags (论文模型架构标志)
    parser.add_argument('--diff_cam', default=False, action='store_true',
                        help='Legacy diff_cam model (separate camera head) (传统的 diff_cam 模型 (独立的相机头))')
    parser.add_argument('--paper_unified_control', default=False, action='store_true',
                        help='Paper §2.1: unified control model (camera deltas in action) (论文 §2.1: 统一控制模型 (动作中包含相机增量))')
    parser.add_argument('--paper_cam_obs', default=False, action='store_true',
                        help='Paper §2.1: camera state in observation (论文 §2.1: 观测中包含相机状态)')
    parser.add_argument('--cam_delta_scale', type=float, default=0.05,
                        help='Per-step scale for incremental camera deltas (增量相机参数的每步缩放比例)')
    return parser.parse_args()


@torch.no_grad()
def evaluate_batch(env, model, args, device):
    """Run one batch of episodes, return per-drone statistics. (运行一批回合，返回每架无人机的统计数据)"""
    B = args.batch_size # 批次大小 (Batch size)
    use_cam = args.diff_cam or args.paper_unified_control # 是否使用相机控制 (Whether to use camera control)
    env.reset() # 重置环境 (Reset environment)
    model.reset() # 重置模型 (Reset model)

    # 记录历史数据 (Record history data)
    p_history = []
    distance_history = []
    roll_history = []
    cam_fov_history = []
    cam_exp_history = []
    speed_history_eval = []
    h = None # RNN 隐藏状态 (RNN hidden state)

    act_lag = 1 # 动作延迟 (Action lag)
    act_buffer = [env.act] * (act_lag + 1) # 动作缓冲区 (Action buffer)
    target_v_raw = env.p_target - env.p # 原始目标速度 (Raw target velocity)
    depth_frames = []  # for GIF (用于生成 GIF 的深度图帧)

    # Initialize camera params (初始化相机参数)
    if use_cam:
        cam_fov = torch.full((B,), env._fov_x_half_tan, device=device)
        cam_exposure = torch.full((B,), 0.5, device=device)
        cam_iso = torch.full((B,), 0.5, device=device)
        cam_focus = torch.full((B,), 0.5, device=device)

    for t in range(args.timesteps):
        ctl_dt = 1 / 15  # fixed dt for evaluation (评估时固定的控制时间步长)

        # 渲染深度图 (Render depth map)
        if use_cam:
            depth = env.render_diff(cam_fov)
            depth = apply_camera_effects(depth, cam_exposure, cam_iso, cam_focus)
        else:
            depth, flow = env.render(ctl_dt)
        p_history.append(env.p.clone()) # 记录位置 (Record position)

        # 计算到最近障碍物的距离 (Calculate distance to nearest obstacle)
        vec_to_pt = env.find_vec_to_nearest_pt()
        dist = torch.norm(vec_to_pt, 2, -1)  # (sub_steps, B)
        distance_history.append(dist)

        # Track roll angle (angle of up-vector from vertical) (跟踪滚转角 (向上向量与垂直方向的夹角))
        up_vec = env.R[:, :, 2]  # (B, 3)
        roll_angle = torch.acos(up_vec[:, 2].clamp(-1, 1))  # (B,) radians
        roll_history.append(roll_angle * 180 / math.pi) # 转换为角度 (Convert to degrees)
        speed_history_eval.append(env.v.norm(2, -1)) # 记录速度 (Record speed)
        
        if use_cam:
            cam_fov_history.append(cam_fov.clone())
            cam_exp_history.append(cam_exposure.clone())

        # 保存 GIF 帧 (Save GIF frames)
        if args.save_gif and t % 2 == 0:
            depth_frames.append(depth[0].cpu())

        target_v_raw = env.p_target - env.p # 更新目标速度 (Update target velocity)
        env.run(act_buffer[t], ctl_dt, target_v_raw) # 运行环境一步 (Run environment for one step)

        # 计算无人机局部坐标系 (Calculate drone local coordinate system)
        R = env.R
        fwd = env.R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd = F.normalize(fwd, 2, -1)
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1)

        # 计算目标速度向量 (Calculate target velocity vector)
        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / target_v_norm
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)
        
        # 构建状态观测 (Construct state observation)
        state = [
            torch.squeeze(target_v[:, None] @ R, 1), # 局部目标速度 (Local target velocity)
            env.R[:, 2], # 向上向量 (Up vector)
            env.margin[:, None]] # 碰撞裕度 (Collision margin)
        local_v = torch.squeeze(env.v[:, None] @ R, 1) # 局部速度 (Local velocity)
        if not args.no_odom:
            state.insert(0, local_v)
            
        # Paper: include camera state in observation (论文：在观测中包含相机状态)
        if args.paper_cam_obs and use_cam:
            cam_obs = torch.stack([
                cam_fov / env._fov_x_half_tan - 1.0,
                cam_exposure, cam_iso, cam_focus
            ], -1)
            state.append(cam_obs)
        state = torch.cat(state, -1)

        # 处理深度图输入 (Process depth map input)
        if use_cam:
            x = 3 / depth.clamp(0.3, 24) - 0.6
        else:
            x = 3 / depth.clamp_(0.3, 24) - 0.6
        x = F.max_pool2d(x[:, None], 4, 4) # 最大池化降采样 (Max pooling downsampling)
        
        # 模型前向传播 (Model forward pass)
        act, cam_params, h = model(x, state, h)

        # Update camera parameters (更新相机参数)
        if args.paper_unified_control and cam_params is not None:
            delta_fov, delta_exp, delta_iso, delta_focus = cam_params.unbind(-1)
            scale = args.cam_delta_scale
            cam_fov = (cam_fov + delta_fov * scale * env._fov_x_half_tan).clamp(
                env._fov_x_half_tan * 0.3, env._fov_x_half_tan * 2.0)
            cam_exposure = (cam_exposure + delta_exp * scale).clamp(0.01, 0.99)
            cam_iso = (cam_iso + delta_iso * scale).clamp(0.01, 0.99)
            cam_focus = (cam_focus + delta_focus * scale).clamp(0.01, 0.99)
        elif cam_params is not None:
            fov_delta, exposure, iso, focus_dist = cam_params.unbind(-1)
            cam_fov = env._fov_x_half_tan * (0.5 + fov_delta)
            cam_exposure = exposure
            cam_iso = iso
            cam_focus = focus_dist

        # 计算输出动作 (Calculate output action)
        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        act_out = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act_out)

    # Compute metrics (计算评估指标)
    p_history = torch.stack(p_history)  # (T, B, 3)
    distance_history = torch.stack(distance_history)  # (T, sub_steps, B)
    roll_history = torch.stack(roll_history)  # (T, B)

    # Min distance across all timesteps and sub-steps for each drone (每架无人机在所有时间步和子步中的最小距离)
    min_dist_per_drone = distance_history.flatten(0, 1).min(0).values - env.margin  # (B,)
    no_collision = min_dist_per_drone > 0  # (B,) 是否无碰撞 (Whether no collision occurred)

    # Did drone cross the wall? (无人机是否穿过墙壁？)
    final_x = p_history[-1, :, 0]
    crossed = final_x > env.wall_x  # (B,)

    # Pass = crossed AND no collision (通过 = 穿过墙壁 且 无碰撞)
    passed = crossed & no_collision

    # Time to first cross wall (or -1 if never crossed) (首次穿过墙壁的时间 (如果从未穿过则为 -1))
    cross_time = torch.full((B,), -1.0, device=device)
    for t_idx in range(p_history.shape[0]):
        just_crossed = (p_history[t_idx, :, 0] > env.wall_x) & (cross_time < 0)
        cross_time[just_crossed] = t_idx / 15.0  # convert to seconds (转换为秒)

    # Roll angle near wall (墙壁附近的滚转角)
    dx_wall = (p_history[..., 0] - env.wall_x).abs()  # (T, B)
    near_wall = dx_wall < 1.0
    roll_at_wall = roll_history[near_wall].mean() if near_wall.any() else torch.tensor(0.0)
    max_roll = roll_history.max(0).values  # (B,) 最大滚转角 (Max roll angle)

    results = {
        'no_collision': no_collision.cpu(),
        'crossed': crossed.cpu(),
        'passed': passed.cpu(),
        'min_dist': min_dist_per_drone.cpu(),
        'cross_time': cross_time.cpu(),
        'final_x': final_x.cpu(),
        'wall_x': env.wall_x,
        'roll_at_wall': roll_at_wall.cpu(),
        'max_roll': max_roll.cpu(),
    }

    if args.save_gif and len(depth_frames) > 0:
        results['depth_frames'] = depth_frames

    return results


def main():
    """主函数 (Main function)"""
    args = parse_args()
    device = torch.device('cuda')
    use_cam = args.diff_cam or args.paper_unified_control

    # 初始化环境 (Initialize environment)
    env = Env(args.batch_size, 64, 48, args.grad_decay, device,
              fov_x_half_tan=args.fov_x_half_tan, single=True,
              wall_slit=True, speed_mtp=args.speed_mtp,
              cam_angle=args.cam_angle,
              ellipsoid_a=args.drone_a if args.ellipsoid_collision else 0.0,
              ellipsoid_c=args.drone_c if args.ellipsoid_collision else 0.0)

    # 初始化模型 (Initialize model)
    obs_dim = 7 if args.no_odom else 10
    model = Model(obs_dim, 6,
                  use_diff_cam=args.diff_cam,
                  use_unified_control=args.paper_unified_control,
                  use_cam_obs=args.paper_cam_obs)
    model = model.to(device)

    # 加载模型权重 (Load model weights)
    state_dict = torch.load(args.resume, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
    if missing_keys:
        print("Warning - missing_keys:", missing_keys)
    if unexpected_keys:
        print("Warning - unexpected_keys:", unexpected_keys)
    model.eval() # 设置为评估模式 (Set to evaluation mode)

    # 计算总批次数 (Calculate total number of batches)
    num_batches = (args.num_episodes + args.batch_size - 1) // args.batch_size
    total_episodes = num_batches * args.batch_size

    # 记录所有回合的结果 (Record results for all episodes)
    all_no_collision = []
    all_crossed = []
    all_passed = []
    all_min_dist = []
    all_cross_time = []
    all_roll_at_wall = []
    all_max_roll = []
    gif_frames = None

    # 打印评估模式信息 (Print evaluation mode information)
    mode_str = 'ellipsoid' if args.ellipsoid_collision else 'point'
    if args.paper_unified_control:
        mode_str += '+unified_ctrl'
    if args.diff_cam:
        mode_str += '+diff_cam'

    print(f"\nEvaluating wall-slit with {total_episodes} episodes ({mode_str})...\n")

    # 运行评估循环 (Run evaluation loop)
    for batch_i in tqdm(range(num_batches), desc='Evaluating'):
        results = evaluate_batch(env, model, args, device)
        all_no_collision.append(results['no_collision'])
        all_crossed.append(results['crossed'])
        all_passed.append(results['passed'])
        all_min_dist.append(results['min_dist'])
        all_cross_time.append(results['cross_time'])
        all_roll_at_wall.append(results['roll_at_wall'])
        all_max_roll.append(results['max_roll'])

        # 保存第一个批次的 GIF 帧 (Save GIF frames from the first batch)
        if batch_i == 0 and args.save_gif and 'depth_frames' in results:
            gif_frames = results['depth_frames']

    # 汇总结果 (Aggregate results)
    all_no_collision = torch.cat(all_no_collision)
    all_crossed = torch.cat(all_crossed)
    all_passed = torch.cat(all_passed)
    all_min_dist = torch.cat(all_min_dist)
    all_cross_time = torch.cat(all_cross_time)
    all_max_roll = torch.cat(all_max_roll)

    # 打印最终统计数据 (Print final statistics)
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
    avg_roll_at_wall = torch.stack(all_roll_at_wall).mean() if len(all_roll_at_wall) > 0 else 0
    print(f"  Avg roll at wall:   {avg_roll_at_wall:.1f}°")
    print(f"  Max roll (mean):    {all_max_roll.mean():.1f}° ± {all_max_roll.std():.1f}°")
    print(f"{'='*60}\n")

    # Save GIF (保存 GIF)
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
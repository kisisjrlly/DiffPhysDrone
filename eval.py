"""
DiffPhysDrone evaluation entry point.

目的：
- 复用训练时同一套参数文件与传感器/控制逻辑
- 仅做前向推理，不做训练、不计算 loss、不写 wandb
- 支持 Rerun 实时可视化
"""

import argparse
import os
from random import normalvariate
import time

import torch
from torch.cuda.amp.autocast_mode import autocast

from config import (
    build_parser,
    parse_diff_sensor_impl,
    parse_scenarios,
    set_global_seed,
    validate_args,
    print_runtime_mode,
)
from lqr import build_velocity_tracking_linear_system, solve_batched_dlqr
from model import Model
from rerun_vis import RerunVis
from rollout_ops import (
    render_sensors,
    build_local_frame,
    build_state_vector,
    compute_target_velocity,
    decode_action_direct,
    decode_action_lqr,
    update_camera_params,
    diff_depth_exposure_to_time,
    init_camera_params,
    compute_camera_param_stats,
    compute_diff_depth_proxies,
    compute_depth_fill_rate,
    diff_depth_fill_softness,
)
from train_utils import build_env, make_yaw_drift_R


def parse_eval_args():
    parser = build_parser()
    parser.add_argument('--eval_episodes', type=int, default=1,
                        help='评估 episode 数（每个 episode 重置一次环境并做 timesteps 步推理）')
    args = parser.parse_args()

    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.scenarios = parse_scenarios(args.scenarios)
    set_global_seed(args.seed, args.deterministic)
    validate_args(args)

    if args.eval_episodes < 1:
        raise ValueError('--eval_episodes 必须 >= 1')

    return args


def run_one_episode(ep_idx, scene_name, args, model, env, vis, device):
    B = env.batch_size
    use_amp = bool(args.amp and device.type == 'cuda')

    env.reset(scene_name=scene_name)
    model.reset()

    if vis.enabled:
        vis.begin_episode(ep_idx)
        j = int(min(max(args.vis_env_idx, 0), B - 1))
        vis.log_environment(
            phase='student',
            balls=env.balls[j].detach().cpu().numpy(),
            voxels=env.voxels[j].detach().cpu().numpy(),
            cyl=env.cyl[j].detach().cpu().numpy(),
            cyl_h=env.cyl_h[j].detach().cpu().numpy(),
            start=env.p[j].detach().cpu().numpy(),
            target=env.p_target[j].detach().cpu().numpy(),
            scene_name=getattr(env, 'current_scene_name', None),
            scene_effects=getattr(env, 'current_scene_effects', None),
            step_idx=0,
        )

    h = None
    act_buffer = [env.act] * 2
    target_v_raw = env.p_target - env.p
    yaw_drift_R = make_yaw_drift_R(B, device) if args.yaw_drift else None

    power, exposure, gain = init_camera_params(env, B, device)

    # 与训练推理路径一致（目前训练中也是固定 1/15 构造 LQR 离散系统）
    A_lqr, B_lqr = build_velocity_tracking_linear_system(B, 1 / 15, device)

    min_margin_hist = []
    speed_hist = []
    power_hist = []
    exposure_hist = []
    gain_hist = []
    fill_rate_hist = []
    fill_rate_soft_hist = []
    goal_dist_hist = []
    collided = False
    collided_step = None

    for t in range(args.timesteps):
        print("timestep:", t)
        base_dt = normalvariate(1 / args.base_control_freq, 0.1 / args.base_control_freq)
        exposure_delay = float(diff_depth_exposure_to_time(
            exposure.mean().detach(),
            camera_semantics=env.cam_sem,
        )) * 0.01
        ctl_dt = base_dt + exposure_delay

        depth_obs, quality = render_sensors(env, ctl_dt, power, exposure, gain, differentiable=False)
        fill_rate_hist.append(compute_depth_fill_rate(
            depth_obs,
            min_valid_depth=args.depth_min_valid,
        ).detach())
        fill_rate_soft_hist.append(compute_depth_fill_rate(
            depth_obs,
            min_valid_depth=args.depth_min_valid,
            softness=diff_depth_fill_softness(args.depth_min_valid),
        ).detach())

        # 记录推进前的最小安全边距（<=0 视为碰撞）
        vec_now = env.find_vec_to_nearest_pt()
        min_margin_before = (torch.norm(vec_now, 2, -1) - env.margin)

        if args.yaw_drift and yaw_drift_R is not None:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ yaw_drift_R, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()

        env.run(act_buffer[t], ctl_dt, target_v_raw)

        # 记录推进后的最小安全边距，避免漏检“本步内发生”的碰撞
        vec_after = env.find_vec_to_nearest_pt()
        min_margin_after = (torch.norm(vec_after, 2, -1) - env.margin)
        min_margin_now = torch.minimum(min_margin_before, min_margin_after)
        min_margin_hist.append(min_margin_now)

        # eval 规则：一旦发生碰撞，立即结束当前 episode
        # 当前 batch 并行评估时，只要任一无人机碰撞就提前终止。
        if bool((min_margin_now <= 0).any().item()):
            collided = True
            collided_step = t

        R = build_local_frame(env)
        target_v = compute_target_velocity(target_v_raw, env)
        state, local_v = build_state_vector(
            env, target_v, R, power, exposure, gain,
            args.no_odom, args.include_camera_state_in_obs,
        )

        if args.policy_output_intent:
            with autocast(enabled=use_amp):
                act_raw, cam_params, h, intent = model(
                    state, h, return_intent=True, depth_obs=depth_obs, add_noise=False)
            act_raw = act_raw.float()
            intent = intent.float()
        else:
            with autocast(enabled=use_amp):
                act_raw, cam_params, h = model(
                    state, h, depth_obs=depth_obs, add_noise=False)
            act_raw = act_raw.float()
            intent = None

        cam_params = cam_params.float()

        power, exposure, gain, _ = update_camera_params(cam_params, power, exposure, gain, env)
        power_hist.append(power.detach())
        exposure_hist.append(exposure.detach())
        gain_hist.append(gain.detach())

        if args.use_dmpc and args.policy_output_intent and intent is not None:
            act_final, _ = decode_action_lqr(
                intent, R, env, local_v, B,
                A_lqr, B_lqr,
                args.lqr_horizon, args.lqr_reg, args.max_acc_cmd,
                args.inject_depth_into_lqr, args.depth_safe_dist, args.depth_repel_gain,
                vec_now, solve_batched_dlqr,
            )
        else:
            act_final, _ = decode_action_direct(act_raw, R, env, B, args.max_acc_cmd)

        act_buffer.append(act_final)
        speed_hist.append(env.v.norm(2, -1))
        goal_dist_hist.append((env.p_target - env.p).norm(2, -1))

        if vis.enabled:
            j = int(min(max(args.vis_env_idx, 0), B - 1))
            cam_vals = (
                float(power[j].detach().cpu()),
                float(exposure[j].detach().cpu()),
                float(gain[j].detach().cpu()),
            )

            main_img_np = None
            main_img_mode = 'depth'
            depth_img_np = depth_obs[j].detach().cpu().numpy()

            # 评估阶段按 step 记录无人机动力学指标（替代训练 loss/fps 指标）
            vj = env.v[j]
            speed_mps = float(vj.norm(2).detach().cpu())

            # 由旋转增量估算角速度幅值：|omega| ~= theta / dt
            R_old_j = env.R_old[j]
            R_j = env.R[j]
            R_delta = R_old_j.transpose(0, 1) @ R_j
            trace_val = float((R_delta[0, 0] + R_delta[1, 1] + R_delta[2, 2]).detach().cpu())
            cos_theta = max(-1.0, min(1.0, 0.5 * (trace_val - 1.0)))
            theta = float(torch.acos(torch.tensor(cos_theta)).item())
            angular_speed_rps = theta / max(float(ctl_dt), 1e-6)

            thrust_norm_mps2 = float(env.act[j].norm(2).detach().cpu())
            accel_norm_mps2 = float(env.a[j].norm(2).detach().cpu())
            dist_to_goal_m = float((env.p_target[j] - env.p[j]).norm(2).detach().cpu())

            step_scalars = {
                'speed_mps': speed_mps,
                'angular_speed_rps': angular_speed_rps,
                'thrust_norm_mps2': thrust_norm_mps2,
                'accel_norm_mps2': accel_norm_mps2,
                'dist_to_goal_m': dist_to_goal_m,
            }
            scene_debug = env.export_last_diff_depth_debug(j)
            step_scalars.update(scene_debug.get('scalars', {}))
            step_scalars['scene_id'] = float(getattr(env, 'current_scene_id', 0))

            vis.log_step(
                phase='student',
                step_idx=t,
                pos=env.p[j].detach().cpu().numpy(),
                target=env.p_target[j].detach().cpu().numpy(),
                depth=None,
                cam=cam_vals,
                scalars=step_scalars,
                main_img=main_img_np,
                main_img_mode=main_img_mode,
                depth_img=depth_img_np,
                quality_img=scene_debug.get('images', {}).get('quality_map'),
                invalid_img=scene_debug.get('images', {}).get('invalid_mask'),
                scene_effect_img=scene_debug.get('images', {}).get('scene_effect_map'),
                drone_R=env.R[j].detach().cpu().numpy(),
                cam_R=env.R_cam[j].detach().cpu().numpy(),
                main_fov_half_tan=float(env._fov_x_half_tan),
                main_hw=(int(env.height), int(env.width)),
                depth_hw=(int(env.height), int(env.width)),
            )

        if collided:
            print(f"[eval] collision detected at step={t}, early stop this episode.")
            break
        if vis.enabled:
            time.sleep(1.0/15)

    min_margin_all = torch.stack(min_margin_hist).amin(dim=0)
    success_mask = min_margin_all > 0
    success_rate = float(success_mask.float().mean().detach().cpu())
    collision_rate = float((~success_mask).float().mean().detach().cpu())

    if len(speed_hist) > 0:
        speed_all = torch.stack(speed_hist)
        avg_speed = float(speed_all.mean().detach().cpu())
        max_speed = float(speed_all.max().detach().cpu())
    else:
        avg_speed = 0.0
        max_speed = 0.0

    fill_mean = float(torch.stack(fill_rate_hist).mean().item()) if fill_rate_hist else 0.0
    fill_soft_mean = float(torch.stack(fill_rate_soft_hist).mean().item()) if fill_rate_soft_hist else fill_mean
    hole_mean = 1.0 - fill_mean
    hole_soft_mean = 1.0 - fill_soft_mean

    cam_stats = compute_camera_param_stats(power_hist, exposure_hist, gain_hist)
    proxy_stats = compute_diff_depth_proxies(
        power_hist,
        exposure_hist,
        gain_hist,
        speed_hist,
        camera_semantics=env.cam_sem,
    )

    if goal_dist_hist:
        goal_dist_all = torch.stack(goal_dist_hist)
        reached = goal_dist_all < 0.5
        reached_any = reached.any(dim=0)
        first_hit = torch.full((B,), args.timesteps, device=device, dtype=torch.long)
        if reached_any.any():
            hit_idx = reached.float().argmax(dim=0)
            first_hit = torch.where(reached_any, hit_idx, first_hit)
        time_to_goal = float(first_hit.float().mean().item() / max(args.base_control_freq, 1e-6))
    else:
        time_to_goal = float(args.timesteps / max(args.base_control_freq, 1e-6))

    metrics = {
        'scene_name': str(getattr(env, 'current_scene_name', scene_name)),
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'avg_speed': avg_speed,
        'max_speed': max_speed,
        'fill_rate': fill_mean,
        'fill_rate_soft': fill_soft_mean,
        'hole_rate': hole_mean,
        'hole_rate_soft': hole_soft_mean,
        'time_to_goal': time_to_goal,
        'collided': float(collided),
    }
    metrics.update(cam_stats)
    metrics.update(proxy_stats)

    print(
        f"[eval] episode={ep_idx + 1}/{args.eval_episodes} "
        f"scene={metrics['scene_name']} "
        f"success_rate={success_rate:.3f} collision_rate={collision_rate:.3f} "
        f"fill_rate={fill_mean:.3f} fill_rate_soft={fill_soft_mean:.3f} "
        f"avg_speed={avg_speed:.3f} max_speed={max_speed:.3f} "
        f"collided={collided}" + (f" collided_step={collided_step}" if collided_step is not None else "")
    )

    # 评估汇总指标保留在控制台输出；Rerun 侧重点为 step 级飞行状态。
    return metrics


def main():
    args = parse_eval_args()

    if not args.resume:
        raise ValueError('评估必须提供 --resume <checkpoint_path>')
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f'checkpoint 不存在: {args.resume}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("\n" + "=" * 30 + " Eval Configuration " + "=" * 30)
    for k, v in vars(args).items():
        print(f"{k:<30}: {v}")
    print("=" * 80 + "\n")
    print_runtime_mode(args)

    env = build_env(args.batch_size, args, device, eval_mode=True)

    obs_dim = 7 if args.no_odom else 10
    model = Model(
        obs_dim, 6,
        include_camera_state_in_obs=args.include_camera_state_in_obs,
        use_policy_intent=args.policy_output_intent,
        intent_dim=9,
        depth_nn_width=args.depth_nn_width,
        depth_nn_height=args.depth_nn_height,
        depth_use_pipeline=args.depth_use_pipeline,
        depth_min_valid=args.depth_min_valid,
        depth_max_range=args.depth_max_range,
    ).to(device)

    print(f"[eval] loading checkpoint: {args.resume}")
    state_dict = torch.load(args.resume, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print('[eval][warn] missing_keys:', missing)
    if unexpected:
        print('[eval][warn] unexpected_keys:', unexpected)

    model.eval()

    vis = RerunVis(
        enabled=(args.vis_enable and args.vis_backend == 'rerun'),
        app_id='DiffPhysDrone-Eval',
        spawn=args.vis_spawn,
    )

    with torch.no_grad():
        ep_metrics = []
        eval_scenes = list(args.scenarios)
        for ep_idx in range(args.eval_episodes):
            scene_name = eval_scenes[ep_idx % len(eval_scenes)]
            ep_metrics.append(run_one_episode(ep_idx, scene_name, args, model, env, vis, device))

    if ep_metrics:
        keys = [
            'success_rate', 'collision_rate', 'time_to_goal',
            'fill_rate', 'fill_rate_soft', 'hole_rate', 'hole_rate_soft',
            'energy_proxy', 'blur_proxy', 'noise_proxy',
            'avg_speed', 'max_speed',
        ]
        print('[eval] overall summary:')
        for key in keys:
            vals = [float(m[key]) for m in ep_metrics if key in m]
            if vals:
                print(f'  {key:<16}: {sum(vals) / len(vals):.4f}')

        if len(eval_scenes) > 1:
            print('[eval] per-scene summary:')
            for scene_name in eval_scenes:
                scene_eps = [m for m in ep_metrics if m.get('scene_name') == scene_name]
                if not scene_eps:
                    continue
                parts = []
                for key in keys:
                    vals = [float(m[key]) for m in scene_eps if key in m]
                    if vals:
                        parts.append(f'{key}={sum(vals) / len(vals):.4f}')
                print(f'  {scene_name}: ' + ' '.join(parts))

    print('[eval] done.')


if __name__ == '__main__':
    main()

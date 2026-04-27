"""
DiffPhysDrone evaluation entry point.

目的：
- 复用训练时同一套参数文件与传感器/控制逻辑
- 仅做前向推理，不做训练、不计算 loss、不写 wandb
- 支持 Rerun 实时可视化
"""

import argparse
import csv
import os
from random import normalvariate
import time

import torch
from torch.cuda.amp.autocast_mode import autocast

from config import (
    build_parser,
    parse_diff_sensor_impl,
    parse_scenarios,
    parse_sun_glare_levels,
    canonicalize_sun_glare_level,
    set_global_seed,
    validate_args,
    print_runtime_mode,
)
from lqr import build_velocity_tracking_linear_system, solve_batched_dlqr
from model import Model
from rerun_vis import RerunVis
from rollout_ops import (
    render_sensors,
    select_policy_depth_obs,
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
    parser.add_argument('--vis_episode_idx', type=int, default=-1,
                        help='Rerun 只记录指定 episode，0-based；-1 表示记录全部 episode')
    parser.add_argument('--eval_trace_csv', type=str, default=None,
                        help='可选：保存 step 级 eval trace CSV')
    parser.add_argument('--eval_episode_csv', type=str, default=None,
                        help='可选：保存 episode 级 eval metrics CSV')
    args = parser.parse_args()

    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.scenarios = parse_scenarios(args.scenarios)
    args.sun_glare_levels = parse_sun_glare_levels(args.sun_glare_levels)
    if args.sun_glare_eval_level is not None:
        args.sun_glare_eval_level = canonicalize_sun_glare_level(args.sun_glare_eval_level)
    set_global_seed(args.seed, args.deterministic)
    validate_args(args)

    if args.eval_episodes < 1:
        raise ValueError('--eval_episodes 必须 >= 1')
    if args.vis_episode_idx < -1:
        raise ValueError('--vis_episode_idx 必须为 -1 或 >= 0')
    if args.vis_episode_idx >= args.eval_episodes:
        raise ValueError('--vis_episode_idx 必须小于 --eval_episodes')

    return args


def _write_csv_rows(path, rows):
    if not path or not rows:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_one_episode(ep_idx, scene_name, scene_variant, args, model, env, vis, device, collect_trace=False):
    B = env.batch_size
    use_amp = bool(args.amp and device.type == 'cuda')

    env.reset(scene_name=scene_name, scene_variant=scene_variant)
    model.reset()
    vis_episode_idx = int(getattr(args, 'vis_episode_idx', -1))
    log_vis = bool(vis.enabled and (vis_episode_idx < 0 or int(ep_idx) == vis_episode_idx))
    vis_all_episode_paths = bool(log_vis and vis_episode_idx < 0 and int(args.eval_episodes) > 1)
    # For multi-episode Rerun eval, log each episode to an independent entity
    # namespace. This lets the viewer hide/show one episode after the run.
    vis_phase = f"episodes/ep_{int(ep_idx):03d}/student" if vis_all_episode_paths else "student"
    # Episode-specific paths can reuse local step 0..T-1. If all episodes are
    # deliberately logged to one path, keep a monotonic global step.
    episode_step_base = 0 if vis_all_episode_paths or vis_episode_idx >= 0 else int(ep_idx) * int(args.timesteps)

    if log_vis:
        vis.begin_episode(ep_idx, step_base=episode_step_base)
        j = int(min(max(args.vis_env_idx, 0), B - 1))
        vis.log_environment(
            phase=vis_phase,
            balls=env.balls[j].detach().cpu().numpy(),
            voxels=env.voxels[j].detach().cpu().numpy(),
            cyl=env.cyl[j].detach().cpu().numpy(),
            cyl_h=env.cyl_h[j].detach().cpu().numpy(),
            start=env.p[j].detach().cpu().numpy(),
            target=env.p_target[j].detach().cpu().numpy(),
            scene_name=getattr(env, 'current_scene_tag', getattr(env, 'current_scene_name', None)),
            scene_effects=getattr(env, 'current_scene_effects', None),
            step_idx=episode_step_base,
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
    x_hist = []
    scene_effect_hist = []
    glare_quality_hist = []
    glare_invalid_hist = []
    collided = False
    collided_step = None
    trace_rows = [] if collect_trace else None
    depth_input_mode = str(
        getattr(args, 'eval_depth_mode', getattr(args, 'policy_depth_mode', 'depth'))
    ).strip().lower()

    for t in range(args.timesteps):
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
        policy_depth_obs = select_policy_depth_obs(depth_obs, depth_input_mode)

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
                    state, h, return_intent=True, depth_obs=policy_depth_obs, add_noise=False)
            act_raw = act_raw.float()
            intent = intent.float()
        else:
            with autocast(enabled=use_amp):
                act_raw, cam_params, h = model(
                    state, h, depth_obs=policy_depth_obs, add_noise=False)
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
        x_hist.append(env.p[:, 0].detach())
        scene_debug_for_metrics = env.export_last_diff_depth_debug(0)
        if 'scene_effect_mean' in scene_debug_for_metrics.get('scalars', {}):
            scene_effect_hist.append(float(scene_debug_for_metrics['scalars']['scene_effect_mean']))
        if 'glare_quality_mean' in scene_debug_for_metrics.get('scalars', {}):
            glare_quality_hist.append(float(scene_debug_for_metrics['scalars']['glare_quality_mean']))
        if 'glare_invalid_rate' in scene_debug_for_metrics.get('scalars', {}):
            glare_invalid_hist.append(float(scene_debug_for_metrics['scalars']['glare_invalid_rate']))

        if log_vis:
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
                phase=vis_phase,
                step_idx=episode_step_base + t,
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

        if collect_trace:
            j = 0
            scene_debug = env.export_last_diff_depth_debug(j)
            fill_rate_last = fill_rate_hist[-1]
            fill_rate_soft_last = fill_rate_soft_hist[-1]
            if torch.is_tensor(fill_rate_last) and fill_rate_last.ndim == 0:
                fill_rate_scalar = float(fill_rate_last.detach().cpu().item())
            else:
                fill_rate_scalar = float(fill_rate_last[j].detach().cpu().item())
            if torch.is_tensor(fill_rate_soft_last) and fill_rate_soft_last.ndim == 0:
                fill_rate_soft_scalar = float(fill_rate_soft_last.detach().cpu().item())
            else:
                fill_rate_soft_scalar = float(fill_rate_soft_last[j].detach().cpu().item())
            trace_rows.append({
                'episode_idx': int(ep_idx),
                'step_idx': int(t),
                'scene_name': str(getattr(env, 'current_scene_tag', getattr(env, 'current_scene_name', scene_name))),
                'glare_level': getattr(env, 'current_sun_glare_level', None) or '',
                'x': float(env.p[j, 0].detach().cpu().item()),
                'y': float(env.p[j, 1].detach().cpu().item()),
                'z': float(env.p[j, 2].detach().cpu().item()),
                'speed_mps': float(env.v[j].norm(2).detach().cpu().item()),
                'accel_norm_mps2': float(env.a[j].norm(2).detach().cpu().item()),
                'power': float(power[j].detach().cpu().item()),
                'exposure': float(exposure[j].detach().cpu().item()),
                'gain': float(gain[j].detach().cpu().item()),
                'fill_rate': fill_rate_scalar,
                'fill_rate_soft': fill_rate_soft_scalar,
                'scene_effect_mean': float(scene_debug.get('scalars', {}).get('scene_effect_mean', 0.0)),
                'glare_quality_mean': float(scene_debug.get('scalars', {}).get('glare_quality_mean', 0.0)),
                'glare_invalid_rate': float(scene_debug.get('scalars', {}).get('glare_invalid_rate', 0.0)),
                'glare_level_id': float(scene_debug.get('scalars', {}).get('glare_level_id', -1.0)),
                'sensor_regime_id': float(scene_debug.get('scalars', {}).get('sensor_regime_id', -1.0)),
                'decision_open_side_id': float(scene_debug.get('scalars', {}).get('decision_open_side_id', 0.0)),
                'zone_enter_x': float(getattr(env, 'current_scene_effects', {}).get('zone_enter_x', 0.0)),
                'dist_to_goal_m': float((env.p_target[j] - env.p[j]).norm(2).detach().cpu().item()),
                'collided': float(collided),
                'depth_input_mode': depth_input_mode,
            })

        if collided:
            print(f"[eval] collision detected at step={t}, early stop this episode.")
            break
        if log_vis:
            time.sleep(1.0/5)

    min_margin_all = torch.stack(min_margin_hist).amin(dim=0)

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
        final_goal_dist = goal_dist_all[-1]
        first_hit = torch.full((B,), args.timesteps, device=device, dtype=torch.long)
        if reached_any.any():
            hit_idx = reached.float().argmax(dim=0)
            first_hit = torch.where(reached_any, hit_idx, first_hit)
        time_to_goal = float(first_hit.float().mean().item() / max(args.base_control_freq, 1e-6))
    else:
        reached_any = torch.zeros((B,), device=device, dtype=torch.bool)
        final_goal_dist = torch.full((B,), float('inf'), device=device)
        time_to_goal = float(args.timesteps / max(args.base_control_freq, 1e-6))

    success_mask = (min_margin_all > 0) & reached_any
    success_rate = float(success_mask.float().mean().detach().cpu())
    collision_rate = float((min_margin_all <= 0).float().mean().detach().cpu())
    goal_reach_rate = float(reached_any.float().mean().detach().cpu())
    final_goal_dist_mean = float(final_goal_dist.mean().detach().cpu().item())

    stop_before_glare = 0.0
    if getattr(env, 'current_scene_name', None) == 'sun_glare' and x_hist:
        tail_k = min(10, len(speed_hist))
        tail_speed = 0.0
        if tail_k > 0:
            tail_speed = float(torch.stack(speed_hist[-tail_k:]).mean().detach().cpu().item())
        entered_glare = any(v > 0.02 for v in scene_effect_hist)
        stop_before_glare = 1.0 if (not entered_glare and tail_speed < 0.15) else 0.0

    metrics = {
        'scene_name': str(getattr(env, 'current_scene_tag', getattr(env, 'current_scene_name', scene_name))),
        'glare_level': str(getattr(env, 'current_sun_glare_level', '') or ''),
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'goal_reach_rate': goal_reach_rate,
        'avg_speed': avg_speed,
        'max_speed': max_speed,
        'fill_rate': fill_mean,
        'fill_rate_soft': fill_soft_mean,
        'hole_rate': hole_mean,
        'hole_rate_soft': hole_soft_mean,
        'time_to_goal': time_to_goal,
        'final_goal_dist': final_goal_dist_mean,
        'stop_before_glare_rate': stop_before_glare,
        'local_glare_quality': float(sum(glare_quality_hist) / len(glare_quality_hist)) if glare_quality_hist else 0.0,
        'local_glare_invalid_rate': float(sum(glare_invalid_hist) / len(glare_invalid_hist)) if glare_invalid_hist else 0.0,
        'collided': float(collided),
        'depth_input_mode': depth_input_mode,
    }
    metrics.update(cam_stats)
    metrics.update(proxy_stats)

    total_eval_episodes = getattr(args, 'eval_episodes', '?')
    print(
        f"[eval] episode={ep_idx + 1}/{total_eval_episodes} "
        f"scene={metrics['scene_name']} "
        f"success_rate={success_rate:.3f} collision_rate={collision_rate:.3f} goal_reach_rate={goal_reach_rate:.3f} "
        f"fill_rate={fill_mean:.3f} fill_rate_soft={fill_soft_mean:.3f} "
        f"avg_speed={avg_speed:.3f} max_speed={max_speed:.3f} "
        f"final_goal_dist={final_goal_dist_mean:.3f} "
        f"collided={collided}" + (f" collided_step={collided_step}" if collided_step is not None else "")
    )

    # 评估汇总指标保留在控制台输出；Rerun 侧重点为 step 级飞行状态。
    if collect_trace:
        return metrics, trace_rows
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
    if vis.enabled:
        vis.send_eval_episode_blueprint(args.eval_episodes, args.vis_episode_idx)
    if vis.enabled and int(args.vis_episode_idx) >= 0:
        print(f"[eval] rerun visualizes only episode index {int(args.vis_episode_idx)}")
    elif vis.enabled and int(args.eval_episodes) > 1:
        print("[eval] rerun logs episodes under /episodes/ep_XXX/student for post-run selection")

    with torch.no_grad():
        ep_metrics = []
        trace_rows_all = []
        eval_scenes = list(args.scenarios)
        for ep_idx in range(args.eval_episodes):
            scene_name = eval_scenes[ep_idx % len(eval_scenes)]
            scene_variant = None
            if scene_name == 'sun_glare':
                if args.sun_glare_eval_level is not None:
                    scene_variant = args.sun_glare_eval_level
                elif len(eval_scenes) == 1:
                    scene_variant = args.sun_glare_levels[ep_idx % len(args.sun_glare_levels)]
                else:
                    scene_variant = args.sun_glare_levels[0]
            if args.eval_trace_csv:
                metrics, trace_rows = run_one_episode(
                    ep_idx, scene_name, scene_variant, args, model, env, vis, device,
                    collect_trace=True,
                )
                trace_rows_all.extend(trace_rows)
            else:
                metrics = run_one_episode(ep_idx, scene_name, scene_variant, args, model, env, vis, device)
            metrics = dict(metrics)
            metrics['episode_idx'] = int(ep_idx)
            ep_metrics.append(metrics)

    if ep_metrics:
        keys = [
            'success_rate', 'collision_rate', 'goal_reach_rate', 'stop_before_glare_rate', 'time_to_goal',
            'local_glare_quality', 'local_glare_invalid_rate',
            'fill_rate', 'fill_rate_soft', 'hole_rate', 'hole_rate_soft',
            'energy_proxy', 'blur_proxy', 'noise_proxy',
            'avg_speed', 'max_speed', 'final_goal_dist',
        ]
        print('[eval] overall summary:')
        for key in keys:
            vals = [float(m[key]) for m in ep_metrics if key in m]
            if vals:
                print(f'  {key:<16}: {sum(vals) / len(vals):.4f}')

        unique_scene_names = []
        for m in ep_metrics:
            name = m.get('scene_name')
            if name not in unique_scene_names:
                unique_scene_names.append(name)
        if len(unique_scene_names) > 1:
            print('[eval] per-scene summary:')
            for scene_name in unique_scene_names:
                scene_eps = [m for m in ep_metrics if m.get('scene_name') == scene_name]
                if not scene_eps:
                    continue
                parts = []
                for key in keys:
                    vals = [float(m[key]) for m in scene_eps if key in m]
                    if vals:
                        parts.append(f'{key}={sum(vals) / len(vals):.4f}')
                print(f'  {scene_name}: ' + ' '.join(parts))

    if args.eval_episode_csv and ep_metrics:
        _write_csv_rows(args.eval_episode_csv, ep_metrics)
        print(f"[eval] wrote episode metrics: {args.eval_episode_csv}")
    if args.eval_trace_csv and trace_rows_all:
        _write_csv_rows(args.eval_trace_csv, trace_rows_all)
        print(f"[eval] wrote trace rows: {args.eval_trace_csv}")

    print('[eval] done.')


if __name__ == '__main__':
    main()

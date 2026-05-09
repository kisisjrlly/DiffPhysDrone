"""Minimal evaluation entry point for active-sensing simulation."""
import csv
import os
from random import normalvariate

import torch
from torch.cuda.amp.autocast_mode import autocast

from config import build_parser, parse_diff_sensor_impl, parse_scenarios, set_global_seed, validate_args, canonicalize_sun_glare_slot, print_runtime_mode
from model import Model
from rerun_vis import RerunVis
from rollout_ops import (
    render_sensors,
    select_policy_depth_obs,
    build_local_frame,
    build_state_vector,
    compute_target_velocity,
    decode_action_direct,
    update_camera_params,
    diff_depth_exposure_to_time,
    init_camera_params,
    compute_camera_param_stats,
    compute_depth_fill_rate,
)
from train_utils import build_env


def parse_eval_args():
    parser = build_parser()
    parser.add_argument('--eval_episodes', type=int, default=1)
    parser.add_argument('--vis_episode_idx', type=int, default=-1)
    parser.add_argument('--eval_trace_csv', type=str, default=None)
    parser.add_argument('--eval_episode_csv', type=str, default=None)
    args = parser.parse_args()
    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.scenarios = parse_scenarios(args.scenarios)
    args.sun_glare_eval_slot = canonicalize_sun_glare_slot(args.sun_glare_eval_slot)
    set_global_seed(args.seed, args.deterministic)
    validate_args(args)
    if args.eval_episodes < 1:
        raise ValueError('--eval_episodes must be >= 1')
    return args


def _write_csv_rows(path, rows):
    if not path or not rows:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    keys, seen = [], set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _append_csv_rows(path, rows):
    if not path or not rows:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    path_exists = os.path.exists(path)
    keys, seen = [], set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not path_exists:
            writer.writeheader()
        writer.writerows(rows)


def _min_clearance_from_vec(vec_now, env):
    """Return per-env physical clearance after subtracting drone radius in the CUDA kernel."""
    dist = torch.norm(vec_now, 2, -1)
    batch_size = int(env.batch_size)
    if dist.ndim == 1:
        per_env = dist
    elif dist.ndim == 2:
        if dist.shape[0] == batch_size and dist.shape[1] != batch_size:
            per_env = dist.min(dim=1).values
        elif dist.shape[1] == batch_size:
            per_env = dist.min(dim=0).values
        else:
            per_env = dist.min(dim=0).values
    else:
        per_env = dist.reshape(-1, batch_size).min(dim=0).values
    return per_env


def _collision_from_clearance(min_clearance, args):
    return min_clearance <= float(args.collision_clearance)


def _eval_scalars(scene_debug, min_clearance, goal_dist, env, args, *, env_idx=0, collided=False):
    scalars = dict(scene_debug.get('scalars', {}) if scene_debug else {})
    idx = int(min(max(env_idx, 0), int(env.batch_size) - 1))
    speed = env.v.norm(2, -1)
    thrust = env.act.norm(2, -1)
    accel = env.a.norm(2, -1)
    scalars.update({
        'min_margin_m': float((min_clearance[idx] - env.margin[idx]).detach().cpu()),
        'safety_margin_m': float((min_clearance[idx] - env.margin[idx]).detach().cpu()),
        'clearance_m': float(min_clearance[idx].detach().cpu()),
        'collision_clearance_m': float(args.collision_clearance),
        'goal_dist_m': float(goal_dist[idx].detach().cpu()),
        'dist_to_goal_m': float(goal_dist[idx].detach().cpu()),
        'speed_mps': float(speed[idx].detach().cpu()),
        'angular_speed_rps': 0.0,
        'thrust_norm_mps2': float(thrust[idx].detach().cpu()),
        'accel_norm_mps2': float(accel[idx].detach().cpu()),
        'pos_x_m': float(env.p[idx, 0].detach().cpu()),
        'pos_y_m': float(env.p[idx, 1].detach().cpu()),
        'pos_z_m': float(env.p[idx, 2].detach().cpu()),
        'collision': 1.0 if collided else 0.0,
    })
    return scalars


def run_one_episode(ep_idx, scene_name, args, model, env, vis, device, collect_trace=False):
    B = env.batch_size
    use_amp = bool(args.amp and device.type == 'cuda')
    env.reset(scene_name=scene_name)
    model.reset()
    log_vis = bool(vis.enabled and (args.vis_episode_idx < 0 or int(ep_idx) == int(args.vis_episode_idx)))
    vis_phase = f'episodes/ep_{int(ep_idx):03d}/student' if log_vis and args.vis_episode_idx < 0 and args.eval_episodes > 1 else 'student'
    if log_vis:
        vis.begin_episode(ep_idx, step_base=0)
        j = int(min(max(args.vis_env_idx, 0), B - 1))
        vis.log_environment(
            phase=vis_phase,
            balls=env.get_world_balls_for_env(j),
            voxels=env.get_world_voxels_for_env(j),
            cyl=env.get_world_cyl_for_env(j),
            cyl_h=env.get_world_cyl_h_for_env(j),
            start=env.p[j].detach().cpu().numpy(),
            target=env.p_target[j].detach().cpu().numpy(),
            scene_name=getattr(env, 'current_scene_name', None),
            scene_effects=env.get_scene_effects_for_env(j),
            scene_yaw=env.get_scene_yaw_for_env(j),
            step_idx=0,
        )

    h = None
    cam_h = None
    act_buffer = [env.act] * 2
    power, exposure, gain = init_camera_params(env, B, device)
    min_clearance_hist, speed_hist, goal_dist_hist = [], [], []
    power_hist, exposure_hist, gain_hist, fill_hist = [], [], [], []
    trace_rows = [] if collect_trace else None
    collided_cum = torch.zeros((B,), dtype=torch.bool, device=device)
    reached_cum = torch.zeros((B,), dtype=torch.bool, device=device)
    final_goal_dist = torch.norm(env.p_target - env.p, dim=-1).detach()
    stop_reason = 'timeout'

    for t in range(args.timesteps):
        base_dt = normalvariate(1 / args.base_control_freq, 0.1 / args.base_control_freq)
        exposure_delay = float(diff_depth_exposure_to_time(exposure.mean().detach(), camera_semantics=env.cam_sem)) * 0.01
        ctl_dt = base_dt + exposure_delay
        vec_now = env.find_vec_to_nearest_pt()
        min_clearance_now = _min_clearance_from_vec(vec_now, env)
        goal_dist_now = torch.norm(env.p_target - env.p, dim=-1).detach()
        collided_cum |= _collision_from_clearance(min_clearance_now, args)
        reached_cum |= (goal_dist_now < 0.35)
        final_goal_dist = goal_dist_now
        if bool(collided_cum.any().item()):
            min_clearance_hist.append(min_clearance_now.detach())
            goal_dist_hist.append(goal_dist_now.detach())
            speed_hist.append(env.v.norm(2, -1).detach())
            stop_reason = 'collision'
            break

        depth_obs, _ = render_sensors(env, ctl_dt, power, exposure, gain, differentiable=False)
        policy_depth_obs = select_policy_depth_obs(depth_obs, args.policy_depth_mode)
        target_v_raw = env.p_target - env.p.detach()
        R = build_local_frame(env)
        target_v = compute_target_velocity(target_v_raw, env)
        state, local_v, camera_state, camera_motion_state = build_state_vector(
            env, target_v, R, power, exposure, gain,
            args.no_odom, args.include_camera_state_in_obs,
        )
        _ = local_v
        with autocast(enabled=use_amp):
            act_raw, cam_params, h, cam_h = model(
                state, h,
                depth_obs=policy_depth_obs,
                add_noise=False,
                cam_hx=cam_h,
                camera_state=camera_state,
                camera_motion_state=camera_motion_state,
            )
        act_final = decode_action_direct(act_raw.float(), R, env, B, args.max_acc_cmd)
        render_power, render_exposure, render_gain = power, exposure, gain
        power, exposure, gain, _ = update_camera_params(cam_params.float(), power, exposure, gain, env)
        act_buffer.append(act_final)

        speed_hist.append(env.v.norm(2, -1).detach())
        goal_dist_hist.append(goal_dist_now.detach())
        min_clearance_hist.append(min_clearance_now.detach())
        power_hist.append(render_power.detach())
        exposure_hist.append(render_exposure.detach())
        gain_hist.append(render_gain.detach())
        fill_hist.append(compute_depth_fill_rate(depth_obs, args.depth_min_valid).detach())

        scene_debug = env.export_last_diff_depth_debug(0)
        if collect_trace:
            p_local = torch.bmm(env.R_scene_T, env.p[:, :, None])[:, :, 0]
            trace_rows.append({
                'episode_idx': ep_idx,
                'step': t,
                'scene_name': scene_name,
                'slit_slot': env.get_scene_effects_for_env(0).get('slit_slot_name'),
                'x': float(env.p[0, 0].detach().cpu()),
                'y': float(env.p[0, 1].detach().cpu()),
                'z': float(env.p[0, 2].detach().cpu()),
                'local_x': float(p_local[0, 0].detach().cpu()),
                'local_y': float(p_local[0, 1].detach().cpu()),
                'local_z': float(p_local[0, 2].detach().cpu()),
                'power': float(render_power[0].detach().cpu()),
                'exposure': float(render_exposure[0].detach().cpu()),
                'gain': float(render_gain[0].detach().cpu()),
                'goal_dist': float(goal_dist_hist[-1][0].detach().cpu()),
                'min_margin': float((min_clearance_now[0] - env.margin[0]).detach().cpu()),
                'clearance': float(min_clearance_now[0].detach().cpu()),
                'scene_effect_mean': scene_debug.get('scalars', {}).get('scene_effect_mean', 0.0),
            })

        if log_vis and (t % max(args.vis_every_steps, 1) == 0):
            j = int(min(max(args.vis_env_idx, 0), B - 1))
            vis_scalars = _eval_scalars(scene_debug, min_clearance_now, goal_dist_now, env, args, env_idx=j)
            vis.log_step(
                phase=vis_phase,
                step_idx=t,
                pos=env.p[j].detach().cpu().numpy(),
                target=env.p_target[j].detach().cpu().numpy(),
                depth=depth_obs[j].detach().cpu().numpy(),
                cam=(float(render_power[j].cpu()), float(render_exposure[j].cpu()), float(render_gain[j].cpu())),
                scalars=vis_scalars,
                raw_depth_img=scene_debug.get('images', {}).get('raw_depth_map'),
                quality_img=scene_debug.get('images', {}).get('quality_map'),
                invalid_img=scene_debug.get('images', {}).get('invalid_mask'),
                scene_effect_img=scene_debug.get('images', {}).get('scene_effect_map'),
                drone_R=env.R[j].detach().cpu().numpy(),
                cam_R=env.R_cam[j].detach().cpu().numpy(),
                main_fov_half_tan=float(env._fov_x_half_tan),
                main_hw=(int(env.height), int(env.width)),
                depth_hw=(int(env.height), int(env.width)),
            )

        env.run(act_buffer[t], ctl_dt, target_v_raw)
        vec_after = env.find_vec_to_nearest_pt()
        min_clearance_after = _min_clearance_from_vec(vec_after, env)
        goal_dist_after = torch.norm(env.p_target - env.p, dim=-1).detach()
        collided_cum |= _collision_from_clearance(min_clearance_after, args)
        reached_cum |= (goal_dist_after < 0.35)
        final_goal_dist = goal_dist_after
        if bool(collided_cum.any().item()):
            min_clearance_hist.append(min_clearance_after.detach())
            goal_dist_hist.append(goal_dist_after.detach())
            if collect_trace:
                p_local = torch.bmm(env.R_scene_T, env.p[:, :, None])[:, :, 0]
                trace_rows.append({
                    'episode_idx': ep_idx,
                    'step': t + 1,
                    'event': 'collision_after_run',
                    'scene_name': scene_name,
                    'slit_slot': env.get_scene_effects_for_env(0).get('slit_slot_name'),
                    'x': float(env.p[0, 0].detach().cpu()),
                    'y': float(env.p[0, 1].detach().cpu()),
                    'z': float(env.p[0, 2].detach().cpu()),
                    'local_x': float(p_local[0, 0].detach().cpu()),
                    'local_y': float(p_local[0, 1].detach().cpu()),
                    'local_z': float(p_local[0, 2].detach().cpu()),
                    'power': float(power[0].detach().cpu()),
                    'exposure': float(exposure[0].detach().cpu()),
                    'gain': float(gain[0].detach().cpu()),
                    'goal_dist': float(goal_dist_after[0].detach().cpu()),
                    'min_margin': float((min_clearance_after[0] - env.margin[0]).detach().cpu()),
                    'clearance': float(min_clearance_after[0].detach().cpu()),
                    'scene_effect_mean': scene_debug.get('scalars', {}).get('scene_effect_mean', 0.0),
                })
            if log_vis:
                j = int(min(max(args.vis_env_idx, 0), B - 1))
                vis.log_step(
                    phase=vis_phase,
                    step_idx=t + 1,
                    pos=env.p[j].detach().cpu().numpy(),
                    target=env.p_target[j].detach().cpu().numpy(),
                    cam=(float(power[j].detach().cpu()), float(exposure[j].detach().cpu()), float(gain[j].detach().cpu())),
                    scalars=_eval_scalars({}, min_clearance_after, goal_dist_after, env, args, env_idx=j, collided=True),
                    drone_R=env.R[j].detach().cpu().numpy(),
                    cam_R=env.R_cam[j].detach().cpu().numpy(),
                    main_fov_half_tan=float(env._fov_x_half_tan),
                    main_hw=(int(env.height), int(env.width)),
                    depth_hw=(int(env.height), int(env.width)),
                )
            stop_reason = 'collision'
            break

    min_clearance = torch.stack(min_clearance_hist)
    goal_dist = torch.stack(goal_dist_hist)
    collided = collided_cum | torch.any(_collision_from_clearance(min_clearance, args), dim=0)
    reached = reached_cum | torch.any(goal_dist < 0.35, dim=0)
    success = reached & (~collided)
    cam_stats = compute_camera_param_stats(power_hist, exposure_hist, gain_hist)
    row = {
        'scene_name': scene_name,
        'success_rate': float(success.float().mean().cpu()),
        'collision_rate': float(collided.float().mean().cpu()),
        'goal_reach_rate': float(reached.float().mean().cpu()),
        'final_goal_dist': float(final_goal_dist.mean().cpu()),
        'avg_speed': float(torch.stack(speed_hist).mean().cpu()),
        'fill_rate': float(torch.stack(fill_hist).mean().cpu()) if fill_hist else 0.0,
        'slit_slot': env.get_scene_effects_for_env(0).get('slit_slot_name'),
        'power_mean': cam_stats.get('power_mean', 0.0),
        'exposure_mean': cam_stats.get('exposure_mean', 0.0),
        'gain_mean': cam_stats.get('gain_mean', 0.0),
        'steps': int(min_clearance.shape[0]),
        'stop_reason': stop_reason,
    }
    print(
        f"[eval] episode={ep_idx + 1}/{args.eval_episodes} scene={scene_name} "
        f"success_rate={row['success_rate']:.3f} collision_rate={row['collision_rate']:.3f} "
        f"final_goal_dist={row['final_goal_dist']:.3f}"
    )
    return row, (trace_rows or [])


def main():
    args = parse_eval_args()
    print_runtime_mode(args)
    if not args.resume:
        raise ValueError('eval requires --resume')
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(args.resume)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    env = build_env(args.batch_size, args, device, eval_mode=True)
    obs_dim = 7 if args.no_odom else 10
    model = Model(
        obs_dim,
        3,
        include_camera_state_in_obs=args.include_camera_state_in_obs,
        use_policy_intent=False,
        depth_nn_width=args.depth_nn_width,
        depth_nn_height=args.depth_nn_height,
        depth_use_pipeline=args.depth_use_pipeline,
        depth_min_valid=args.depth_min_valid,
        depth_max_range=args.depth_max_range,
    ).to(device)
    print(f'[eval] loading checkpoint: {args.resume}')
    state_dict = torch.load(args.resume, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    vis = RerunVis(
        enabled=(args.vis_enable and args.vis_backend == 'rerun'),
        app_id='DiffPhysDrone-Eval',
        spawn=args.vis_spawn,
    )
    if vis.enabled:
        vis.send_eval_episode_blueprint(
            num_episodes=int(args.eval_episodes),
            vis_episode_idx=int(args.vis_episode_idx),
        )
    episode_rows, trace_rows = [], []
    if args.eval_episode_csv and os.path.exists(args.eval_episode_csv):
        os.remove(args.eval_episode_csv)
    if args.eval_trace_csv and os.path.exists(args.eval_trace_csv):
        os.remove(args.eval_trace_csv)
    with torch.no_grad():
        for ep in range(args.eval_episodes):
            scene = args.scenarios[ep % len(args.scenarios)]
            row, trace = run_one_episode(ep, scene, args, model, env, vis, device, collect_trace=bool(args.eval_trace_csv))
            episode_rows.append(row)
            trace_rows.extend(trace)
            if args.eval_episode_csv:
                _append_csv_rows(args.eval_episode_csv, [row])
            if args.eval_trace_csv and trace:
                _append_csv_rows(args.eval_trace_csv, trace)
    _write_csv_rows(args.eval_episode_csv, episode_rows)
    _write_csv_rows(args.eval_trace_csv, trace_rows)
    if episode_rows:
        keys = ['success_rate', 'collision_rate', 'goal_reach_rate', 'final_goal_dist', 'avg_speed', 'fill_rate']
        print('[eval] overall summary:')
        for key in keys:
            print(f"  {key:<16}: {sum(float(r[key]) for r in episode_rows) / len(episode_rows):.4f}")
    print('[eval] done.')


if __name__ == '__main__':
    main()

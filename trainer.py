"""Minimal full-BPTT trainer for active-sensing simulation."""
from random import normalvariate
import math
import time

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from losses import compute_camera_losses, compute_physics_losses, aggregate_loss
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
    compute_render_health_metrics,
)
from train_utils import MetricSmoother, periodic_tail_ops


def _stack_or_none(values):
    if values is None or len(values) == 0:
        return None
    return torch.stack(values)


def _compute_success_collision(p_history, distance, env):
    collision = torch.any(distance.flatten(0, 1) <= 0, dim=0)
    final_dist = torch.norm(env.p_target - p_history[-1], dim=-1)
    reached = final_dist < 0.35
    success = reached & (~collision)
    return success.float().mean(), collision.float().mean(), final_dist.mean()


def _rollout(env, model, args, B, device, use_amp, vis, should_vis):
    h = None
    act_buffer = [env.act] * 2
    power, exposure, gain = init_camera_params(env, B, device)
    camera_initial = torch.stack([power.detach(), exposure.detach(), gain.detach()], -1)

    p_history, v_history, target_v_history = [], [], []
    vec_history, act_history, cam_history = [], [], []
    power_history, exposure_history, gain_history, speed_history, fill_history = [], [], [], [], []

    sensor_differentiable = getattr(args, 'sensor_grad_mode', 'full') == 'full'

    for t in range(args.timesteps):
        base_dt = normalvariate(1 / args.base_control_freq, 0.1 / args.base_control_freq)
        exposure_delay = float(diff_depth_exposure_to_time(exposure.mean().detach(), camera_semantics=env.cam_sem)) * 0.01
        ctl_dt = base_dt + exposure_delay

        depth_obs, _ = render_sensors(env, ctl_dt, power, exposure, gain, differentiable=sensor_differentiable)
        fill_hard, fill_soft, fill_health = compute_render_health_metrics(env, depth_obs, min_valid_depth=args.depth_min_valid)
        _ = fill_hard, fill_soft
        policy_depth_obs = select_policy_depth_obs(depth_obs, args.policy_depth_mode)

        vec_now = env.find_vec_to_nearest_pt()
        target_v_raw = env.p_target - env.p.detach()
        R = build_local_frame(env)
        target_v = compute_target_velocity(target_v_raw, env)
        state, local_v = build_state_vector(
            env, target_v, R, power, exposure, gain,
            args.no_odom, args.include_camera_state_in_obs,
        )
        _ = local_v

        with autocast(enabled=use_amp):
            act_raw, cam_params, h = model(state, h, depth_obs=policy_depth_obs, add_noise=True)
        act_raw = act_raw.float()
        cam_params = cam_params.float()

        render_power, render_exposure, render_gain = power, exposure, gain
        act, v_pred = decode_action_direct(act_raw, R, env, B, args.max_acc_cmd)
        power, exposure, gain, cam_hist_entry = update_camera_params(cam_params, power, exposure, gain, env)
        act_buffer.append(act)

        p_history.append(env.p)
        v_history.append(env.v)
        target_v_history.append(target_v)
        vec_history.append(vec_now)
        act_history.append(act)
        cam_history.append(cam_hist_entry)
        power_history.append(render_power)
        exposure_history.append(render_exposure)
        gain_history.append(render_gain)
        speed_history.append(env.v.norm(2, -1))
        fill_history.append(fill_health)

        if should_vis and args.vis_student and (t % max(args.vis_every_steps, 1) == 0):
            j = int(min(max(args.vis_env_idx, 0), B - 1))
            scene_debug = env.export_last_diff_depth_debug(j)
            vis.log_step(
                phase='student',
                step_idx=t,
                pos=env.p[j].detach().cpu().numpy(),
                target=env.p_target[j].detach().cpu().numpy(),
                depth=depth_obs[j].detach().cpu().numpy(),
                cam=(float(render_power[j].detach().cpu()), float(render_exposure[j].detach().cpu()), float(render_gain[j].detach().cpu())),
                scalars=scene_debug.get('scalars', {}),
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

    return {
        'p_history': p_history,
        'v_history': v_history,
        'target_v_history': target_v_history,
        'vec_history': vec_history,
        'act_history': act_history,
        'cam_history': cam_history,
        'camera_initial': camera_initial,
        'power_history': power_history,
        'exposure_history': exposure_history,
        'gain_history': gain_history,
        'speed_history': speed_history,
        'fill_history': fill_history,
        'act_buffer': act_buffer,
    }


def _loss_from_rollout(rollout, env, args):
    p_history = torch.stack(rollout['p_history'])
    v_history = torch.stack(rollout['v_history'])
    target_v_history = torch.stack(rollout['target_v_history'])
    vec_history = torch.stack(rollout['vec_history'])
    act_history = torch.stack(rollout['act_history'])
    prev_act_tail = rollout['act_buffer'][1]

    physics_losses = compute_physics_losses(
        v_history,
        target_v_history,
        act_history,
        vec_history,
        p_history,
        env.margin,
        prev_act_tail,
        win=args.loss_v_window,
    )
    camera_losses = compute_camera_losses(
        _stack_or_none(rollout['cam_history']),
        _stack_or_none(rollout['power_history']),
        _stack_or_none(rollout['exposure_history']),
        _stack_or_none(rollout['gain_history']),
        _stack_or_none(rollout['speed_history']),
        _stack_or_none(rollout['fill_history']),
        min_fill_rate=args.diff_depth_min_fill_rate,
        camera_semantics=env.cam_sem,
        power_baseline=args.cam_power_baseline,
        cam_initial=rollout['camera_initial'],
    )
    loss, loss_terms = aggregate_loss(physics_losses, camera_losses, args)
    distance = torch.norm(vec_history + 1e-6, 2, -1) - env.margin
    return loss, loss_terms, p_history, distance


def train(args, model, env_train, env_full, optim, sched, scaler, vis, checkpoint_dir, device):
    _ = env_full
    use_amp = bool(args.amp and device.type == 'cuda')
    smoother = MetricSmoother(args)
    pbar = tqdm(range(args.num_iters), ncols=80)

    for i in pbar:
        iter_tic = time.time()
        env_train.reset()
        model.reset()
        B = env_train.batch_size
        should_vis = bool(args.vis_enable and (i % max(args.vis_every_iters, 1) == 0))
        if should_vis:
            vis.begin_iter(i)
            j = int(min(max(args.vis_env_idx, 0), B - 1))
            vis.log_environment(
                phase='student',
                balls=env_train.balls[j].detach().cpu().numpy(),
                voxels=env_train.voxels[j].detach().cpu().numpy(),
                cyl=env_train.cyl[j].detach().cpu().numpy(),
                cyl_h=env_train.cyl_h[j].detach().cpu().numpy(),
                start=env_train.p[j].detach().cpu().numpy(),
                target=env_train.p_target[j].detach().cpu().numpy(),
                scene_name=getattr(env_train, 'current_scene_name', None),
                scene_effects=env_train.get_scene_effects_for_env(j),
            )

        rollout = _rollout(env_train, model, args, B, device, use_amp, vis, should_vis)
        loss, loss_terms, p_history, distance = _loss_from_rollout(rollout, env_train, args)

        optim.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optim.step()
            sched.step()
        else:
            print(f'[warn] non-finite loss at iter {i}; skip optimizer step')
        if device.type == 'cuda':
            torch.cuda.synchronize()

        success_rate, collision_rate, final_goal_dist = _compute_success_collision(p_history.detach(), distance.detach(), env_train)
        cam_stats = compute_camera_param_stats(
            rollout['power_history'], rollout['exposure_history'], rollout['gain_history'])
        iter_time = time.time() - iter_tic
        iter_per_sec = 1.0 / max(iter_time, 1e-6)
        sim_fps = iter_per_sec * args.timesteps * B

        pbar.set_description_str(f'loss: {float(loss.detach()):.3f}')
        log = {
            'loss': float(loss.detach()),
            'loss_v': float(loss_terms['loss_v'].detach()),
            'loss_obj_avoidance': float(loss_terms['loss_obj_avoidance'].detach()),
            'loss_collide': float(loss_terms['loss_collide'].detach()),
            'loss_d_acc': float(loss_terms['loss_d_acc'].detach()),
            'loss_d_jerk': float(loss_terms['loss_d_jerk'].detach()),
            'loss_cam_smooth': float(loss_terms['loss_cam_smooth'].detach()),
            'loss_diff_depth_power': float(loss_terms['loss_diff_depth_power'].detach()),
            'collision_rate': float(collision_rate.detach()),
            'success_rate': float(success_rate.detach()),
            'goal_dist/final': float(final_goal_dist.detach()),
            'iter_per_sec': iter_per_sec,
            'sim_fps': sim_fps,
        }
        log.update({f'cam/{k}': v for k, v in cam_stats.items()})
        smoother.add(log)
        if args.vis_enable:
            vis.log_train_scalars({k: v for k, v in log.items() if k in {'loss', 'collision_rate', 'success_rate', 'goal_dist/final'}}, iter_idx=i)
        periodic_tail_ops(i, checkpoint_dir, model, smoother)

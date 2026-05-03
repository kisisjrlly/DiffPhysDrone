"""Minimal full-BPTT trainer for active-sensing simulation."""
from random import normalvariate
import math
import time

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import wandb

try:
    from matplotlib import pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

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
    compute_depth_fill_health,
)
from train_utils import MetricSmoother, periodic_tail_ops


def _stack_or_none(values):
    if values is None or len(values) == 0:
        return None
    return torch.stack(values)


def _compute_success_collision(p_history, clearance, env, args):
    collision = torch.any(clearance.flatten(0, 1) <= float(args.collision_clearance), dim=0)
    final_dist = torch.norm(env.p_target - p_history[-1], dim=-1)
    reached = final_dist < 0.35
    success = reached & (~collision)
    return success.float().mean(), collision.float().mean(), final_dist.mean()


def _build_loss_contrib_metrics(loss_terms, args):
    specs = [
        ('v', 'loss_v', args.coef_v),
        ('obj_avoidance', 'loss_obj_avoidance', args.coef_obj_avoidance),
        ('collide', 'loss_collide', args.coef_collide),
        ('d_acc', 'loss_d_acc', args.coef_d_acc),
        ('d_jerk', 'loss_d_jerk', args.coef_d_jerk),
        ('cam_smooth', 'loss_cam_smooth', args.coef_cam_smooth),
        ('diff_depth_power', 'loss_diff_depth_power', args.coef_diff_depth_power),
        ('diff_depth_blur', 'loss_diff_depth_blur', args.coef_diff_depth_blur),
        ('diff_depth_noise', 'loss_diff_depth_noise', args.coef_diff_depth_noise),
        ('diff_depth_fill', 'loss_diff_depth_fill', args.coef_diff_depth_fill),
    ]
    contrib = {}
    for name, key, coef in specs:
        coef = float(coef)
        if abs(coef) <= 1e-12:
            continue
        value = loss_terms.get(key, None)
        if value is None:
            continue
        contrib[name] = coef * float(value.detach())
    if not contrib:
        return {}
    total = sum(abs(v) for v in contrib.values())
    total = total if total > 1e-12 else 1e-12
    out = {}
    for name, value in contrib.items():
        out[f'loss_contrib/{name}'] = value
        out[f'loss_share/{name}'] = abs(value) / total
    return out


def _stack_history_for_plot(values):
    if values is None or len(values) == 0:
        return None
    return torch.stack([v.detach() if isinstance(v, torch.Tensor) else torch.as_tensor(v) for v in values])


def _plot_xyz(seq, title, labels=('x', 'y', 'z'), extra=None, ylabel=None, subtitle=None):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for k, label in enumerate(labels):
        ax.plot(seq[:, k].numpy(), label=label)
    if extra:
        for label, values in extra:
            ax.plot(values.numpy(), '--', label=label)
    ax.set_title(title if not subtitle else f'{title}\n{subtitle}')
    ax.set_xlabel('timestep')
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best')
    fig.tight_layout()
    return fig


def _episode_history_metadata(env, env_idx):
    if env is None:
        return '', {}
    fx = {}
    try:
        fx = env.get_scene_effects_for_env(env_idx)
    except Exception:
        fx = {}
    scene_name = str(fx.get('sensor_regime_name', getattr(env, 'current_scene_name', 'unknown')))
    slot = str(fx.get('slit_slot_name', 'unknown'))
    slit_y = fx.get('slit_center_y', None)
    yaw = None
    try:
        yaw = float(env.get_scene_yaw_for_env(env_idx))
    except Exception:
        pass
    parts = [f'scene={scene_name}', f'slot={slot}', f'env={int(env_idx)}']
    if slit_y is not None:
        parts.append(f'slit_y={float(slit_y):+.3f}')
    if yaw is not None:
        parts.append(f'yaw={yaw:+.2f}rad')
    return ' | '.join(parts), {}


def _to_local_position_history(ph, env, env_idx):
    if env is None or not hasattr(env, 'R_scene_T'):
        return None
    try:
        R_scene_T = env.R_scene_T[int(env_idx)].detach().cpu()
        return torch.matmul(ph, R_scene_T.T)
    except Exception:
        return None


def _log_episode_history_plots(rollout, args, iter_idx, env=None):
    """Log one rollout's time-series curves to WandB.

    These are diagnostic plots, not smoothed training scalars.  They restore the
    old p_history/camera-parameter view so a single episode's behavior can be
    inspected directly in WandB.
    """
    p_hist = _stack_history_for_plot(rollout.get('p_history'))
    v_hist = _stack_history_for_plot(rollout.get('v_history'))
    power_hist = _stack_history_for_plot(rollout.get('power_history'))
    exposure_hist = _stack_history_for_plot(rollout.get('exposure_history'))
    gain_hist = _stack_history_for_plot(rollout.get('gain_history'))
    if p_hist is None or v_hist is None:
        return

    B = int(p_hist.shape[1])
    j = int(min(max(getattr(args, 'vis_env_idx', 0), 0), B - 1))
    ph = p_hist[:, j].detach().cpu()
    vh = v_hist[:, j].detach().cpu()
    local_ph = _to_local_position_history(ph, env, j)
    meta_title, meta_scalars = _episode_history_metadata(env, j)

    if not MATPLOTLIB_AVAILABLE:
        return

    speed = vh.norm(2, -1)
    figs = []
    try:
        fig_p = _plot_xyz(ph, 'episode position world xyz', ylabel='m', subtitle=meta_title)
        figs.append(fig_p)
        fig_p_local = None
        if local_ph is not None:
            fig_p_local = _plot_xyz(local_ph, 'episode position local xyz', ylabel='m', subtitle=meta_title)
            figs.append(fig_p_local)
        fig_v = _plot_xyz(vh, 'episode velocity world xyz', extra=[('speed', speed)], ylabel='m/s', subtitle=meta_title)
        figs.append(fig_v)

        log_payload = {
            'episode_history/position_xyz': wandb.Image(fig_p, caption=meta_title),
            'episode_history/v_history': wandb.Image(fig_v, caption=meta_title),
        }
        if fig_p_local is not None:
            log_payload['episode_history/position_xyz_local'] = wandb.Image(fig_p_local, caption=meta_title)
        log_payload.update(meta_scalars)
        if power_hist is not None and exposure_hist is not None and gain_hist is not None:
            cam = torch.stack([
                power_hist[:, j].detach().cpu(),
                exposure_hist[:, j].detach().cpu(),
                gain_hist[:, j].detach().cpu(),
            ], -1)
            fig_cam = _plot_xyz(
                cam,
                'episode camera params',
                labels=('power', 'exposure', 'gain'),
                ylabel='0..1',
                subtitle=meta_title,
            )
            figs.append(fig_cam)
            log_payload['episode_history/camera_params'] = wandb.Image(fig_cam, caption=meta_title)
        wandb.log(log_payload, step=int(iter_idx) + 1)
    finally:
        for fig in figs:
            plt.close(fig)


def _rollout(env, model, args, B, device, use_amp, vis, should_vis):
    h = None
    cam_h = None
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
        depth_fill = compute_depth_fill_health(
            env,
            depth_obs,
            min_valid_depth=args.depth_min_valid,
            patch_rows=args.diff_depth_health_patch_rows,
            patch_cols=args.diff_depth_health_patch_cols,
            cvar_frac=args.diff_depth_health_cvar_frac,
        )
        policy_depth_obs = select_policy_depth_obs(depth_obs, args.policy_depth_mode)

        vec_now = env.find_vec_to_nearest_pt()
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
                add_noise=True,
                cam_hx=cam_h,
                camera_state=camera_state,
                camera_motion_state=camera_motion_state,
            )
        act_raw = act_raw.float()
        cam_params = cam_params.float()

        render_power, render_exposure, render_gain = power, exposure, gain
        act = decode_action_direct(act_raw, R, env, B, args.max_acc_cmd)
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
        fill_history.append(depth_fill)

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
    clearance = torch.norm(vec_history + 1e-6, 2, -1)
    return loss, loss_terms, p_history, clearance


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
            scene_effects = env_train.get_scene_effects_for_env(j)
            vis.log_environment(
                phase='student',
                balls=env_train.get_world_balls_for_env(j),
                voxels=env_train.get_world_voxels_for_env(j),
                cyl=env_train.get_world_cyl_for_env(j),
                cyl_h=env_train.get_world_cyl_h_for_env(j),
                start=env_train.p[j].detach().cpu().numpy(),
                target=env_train.p_target[j].detach().cpu().numpy(),
                scene_name=scene_effects.get('sensor_regime_name', getattr(env_train, 'current_scene_name', None)),
                scene_effects=scene_effects,
                scene_yaw=env_train.get_scene_yaw_for_env(j),
            )

        rollout = _rollout(env_train, model, args, B, device, use_amp, vis, should_vis)
        loss, loss_terms, p_history, clearance = _loss_from_rollout(rollout, env_train, args)

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

        success_rate, collision_rate, final_goal_dist = _compute_success_collision(
            p_history.detach(), clearance.detach(), env_train, args)
        cam_stats = compute_camera_param_stats(
            rollout['power_history'], rollout['exposure_history'], rollout['gain_history'])
        iter_time = time.time() - iter_tic
        iter_per_sec = 1.0 / max(iter_time, 1e-6)
        sim_fps = iter_per_sec * args.timesteps * B

        pbar.set_description_str(f'loss: {float(loss.detach()):.3f}')
        log = {
            'loss': float(loss.detach()),
            'collision_rate': float(collision_rate.detach()),
            'success_rate': float(success_rate.detach()),
            'charts/goal_dist': float(final_goal_dist.detach()),
            'iter_per_sec': iter_per_sec,
            'sim_fps': sim_fps,
        }
        log.update(_build_loss_contrib_metrics(loss_terms, args))
        log.update({f'cam/{k}': v for k, v in cam_stats.items()})
        smoother.add(log)
        should_log_episode_history = (
            not args.wandb_disabled
            and bool(args.wandb_episode_history)
            and (i % max(args.wandb_episode_history_every_iters, 1) == 0)
        )
        if should_log_episode_history:
            _log_episode_history_plots(rollout, args, i, env=env_train)
        if args.vis_enable:
            vis.log_train_scalars({k: v for k, v in log.items() if k in {'loss', 'collision_rate', 'success_rate', 'charts/goal_dist'}}, iter_idx=i)
        periodic_tail_ops(i, checkpoint_dir, model, smoother)

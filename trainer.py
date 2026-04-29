"""
Main training loop — extracted from monolithic main_cuda.py.

Created incrementally:
  Step 1: imports + teacher_phase()
  Step 2: student_rollout_step() + tbptt chunk
  Step 3: full_bptt_losses() + logging + train()
"""
from collections import defaultdict
import math
from random import normalvariate
import time

try:
    from matplotlib import pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import wandb
from tqdm import tqdm

from lqr import build_velocity_tracking_linear_system, solve_batched_dlqr
from losses import (
    compute_physics_losses,
    compute_camera_losses,
    compute_distill_loss,
    aggregate_loss,
)
from rollout_ops import (
    render_sensors, build_local_frame, build_state_vector,
    compute_target_velocity, decode_action_direct, decode_action_lqr,
    update_camera_params, diff_depth_exposure_to_time,
    init_camera_params, compute_camera_param_stats, compute_diff_depth_proxies,
    compute_render_health_metrics, select_policy_depth_obs,
)
from train_utils import (
    MetricSmoother, periodic_tail_ops, is_save_iter,
    detach_env_graph, distill_coef_at_iter, teacher_dt_like_student,
    make_yaw_drift_R, active_loss_term_specs, filter_active_loss_scalars,
)


def _stack_or_none(seq):
    return torch.stack(seq) if seq else None


def _accumulate_tbptt_stats(stats, total_loss, loss_terms):
    stats['loss'] += float(total_loss.detach())
    for key, val in loss_terms.items():
        stats[key] += float(val.detach())


def _optimizer_step_and_maybe_advance_scheduler(optim, sched, scaler, use_amp: bool) -> bool:
    """Step optimizer and scheduler in a PyTorch-safe order."""
    if use_amp:
        scale_before = float(scaler.get_scale())
        scaler.step(optim)
        scaler.update()
        did_step = float(scaler.get_scale()) >= scale_before
    else:
        optim.step()
        did_step = True
    if did_step:
        sched.step()
    return did_step


# =====================================================================
# Teacher Phase I — inner-loop trajectory optimization
# =====================================================================

def _teacher_initial_guess(env, model, args, B, device, use_amp, yaw_drift_R):
    """Roll out current student policy (no grad) to get initial guess for teacher."""
    optimize_intent = bool(args.policy_output_intent and args.use_dmpc)

    init_acts, init_intents, init_cam_deltas = [], [], []
    h_tmp = None
    act_buf_tmp = [env.act] * 2
    tv_raw = env.p_target - env.p

    power, exposure, gain = init_camera_params(env, B, device)

    for t in range(args.timesteps):
        dt = teacher_dt_like_student(
            float(exposure.mean()),
            args.base_control_freq,
            camera_semantics=env.cam_sem,
        )
        depth_obs, _ = render_sensors(env, dt, power, exposure, gain, differentiable=False)
        policy_depth_obs = select_policy_depth_obs(depth_obs, getattr(args, 'policy_depth_mode', 'depth'))
        # torch.cuda.synchronize()
        if args.yaw_drift and yaw_drift_R is not None:
            tv_raw = torch.squeeze(tv_raw[:, None] @ yaw_drift_R, 1)
        else:
            tv_raw = env.p_target - env.p

        env.run(act_buf_tmp[t], dt, tv_raw)

        R = build_local_frame(env)
        target_v = compute_target_velocity(tv_raw, env)
        state, local_v = build_state_vector(
            env, target_v, R, power, exposure, gain,
            args.no_odom, args.include_camera_state_in_obs)

        if args.policy_output_intent:
            with autocast(enabled=use_amp):
                a_out, c_out, h_tmp, y_out = model(
                    state, h_tmp, return_intent=True, depth_obs=policy_depth_obs, add_noise=False)
            a_out, y_out = a_out.float(), y_out.float()
            c_out = c_out.float()
            if optimize_intent:
                init_intents.append(y_out.clone())
        else:
            with autocast(enabled=use_amp):
                a_out, c_out, h_tmp = model(
                    state, h_tmp, depth_obs=policy_depth_obs, add_noise=False)
            a_out = a_out.float()
            c_out = c_out.float()

        init_acts.append(a_out.clone())
        init_cam_deltas.append(c_out.clone())

        if optimize_intent:
            A_lqr_t, B_lqr_t = build_velocity_tracking_linear_system(B, dt, device)
            a_final, _ = decode_action_lqr(
                y_out, R, env, local_v, B,
                A_lqr_t, B_lqr_t,
                args.lqr_horizon, args.lqr_reg, args.max_acc_cmd,
                args.inject_depth_into_lqr, args.depth_safe_dist, args.depth_repel_gain,
                env.find_vec_to_nearest_pt(), solve_batched_dlqr)
        else:
            a_final, _ = decode_action_direct(a_out, R, env, B, args.max_acc_cmd)
        act_buf_tmp.append(a_final)

        power, exposure, gain, _ = update_camera_params(c_out, power, exposure, gain, env)

    return init_acts, init_intents, init_cam_deltas


def _teacher_inner_loop(env, env_snapshot, args,
                        init_acts, init_intents, init_cam_deltas,
                        B, device, yaw_drift_R, vis, should_vis_iter):
    """Run teacher inner optimization (TBPTT) and return u_star / y_star / u_star_cam."""
    optimize_intent = bool(args.policy_output_intent and args.use_dmpc)
    sensor_differentiable = (getattr(args, 'sensor_grad_mode', 'full') == 'full')

    # Build optimizable parameters
    u_guess, y_guess = None, None
    if optimize_intent:
        y_guess = [y.clone().requires_grad_(True) for y in init_intents]
    else:
        u_guess = [a.clone().requires_grad_(True) for a in init_acts]

    u_cam_guess = [c.clone().requires_grad_(True) for c in init_cam_deltas]

    base_params = y_guess if y_guess is not None else u_guess
    assert base_params is not None
    inner_params = list(base_params) + (list(u_cam_guess) if u_cam_guess else [])
    inner_optim = torch.optim.Adam(inner_params, lr=args.teacher_inner_lr)

    teacher_chunk_steps = max(2, args.teacher_tbptt_chunk_steps)
    teacher_chunk_count = max(1, math.ceil(args.timesteps / teacher_chunk_steps))

    for k in range(args.teacher_inner_steps):
        inner_optim.zero_grad()
        env.restore_state(env_snapshot)

        act_buf_k = [env.act.detach()] * 2
        tv_raw_k = env.p_target - env.p
        prev_act_tail_k = env.act.detach()
        v_roll_k, tv_roll_k = [], []
        # chunk accumulators
        c_p, c_v, c_tv, c_vtp, c_act = [], [], [], [], []
        c_sensor_hist = []
        c_exposure, c_gain, c_power, c_speed, c_fill = [], [], [], [], []

        power_k, exposure_k, gain_k = init_camera_params(env, B, device)
        chunk_cam_initial_k = torch.stack([
            power_k.detach(),
            exposure_k.detach(),
            gain_k.detach(),
        ], dim=-1)

        for t in range(args.timesteps):
            dt_k = teacher_dt_like_student(
                float(exposure_k.mean().detach()),
                args.base_control_freq,
                camera_semantics=env.cam_sem,
            )
            depth_obs_k, quality_k = render_sensors(
                env, dt_k, power_k, exposure_k, gain_k, differentiable=sensor_differentiable)
            _, _, fill_health_k = compute_render_health_metrics(
                env,
                depth_obs_k,
                min_valid_depth=args.depth_min_valid,
                patch_rows=args.diff_depth_health_patch_rows,
                patch_cols=args.diff_depth_health_patch_cols,
                cvar_frac=args.diff_depth_health_cvar_frac,
            )
            c_p.append(env.p)
            vec_now_k = env.find_vec_to_nearest_pt()
            c_vtp.append(vec_now_k)

            if args.yaw_drift and yaw_drift_R is not None:
                tv_raw_k = torch.squeeze(tv_raw_k[:, None] @ yaw_drift_R, 1)
            else:
                tv_raw_k = env.p_target - env.p.detach()

            R_k = build_local_frame(env)
            tv_k = compute_target_velocity(tv_raw_k, env)

            # Decode optimizable action
            if optimize_intent and y_guess is not None:
                local_v_k = torch.squeeze(env.v[:, None] @ R_k, 1)
                A_lqr_k, B_lqr_k = build_velocity_tracking_linear_system(B, dt_k, device)
                a_final_k, _ = decode_action_lqr(
                    y_guess[t], R_k, env, local_v_k, B,
                    A_lqr_k, B_lqr_k,
                    args.lqr_horizon, args.lqr_reg, args.max_acc_cmd,
                    args.inject_depth_into_lqr, args.depth_safe_dist, args.depth_repel_gain,
                    vec_now_k, solve_batched_dlqr)
            else:
                assert u_guess is not None
                a_final_k, _ = decode_action_direct(
                    u_guess[t], R_k, env, B, args.max_acc_cmd)

            act_buf_k.append(a_final_k)

            render_power_k = power_k
            render_exposure_k = exposure_k
            render_gain_k = gain_k

            # Camera update controls the next rendered frame.
            power_k, exposure_k, gain_k, sensor_hist_entry = update_camera_params(
                u_cam_guess[t], power_k, exposure_k, gain_k, env)
            c_sensor_hist.append(sensor_hist_entry)
            c_power.append(render_power_k)
            c_exposure.append(render_exposure_k)
            c_gain.append(render_gain_k)

            c_speed.append(env.v.norm(2, -1))
            c_fill.append(fill_health_k)
            c_v.append(env.v)
            c_tv.append(tv_k)
            c_act.append(a_final_k)

            # Visualization (last inner step only)
            if (should_vis_iter and args.vis_teacher
                    and k == args.teacher_inner_steps - 1
                    and t % max(args.vis_every_steps, 1) == 0):
                j = int(min(max(args.vis_env_idx, 0), B - 1))
                cam_vals = (float(render_power_k[j].detach().cpu()),
                            float(render_exposure_k[j].detach().cpu()),
                            float(render_gain_k[j].detach().cpu()))
                scene_debug = env.export_last_diff_depth_debug(j)
                scene_scalars = dict(scene_debug.get('scalars', {}))
                scene_scalars['scene_id'] = float(getattr(env, 'current_scene_id', 0))
                vis.log_step(
                    phase='teacher', step_idx=t,
                    pos=env.p[j].detach().cpu().numpy(),
                    target=env.p_target[j].detach().cpu().numpy(),
                    depth=None, cam=cam_vals, scalars=scene_scalars,
                    raw_depth_img=scene_debug.get('images', {}).get('raw_depth_map'),
                    quality_img=scene_debug.get('images', {}).get('quality_map'),
                    invalid_img=scene_debug.get('images', {}).get('invalid_mask'),
                    scene_effect_img=scene_debug.get('images', {}).get('scene_effect_map'),
                    drone_R=env.R[j].detach().cpu().numpy(),
                    cam_R=env.R_cam[j].detach().cpu().numpy(),
                    main_fov_half_tan=float(env._fov_x_half_tan),
                    main_hw=(int(env.height), int(env.width)),
                    depth_hw=(int(env.height), int(env.width)))

            env.run(act_buf_k[t], dt_k, tv_raw_k)

            # ---- TBPTT chunk boundary ----
            chunk_end_k = ((t + 1) % teacher_chunk_steps == 0) or (t == args.timesteps - 1)
            if chunk_end_k and len(c_v) > 0:
                v_ck = torch.stack(c_v)
                tv_ck = torch.stack(c_tv)
                vec_ck = torch.stack(c_vtp)
                act_ck = torch.stack(c_act)
                p_ck = torch.stack(c_p)

                physics_losses = compute_physics_losses(
                    v_ck, tv_ck, act_ck, vec_ck, p_ck,
                    env.margin, prev_act_tail_k,
                    v_roll=v_roll_k, tv_roll=tv_roll_k, win=args.loss_v_window,
                )
                camera_losses = compute_camera_losses(
                    _stack_or_none(c_sensor_hist),
                    _stack_or_none(c_power),
                    _stack_or_none(c_exposure),
                    _stack_or_none(c_gain),
                    _stack_or_none(c_speed),
                    _stack_or_none(c_fill),
                    min_fill_rate=args.diff_depth_min_fill_rate,
                    camera_semantics=env.cam_sem,
                    power_baseline=args.cam_power_baseline,
                    cam_initial=chunk_cam_initial_k,
                )
                chunk_loss, _ = aggregate_loss(
                    physics_losses,
                    camera_losses,
                    args,
                    chunk_count=teacher_chunk_count,
                )
                chunk_loss.backward()

                keep = 30
                v_for_loss = physics_losses['v_for_loss']
                tv_for_loss = physics_losses['tv_for_loss']
                v_roll_k = [v_for_loss[-keep:].detach()] if v_for_loss.shape[0] > 0 else []
                tv_roll_k = [tv_for_loss[-keep:].detach()] if tv_for_loss.shape[0] > 0 else []
                prev_act_tail_k = act_ck[-1].detach()
                power_k = power_k.detach()
                exposure_k = exposure_k.detach()
                gain_k = gain_k.detach()
                act_buf_k = [a.detach() for a in act_buf_k]
                detach_env_graph(env)
                c_p.clear(); c_v.clear(); c_tv.clear(); c_vtp.clear(); c_act.clear()
                c_sensor_hist.clear()
                c_exposure.clear(); c_gain.clear(); c_power.clear(); c_speed.clear(); c_fill.clear()
                chunk_cam_initial_k = torch.stack([
                    power_k.detach(),
                    exposure_k.detach(),
                    gain_k.detach(),
                ], dim=-1)

        inner_optim.step()

    # Extract optimised sequences
    u_star, y_star, u_star_cam = None, None, None
    if y_guess is not None:
        y_star = [y.detach() for y in y_guess]
    else:
        assert u_guess is not None
        u_star = [u.detach() for u in u_guess]
    u_star_cam = [c.detach() for c in u_cam_guess]
    return u_star, y_star, u_star_cam


def teacher_phase(env, model, args, B, device, use_amp, vis, should_vis_iter):
    """Full teacher phase: initial guess → inner optimisation → return targets."""
    yaw_drift_R = make_yaw_drift_R(B, device) if args.yaw_drift else None
    env_snapshot = env.save_state()

    with torch.no_grad():
        env.restore_state(env_snapshot)
        init_acts, init_intents, init_cam_deltas = _teacher_initial_guess(
            env, model, args, B, device, use_amp, yaw_drift_R)

    u_star, y_star, u_star_cam = _teacher_inner_loop(
        env, env_snapshot, args,
        init_acts, init_intents, init_cam_deltas,
        B, device, yaw_drift_R, vis, should_vis_iter)

    env.restore_state(env_snapshot)
    return u_star, y_star, u_star_cam


# =====================================================================
# Student Phase II — rollout + TBPTT / full-BPTT loss
# =====================================================================

def student_rollout(env, model, args, B, device, use_amp,
                    scaler, optim, sched,
                    u_star, y_star, u_star_cam,
                    distill_coef_iter,
                    tbptt_this_iter,
                    vis, should_vis_iter):
    """Run the student rollout and compute / backprop losses.

    Returns a dict with all quantities needed for logging:
      loss, per-component losses, histories (detached), metrics, etc.
    """
    vid_idx = min(4, B - 1)

    # Global history lists (used for full-BPTT and logging)
    p_history, v_history, target_v_history = [], [], []
    vec_to_pt_history = []
    raw_act_history, raw_intent_history, raw_cam_history = [], [], []
    cam_params_history = []
    power_history, exposure_history, gain_history = [], [], []
    speed_for_depth_history, R_up_history = [], []
    depth_fill_history = []
    depth_fill_soft_history = []
    depth_fill_health_history = []
    h = None

    # TBPTT state
    chunk_steps = max(args.tbptt_chunk_steps, 1)
    chunk_accum = max(args.tbptt_chunk_accum, 1)
    chunk_counter = 0
    tbptt_stats = defaultdict(float)
    tbptt_chunk_n = 0
    if tbptt_this_iter:
        optim.zero_grad(set_to_none=True)
        v_roll, tv_roll = [], []
        c_v_hist, c_tv_hist, c_vec_hist = [], [], []
        c_act_hist, c_p_hist = [], []
        c_sensor_hist, c_exposure, c_gain, c_power, c_speed, c_fill = [], [], [], [], [], []
        prev_act_tail = env.act.detach()

    act_buffer = [env.act] * 2
    target_v_raw = env.p_target - env.p

    R_drift = make_yaw_drift_R(B, device) if args.yaw_drift else None

    power, exposure, gain = init_camera_params(env, B, device)
    rollout_cam_initial = torch.stack([
        power.detach(),
        exposure.detach(),
        gain.detach(),
    ], dim=-1)
    if tbptt_this_iter:
        chunk_cam_initial = rollout_cam_initial
    sensor_differentiable = (getattr(args, 'sensor_grad_mode', 'full') == 'full')

    # ── Main rollout loop ──
    for t in range(args.timesteps):
        base_dt = normalvariate(1 / args.base_control_freq, 0.1 / args.base_control_freq)
        exposure_delay = float(diff_depth_exposure_to_time(
            exposure.mean().detach(),
            camera_semantics=env.cam_sem,
        )) * 0.01
        ctl_dt = base_dt + exposure_delay
        student_add_noise = (args.student_noise_mode == 'on') if args.enable_teacher_student_training else True

        depth_obs, depth_quality = render_sensors(
            env, ctl_dt, power, exposure, gain, differentiable=sensor_differentiable)
        policy_depth_obs = select_policy_depth_obs(depth_obs, getattr(args, 'policy_depth_mode', 'depth'))
        depth_vis = depth_obs
        # Fill/health 用传感器内部有效概率图计算（0~1 语义），
        # 不再把 quality 当作米制 depth 去套 depth_min_valid 阈值。
        fill_rate_t, fill_rate_soft_t, fill_health_t = compute_render_health_metrics(
            env,
            depth_obs,
            min_valid_depth=args.depth_min_valid,
            patch_rows=args.diff_depth_health_patch_rows,
            patch_cols=args.diff_depth_health_patch_cols,
            cvar_frac=args.diff_depth_health_cvar_frac,
        )
        depth_fill_history.append(fill_rate_t.detach() if tbptt_this_iter else fill_rate_t)
        depth_fill_soft_history.append(fill_rate_soft_t.detach() if tbptt_this_iter else fill_rate_soft_t)
        depth_fill_health_history.append(fill_health_t.detach() if tbptt_this_iter else fill_health_t)

        vec_now = env.find_vec_to_nearest_pt()
        if tbptt_this_iter:
            p_history.append(env.p.detach())
            vec_to_pt_history.append(vec_now.detach())
        else:
            p_history.append(env.p)
            vec_to_pt_history.append(vec_now)

        # Target velocity
        if args.yaw_drift and R_drift is not None:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()

        R = build_local_frame(env)
        target_v = compute_target_velocity(target_v_raw, env)
        state, local_v = build_state_vector(
            env, target_v, R, power, exposure, gain,
            args.no_odom, args.include_camera_state_in_obs)

        # Policy forward
        if args.policy_output_intent:
            with autocast(enabled=use_amp):
                act, cam_params, h, intent = model(
                    state, h, return_intent=True, depth_obs=policy_depth_obs, add_noise=student_add_noise)
            act, intent = act.float(), intent.float()
            if args.enable_teacher_student_training and args.use_dmpc:
                raw_intent_history.append(intent)
        else:
            with autocast(enabled=use_amp):
                act, cam_params, h = model(
                    state, h, depth_obs=policy_depth_obs, add_noise=student_add_noise)
            act = act.float()
            intent = None
        cam_params = cam_params.float()
        if args.enable_teacher_student_training:
            raw_act_history.append(act)
        if args.enable_teacher_student_training:
            raw_cam_history.append(cam_params)

        # Track current-frame camera/kinematic state. The camera update below
        # affects the next rendered frame, so logs and depth stay time-aligned.
        if tbptt_this_iter:
            power_history.append(power.detach())
            exposure_history.append(exposure.detach())
            gain_history.append(gain.detach())
            speed_for_depth_history.append(env.v.norm(2, -1).detach())
            R_up_history.append(env.R[:, :, 2].detach().clone())
        else:
            power_history.append(power)
            exposure_history.append(exposure)
            gain_history.append(gain)
            speed_for_depth_history.append(env.v.norm(2, -1))
            R_up_history.append(env.R[:, :, 2].clone())

        # Action decode (LQR or direct)
        if args.use_dmpc and args.policy_output_intent and intent is not None:
            A_lqr_t, B_lqr_t = build_velocity_tracking_linear_system(B, ctl_dt, device)
            act_final, v_pred = decode_action_lqr(
                intent, R, env, local_v, B, A_lqr_t, B_lqr_t,
                args.lqr_horizon, args.lqr_reg, args.max_acc_cmd,
                args.inject_depth_into_lqr, args.depth_safe_dist, args.depth_repel_gain,
                vec_now, solve_batched_dlqr)
        else:
            act_final, v_pred = decode_action_direct(act, R, env, B, args.max_acc_cmd)

        render_power = power
        render_exposure = exposure
        render_gain = gain

        # Camera update is causal: policy output at this frame controls the next
        # depth render. The action command is appended to the delayed act buffer;
        # the current physics step still consumes act_buffer[t].
        power, exposure, gain, cam_hist_entry = update_camera_params(
            cam_params, power, exposure, gain, env)
        cam_params_history.append(cam_hist_entry)
        act = act_final
        act_buffer.append(act)

        if tbptt_this_iter:
            c_v_hist.append(env.v); c_tv_hist.append(target_v)
            c_vec_hist.append(vec_now)
            c_act_hist.append(act); c_p_hist.append(env.p)
            c_speed.append(env.v.norm(2, -1))
            c_fill.append(fill_health_t)
            c_sensor_hist.append(cam_hist_entry)
            c_exposure.append(render_exposure)
            c_gain.append(render_gain)
            c_power.append(render_power)
            v_history.append(env.v.detach())
            target_v_history.append(target_v.detach())
        else:
            v_history.append(env.v)
            target_v_history.append(target_v)

        # Visualization before physics integration: current position, depth and
        # camera params all describe the same instant.
        if should_vis_iter and args.vis_student and (t % max(args.vis_every_steps, 1) == 0):
            j = int(min(max(args.vis_env_idx, 0), B - 1))
            cam_vals = (float(power_history[-1][j].detach().cpu()),
                        float(exposure_history[-1][j].detach().cpu()),
                        float(gain_history[-1][j].detach().cpu()))
            main_img_np = None
            main_img_mode = 'depth'
            depth_img_np = depth_obs[j].detach().cpu().numpy()
            scene_debug = env.export_last_diff_depth_debug(j)
            step_scalars = dict(scene_debug.get('scalars', {}))
            step_scalars['scene_id'] = float(getattr(env, 'current_scene_id', 0))
            vis.log_step(
                phase='student', step_idx=t,
                pos=env.p[j].detach().cpu().numpy(),
                target=env.p_target[j].detach().cpu().numpy(),
                depth=depth_vis[j].detach().cpu().numpy(),
                cam=cam_vals, scalars=step_scalars, main_img=main_img_np,
                main_img_mode=main_img_mode, depth_img=depth_img_np,
                raw_depth_img=scene_debug.get('images', {}).get('raw_depth_map'),
                quality_img=scene_debug.get('images', {}).get('quality_map'),
                invalid_img=scene_debug.get('images', {}).get('invalid_mask'),
                scene_effect_img=scene_debug.get('images', {}).get('scene_effect_map'),
                drone_R=env.R[j].detach().cpu().numpy(),
                cam_R=env.R_cam[j].detach().cpu().numpy(),
                main_fov_half_tan=float(env._fov_x_half_tan),
                main_hw=(int(env.height), int(env.width)),
                depth_hw=(int(env.height), int(env.width)))

        env.run(act_buffer[t], ctl_dt, target_v_raw)

        # ── TBPTT chunk accumulation & backward ──
        if tbptt_this_iter:
            chunk_end = ((t + 1) % chunk_steps == 0) or (t == args.timesteps - 1)
            if chunk_end and len(c_v_hist) > 0:
                v_ck = torch.stack(c_v_hist); tv_ck = torch.stack(c_tv_hist)
                vec_ck = torch.stack(c_vec_hist)
                act_ck = torch.stack(c_act_hist); p_ck = torch.stack(c_p_hist)

                physics_losses = compute_physics_losses(
                    v_ck, tv_ck, act_ck, vec_ck, p_ck,
                    env.margin, prev_act_tail,
                    v_roll=v_roll, tv_roll=tv_roll, win=args.loss_v_window,
                )
                camera_losses = compute_camera_losses(
                    _stack_or_none(c_sensor_hist),
                    _stack_or_none(c_power),
                    _stack_or_none(c_exposure),
                    _stack_or_none(c_gain),
                    _stack_or_none(c_speed),
                    _stack_or_none(c_fill),
                    min_fill_rate=args.diff_depth_min_fill_rate,
                    camera_semantics=env.cam_sem,
                    power_baseline=args.cam_power_baseline,
                    cam_initial=chunk_cam_initial,
                )

                loss_distill_c = None
                if args.enable_teacher_student_training and (u_star is not None or y_star is not None):
                    cl = int(act_ck.shape[0])
                    si, ei = int(t + 1 - cl), int(t + 1)
                    loss_distill_c = compute_distill_loss(
                        raw_act_history,
                        raw_intent_history,
                        raw_cam_history,
                        u_star,
                        y_star,
                        u_star_cam,
                        start_idx=si,
                        end_idx=ei,
                        device=device,
                    )

                chunk_loss, loss_terms = aggregate_loss(
                    physics_losses,
                    camera_losses,
                    args,
                    loss_distill=loss_distill_c,
                    distill_coef_iter=(
                        distill_coef_iter if args.enable_teacher_student_training else None
                    ),
                )

                # Determine if we should optimize this chunk
                chunk_counter += 1
                do_step = (chunk_counter % chunk_accum == 0) or (t == args.timesteps - 1)

                # Backprop through this chunk (graphs are independent due to h.detach())
                if use_amp:
                    scaler.scale(chunk_loss).backward()
                else:
                    chunk_loss.backward()

                if do_step:
                    if use_amp:
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        _optimizer_step_and_maybe_advance_scheduler(optim, sched, scaler, use_amp=True)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        _optimizer_step_and_maybe_advance_scheduler(optim, sched, scaler, use_amp=False)
                    optim.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()  # flush GPU queue after optimizer step

                # stats
                _accumulate_tbptt_stats(tbptt_stats, chunk_loss, loss_terms)
                tbptt_chunk_n += 1

                keep = 30
                v_for_loss = physics_losses['v_for_loss']
                tv_for_loss = physics_losses['tv_for_loss']
                v_roll = [v_for_loss[-keep:].detach()] if v_for_loss.shape[0] > 0 else []
                tv_roll = [tv_for_loss[-keep:].detach()] if tv_for_loss.shape[0] > 0 else []
                prev_act_tail = act_ck[-1].detach()
                if h is not None:
                    h = h.detach()
                power = power.detach(); exposure = exposure.detach()
                gain = gain.detach()
                act_buffer = [a.detach() for a in act_buffer]
                detach_env_graph(env)
                c_v_hist.clear(); c_tv_hist.clear()
                c_vec_hist.clear(); c_act_hist.clear(); c_p_hist.clear()
                c_sensor_hist.clear(); c_exposure.clear(); c_gain.clear()
                c_power.clear(); c_speed.clear(); c_fill.clear()
                chunk_cam_initial = torch.stack([
                    power.detach(),
                    exposure.detach(),
                    gain.detach(),
                ], dim=-1)

    # ── End of rollout loop ──
    # Package results for the caller (train loop)
    return {
        'tbptt_this_iter': tbptt_this_iter,
        'tbptt_stats': tbptt_stats,
        'tbptt_chunk_n': tbptt_chunk_n,
        'p_history': p_history,
        'v_history': v_history,
        'target_v_history': target_v_history,
        'vec_to_pt_history': vec_to_pt_history,
        'vid_idx': vid_idx,
        'act_buffer': act_buffer,
        'raw_act_history': raw_act_history,
        'raw_intent_history': raw_intent_history,
        'raw_cam_history': raw_cam_history,
        'cam_params_history': cam_params_history,
        'camera_initial': rollout_cam_initial,
        'power_history': power_history,
        'exposure_history': exposure_history,
        'gain_history': gain_history,
        'speed_for_depth_history': speed_for_depth_history,
        'depth_fill_history': depth_fill_history,
        'depth_fill_soft_history': depth_fill_soft_history,
        'depth_fill_health_history': depth_fill_health_history,
        'R_up_history': R_up_history,
    }


# =====================================================================
# Full-BPTT loss + backward (non-TBPTT iterations)
# =====================================================================

def full_bptt_losses(rollout, env, args, device, u_star, y_star, u_star_cam, distill_coef_iter):
    """Compute all losses for full-BPTT iterations and return (loss, loss_dict)."""
    p_history = torch.stack(rollout['p_history'])
    v_history = torch.stack(rollout['v_history'])
    target_v_history = torch.stack(rollout['target_v_history'])
    vec_to_pt_history = torch.stack(rollout['vec_to_pt_history'])
    prev_act_tail = rollout['act_buffer'][1]
    act_history = torch.stack(rollout['act_buffer'][2:])

    physics_losses = compute_physics_losses(
        v_history, target_v_history, act_history, vec_to_pt_history, p_history,
        env.margin, prev_act_tail, win=args.loss_v_window,
    )

    distance = torch.norm(vec_to_pt_history + 1e-6, 2, -1) - env.margin
    speed_history = v_history.norm(2, -1)

    camera_losses = compute_camera_losses(
        _stack_or_none(rollout['cam_params_history']),
        _stack_or_none(rollout['power_history']),
        _stack_or_none(rollout['exposure_history']),
        _stack_or_none(rollout['gain_history']),
        _stack_or_none(rollout['speed_for_depth_history']),
        _stack_or_none(rollout['depth_fill_health_history']),
        min_fill_rate=args.diff_depth_min_fill_rate,
        camera_semantics=env.cam_sem,
        power_baseline=args.cam_power_baseline,
        cam_initial=rollout.get('camera_initial'),
    )

    loss_distill = None
    if args.enable_teacher_student_training and (u_star is not None or y_star is not None):
        loss_distill = compute_distill_loss(
            rollout['raw_act_history'],
            rollout['raw_intent_history'],
            rollout['raw_cam_history'],
            u_star,
            y_star,
            u_star_cam,
            device=device,
        )

    loss, loss_terms = aggregate_loss(
        physics_losses,
        camera_losses,
        args,
        loss_distill=loss_distill,
        distill_coef_iter=(
            distill_coef_iter if args.enable_teacher_student_training else None
        ),
    )

    loss_dict = {
        **loss_terms,
        'distance': distance, 'speed_history': speed_history,
        'p_history': p_history, 'v_history': v_history,
        'vec_to_pt_history': vec_to_pt_history,
        'act_buffer': act_history,
    }
    return loss, loss_dict


# =====================================================================
# Logging helpers
# =====================================================================

def _compute_emerging_metrics(rollout, loss_dict, env, args, smoother):
    """Compute and log emerging-behavior metrics for the fixed small-map scenarios."""
    p_history = loss_dict.get('p_history')
    vec_to_pt_history = loss_dict.get('vec_to_pt_history')
    distance = loss_dict.get('distance')

    # Roll angle
    if rollout['R_up_history']:
        up_hist = torch.stack([x.detach() if x.requires_grad else x for x in rollout['R_up_history']])
        roll_angle = torch.acos(up_hist[:, :, 2].clamp(-1, 1))
        roll_deg = roll_angle * 180 / math.pi
        smoother.add({'roll_max_deg': roll_deg.max().item(), 'roll_mean_deg': roll_deg.mean().item()})

    # Speed-exposure correlation
    if rollout['exposure_history']:
        _sp = torch.stack([x.detach() if x.requires_grad else x for x in rollout['speed_for_depth_history']])
        _ex = torch.stack([x.detach() if x.requires_grad else x for x in rollout['exposure_history']])
        sp_m = _sp.mean(0, keepdim=True); ex_m = _ex.mean(0, keepdim=True)
        cov = ((_sp - sp_m) * (_ex - ex_m)).mean(0)
        sp_s = (_sp - sp_m).pow(2).mean(0).sqrt().clamp(min=1e-6)
        ex_s = (_ex - ex_m).pow(2).mean(0).sqrt().clamp(min=1e-6)
        smoother.add({'speed_exposure_corr': (cov / (sp_s * ex_s)).mean().item()})

        _pw = torch.stack([x.detach() if x.requires_grad else x for x in rollout['power_history']])
        _dn = torch.norm(vec_to_pt_history, 2, -1).min(1).values if vec_to_pt_history is not None else _pw
        pw_m = _pw.mean(0, keepdim=True); dn_m = _dn.mean(0, keepdim=True)
        cov_pd = ((_pw - pw_m) * (_dn - dn_m)).mean(0)
        pw_s = (_pw - pw_m).pow(2).mean(0).sqrt().clamp(min=1e-6)
        dn_s = (_dn - dn_m).pow(2).mean(0).sqrt().clamp(min=1e-6)
        smoother.add({'power_obstacle_corr': (cov_pd / (pw_s * dn_s)).mean().item()})

    if rollout.get('depth_fill_history'):
        fill_hist = torch.stack([
            x.detach() if isinstance(x, torch.Tensor) and x.requires_grad else x
            for x in rollout['depth_fill_history']
        ])
        fill_mean = fill_hist.mean()
        smoother.add({
            'diff_depth_fill_rate': float(fill_mean.item()),
            'diff_depth_hole_rate': float((1.0 - fill_mean).item()),
        })
    if rollout.get('depth_fill_soft_history'):
        fill_soft_hist = torch.stack([
            x.detach() if isinstance(x, torch.Tensor) and x.requires_grad else x
            for x in rollout['depth_fill_soft_history']
        ])
        fill_soft_mean = fill_soft_hist.mean()
        smoother.add({
            'diff_depth_fill_rate_soft': float(fill_soft_mean.item()),
            'diff_depth_hole_rate_soft': float((1.0 - fill_soft_mean).item()),
        })
    if rollout.get('depth_fill_health_history'):
        health_hist = torch.stack([
            x.detach() if isinstance(x, torch.Tensor) and x.requires_grad else x
            for x in rollout['depth_fill_health_history']
        ])
        health_mean = health_hist.mean()
        smoother.add({
            'diff_depth_sensor_health_cvar': float(health_mean.item()),
            'diff_depth_sensor_health_gap': float((1.0 - health_mean).item()),
        })
    cam_stats = compute_camera_param_stats(
        rollout.get('power_history'),
        rollout.get('exposure_history'),
        rollout.get('gain_history'),
    )
    if cam_stats:
        smoother.add({f'cam/{k}': v for k, v in cam_stats.items()})

    proxy_stats = compute_diff_depth_proxies(
        rollout.get('power_history'),
        rollout.get('exposure_history'),
        rollout.get('gain_history'),
        rollout.get('speed_for_depth_history'),
        camera_semantics=env.cam_sem,
    )
    if proxy_stats:
        smoother.add(proxy_stats)

    scene_name = getattr(env, 'current_scene_name', None)
    scene_tag = getattr(env, 'current_scene_tag', scene_name)
    if scene_name is not None:
        smoother.add({'scene/current_id': float(getattr(env, 'current_scene_id', -1))})
        for name in getattr(env, 'scene_name_to_id', {}):
            smoother.add({f'scene/is_{name}': 1.0 if name == scene_name else 0.0})
    if scene_tag is not None and scene_tag != scene_name:
        smoother.add({f'scene/is_{scene_tag}': 1.0})


def _compute_final_goal_distance_metric(rollout, env):
    """Return final train-time goal distance in meters."""
    positions = []
    for p in rollout.get('p_history', []):
        if isinstance(p, torch.Tensor):
            positions.append(p.detach())
    if isinstance(getattr(env, 'p', None), torch.Tensor):
        positions.append(env.p.detach())

    if not positions:
        return {}

    p_seq = torch.stack(positions)
    target = env.p_target.detach().unsqueeze(0)
    final_goal_dist = torch.norm(target - p_seq[-1:].detach(), dim=-1)
    return {'goal_dist/final': float(final_goal_dist.mean().item())}


def _build_loss_share_metrics(loss_scalars: dict, args, distill_coef_iter: float) -> dict:
    """Build weighted loss contribution and share metrics for WandB.

    Returns keys like:
      - loss_contrib/<name>
      - loss_share/<name>
    """
    contrib = {}
    for name, loss_key, coef in active_loss_term_specs(
        args,
        distill_coef_iter=(
            float(distill_coef_iter)
            if args.enable_teacher_student_training
            else None
        ),
    ):
        raw_v = float(loss_scalars.get(loss_key, 0.0))
        if name == 'distill':
            contrib[name] = coef * raw_v
        else:
            physics_scale = float(args.student_physics_coef) if args.enable_teacher_student_training else 1.0
            contrib[name] = physics_scale * coef * raw_v

    if not contrib:
        return {}

    total = sum(contrib.values())
    eps = 1e-12
    if abs(total) < eps:
        total = eps

    physics_total = sum(v for k, v in contrib.items() if k != 'distill')
    out = {}
    for name, val in contrib.items():
        out[f'loss_contrib/{name}'] = float(val)
        out[f'loss_share/{name}'] = float(val / total)
    if 'distill' in contrib:
        out['loss_share/physics_total'] = float(physics_total / total)
    return out


def _log_save_iter(rollout, loss_dict, env, args, i):
    """Save plots to WandB on checkpoint iterations (video logging removed)."""
    if not MATPLOTLIB_AVAILABLE:
        print('[warn] matplotlib not installed: skip figure logging.')
        return
    vid_idx = rollout['vid_idx']
    p_history = loss_dict['p_history']
    v_history = loss_dict['v_history']
    act_buffer = loss_dict['act_buffer']
    print("save checkpoint figures:", i)

    fig_p, ax = plt.subplots()
    ph = p_history[:, vid_idx].detach().cpu()
    ax.plot(ph[:, 0], label='x'); ax.plot(ph[:, 1], label='y'); ax.plot(ph[:, 2], label='z')
    ax.legend()

    fig_v, ax = plt.subplots()
    vh = v_history[:, vid_idx].detach().cpu()
    ax.plot(vh[:, 0], label='x'); ax.plot(vh[:, 1], label='y'); ax.plot(vh[:, 2], label='z')
    ax.legend()

    fig_a, ax = plt.subplots()
    ah = act_buffer[:, vid_idx].detach().cpu()
    ax.plot(ah[:, 0], label='x'); ax.plot(ah[:, 1], label='y'); ax.plot(ah[:, 2], label='z')
    ax.legend()

    wandb.log({
        "p_history": wandb.Image(fig_p),
        "v_history": wandb.Image(fig_v),
        "a_reals": wandb.Image(fig_a),
    }, step=i + 1)

    plt.close(fig_p); plt.close(fig_v); plt.close(fig_a)

    # Camera params plot
    if MATPLOTLIB_AVAILABLE and rollout['cam_params_history']:
        ch = torch.stack(rollout['cam_params_history'])[:, vid_idx].detach().cpu()
        fig_cam, axes = plt.subplots(1, 3, figsize=(12, 3))
        labels = ['Power', 'Exposure', 'Gain']
        for ci, (ax_c, lb) in enumerate(zip(axes.flatten(), labels)):
            ax_c.plot(ch[:, ci].numpy(), label=lb); ax_c.set_title(lb)
            ax_c.set_ylim(-0.05, 1.05)
        fig_cam.tight_layout()
        wandb.log({'cam_params': wandb.Image(fig_cam)}, step=i + 1)
        plt.close(fig_cam)

    # Roll + speed/exposure plot
    if MATPLOTLIB_AVAILABLE and rollout['R_up_history']:
        up_h = torch.stack([x.detach() if x.requires_grad else x for x in rollout['R_up_history']])[:, vid_idx].cpu()
        roll_rad = torch.acos(up_h[:, 2].clamp(-1, 1))
        roll_deg_plot = roll_rad * 180 / math.pi
        fig_roll, ax_roll = plt.subplots(figsize=(6, 3))
        ax_roll.plot(roll_deg_plot.numpy(), label='Roll angle (deg)')
        ax_roll.set_ylabel('Roll (deg)'); ax_roll.set_xlabel('Timestep')
        if rollout['exposure_history']:
            ax2 = ax_roll.twinx()
            sp_p = torch.stack([x.detach() if x.requires_grad else x for x in rollout['speed_for_depth_history']])[:, vid_idx].cpu()
            ex_p = torch.stack([x.detach() if x.requires_grad else x for x in rollout['exposure_history']])[:, vid_idx].cpu()
            ax2.plot(sp_p.numpy(), 'r--', alpha=0.7, label='Speed')
            ax2.plot(ex_p.numpy(), 'g-.', alpha=0.7, label='Exposure')
            ax2.set_ylabel('Speed / Exposure'); ax2.legend(loc='upper right')
        ax_roll.legend(loc='upper left'); fig_roll.tight_layout()
        wandb.log({'roll_speed_exposure': wandb.Image(fig_roll)}, step=i + 1)
        plt.close(fig_roll)


# =====================================================================
# Main training entry point
# =====================================================================

def train(args, model, env_train, env_full, optim, sched, scaler, vis, checkpoint_dir, device):
    """Main training loop — drop-in replacement for the old main_cuda.py loop."""
    use_amp = bool(args.amp and device.type == 'cuda')
    smoother = MetricSmoother(args)

    pbar = tqdm(range(args.num_iters), ncols=80)
    for i in pbar:
        use_hybrid_full = (args.hybrid_full_bptt_every > 0) and ((i + 1) % args.hybrid_full_bptt_every == 0)
        use_full_bptt_iter = (not args.tbptt_enable) or use_hybrid_full
        env = env_full if use_hybrid_full else env_train
        B = env.batch_size
        dc_iter = (distill_coef_at_iter(i, args) if args.enable_teacher_student_training
                   else float(args.distill_coef))

        iter_tic = time.time()
        env.reset()
        model.reset()
        should_vis = args.vis_enable and (i % max(args.vis_every_iters, 1) == 0)
        if should_vis:
            vis.begin_iter(i)
            j = int(min(max(args.vis_env_idx, 0), B - 1))
            vis.log_environment(
                phase='student',
                balls=env.balls[j].detach().cpu().numpy(),
                voxels=env.voxels[j].detach().cpu().numpy(),
                cyl=env.cyl[j].detach().cpu().numpy(),
                cyl_h=env.cyl_h[j].detach().cpu().numpy(),
                start=env.p[j].detach().cpu().numpy(),
                target=env.p_target[j].detach().cpu().numpy(),
                scene_name=getattr(env, 'current_scene_tag', getattr(env, 'current_scene_name', None)),
                scene_effects=env.get_scene_effects_for_env(j) if hasattr(env, 'get_scene_effects_for_env') else getattr(env, 'current_scene_effects', None))

        # Teacher phase
        u_star, y_star, u_star_cam = None, None, None
        if args.enable_teacher_student_training:
            u_star, y_star, u_star_cam = teacher_phase(
                env, model, args, B, device, use_amp, vis, should_vis)

        # Student rollout
        tbptt_this_iter = args.tbptt_enable and not use_full_bptt_iter
        rollout = student_rollout(
            env, model, args, B, device, use_amp,
            scaler, optim, sched,
            u_star, y_star, u_star_cam, dc_iter,
            tbptt_this_iter, vis, should_vis)

        if tbptt_this_iter:
            # TBPTT: backward already done inside student_rollout
            denom = max(rollout['tbptt_chunk_n'], 1)
            loss = torch.tensor(rollout['tbptt_stats']['loss'] / denom, device=device)
            loss_distill = torch.tensor(rollout['tbptt_stats']['loss_distill'] / denom, device=device)

            # Detached metrics
            vec_det = torch.stack([x.detach() if isinstance(x, torch.Tensor) and x.requires_grad else x
                                   for x in rollout['vec_to_pt_history']])
            distance_det = torch.norm(vec_det, 2, -1) - env.margin
            collision_free = torch.all(distance_det.flatten(0, 1) > 0, 0)
            collision_free_rate = collision_free.sum() / B
            v_det = torch.stack([x.detach() if isinstance(x, torch.Tensor) and x.requires_grad else x
                                 for x in rollout['v_history']])
            speed_history = v_det.norm(2, -1)
            avg_speed = speed_history.mean(0)

            pbar.set_description_str(f'loss: {float(loss):.3f} (tbptt)')
        else:
            # Full BPTT: compute loss and backward
            loss, loss_dict = full_bptt_losses(
                rollout, env, args, device,
                u_star, y_star, u_star_cam, dc_iter)
            loss_distill = loss_dict['loss_distill']
            distance = loss_dict['distance']
            speed_history = loss_dict['speed_history']

            pbar.set_description_str(f'loss: {loss:.3f}')
            optim.zero_grad()
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] loss is nan/inf at iter {i}, skipping optimizer step")
                optim.zero_grad(set_to_none=True)
            elif use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                _optimizer_step_and_maybe_advance_scheduler(optim, sched, scaler, use_amp=True)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                _optimizer_step_and_maybe_advance_scheduler(optim, sched, scaler, use_amp=False)
            torch.cuda.synchronize()  # flush GPU queue to prevent display starvation

            collision_free = torch.all(distance.flatten(0, 1) > 0, 0)
            collision_free_rate = collision_free.sum() / B
            avg_speed = speed_history.mean(0)
        # wandb.watch(model, log="all")
        # wandb.watch(model, log=["gradients", "parameters"])
        # Timing
        iter_toc = time.time()
        iter_time = iter_toc - iter_tic
        iter_per_sec = 1.0 / max(iter_time, 1e-6)
        sim_fps = iter_per_sec * args.timesteps * B

        # Logging
        with torch.no_grad():
            smoother.add({'iter_per_sec': iter_per_sec, 'sim_fps': sim_fps,
                          'iter_time_ms': iter_time * 1000})
            if args.vis_enable:
                vis.log_train_scalars({
                    'loss': float(loss.detach().cpu()),
                    'loss_distill': float(loss_distill.detach().cpu()),
                    'iter_per_sec': float(iter_per_sec),
                    'sim_fps': float(sim_fps),
                }, iter_idx=i)

            if tbptt_this_iter:
                stats = rollout['tbptt_stats']
                dn = max(rollout['tbptt_chunk_n'], 1)
                loss_scalars = {k: stats[k] / dn for k in stats}
            else:
                loss_scalars = {k: float(v.detach()) for k, v in loss_dict.items()
                                if isinstance(v, torch.Tensor) and v.dim() == 0}

            smoother.add(filter_active_loss_scalars(loss_scalars, args))
            # Weighted loss composition (contribution + share)
            smoother.add(_build_loss_share_metrics(loss_scalars, args, dc_iter))

            smoother.add({
                'loss': float(loss.detach()),
                'collision_free_rate': float(collision_free_rate),
                'collision_rate': float(1.0 - collision_free_rate),
                'max_speed': float(speed_history.max(0).values.mean()),
                'avg_speed': float(avg_speed.mean()),
                'ar': float((collision_free * avg_speed).mean()),
            })
            smoother.add(_compute_final_goal_distance_metric(rollout, env))

            # Emerging metrics (shared for both branches)
            if tbptt_this_iter:
                p_det = torch.stack([x for x in rollout['p_history']])
                _compute_emerging_metrics(
                    rollout,
                    {'p_history': p_det, 'vec_to_pt_history': vec_det,
                     'distance': distance_det, 'speed_history': speed_history},
                    env, args, smoother)
            else:
                _compute_emerging_metrics(rollout, loss_dict, env, args, smoother)

            # Save visualizations
            if is_save_iter(i) and not tbptt_this_iter:
                _log_save_iter(rollout, loss_dict, env, args, i)

            periodic_tail_ops(i, checkpoint_dir, model, smoother)

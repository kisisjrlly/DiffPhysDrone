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
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm

from env_cuda import Env
from lqr import build_velocity_tracking_linear_system, solve_batched_dlqr
from losses import velocity_tracking_loss, barrier
from rollout_ops import (
    render_sensors, build_local_frame, build_state_vector,
    compute_target_velocity, decode_action_direct, decode_action_lqr,
    update_camera_params, camera_exposure_to_time,
)
from train_utils import (
    MetricSmoother, periodic_tail_ops, is_save_iter,
    detach_env_graph, distill_coef_at_iter, teacher_dt_like_student,
    make_yaw_drift_R,
)


# =====================================================================
# Teacher Phase I — inner-loop trajectory optimization
# =====================================================================

def _teacher_initial_guess(env, model, args, sensor_flags, B, device,
                           use_amp, yaw_drift_R):
    """Roll out current student policy (no grad) to get initial guess for teacher."""
    sf = sensor_flags
    use_camera_control = sf['use_camera_control']
    optimize_intent = bool(args.policy_output_intent and args.use_dmpc)

    init_acts, init_intents, init_cam_deltas = [], [], []
    h_tmp = None
    act_buf_tmp = [env.act] * 2
    tv_raw = env.p_target - env.p

    cam_fov = torch.full((B,), env._fov_x_half_tan, device=device)
    cam_exp = torch.full((B,), 0.5, device=device)
    cam_iso = torch.full((B,), 0.5, device=device)

    for t in range(args.timesteps):
        dt = teacher_dt_like_student(float(cam_exp.mean()), use_camera_control,
                                     args.base_control_freq, env.cam_sem)
        main_obs, depth_obs = render_sensors(
            env, dt, cam_fov, cam_exp, cam_iso,
            sf['use_depth_only'], sf['use_camera_luma'],
            sf['use_diff_depth'], sf['use_depth_aux'],
            use_camera_control, differentiable=False)
        # torch.cuda.synchronize() 
        if args.yaw_drift and yaw_drift_R is not None:
            tv_raw = torch.squeeze(tv_raw[:, None] @ yaw_drift_R, 1)
        else:
            tv_raw = env.p_target - env.p

        env.run(act_buf_tmp[t], dt, tv_raw)

        R = build_local_frame(env)
        target_v = compute_target_velocity(tv_raw, env)
        state, local_v = build_state_vector(
            env, target_v, R, cam_fov, cam_exp, cam_iso,
            args.no_odom, args.include_camera_state_in_obs, use_camera_control)

        if args.policy_output_intent:
            with autocast(enabled=use_amp):
                a_out, c_out, h_tmp, y_out = model(
                    state, h_tmp, return_intent=True,
                    main_obs=main_obs, depth_obs=depth_obs,
                    add_noise=False)
            a_out, y_out = a_out.float(), y_out.float()
            if c_out is not None:
                c_out = c_out.float()
            if optimize_intent:
                init_intents.append(y_out.clone())
        else:
            with autocast(enabled=use_amp):
                a_out, c_out, h_tmp = model(
                    state, h_tmp, main_obs=main_obs,
                    depth_obs=depth_obs, add_noise=False)
            a_out = a_out.float()
            if c_out is not None:
                c_out = c_out.float()

        init_acts.append(a_out.clone())
        if c_out is not None:
            init_cam_deltas.append(c_out.clone())

        a_final, _ = decode_action_direct(a_out, R, env, B, args.max_acc_cmd)
        act_buf_tmp.append(a_final)

        cam_fov, cam_exp, cam_iso, _ = update_camera_params(
            c_out, cam_fov, cam_exp, cam_iso, env)

    return init_acts, init_intents, init_cam_deltas


def _teacher_inner_loop(env, env_snapshot, args, sensor_flags,
                        init_acts, init_intents, init_cam_deltas,
                        B, device, yaw_drift_R, vis, should_vis_iter, i):
    """Run teacher inner optimization (TBPTT) and return u_star / y_star / u_star_cam."""
    sf = sensor_flags
    use_camera_control = sf['use_camera_control']
    use_depth_aux = sf['use_depth_aux']
    use_diff_depth = sf['use_diff_depth']
    optimize_intent = bool(args.policy_output_intent and args.use_dmpc)

    # Build optimizable parameters
    u_guess, y_guess = None, None
    if optimize_intent:
        y_guess = [y.clone().requires_grad_(True) for y in init_intents]
    else:
        u_guess = [a.clone().requires_grad_(True) for a in init_acts]

    u_cam_guess = None
    if use_camera_control and len(init_cam_deltas) > 0:
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
        c_cam_exp, c_cam_iso, c_cam_fov, c_speed = [], [], [], []

        cam_fov_k = torch.full((B,), env._fov_x_half_tan, device=device)
        cam_exp_k = torch.full((B,), 0.5, device=device)
        cam_iso_k = torch.full((B,), 0.5, device=device)

        for t in range(args.timesteps):
            dt_k = teacher_dt_like_student(float(cam_exp_k.mean().detach()),
                                           use_camera_control, args.base_control_freq, env.cam_sem)
            c_p.append(env.p)
            vec_now_k = env.find_vec_to_nearest_pt()
            c_vtp.append(vec_now_k)

            if args.yaw_drift and yaw_drift_R is not None:
                tv_raw_k = torch.squeeze(tv_raw_k[:, None] @ yaw_drift_R, 1)
            else:
                tv_raw_k = env.p_target - env.p.detach()

            env.run(act_buf_k[t], dt_k, tv_raw_k)

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

            # Camera update
            if use_camera_control and u_cam_guess is not None:
                cam_fov_k, cam_exp_k, cam_iso_k, _ = update_camera_params(
                    u_cam_guess[t], cam_fov_k, cam_exp_k, cam_iso_k, env)
                c_cam_fov.append(cam_fov_k)
                c_cam_exp.append(cam_exp_k)
                c_cam_iso.append(cam_iso_k)

            c_speed.append(env.v.norm(2, -1))
            c_v.append(env.v)
            c_tv.append(tv_k)
            c_act.append(a_final_k)

            # Visualization (last inner step only)
            if (should_vis_iter and args.vis_teacher
                    and k == args.teacher_inner_steps - 1
                    and t % max(args.vis_every_steps, 1) == 0):
                j = int(min(max(args.vis_env_idx, 0), B - 1))
                cam_vals = None
                if use_camera_control:
                    cam_vals = (float(cam_fov_k[j].detach().cpu()),
                                float(cam_exp_k[j].detach().cpu()),
                                float(cam_iso_k[j].detach().cpu()))
                vis.log_step(
                    phase='teacher', step_idx=t,
                    pos=env.p[j].detach().cpu().numpy(),
                    target=env.p_target[j].detach().cpu().numpy(),
                    depth=None, cam=cam_vals,
                    drone_R=env.R[j].detach().cpu().numpy(),
                    cam_R=env.R_cam[j].detach().cpu().numpy(),
                    main_fov_half_tan=(float(cam_fov_k[j].detach().cpu())
                                      if use_camera_control else float(env._fov_x_half_tan)),
                    main_hw=(int(env.height), int(env.width)),
                    depth_hw=(int(env.depth_height), int(env.depth_width)))

            # ---- TBPTT chunk boundary ----
            chunk_end_k = ((t + 1) % teacher_chunk_steps == 0) or (t == args.timesteps - 1)
            if chunk_end_k and len(c_v) > 0:
                v_ck = torch.stack(c_v)
                tv_ck = torch.stack(c_tv)
                vec_ck = torch.stack(c_vtp)
                act_ck = torch.stack(c_act)
                p_ck = torch.stack(c_p)

                v_fl = torch.cat(v_roll_k + [v_ck], 0) if v_roll_k else v_ck
                tv_fl = torch.cat(tv_roll_k + [tv_ck], 0) if tv_roll_k else tv_ck
                l_v = velocity_tracking_loss(v_fl, tv_fl, win=30)

                act_sm = torch.cat([prev_act_tail_k[None], act_ck], 0)
                jk = act_sm.diff(1, 0).mul(15)
                l_acc = act_ck.pow(2).sum(-1).mean()
                l_jerk = jk.pow(2).sum(-1).mean()

                dist_k = torch.norm(vec_ck + 1e-6, 2, -1) - env.margin
                with torch.no_grad():
                    v_to = (-torch.diff(dist_k, 1, 1) * 135).clamp_min(1)
                l_avoid = barrier(dist_k[:, 1:], v_to)
                l_coll = F.softplus(dist_k[:, 1:].clamp(min=-3.0).mul(-32)).mul(v_to).mean()
                l_ga = p_ck[..., 2].relu().pow(2).mean()

                chunk_loss = (args.coef_v * l_v
                              + args.coef_obj_avoidance * l_avoid
                              + args.coef_d_acc * l_acc
                              + args.coef_d_jerk * l_jerk
                              + args.coef_collide * l_coll
                              + args.coef_ground_affinity * l_ga)

                if use_camera_control and c_cam_exp:
                    sp = torch.stack(c_speed); ex = torch.stack(c_cam_exp)
                    iso_t = torch.stack(c_cam_iso); fov_t = torch.stack(c_cam_fov)
                    if use_diff_depth:
                        chunk_loss = chunk_loss + args.coef_diff_depth_power * fov_t.pow(2).mean()
                        chunk_loss = chunk_loss + args.coef_diff_depth_blur * (sp * ex).mean()
                    elif args.enable_camera_quality_loss:
                        ep = env.cam_sem.exposure_to_time(ex)
                        ef = 1.0 / fov_t.clamp(min=0.1)
                        chunk_loss = chunk_loss + args.coef_blur * (sp.pow(2) * ep.pow(2) * ef.pow(2)).mean()
                        iso_gain = env.cam_sem.iso_to_gain(iso_t)
                        ns = env.cam_sem.shot_noise_base * iso_gain / ep.clamp_min(1e-3)
                        chunk_loss = chunk_loss + args.coef_noise * ns.pow(2).mean()

                (chunk_loss / teacher_chunk_count).backward()

                keep = 30
                v_roll_k = [v_fl[-keep:].detach()] if v_fl.shape[0] > 0 else []
                tv_roll_k = [tv_fl[-keep:].detach()] if tv_fl.shape[0] > 0 else []
                prev_act_tail_k = act_ck[-1].detach()
                cam_fov_k = cam_fov_k.detach()
                cam_exp_k = cam_exp_k.detach()
                cam_iso_k = cam_iso_k.detach()
                act_buf_k = [a.detach() for a in act_buf_k]
                detach_env_graph(env)
                c_p.clear(); c_v.clear(); c_tv.clear(); c_vtp.clear(); c_act.clear()
                c_cam_exp.clear(); c_cam_iso.clear(); c_cam_fov.clear(); c_speed.clear()

        inner_optim.step()

    # Extract optimised sequences
    u_star, y_star, u_star_cam = None, None, None
    if y_guess is not None:
        y_star = [y.detach() for y in y_guess]
    else:
        assert u_guess is not None
        u_star = [u.detach() for u in u_guess]
    if u_cam_guess is not None:
        u_star_cam = [c.detach() for c in u_cam_guess]
    return u_star, y_star, u_star_cam


def teacher_phase(env, model, args, sensor_flags, B, device, use_amp,
                  vis, should_vis_iter, i):
    """Full teacher phase: initial guess → inner optimisation → return targets."""
    yaw_drift_R = make_yaw_drift_R(B, device) if args.yaw_drift else None
    env_snapshot = env.save_state()

    with torch.no_grad():
        env.restore_state(env_snapshot)
        init_acts, init_intents, init_cam_deltas = _teacher_initial_guess(
            env, model, args, sensor_flags, B, device, use_amp, yaw_drift_R)

    u_star, y_star, u_star_cam = _teacher_inner_loop(
        env, env_snapshot, args, sensor_flags,
        init_acts, init_intents, init_cam_deltas,
        B, device, yaw_drift_R, vis, should_vis_iter, i)

    env.restore_state(env_snapshot)
    return u_star, y_star, u_star_cam


# =====================================================================
# Student Phase II — rollout + TBPTT / full-BPTT loss
# =====================================================================

def student_rollout(env, model, args, sensor_flags, B, device, use_amp,
                    scaler, optim, sched,
                    u_star, y_star, u_star_cam,
                    distill_coef_iter,
                    tbptt_this_iter, use_full_bptt_iter,
                    vis, should_vis_iter, i):
    """Run the student rollout and compute / backprop losses.

    Returns a dict with all quantities needed for logging:
      loss, per-component losses, histories (detached), metrics, etc.
    """
    sf = sensor_flags
    use_depth_only = sf['use_depth_only']
    use_camera_luma = sf['use_camera_luma']
    use_depth_aux = sf['use_depth_aux']
    use_diff_depth = sf['use_diff_depth']
    use_camera_control = sf['use_camera_control']
    effective_include_camera_state = sf['effective_include_camera_state']

    vid_idx = min(4, B - 1)

    # Global history lists (used for full-BPTT and logging)
    p_history, v_history, target_v_history = [], [], []
    vec_to_pt_history, v_preds = [], []
    raw_act_history, raw_intent_history, raw_cam_history = [], [], []
    cam_params_history = []
    cam_fov_history, cam_exposure_history, cam_iso_history = [], [], []
    speed_for_cam_history, R_up_history = [], []
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
        c_v_hist, c_tv_hist, c_vpred_hist, c_vec_hist = [], [], [], []
        c_act_hist, c_p_hist = [], []
        c_cam_hist, c_cam_exp, c_cam_iso, c_cam_fov, c_speed = [], [], [], [], []
        c_distill = []
        prev_act_tail = env.act.detach()

    act_buffer = [env.act] * 2
    target_v_raw = env.p_target - env.p

    R_drift = make_yaw_drift_R(B, device) if args.yaw_drift else None

    cam_fov = torch.full((B,), env._fov_x_half_tan, device=device)
    cam_exposure = torch.full((B,), 0.5, device=device)
    cam_iso = torch.full((B,), 0.5, device=device)

    A_lqr, B_lqr = build_velocity_tracking_linear_system(B, 1 / 15, device)

    # ── Main rollout loop ──
    for t in range(args.timesteps):
        base_dt = normalvariate(1 / args.base_control_freq, 0.1 / args.base_control_freq)
        exposure_delay = float(env.cam_sem.exposure_to_time(cam_exposure.mean().detach())) * 0.01 if use_camera_control else 0.015
        ctl_dt = base_dt + exposure_delay
        student_add_noise = (args.student_noise_mode == 'on') if args.enable_teacher_student_training else True

        # Render sensors
        # 可微渲染策略：
        # - camera_luma*：主亮度分支可微（FOV/Exposure/ISO 可通过渲染链路反传）
        # - camera_luma_plus_depth：depth_aux 由同次主渲染复用 depth_raw，并 detached
        # - diff_depth：保持原可微路径
        diff_cam = ((use_camera_luma and use_camera_control)
                or (use_diff_depth and use_camera_control))
        main_obs, depth_obs = render_sensors(
            env, ctl_dt, cam_fov, cam_exposure, cam_iso,
            use_depth_only, use_camera_luma, use_diff_depth,
            use_depth_aux, use_camera_control,
            differentiable=diff_cam)
        # torch.cuda.synchronize() 
        depth_vis = main_obs if main_obs is not None else depth_obs
        assert depth_vis is not None

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

        env.run(act_buffer[t], ctl_dt, target_v_raw)

        R = build_local_frame(env)
        target_v = compute_target_velocity(target_v_raw, env)
        state, local_v = build_state_vector(
            env, target_v, R, cam_fov, cam_exposure, cam_iso,
            args.no_odom, effective_include_camera_state, use_camera_control)

        # Detach non-differentiable sensor inputs
        main_obs_in = main_obs if diff_cam and use_camera_luma else (main_obs.detach() if main_obs is not None else None)
        depth_obs_in = depth_obs.detach() if depth_obs is not None else None

        # Policy forward
        if args.policy_output_intent:
            with autocast(enabled=use_amp):
                act, cam_params, h, intent = model(
                    state, h, return_intent=True,
                    main_obs=main_obs_in, depth_obs=depth_obs_in,
                    add_noise=student_add_noise)
            act, intent = act.float(), intent.float()
            if args.enable_teacher_student_training and args.use_dmpc:
                raw_intent_history.append(intent)
        else:
            with autocast(enabled=use_amp):
                act, cam_params, h = model(
                    state, h, main_obs=main_obs_in,
                    depth_obs=depth_obs_in,
                    add_noise=student_add_noise)
            act = act.float()
            intent = None
        if cam_params is not None:
            cam_params = cam_params.float()
        if args.enable_teacher_student_training:
            raw_act_history.append(act)
        if cam_params is not None and args.enable_teacher_student_training:
            raw_cam_history.append(cam_params)

        # Camera update
        cam_fov, cam_exposure, cam_iso, cam_hist_entry = update_camera_params(
            cam_params, cam_fov, cam_exposure, cam_iso, env)
        if cam_hist_entry is not None:
            cam_params_history.append(cam_hist_entry)

        # Track histories
        if use_camera_control:
            if tbptt_this_iter:
                cam_fov_history.append(cam_fov.detach())
                cam_exposure_history.append(cam_exposure.detach())
                cam_iso_history.append(cam_iso.detach())
            else:
                cam_fov_history.append(cam_fov)
                cam_exposure_history.append(cam_exposure)
                cam_iso_history.append(cam_iso)
        if tbptt_this_iter:
            speed_for_cam_history.append(env.v.norm(2, -1).detach())
            R_up_history.append(env.R[:, :, 2].detach().clone())
        else:
            speed_for_cam_history.append(env.v.norm(2, -1))
            R_up_history.append(env.R[:, :, 2].clone())

        # Action decode (LQR or direct)
        if args.use_dmpc and args.policy_output_intent and intent is not None:
            act_final, v_pred = decode_action_lqr(
                intent, R, env, local_v, B, A_lqr, B_lqr,
                args.lqr_horizon, args.lqr_reg, args.max_acc_cmd,
                args.inject_depth_into_lqr, args.depth_safe_dist, args.depth_repel_gain,
                vec_now, solve_batched_dlqr)
        else:
            act_final, v_pred = decode_action_direct(act, R, env, B, args.max_acc_cmd)
        v_preds.append(v_pred)
        act = act_final
        act_buffer.append(act)

        if tbptt_this_iter:
            v_history.append(env.v.detach())
            target_v_history.append(target_v.detach())
        else:
            v_history.append(env.v)
            target_v_history.append(target_v)

        # ── TBPTT chunk accumulation & backward ──
        if tbptt_this_iter:
            c_v_hist.append(env.v); c_tv_hist.append(target_v)
            c_vpred_hist.append(v_pred); c_vec_hist.append(vec_now)
            c_act_hist.append(act); c_p_hist.append(env.p)
            c_speed.append(env.v.norm(2, -1))
            if use_camera_control and cam_hist_entry is not None:
                c_cam_hist.append(cam_hist_entry)
                c_cam_exp.append(cam_exposure); c_cam_iso.append(cam_iso)
                c_cam_fov.append(cam_fov)

            chunk_end = ((t + 1) % chunk_steps == 0) or (t == args.timesteps - 1)
            if chunk_end and len(c_v_hist) > 0:
                v_ck = torch.stack(c_v_hist); tv_ck = torch.stack(c_tv_hist)
                vpred_ck = torch.stack(c_vpred_hist)
                vec_ck = torch.stack(c_vec_hist)
                act_ck = torch.stack(c_act_hist); p_ck = torch.stack(c_p_hist)

                v_fl = torch.cat(v_roll + [v_ck], 0) if v_roll else v_ck
                tv_fl = torch.cat(tv_roll + [tv_ck], 0) if tv_roll else tv_ck
                loss_v_c = velocity_tracking_loss(v_fl, tv_fl, win=30)
                loss_v_pred_c = F.mse_loss(vpred_ck, v_ck.detach())

                act_sm = torch.cat([prev_act_tail[None], act_ck], 0)
                jk = act_sm.diff(1, 0).mul(15)
                loss_d_acc_c = act_ck.pow(2).sum(-1).mean()
                loss_d_jerk_c = jk.pow(2).sum(-1).mean()

                dist_c = torch.norm(vec_ck + 1e-6, 2, -1) - env.margin
                with torch.no_grad():
                    v_to_c = (-torch.diff(dist_c, 1, 1) * 135).clamp_min(1)
                loss_avoid_c = barrier(dist_c[:, 1:], v_to_c)
                loss_collide_c = F.softplus(dist_c[:, 1:].clamp(min=-3.0).mul(-32)).mul(v_to_c).mean()
                loss_ground_c = p_ck[..., 2].relu().pow(2).mean()

                loss_cam_smooth_c = torch.zeros((), device=device)
                loss_fov_reg_c = torch.zeros((), device=device)
                loss_cam_range_c = torch.zeros((), device=device)
                if use_camera_control and len(c_cam_hist) > 1:
                    ch = torch.stack(c_cam_hist)
                    loss_cam_smooth_c = ch.diff(1, 0).pow(2).mean()
                    loss_fov_reg_c = (ch[:, :, 0] - 0.5).pow(2).mean()
                    loss_cam_range_c = (ch - 0.5).pow(2).mean()

                loss_blur_c = torch.zeros((), device=device)
                loss_noise_c = torch.zeros((), device=device)
                loss_adp_c = torch.zeros((), device=device)
                loss_adb_c = torch.zeros((), device=device)
                if use_camera_control and c_cam_exp:
                    sp = torch.stack(c_speed); ex = torch.stack(c_cam_exp)
                    iso_h = torch.stack(c_cam_iso); fov_h = torch.stack(c_cam_fov)
                    if use_diff_depth:
                        loss_adp_c = fov_h.pow(2).mean()
                        loss_adb_c = (sp * ex).mean()
                    elif args.enable_camera_quality_loss:
                        ep = env.cam_sem.exposure_to_time(ex); ef = 1.0 / fov_h.clamp(min=0.1)
                        loss_blur_c = (sp.pow(2) * ep.pow(2) * ef.pow(2)).mean()
                        iso_gain = env.cam_sem.iso_to_gain(iso_h)
                        ns = env.cam_sem.shot_noise_base * iso_gain / ep.clamp_min(1e-3)
                        loss_noise_c = ns.pow(2).mean()

                loss_distill_c = torch.zeros((), device=device)
                if args.enable_teacher_student_training and (u_star is not None or y_star is not None):
                    cl = int(act_ck.shape[0])
                    si, ei = int(t + 1 - cl), int(t + 1)
                    if y_star is not None and len(raw_intent_history) >= ei:
                        loss_distill_c = loss_distill_c + F.mse_loss(
                            torch.stack(raw_intent_history[si:ei]),
                            torch.stack(y_star[si:ei]))
                    elif u_star is not None and len(raw_act_history) >= ei:
                        loss_distill_c = loss_distill_c + F.mse_loss(
                            torch.stack(raw_act_history[si:ei]),
                            torch.stack(u_star[si:ei]))
                    if u_star_cam is not None and len(raw_cam_history) >= ei:
                        loss_distill_c = loss_distill_c + F.mse_loss(
                            torch.stack(raw_cam_history[si:ei]),
                            torch.stack(u_star_cam[si:ei]))

                chunk_loss = (args.coef_v * loss_v_c
                    + args.coef_obj_avoidance * loss_avoid_c
                    + args.coef_d_acc * loss_d_acc_c
                    + args.coef_d_jerk * loss_d_jerk_c
                    + args.coef_v_pred * loss_v_pred_c
                    + args.coef_collide * loss_collide_c
                    + args.coef_ground_affinity * loss_ground_c
                    + args.coef_cam_smooth * loss_cam_smooth_c
                    + args.coef_fov_reg * loss_fov_reg_c
                    + args.coef_cam_range * loss_cam_range_c
                    + args.coef_blur * loss_blur_c
                    + args.coef_noise * loss_noise_c
                    + args.coef_diff_depth_power * loss_adp_c
                    + args.coef_diff_depth_blur * loss_adb_c)

                if args.enable_teacher_student_training:
                    chunk_loss = distill_coef_iter * loss_distill_c + args.student_physics_coef * chunk_loss

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
                        scaler.step(optim); scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()  # flush GPU queue after optimizer step

                # stats
                tbptt_stats['loss'] += float(chunk_loss.detach())
                tbptt_stats['loss_v'] += float(loss_v_c.detach())
                tbptt_stats['loss_v_pred'] += float(loss_v_pred_c.detach())
                tbptt_stats['loss_obj_avoidance'] += float(loss_avoid_c.detach())
                tbptt_stats['loss_d_acc'] += float(loss_d_acc_c.detach())
                tbptt_stats['loss_d_jerk'] += float(loss_d_jerk_c.detach())
                tbptt_stats['loss_collide'] += float(loss_collide_c.detach())
                tbptt_stats['loss_ground_affinity'] += float(loss_ground_c.detach())
                tbptt_stats['loss_cam_smooth'] += float(loss_cam_smooth_c.detach())
                tbptt_stats['loss_fov_reg'] += float(loss_fov_reg_c.detach())
                tbptt_stats['loss_cam_range'] += float(loss_cam_range_c.detach())
                tbptt_stats['loss_tilt'] += 0.0
                tbptt_stats['loss_blur'] += float(loss_blur_c.detach())
                tbptt_stats['loss_noise'] += float(loss_noise_c.detach())
                tbptt_stats['loss_diff_depth_power'] += float(loss_adp_c.detach())
                tbptt_stats['loss_diff_depth_blur'] += float(loss_adb_c.detach())
                tbptt_stats['loss_distill'] += float(loss_distill_c.detach())
                tbptt_chunk_n += 1

                keep = 30
                v_roll = [v_fl[-keep:].detach()] if v_fl.shape[0] > 0 else []
                tv_roll = [tv_fl[-keep:].detach()] if tv_fl.shape[0] > 0 else []
                prev_act_tail = act_ck[-1].detach()
                if h is not None:
                    h = h.detach()
                cam_fov = cam_fov.detach(); cam_exposure = cam_exposure.detach()
                cam_iso = cam_iso.detach()
                act_buffer = [a.detach() for a in act_buffer]
                detach_env_graph(env)
                c_v_hist.clear(); c_tv_hist.clear(); c_vpred_hist.clear()
                c_vec_hist.clear(); c_act_hist.clear(); c_p_hist.clear()
                c_cam_hist.clear(); c_cam_exp.clear(); c_cam_iso.clear()
                c_cam_fov.clear(); c_speed.clear(); c_distill.clear()

        # Visualization
        if should_vis_iter and args.vis_student and (t % max(args.vis_every_steps, 1) == 0):
            j = int(min(max(args.vis_env_idx, 0), B - 1))
            cam_vals = None
            if use_camera_control:
                cam_vals = (float(cam_fov[j].detach().cpu()),
                            float(cam_exposure[j].detach().cpu()),
                            float(cam_iso[j].detach().cpu()))
            main_img_np = main_obs[j].detach().cpu().numpy() if main_obs is not None else None
            main_img_mode = 'luma' if use_camera_luma else 'depth'
            depth_img_np = depth_obs[j].detach().cpu().numpy() if depth_obs is not None else None
            vis.log_step(
                phase='student', step_idx=t,
                pos=env.p[j].detach().cpu().numpy(),
                target=env.p_target[j].detach().cpu().numpy(),
                depth=depth_vis[j].detach().cpu().numpy(),
                cam=cam_vals, main_img=main_img_np,
                main_img_mode=main_img_mode, depth_img=depth_img_np,
                drone_R=env.R[j].detach().cpu().numpy(),
                cam_R=env.R_cam[j].detach().cpu().numpy(),
                main_fov_half_tan=(float(cam_fov[j].detach().cpu())
                                   if use_camera_control else float(env._fov_x_half_tan)),
                main_hw=(int(env.height), int(env.width)),
                depth_hw=(int(env.depth_height), int(env.depth_width)))

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
        'v_preds': v_preds,
        'vid_idx': vid_idx,
        'act_buffer': act_buffer,
        'raw_act_history': raw_act_history,
        'raw_intent_history': raw_intent_history,
        'raw_cam_history': raw_cam_history,
        'cam_params_history': cam_params_history,
        'cam_fov_history': cam_fov_history,
        'cam_exposure_history': cam_exposure_history,
        'cam_iso_history': cam_iso_history,
        'speed_for_cam_history': speed_for_cam_history,
        'R_up_history': R_up_history,
    }


# =====================================================================
# Full-BPTT loss + backward (non-TBPTT iterations)
# =====================================================================

def full_bptt_losses(rollout, env, args, sensor_flags, device,
                     u_star, y_star, u_star_cam, distill_coef_iter):
    """Compute all losses for full-BPTT iterations and return (loss, loss_dict)."""
    sf = sensor_flags
    use_camera_control = sf['use_camera_control']
    use_diff_depth = sf['use_diff_depth']

    p_history = torch.stack(rollout['p_history'])
    v_history = torch.stack(rollout['v_history'])
    target_v_history = torch.stack(rollout['target_v_history'])
    vec_to_pt_history = torch.stack(rollout['vec_to_pt_history'])
    v_preds = torch.stack(rollout['v_preds'])
    act_buffer = torch.stack(rollout['act_buffer'])

    loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()
    loss_v = velocity_tracking_loss(v_history, target_v_history, win=30)
    loss_v_pred = F.mse_loss(v_preds, v_history.detach())

    jerk = act_buffer.diff(1, 0).mul(15)
    loss_d_acc = act_buffer.pow(2).sum(-1).mean()
    loss_d_jerk = jerk.pow(2).sum(-1).mean()

    distance = torch.norm(vec_to_pt_history + 1e-6, 2, -1) - env.margin
    with torch.no_grad():
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
    loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)
    loss_collide = F.softplus(distance[:, 1:].clamp(min=-3.0).mul(-32)).mul(v_to_pt).mean()

    speed_history = v_history.norm(2, -1)

    # Camera losses
    loss_cam_smooth = torch.tensor(0.0, device=device)
    loss_fov_reg = torch.tensor(0.0, device=device)
    loss_cam_range = torch.tensor(0.0, device=device)
    cam_params_history = rollout['cam_params_history']
    if use_camera_control and len(cam_params_history) > 1:
        cam_hist = torch.stack(cam_params_history)
        loss_cam_smooth = cam_hist.diff(1, 0).pow(2).mean()
        loss_fov_reg = (cam_hist[:, :, 0] - 0.5).pow(2).mean()
        loss_cam_range = (cam_hist - 0.5).pow(2).mean()

    # Optical losses
    loss_blur = torch.tensor(0.0, device=device)
    loss_noise = torch.tensor(0.0, device=device)
    loss_adp = torch.tensor(0.0, device=device)
    loss_adb = torch.tensor(0.0, device=device)
    if use_camera_control and rollout['cam_exposure_history']:
        sp = torch.stack(rollout['speed_for_cam_history'])
        ex = torch.stack(rollout['cam_exposure_history'])
        iso_h = torch.stack(rollout['cam_iso_history'])
        fov_h = torch.stack(rollout['cam_fov_history'])
        if use_diff_depth:
            loss_adp = fov_h.pow(2).mean()
            loss_adb = (sp * ex).mean()
        elif args.enable_camera_quality_loss:
            ep = env.cam_sem.exposure_to_time(ex)
            ef = 1.0 / fov_h.clamp(min=0.1)
            loss_blur = (sp.pow(2) * ep.pow(2) * ef.pow(2)).mean()
            iso_gain = env.cam_sem.iso_to_gain(iso_h)
            ns = env.cam_sem.shot_noise_base * iso_gain / ep.clamp_min(1e-3)
            loss_noise = ns.pow(2).mean()

    loss_tilt = torch.tensor(0.0, device=device)

    loss = (args.coef_v * loss_v
            + args.coef_obj_avoidance * loss_obj_avoidance
            + args.coef_d_acc * loss_d_acc
            + args.coef_d_jerk * loss_d_jerk
            + args.coef_v_pred * loss_v_pred
            + args.coef_collide * loss_collide
            + args.coef_ground_affinity * loss_ground_affinity
            + args.coef_cam_smooth * loss_cam_smooth
            + args.coef_fov_reg * loss_fov_reg
            + args.coef_cam_range * loss_cam_range
            + args.coef_tilt * loss_tilt
            + args.coef_blur * loss_blur
            + args.coef_noise * loss_noise
            + args.coef_diff_depth_power * loss_adp
            + args.coef_diff_depth_blur * loss_adb)

    # Distillation
    loss_distill = torch.tensor(0.0, device=device)
    if args.enable_teacher_student_training and (u_star is not None or y_star is not None):
        rih = rollout['raw_intent_history']
        rah = rollout['raw_act_history']
        rch = rollout['raw_cam_history']
        if y_star is not None and len(rih) > 0:
            loss_distill = loss_distill + F.mse_loss(torch.stack(rih), torch.stack(y_star))
        elif u_star is not None and len(rah) > 0:
            loss_distill = loss_distill + F.mse_loss(torch.stack(rah), torch.stack(u_star))
        if u_star_cam is not None and len(rch) > 0:
            loss_distill = loss_distill + F.mse_loss(torch.stack(rch), torch.stack(u_star_cam))
        loss = distill_coef_iter * loss_distill + args.student_physics_coef * loss

    loss_dict = {
        'loss_v': loss_v, 'loss_v_pred': loss_v_pred,
        'loss_obj_avoidance': loss_obj_avoidance,
        'loss_d_acc': loss_d_acc, 'loss_d_jerk': loss_d_jerk,
        'loss_collide': loss_collide,
        'loss_ground_affinity': loss_ground_affinity,
        'loss_cam_smooth': loss_cam_smooth,
        'loss_fov_reg': loss_fov_reg, 'loss_cam_range': loss_cam_range,
        'loss_tilt': loss_tilt,
        'loss_blur': loss_blur, 'loss_noise': loss_noise,
        'loss_diff_depth_power': loss_adp,
        'loss_diff_depth_blur': loss_adb,
        'loss_distill': loss_distill,
        'distance': distance, 'speed_history': speed_history,
        'p_history': p_history, 'v_history': v_history,
        'vec_to_pt_history': vec_to_pt_history,
        'act_buffer': act_buffer,
    }
    return loss, loss_dict


# =====================================================================
# Logging helpers
# =====================================================================

def _compute_emerging_metrics(rollout, loss_dict, env, args, sensor_flags, smoother):
    """Compute and log emerging-behavior metrics (roll, correlations, slit)."""
    sf = sensor_flags
    use_camera_control = sf['use_camera_control']
    use_diff_depth = sf['use_diff_depth']
    p_history = loss_dict.get('p_history')
    vec_to_pt_history = loss_dict.get('vec_to_pt_history')
    distance = loss_dict.get('distance')
    speed_history = loss_dict.get('speed_history')
    B = p_history.shape[1] if p_history is not None else 1

    # Roll angle
    if rollout['R_up_history']:
        up_hist = torch.stack([x.detach() if x.requires_grad else x for x in rollout['R_up_history']])
        roll_angle = torch.acos(up_hist[:, :, 2].clamp(-1, 1))
        roll_deg = roll_angle * 180 / math.pi
        smoother.add({'roll_max_deg': roll_deg.max().item(), 'roll_mean_deg': roll_deg.mean().item()})
        if args.wall_slit and p_history is not None:
            dx = (p_history[..., 0] - env.wall_x).abs()
            near_wall = dx < 1.0
            if near_wall.any():
                smoother.add({'roll_at_wall_deg': roll_deg[near_wall].mean().item()})

    # Speed-exposure correlation
    if use_camera_control and rollout['cam_exposure_history']:
        _sp = torch.stack([x.detach() if x.requires_grad else x for x in rollout['speed_for_cam_history']])
        _ex = torch.stack([x.detach() if x.requires_grad else x for x in rollout['cam_exposure_history']])
        sp_m = _sp.mean(0, keepdim=True); ex_m = _ex.mean(0, keepdim=True)
        cov = ((_sp - sp_m) * (_ex - ex_m)).mean(0)
        sp_s = (_sp - sp_m).pow(2).mean(0).sqrt().clamp(min=1e-6)
        ex_s = (_ex - ex_m).pow(2).mean(0).sqrt().clamp(min=1e-6)
        smoother.add({'speed_exposure_corr': (cov / (sp_s * ex_s)).mean().item()})

        _fv = torch.stack([x.detach() if x.requires_grad else x for x in rollout['cam_fov_history']])
        _dn = torch.norm(vec_to_pt_history, 2, -1).min(1).values if vec_to_pt_history is not None else _fv
        fv_m = _fv.mean(0, keepdim=True); dn_m = _dn.mean(0, keepdim=True)
        cov_fd = ((_fv - fv_m) * (_dn - dn_m)).mean(0)
        fv_s = (_fv - fv_m).pow(2).mean(0).sqrt().clamp(min=1e-6)
        dn_s = (_dn - dn_m).pow(2).mean(0).sqrt().clamp(min=1e-6)
        corr_key = 'power_obstacle_corr' if use_diff_depth else 'fov_obstacle_corr'
        smoother.add({corr_key: (cov_fd / (fv_s * dn_s)).mean().item()})

    # Wall slit
    if args.wall_slit and p_history is not None and distance is not None:
        final_x = p_history[-1, :, 0]
        success = torch.all(distance.flatten(0, 1) > 0, 0)
        crossed = (final_x > env.wall_x).float()
        smoother.add({
            'slit_crossed': crossed.mean().item(),
            'slit_pass_rate': (crossed * success.float()).mean().item(),
        })


def _build_loss_share_metrics(loss_scalars: dict, args, distill_coef_iter: float) -> dict:
    """Build weighted loss contribution and share metrics for WandB.

    Returns keys like:
      - loss_contrib/<name>
      - loss_share/<name>
            - loss_share/physics_total
    """
    coeff_map = {
        'v': ('loss_v', float(args.coef_v)),
        'obj_avoidance': ('loss_obj_avoidance', float(args.coef_obj_avoidance)),
        'd_acc': ('loss_d_acc', float(args.coef_d_acc)),
        'd_jerk': ('loss_d_jerk', float(args.coef_d_jerk)),
        'v_pred': ('loss_v_pred', float(args.coef_v_pred)),
        'collide': ('loss_collide', float(args.coef_collide)),
        'ground_affinity': ('loss_ground_affinity', float(args.coef_ground_affinity)),
        'cam_smooth': ('loss_cam_smooth', float(args.coef_cam_smooth)),
        'fov_reg': ('loss_fov_reg', float(args.coef_fov_reg)),
        'cam_range': ('loss_cam_range', float(args.coef_cam_range)),
        'tilt': ('loss_tilt', float(args.coef_tilt)),
        'blur': ('loss_blur', float(args.coef_blur)),
        'noise': ('loss_noise', float(args.coef_noise)),
        'diff_depth_power': ('loss_diff_depth_power', float(args.coef_diff_depth_power)),
        'diff_depth_blur': ('loss_diff_depth_blur', float(args.coef_diff_depth_blur)),
    }

    physics_scale = float(args.student_physics_coef) if args.enable_teacher_student_training else 1.0
    contrib = {}
    for name, (loss_key, coef) in coeff_map.items():
        raw_v = float(loss_scalars.get(loss_key, 0.0))
        contrib[name] = physics_scale * coef * raw_v

    distill_contrib = 0.0
    if args.enable_teacher_student_training:
        distill_contrib = float(distill_coef_iter) * float(loss_scalars.get('loss_distill', 0.0))
    contrib['distill'] = distill_contrib

    total = sum(contrib.values())
    eps = 1e-12
    if abs(total) < eps:
        total = eps

    physics_total = sum(v for k, v in contrib.items() if k != 'distill')
    out = {}
    for name, val in contrib.items():
        out[f'loss_contrib/{name}'] = float(val)
        out[f'loss_share/{name}'] = float(val / total)
    out['loss_share/physics_total'] = float(physics_total / total)
    return out


def _log_save_iter(rollout, loss_dict, env, args, sensor_flags, i):
    """Save plots to WandB on checkpoint iterations (video logging removed)."""
    if not MATPLOTLIB_AVAILABLE:
        print('[warn] matplotlib not installed: skip figure logging.')
        return
    sf = sensor_flags
    use_camera_control = sf['use_camera_control']
    use_diff_depth = sf['use_diff_depth']
    vid_idx = rollout['vid_idx']
    p_history = loss_dict['p_history']
    v_history = loss_dict['v_history']
    act_buffer = loss_dict['act_buffer']
    print("save check success:", i)

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
    if MATPLOTLIB_AVAILABLE and use_camera_control and rollout['cam_params_history']:
        ch = torch.stack(rollout['cam_params_history'])[:, vid_idx].detach().cpu()
        fig_cam, axes = plt.subplots(1, 3, figsize=(12, 3))
        if use_diff_depth:
            labels = ['Power', 'Exposure', 'Gain']
        else:
            labels = ['FOV delta', 'Exposure', 'ISO']
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
        if use_camera_control and rollout['cam_exposure_history']:
            ax2 = ax_roll.twinx()
            sp_p = torch.stack([x.detach() if x.requires_grad else x for x in rollout['speed_for_cam_history']])[:, vid_idx].cpu()
            ex_p = torch.stack([x.detach() if x.requires_grad else x for x in rollout['cam_exposure_history']])[:, vid_idx].cpu()
            ax2.plot(sp_p.numpy(), 'r--', alpha=0.7, label='Speed')
            ax2.plot(ex_p.numpy(), 'g-.', alpha=0.7, label='Exposure')
            ax2.set_ylabel('Speed / Exposure'); ax2.legend(loc='upper right')
        ax_roll.legend(loc='upper left'); fig_roll.tight_layout()
        wandb.log({'roll_speed_exposure': wandb.Image(fig_roll)}, step=i + 1)
        plt.close(fig_roll)


# =====================================================================
# Main training entry point
# =====================================================================

def train(args, sensor_flags, model, env_train, env_full,
          optim, sched, scaler, vis, checkpoint_dir, device):
    """Main training loop — drop-in replacement for the old main_cuda.py loop."""
    sf = sensor_flags
    use_amp = bool(args.amp and device.type == 'cuda')
    smoother = MetricSmoother(sf, args)

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
            # 从环境中提取缩放参数（用于动态AABB计算）
            max_speed_j = env.max_speed[j:j+1] if hasattr(env, 'max_speed') else None
            y_stretch_j = getattr(env, '_current_y_stretch', None)
            scale_j = getattr(env, '_current_scale', None)
            vis.log_environment(
                phase='student',
                balls=env.balls[j].detach().cpu().numpy(),
                voxels=env.voxels[j].detach().cpu().numpy(),
                cyl=env.cyl[j].detach().cpu().numpy(),
                cyl_h=env.cyl_h[j].detach().cpu().numpy(),
                start=env.p[j].detach().cpu().numpy(),
                target=env.p_target[j].detach().cpu().numpy(),
                max_speed=max_speed_j,
                y_stretch=y_stretch_j,
                scale=scale_j)

        # Teacher phase
        u_star, y_star, u_star_cam = None, None, None
        if args.enable_teacher_student_training:
            u_star, y_star, u_star_cam = teacher_phase(
                env, model, args, sf, B, device, use_amp,
                vis, should_vis, i)

        # Student rollout
        tbptt_this_iter = args.tbptt_enable and not use_full_bptt_iter
        rollout = student_rollout(
            env, model, args, sf, B, device, use_amp,
            scaler, optim, sched,
            u_star, y_star, u_star_cam, dc_iter,
            tbptt_this_iter, use_full_bptt_iter,
            vis, should_vis, i)

        if tbptt_this_iter:
            # TBPTT: backward already done inside student_rollout
            denom = max(rollout['tbptt_chunk_n'], 1)
            loss = torch.tensor(rollout['tbptt_stats']['loss'] / denom, device=device)
            loss_distill = torch.tensor(rollout['tbptt_stats']['loss_distill'] / denom, device=device)

            # Detached metrics
            vec_det = torch.stack([x.detach() if isinstance(x, torch.Tensor) and x.requires_grad else x
                                   for x in rollout['vec_to_pt_history']])
            distance_det = torch.norm(vec_det, 2, -1) - env.margin
            success = torch.all(distance_det.flatten(0, 1) > 0, 0)
            _success = success.sum() / B
            v_det = torch.stack([x.detach() if isinstance(x, torch.Tensor) and x.requires_grad else x
                                 for x in rollout['v_history']])
            speed_history = v_det.norm(2, -1)
            avg_speed = speed_history.mean(0)

            pbar.set_description_str(f'loss: {float(loss):.3f} (tbptt)')
        else:
            # Full BPTT: compute loss and backward
            loss, loss_dict = full_bptt_losses(
                rollout, env, args, sf, device,
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
                scaler.step(optim); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optim.step()
            sched.step()
            torch.cuda.synchronize()  # flush GPU queue to prevent display starvation

            success = torch.all(distance.flatten(0, 1) > 0, 0)
            _success = success.sum() / B
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

            # Weighted loss composition (contribution + share)
            smoother.add(_build_loss_share_metrics(loss_scalars, args, dc_iter))

            smoother.add({
                'loss': float(loss.detach()),
                'success': float(_success),
                'max_speed': float(speed_history.max(0).values.mean()),
                'avg_speed': float(avg_speed.mean()),
                'ar': float((success * avg_speed).mean()),
            })

            # Emerging metrics (shared for both branches)
            if tbptt_this_iter:
                p_det = torch.stack([x for x in rollout['p_history']])
                _compute_emerging_metrics(
                    rollout,
                    {'p_history': p_det, 'vec_to_pt_history': vec_det,
                     'distance': distance_det, 'speed_history': speed_history},
                    env, args, sf, smoother)
            else:
                _compute_emerging_metrics(rollout, loss_dict, env, args, sf, smoother)

            # Save visualizations
            if is_save_iter(i) and not tbptt_this_iter:
                _log_save_iter(rollout, loss_dict, env, args, sf, i)

            periodic_tail_ops(i, checkpoint_dir, model, smoother)

"""
Rollout building blocks shared by Teacher and Student phases.

All functions are stateless / pure, taking explicit arguments.
"""
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Sensor rendering
# ---------------------------------------------------------------------------
def render_sensors(env, ctl_dt, cam_fov, cam_exposure, cam_iso,
                   use_passive_depth, use_camera_luma, use_active_depth,
                   use_depth_channel, use_camera_control,
                   differentiable=False):
    """Render main observation and optional ToF.

    Args:
        differentiable: If True, keep the computation graph for the main camera
                        path (required for student's camera-control gradient).
                        If False, wrap everything in torch.no_grad().
    Returns:
        main_obs, tof_depth, tof_conf
    """
    main_obs = None
    tof_depth = None
    tof_conf = None

    if differentiable and use_camera_luma and use_camera_control:
        main_obs = env.render_main_luma_diff(cam_fov, cam_exposure, cam_iso)
        if use_depth_channel:
            with torch.no_grad():
                tof_depth, tof_conf, _, _ = env.render_tof(ctl_dt, return_meta=True)
        return main_obs, tof_depth, tof_conf

    if differentiable and use_active_depth and use_camera_control:
        active_power, active_exposure, active_gain = cam_fov, cam_exposure, cam_iso
        tof_depth, tof_conf = env.render_active_tof_diff(active_power, active_exposure, active_gain)
        return main_obs, tof_depth, tof_conf

    # Non-differentiable / teacher paths
    with torch.no_grad():
        if use_passive_depth:
            main_depth, _ = env.render(ctl_dt)
            main_obs = main_depth
        elif use_camera_luma:
            if use_camera_control:
                main_obs = env.render_main_luma_diff(cam_fov, cam_exposure, cam_iso)
            else:
                main_obs = env.render_main_luma(ctl_dt)
        elif use_active_depth:
            if use_camera_control:
                active_power, active_exposure, active_gain = cam_fov, cam_exposure, cam_iso
                tof_depth, tof_conf = env.render_active_tof_diff(active_power, active_exposure, active_gain)
            else:
                raise NotImplementedError("active_depth requires camera_action_mode != off")

        if use_depth_channel and not use_active_depth:
            tof_depth, tof_conf, _, _ = env.render_tof(ctl_dt, return_meta=True)

    return main_obs, tof_depth, tof_conf


# ---------------------------------------------------------------------------
# 2. Local coordinate frame & state vector
# ---------------------------------------------------------------------------
def build_local_frame(env):
    """Build the heading-aligned local frame R (yaw only, no pitch/roll)."""
    fwd = env.R[:, :, 0].clone()
    fwd[:, 2] = 0
    up = torch.zeros_like(fwd)
    up[:, 2] = 1
    fwd = F.normalize(fwd, 2, -1)
    R = torch.stack([fwd, torch.cross(up, fwd), up], -1)
    return R


def build_state_vector(env, target_v, R, cam_fov, cam_exposure, cam_iso,
                       no_odom, include_camera_state, use_camera_control):
    """Construct the observation vector fed to the policy network.

    Returns:
        state: (B, obs_dim) tensor
        local_v: (B, 3) local velocity
    """
    tv_local = torch.squeeze(target_v[:, None] @ R, 1)
    local_v = torch.squeeze(env.v[:, None] @ R, 1)
    st = [tv_local, env.R[:, 2], env.margin[:, None]]
    if not no_odom:
        st.insert(0, local_v)
    if include_camera_state and use_camera_control:
        co = torch.stack([
            cam_fov / env._fov_x_half_tan - 1.0,
            cam_exposure,
            cam_iso,
        ], -1)
        st.append(co)
    return torch.cat(st, -1), local_v


# ---------------------------------------------------------------------------
# 3. Target velocity
# ---------------------------------------------------------------------------
def compute_target_velocity(target_v_raw, env):
    """Clamp raw target direction to max speed."""
    tv_n = torch.norm(target_v_raw, 2, -1, keepdim=True).clamp_min(1e-6)
    tv_u = target_v_raw / tv_n
    return tv_u * torch.minimum(tv_n, env.max_speed)


# ---------------------------------------------------------------------------
# 4. Action decoding
# ---------------------------------------------------------------------------
def decode_action_direct(act_raw, R, env, B, max_acc_cmd):
    """Decode policy output in *action domain* (no dLQR).

    Returns:
        a_final: clamped thrust command  (B, 3)
        v_pred:  auxiliary velocity prediction  (B, 3)
    """
    a_pred, v_pred, *_ = (R @ act_raw.reshape(B, 3, -1)).unbind(-1)
    a_final = (a_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
    a_final = a_final.clamp(-max_acc_cmd, max_acc_cmd)
    return a_final, v_pred


def decode_action_lqr(intent, R, env, local_v, B,
                      A_lqr, B_lqr,
                      lqr_horizon, lqr_reg, max_acc_cmd,
                      inject_tof, tof_safe_dist, tof_repel_gain,
                      use_depth_channel, vec_now,
                      solve_batched_dlqr_fn):
    """Decode policy output in *intent domain* via dLQR.

    Returns:
        a_final: clamped thrust command  (B, 3)
        v_pred:  zeros (no auxiliary prediction in LQR mode)  (B, 3)
    """
    v_ref_local = torch.tanh(intent[:, :3]) * env.max_speed
    q_diag = (F.softplus(intent[:, 3:6]) + 1e-3).clamp(1e-3, 20.0)
    r_diag = (F.softplus(intent[:, 6:9]) + 1e-3).clamp(1e-3, 20.0)
    Q_lqr = torch.diag_embed(q_diag)
    R_lqr = torch.diag_embed(r_diag)

    u_local, _, _ = solve_batched_dlqr_fn(
        A_lqr, B_lqr, Q_lqr, R_lqr,
        local_v, v_ref_local,
        horizon=lqr_horizon,
        reg=lqr_reg,
    )
    u_local = u_local.clamp(-max_acc_cmd, max_acc_cmd)

    if inject_tof and use_depth_channel and vec_now is not None:
        vec_now_lqr = vec_now[0]  # (B, 3)
        dist_now = torch.norm(vec_now_lqr, 2, -1)
        repel_mag = F.softplus(tof_safe_dist - dist_now) * tof_repel_gain
        vec_local = torch.squeeze(vec_now_lqr[:, None] @ R, 1)
        repel_dir = -F.normalize(vec_local, 2, -1)
        u_local = u_local + repel_dir * repel_mag[:, None]

    a_pred = torch.squeeze(R @ u_local[:, :, None], -1)
    v_pred = torch.zeros_like(a_pred)
    a_final = (a_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
    a_final = a_final.clamp(-max_acc_cmd, max_acc_cmd)
    return a_final, v_pred


# ---------------------------------------------------------------------------
# 5. Camera parameter update
# ---------------------------------------------------------------------------
def update_camera_params(cam_params, cam_fov, cam_exposure, cam_iso,
                         camera_action_mode, cam_delta_scale, env):
    """Apply policy camera output and return updated params + history entry.

    Returns:
        cam_fov, cam_exposure, cam_iso, cam_hist_entry (or None)
    """
    if cam_params is None:
        return cam_fov, cam_exposure, cam_iso, None

    if camera_action_mode == 'incremental':
        df, de, di = cam_params.unbind(-1)
        sc = cam_delta_scale
        cam_fov = (cam_fov + df * sc * env._fov_x_half_tan).clamp(
            env._fov_x_half_tan * 0.08, env._fov_x_half_tan * 1.5)
        cam_exposure = (cam_exposure + de * sc).clamp(0.01, 0.99)
        cam_iso = (cam_iso + di * sc).clamp(0.01, 0.99)
        hist = torch.stack([cam_fov / env._fov_x_half_tan, cam_exposure, cam_iso], -1)
    else:
        fd, ex, iso_v = cam_params.unbind(-1)
        cam_fov = env._fov_x_half_tan * 0.08 + fd * env._fov_x_half_tan * 1.42
        cam_exposure = ex
        cam_iso = iso_v
        hist = cam_params

    return cam_fov, cam_exposure, cam_iso, hist

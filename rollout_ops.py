"""
Rollout building blocks shared by Teacher and Student phases.

All functions are stateless / pure, taking explicit arguments.
"""
import torch
import torch.nn.functional as F

from camera_semantics import CameraSemantics


_DEFAULT_CAMERA_SEMANTICS = CameraSemantics()


def diff_depth_exposure_to_time(exposure01, camera_semantics=None):
    """Map normalized diff_depth exposure [0,1] to a unified effective exposure time."""
    cam_sem = camera_semantics if camera_semantics is not None else _DEFAULT_CAMERA_SEMANTICS
    return cam_sem.exposure_to_time(exposure01)


def diff_depth_fill_softness(min_valid_depth: float) -> float:
    """Choose a smooth fill-proxy transition around the valid-depth threshold."""
    return max(0.04, 0.15 * float(min_valid_depth))


def compute_depth_fill_rate(depth_obs, min_valid_depth: float = 0.3, softness=None):
    """Return diff_depth fill rate using the same validity semantics everywhere."""
    threshold = float(min_valid_depth)
    if softness is None:
        return (depth_obs >= threshold).float().mean()
    softness = float(softness)
    if softness <= 0.0:
        return (depth_obs >= threshold).float().mean()
    return torch.sigmoid((depth_obs - threshold) / softness).mean()


def select_policy_depth_obs(depth_obs, mode: str = 'depth'):
    """Choose what depth tensor is actually fed into the policy network."""
    if depth_obs is None:
        return None
    mode = str(mode).strip().lower()
    if mode in {'zero', 'blind', 'none'}:
        return torch.zeros_like(depth_obs)
    return depth_obs


def _stack_history_or_tensor(values):
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return values
    if len(values) == 0:
        return None
    return torch.stack([
        x.detach() if isinstance(x, torch.Tensor) and x.requires_grad else x
        for x in values
    ])


def compute_camera_param_stats(power_seq, exposure_seq, gain_seq):
    """Return mean/std/min/max for power / exposure / gain histories."""
    power_seq = _stack_history_or_tensor(power_seq)
    exposure_seq = _stack_history_or_tensor(exposure_seq)
    gain_seq = _stack_history_or_tensor(gain_seq)

    if power_seq is None or exposure_seq is None or gain_seq is None:
        return {}

    out = {}
    stats = {
        'power': power_seq,
        'exposure': exposure_seq,
        'gain': gain_seq,
    }
    for name, seq in stats.items():
        out[f'{name}_mean'] = float(seq.mean().item())
        out[f'{name}_std'] = float(seq.std(unbiased=False).item())
        out[f'{name}_min'] = float(seq.min().item())
        out[f'{name}_max'] = float(seq.max().item())
    return out


def compute_diff_depth_proxies(power_seq, exposure_seq, gain_seq, speed_seq, camera_semantics=None):
    """Shared diff_depth proxy metrics for training/evaluation."""
    power_seq = _stack_history_or_tensor(power_seq)
    exposure_seq = _stack_history_or_tensor(exposure_seq)
    gain_seq = _stack_history_or_tensor(gain_seq)
    speed_seq = _stack_history_or_tensor(speed_seq)

    if power_seq is None or exposure_seq is None or gain_seq is None or speed_seq is None:
        return {}

    exp_phys = diff_depth_exposure_to_time(exposure_seq, camera_semantics=camera_semantics)
    return {
        'energy_proxy': float(power_seq.pow(2).mean().item()),
        'blur_proxy': float((speed_seq * exp_phys).pow(2).mean().item()),
        'noise_proxy': float(gain_seq.pow(2).mean().item()),
    }


def init_camera_params(env, B, device):
    """Initial diff_depth sensor-control state: power / exposure / gain."""
    mode = getattr(env, 'camera_control_mode', 'learned')
    power_baseline = float(getattr(env, 'cam_power_baseline', 0.55))
    if mode == 'fixed':
        power0 = float(getattr(env, 'fixed_camera_power', power_baseline))
        exposure0 = float(getattr(env, 'fixed_camera_exposure', 0.5))
        gain0 = float(getattr(env, 'fixed_camera_gain', 0.5))
    elif mode == 'fixed_random_static':
        p_lo, p_hi = getattr(env, 'fixed_random_power_range', (0.55, 0.90))
        e_lo, e_hi = getattr(env, 'fixed_random_exposure_range', (0.16, 0.60))
        g_lo, g_hi = getattr(env, 'fixed_random_gain_range', (0.02, 0.42))
        power = torch.empty((B,), device=device).uniform_(float(p_lo), float(p_hi))
        exposure = torch.empty((B,), device=device).uniform_(float(e_lo), float(e_hi))
        gain = torch.empty((B,), device=device).uniform_(float(g_lo), float(g_hi))
        return power, exposure, gain
    else:
        power0 = power_baseline
        exposure0 = 0.5
        gain0 = 0.5
    power = torch.full((B,), min(max(power0, 0.0), 1.0), device=device)
    exposure = torch.full((B,), min(max(exposure0, 0.0), 1.0), device=device)
    gain = torch.full((B,), min(max(gain0, 0.0), 1.0), device=device)
    return power, exposure, gain


# ---------------------------------------------------------------------------
# 1. Sensor rendering
# ---------------------------------------------------------------------------
def render_sensors(env, ctl_dt, power, exposure, gain, differentiable=False):
    """Render diff_depth observations only.

    Returns:
        depth_obs: (B, H, W) noisy depth image
        quality: (B, H, W) deterministic quality map (no randn), differentiable
                 w.r.t. power/exposure/gain — use this for fill-rate loss
    """
    _ = ctl_dt
    if differentiable:
        result = env.render_diff_depth(power, exposure, gain)
    else:
        with torch.no_grad():
            result = env.render_diff_depth(power, exposure, gain)
    # render_diff_depth returns (noisy_depth, quality) for python impl
    if isinstance(result, tuple):
        depth_obs, quality = result
    else:
        depth_obs = result
        quality = None
    return depth_obs, quality


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


def build_state_vector(env, target_v, R, power, exposure, gain,
                       no_odom, include_camera_state):
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
    if include_camera_state:
        _ = env
        co = torch.stack([
            power * 2.0 - 1.0,
            exposure * 2.0 - 1.0,
            gain * 2.0 - 1.0,
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
                      inject_depth, depth_safe_dist, depth_repel_gain,
                      vec_now,
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

    if inject_depth and vec_now is not None:
        vec_now_lqr = vec_now[0]  # (B, 3)
        dist_now = torch.norm(vec_now_lqr, 2, -1)
        repel_mag = F.softplus(depth_safe_dist - dist_now) * depth_repel_gain
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
def update_camera_params(cam_params, power, exposure, gain, env):
    """Apply diff_depth camera output and return updated params + history entry.

    EMA 平滑传感器状态（alpha=0.7），让物理传感器参数有时间连续性。
    history entry 存储网络原始输出 cam_params（未经 EMA），使 loss_cam_range /
    loss_cam_smooth 的梯度能完整流回网络，不被 EMA 的 detach 截断。

    Returns:
        power, exposure, gain, cam_hist_entry (= raw cam_params, shape [B, 3])
    """
    if cam_params is None:
        raise ValueError('diff_depth-only 路径要求 cam_params 不为空')

    mode = getattr(env, 'camera_control_mode', 'learned')
    if mode == 'fixed':
        power_baseline = float(getattr(env, 'cam_power_baseline', 0.55))
        fixed_power = torch.full_like(power, float(getattr(env, 'fixed_camera_power', power_baseline)))
        fixed_exposure = torch.full_like(exposure, float(getattr(env, 'fixed_camera_exposure', 0.5)))
        fixed_gain = torch.full_like(gain, float(getattr(env, 'fixed_camera_gain', 0.5)))
        hist = torch.stack([fixed_power, fixed_exposure, fixed_gain], dim=-1)
        return fixed_power, fixed_exposure, fixed_gain, hist
    if mode == 'fixed_random_static':
        hist = torch.stack([power.detach(), exposure.detach(), gain.detach()], dim=-1)
        return power.detach(), exposure.detach(), gain.detach(), hist

    _ = env
    alpha = 0.7
    p_new, e_new, g_new = cam_params.unbind(-1)
    p_new = p_new.clamp(0.0, 1.0)
    e_new = e_new.clamp(0.0, 1.0)
    g_new = g_new.clamp(0.0, 1.0)

    # EMA 平滑物理传感器状态（detach 历史，只保留当前步梯度）
    power = alpha * power.detach() + (1.0 - alpha) * p_new
    exposure = alpha * exposure.detach() + (1.0 - alpha) * e_new
    gain = alpha * gain.detach() + (1.0 - alpha) * g_new

    # hist 存原始网络输出，让 loss_cam_range/loss_cam_smooth 梯度完整流回网络
    hist = cam_params  # shape: [B, 3]
    return power, exposure, gain, hist

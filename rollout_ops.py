"""
Rollout building blocks shared by Teacher and Student phases.

All functions are stateless / pure, taking explicit arguments.
"""
import math

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


def compute_depth_sensor_health(depth_obs, min_valid_depth: float = 0.3,
                                softness=None, patch_rows: int = 6,
                                patch_cols: int = 8,
                                cvar_frac: float = 0.25):
    """Return per-env worst-patch/CVaR fill health for camera optimization.

    This is intentionally scene-agnostic: split each depth/quality image into a
    coarse grid, compute valid-rate per patch, then average the worst
    ``cvar_frac`` patches. It prevents a large valid background region from
    hiding local sensor failures around edges, glare, or dark patches.
    """
    threshold = float(min_valid_depth)
    if softness is None or float(softness) <= 0.0:
        valid = (depth_obs >= threshold).float()
    else:
        valid = torch.sigmoid((depth_obs - threshold) / float(softness))

    if valid.ndim == 2:
        valid = valid.unsqueeze(0)
    elif valid.ndim == 4:
        if valid.shape[1] == 1:
            valid = valid[:, 0]
        else:
            valid = valid.mean(dim=1)
    if valid.ndim != 3:
        raise ValueError(
            f"compute_depth_sensor_health expects [H,W], [B,H,W], or [B,C,H,W], got {tuple(valid.shape)}"
        )

    rows = max(1, int(patch_rows))
    cols = max(1, int(patch_cols))
    patch_fill = F.adaptive_avg_pool2d(valid[:, None], (rows, cols)).flatten(1)
    frac = min(max(float(cvar_frac), 0.0), 1.0)
    if frac <= 0.0:
        k = 1
    else:
        k = max(1, int(math.ceil(patch_fill.shape[1] * frac)))
    worst = torch.topk(patch_fill, k=k, dim=1, largest=False).values
    return worst.mean(dim=1)


def _as_bhw(x):
    if x is None:
        return None
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 4:
        if x.shape[1] == 1:
            return x[:, 0]
        return x.mean(dim=1)
    return x


def compute_validity_fill_rate(valid_map):
    """Return the mean valid-pixel probability for a [0,1] validity map."""
    valid = _as_bhw(valid_map)
    if valid is None:
        return None
    if valid.ndim != 3:
        raise ValueError(
            f"compute_validity_fill_rate expects [H,W], [B,H,W], or [B,C,H,W], got {tuple(valid.shape)}"
        )
    return valid.clamp(0.0, 1.0).mean()


def compute_validity_sensor_health(valid_map, patch_rows: int = 6,
                                   patch_cols: int = 8,
                                   cvar_frac: float = 0.25):
    """Patch/CVaR health for a dimensionless valid-probability map.

    Unlike ``compute_depth_sensor_health``, this function does not apply a
    metric depth threshold.  It is the correct loss-side companion for the
    differentiable sensor model's internal valid probability.
    """
    valid = _as_bhw(valid_map)
    if valid is None:
        return None
    if valid.ndim != 3:
        raise ValueError(
            f"compute_validity_sensor_health expects [H,W], [B,H,W], or [B,C,H,W], got {tuple(valid.shape)}"
        )
    rows = max(1, int(patch_rows))
    cols = max(1, int(patch_cols))
    patch_fill = F.adaptive_avg_pool2d(valid.clamp(0.0, 1.0)[:, None], (rows, cols)).flatten(1)
    frac = min(max(float(cvar_frac), 0.0), 1.0)
    k = max(1, int(math.ceil(patch_fill.shape[1] * frac))) if frac > 0.0 else 1
    worst = torch.topk(patch_fill, k=k, dim=1, largest=False).values
    return worst.mean(dim=1)


def sensor_validity_maps(env):
    """Fetch the latest differentiable validity maps exported by Env, if any."""
    if env is None or not hasattr(env, 'get_last_diff_depth_train_aux'):
        return None, None
    aux = env.get_last_diff_depth_train_aux() or {}
    valid_prob = aux.get('valid_prob_map', None)
    hard_valid = aux.get('hard_valid_map', None)
    return valid_prob, hard_valid


def compute_render_health_metrics(env, depth_obs, min_valid_depth: float = 0.3,
                                  patch_rows: int = 6, patch_cols: int = 8,
                                  cvar_frac: float = 0.25):
    """Compute fill/health metrics from the latest render in consistent units.

    The loss-side signal should use the sensor model's dimensionless validity
    probability when available.  Falling back to ``depth_obs`` keeps CUDA
    backend / legacy paths functional, but those paths are not fully
    differentiable through the Python sensor model.
    """
    valid_prob, hard_valid = sensor_validity_maps(env)
    if valid_prob is not None:
        fill_hard = (
            compute_validity_fill_rate(hard_valid)
            if hard_valid is not None else compute_validity_fill_rate(valid_prob.detach())
        )
        fill_soft = compute_validity_fill_rate(valid_prob)
        fill_health = compute_validity_sensor_health(
            valid_prob,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
            cvar_frac=cvar_frac,
        )
        return fill_hard, fill_soft, fill_health

    fill_hard = compute_depth_fill_rate(depth_obs, min_valid_depth=min_valid_depth)
    fill_soft = compute_depth_fill_rate(
        depth_obs,
        min_valid_depth=min_valid_depth,
        softness=diff_depth_fill_softness(min_valid_depth),
    )
    fill_health = compute_depth_sensor_health(
        depth_obs,
        min_valid_depth=min_valid_depth,
        softness=diff_depth_fill_softness(min_valid_depth),
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        cvar_frac=cvar_frac,
    )
    return fill_hard, fill_soft, fill_health


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

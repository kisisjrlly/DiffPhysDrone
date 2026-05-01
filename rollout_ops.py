"""Minimal rollout helpers for direct-action active-sensing training."""
import torch
import torch.nn.functional as F

from camera_semantics import CameraSemantics


_DEFAULT_CAMERA_SEMANTICS = CameraSemantics()


def diff_depth_exposure_to_time(exposure01, camera_semantics=None):
    cam_sem = camera_semantics if camera_semantics is not None else _DEFAULT_CAMERA_SEMANTICS
    return cam_sem.exposure_to_time(exposure01)


def compute_depth_fill_rate(depth_obs, min_valid_depth: float = 0.3, softness=None):
    if softness is None or float(softness) <= 0.0:
        return (depth_obs >= float(min_valid_depth)).float().mean()
    return torch.sigmoid((depth_obs - float(min_valid_depth)) / float(softness)).mean()


def sensor_validity_map(env):
    aux = env.get_last_diff_depth_train_aux() if env is not None and hasattr(env, 'get_last_diff_depth_train_aux') else {}
    return aux.get('valid_prob_map', None)


def compute_depth_fill_health(env, depth_obs, min_valid_depth: float = 0.3,
                              patch_rows: int = 6, patch_cols: int = 8,
                              cvar_frac: float = 0.25):
    valid_prob = sensor_validity_map(env)
    if valid_prob is not None:
        valid = valid_prob.clamp(0.0, 1.0)
    else:
        valid = (depth_obs >= float(min_valid_depth)).float()
    if valid.ndim == 2:
        valid = valid.unsqueeze(0)
    elif valid.ndim == 4:
        valid = valid[:, 0] if valid.shape[1] == 1 else valid.mean(dim=1)
    if valid.ndim != 3:
        raise ValueError(f'fill health expects [H,W], [B,H,W], or [B,C,H,W], got {tuple(valid.shape)}')
    rows = max(1, int(patch_rows))
    cols = max(1, int(patch_cols))
    patch_fill = F.adaptive_avg_pool2d(valid[:, None], (rows, cols)).flatten(1)
    frac = min(max(float(cvar_frac), 0.0), 1.0)
    num_patches = int(patch_fill.shape[1])
    k = num_patches if frac >= 1.0 else max(1, int(num_patches * frac + 0.999999))
    return torch.topk(patch_fill, k=k, dim=1, largest=False).values.mean(dim=1)


def render_sensors(env, ctl_dt, power, exposure, gain, differentiable=True):
    _ = ctl_dt
    depth_obs, quality = env.render_diff_depth(power, exposure, gain)
    if not differentiable:
        depth_obs = depth_obs.detach()
        quality = quality.detach() if isinstance(quality, torch.Tensor) else quality
    return depth_obs, quality


def select_policy_depth_obs(depth_obs, mode: str = 'depth'):
    if str(mode).strip().lower() in {'zero', 'blind', 'none'}:
        return torch.zeros_like(depth_obs)
    return depth_obs


def build_local_frame(env):
    fwd = env.R[:, :, 0].clone()
    fwd[:, 2] = 0
    up = torch.zeros_like(fwd)
    up[:, 2] = 1
    fwd = F.normalize(fwd, 2, -1)
    return torch.stack([fwd, torch.cross(up, fwd), up], -1)


def build_state_vector(env, target_v, R, power, exposure, gain, no_odom, include_camera_state):
    tv_local = torch.squeeze(target_v[:, None] @ R, 1)
    local_v = torch.squeeze(env.v[:, None] @ R, 1)
    st = [tv_local, env.R[:, 2], env.margin[:, None]]
    if not no_odom:
        st.insert(0, local_v)
    state = torch.cat(st, -1)

    if include_camera_state:
        camera_state = torch.stack([power * 2.0 - 1.0, exposure * 2.0 - 1.0, gain * 2.0 - 1.0], -1)
    else:
        camera_state = torch.zeros_like(local_v)

    speed_scale = getattr(env, 'max_speed', None)
    if speed_scale is None:
        local_v_norm = local_v
    else:
        local_v_norm = local_v / speed_scale.clamp_min(1e-3)
    camera_motion_state = torch.cat([local_v_norm.clamp(-2.0, 2.0), env.R[:, :, 2]], -1)
    return state, local_v, camera_state, camera_motion_state


def compute_target_velocity(target_v_raw, env):
    tv_n = torch.norm(target_v_raw, 2, -1, keepdim=True).clamp_min(1e-6)
    return target_v_raw / tv_n * env.max_speed


def decode_action_direct(raw_act, R, env, B, max_acc_cmd):
    _ = env, B
    act_local = raw_act[..., :3].clamp(-float(max_acc_cmd), float(max_acc_cmd))
    v_pred_local = raw_act[..., 3:6]
    act_world = torch.squeeze(act_local[:, None] @ R.transpose(1, 2), 1)
    v_pred = torch.squeeze(v_pred_local[:, None] @ R.transpose(1, 2), 1)
    return act_world, v_pred


def init_camera_params(env, B, device):
    mode = getattr(env, 'camera_control_mode', 'learned')
    if mode == 'fixed':
        return (
            torch.full((B,), float(env.fixed_camera_power), device=device),
            torch.full((B,), float(env.fixed_camera_exposure), device=device),
            torch.full((B,), float(env.fixed_camera_gain), device=device),
        )
    if mode == 'fixed_random_static':
        p_lo, p_hi = env.fixed_random_power_range
        e_lo, e_hi = env.fixed_random_exposure_range
        g_lo, g_hi = env.fixed_random_gain_range
        return (
            torch.empty((B,), device=device).uniform_(float(p_lo), float(p_hi)),
            torch.empty((B,), device=device).uniform_(float(e_lo), float(e_hi)),
            torch.empty((B,), device=device).uniform_(float(g_lo), float(g_hi)),
        )
    base = float(getattr(env, 'cam_power_baseline', 0.5))
    return (
        torch.full((B,), base, device=device),
        torch.full((B,), 0.35, device=device),
        torch.full((B,), 0.25, device=device),
    )


def update_camera_params(cam_params, power, exposure, gain, env):
    mode = getattr(env, 'camera_control_mode', 'learned')
    if mode == 'fixed':
        p = torch.full_like(power, float(env.fixed_camera_power))
        e = torch.full_like(exposure, float(env.fixed_camera_exposure))
        g = torch.full_like(gain, float(env.fixed_camera_gain))
        return p, e, g, torch.stack([p, e, g], -1)
    if mode == 'fixed_random_static':
        hist = torch.stack([power.detach(), exposure.detach(), gain.detach()], -1)
        return power.detach(), exposure.detach(), gain.detach(), hist

    step = float(getattr(env, 'cam_delta_max', 0.02))
    ret = float(getattr(env, 'cam_return_rate', 0.05))
    delta = cam_params.clamp(-1.0, 1.0) * step
    p_delta, e_delta, g_delta = delta.unbind(-1)
    p_center = torch.full_like(power, float(getattr(env, 'fixed_camera_power', 0.5)))
    e_center = torch.full_like(exposure, float(getattr(env, 'fixed_camera_exposure', 0.5)))
    g_center = torch.full_like(gain, float(getattr(env, 'fixed_camera_gain', 0.5)))
    power = power + p_delta + ret * (p_center - power.detach())
    exposure = exposure + e_delta + ret * (e_center - exposure.detach())
    gain = gain + g_delta + ret * (g_center - gain.detach())
    power = power.clamp(0.08, 0.95)
    exposure = exposure.clamp(0.08, 0.95)
    gain = gain.clamp(0.02, 0.95)
    return power, exposure, gain, torch.stack([power, exposure, gain], -1)


def _stack_history_or_tensor(values):
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return values
    if len(values) == 0:
        return None
    return torch.stack([x.detach() if isinstance(x, torch.Tensor) and x.requires_grad else x for x in values])


def compute_camera_param_stats(power_seq, exposure_seq, gain_seq):
    power_seq = _stack_history_or_tensor(power_seq)
    exposure_seq = _stack_history_or_tensor(exposure_seq)
    gain_seq = _stack_history_or_tensor(gain_seq)
    if power_seq is None or exposure_seq is None or gain_seq is None:
        return {}
    return {
        'power_mean': float(power_seq.mean().item()),
        'exposure_mean': float(exposure_seq.mean().item()),
        'gain_mean': float(gain_seq.mean().item()),
    }

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


def sensor_validity_maps(env):
    aux = env.get_last_diff_depth_train_aux() if env is not None and hasattr(env, 'get_last_diff_depth_train_aux') else {}
    return aux.get('valid_prob_map', None), aux.get('hard_valid_map', None)


def compute_render_health_metrics(env, depth_obs, min_valid_depth: float = 0.3, **_):
    valid_prob, hard_valid = sensor_validity_maps(env)
    if valid_prob is not None:
        fill_soft = valid_prob.clamp(0.0, 1.0).mean()
        fill_hard = hard_valid.clamp(0.0, 1.0).mean() if hard_valid is not None else fill_soft.detach()
        return fill_hard, fill_soft, fill_soft
    fill = compute_depth_fill_rate(depth_obs, min_valid_depth=min_valid_depth)
    return fill, fill, fill


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
    if include_camera_state:
        st.append(torch.stack([power * 2.0 - 1.0, exposure * 2.0 - 1.0, gain * 2.0 - 1.0], -1))
    return torch.cat(st, -1), local_v


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

    alpha = 0.7
    p_new, e_new, g_new = cam_params.clamp(0.0, 1.0).unbind(-1)
    power = alpha * power.detach() + (1.0 - alpha) * p_new
    exposure = alpha * exposure.detach() + (1.0 - alpha) * e_new
    gain = alpha * gain.detach() + (1.0 - alpha) * g_new
    return power, exposure, gain, cam_params


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

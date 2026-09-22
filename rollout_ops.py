"""Rollout helpers for grayscale active sensing."""

import torch
import torch.nn.functional as F


def render_gray_sensor(env, exposure, gain, differentiable=True):
    if not hasattr(env, "gray_camera"):
        raise RuntimeError("env.gray_camera is not configured")
    ideal, render_aux = env.render_gray_ideal(return_aux=True)
    sensor_exposure = exposure if differentiable else exposure.detach()
    sensor_gain = gain if differentiable else gain.detach()
    motion = env.v.norm(2, -1).detach()
    gray, camera_aux = env.gray_camera(
        ideal,
        sensor_exposure,
        sensor_gain,
        motion=motion,
        enable_noise=bool(getattr(env, "gray_enable_noise", True)),
    )
    aux = dict(render_aux or {})
    aux.update(camera_aux or {})
    aux["ideal_gray"] = ideal.detach()
    return gray, aux


def select_policy_gray_obs(gray_obs, mode="gray"):
    if str(mode).strip().lower() in {"zero", "blind", "none"}:
        return torch.zeros_like(gray_obs)
    return gray_obs


def build_local_frame(env):
    fwd = env.R[:, :, 0].clone()
    fwd[:, 2] = 0
    up = torch.zeros_like(fwd)
    up[:, 2] = 1
    fwd = F.normalize(fwd, 2, -1)
    return torch.stack([fwd, torch.cross(up, fwd, dim=-1), up], -1)


def build_state_vector(env, target_v, R, exposure, gain, no_odom, include_camera_state):
    tv_local = torch.squeeze(target_v[:, None] @ R, 1)
    local_v = torch.squeeze(env.v[:, None] @ R, 1)
    parts = [tv_local, env.R[:, 2], env.margin[:, None]]
    if not no_odom:
        parts.insert(0, local_v)

    camera_state = torch.stack([exposure * 2.0 - 1.0, gain * 2.0 - 1.0], -1)
    if include_camera_state:
        parts.append(camera_state)
    state = torch.cat(parts, -1)

    speed_scale = getattr(env, "max_speed", None)
    local_v_norm = local_v if speed_scale is None else local_v / speed_scale.clamp_min(1e-3)
    camera_motion_state = torch.cat(
        [local_v_norm.clamp(-2.0, 2.0), env.R[:, :, 2]],
        -1,
    )
    return state, local_v, camera_state, camera_motion_state


def compute_target_velocity(target_v_raw, env):
    goal_dist = torch.norm(target_v_raw, 2, -1, keepdim=True)
    tv_dir = target_v_raw / goal_dist.clamp_min(1e-6)
    slow_radius = float(getattr(env, "target_slow_radius", 0.8))
    speed_scale = (goal_dist / max(slow_radius, 1e-6)).clamp(0.0, 1.0)
    return tv_dir * env.max_speed * speed_scale


def decode_action_direct(raw_act, R, env, B, max_acc_cmd):
    _ = env, B
    act_local = raw_act[..., :3].clamp(-float(max_acc_cmd), float(max_acc_cmd))
    return torch.squeeze(act_local[:, None] @ R.transpose(1, 2), 1)


def init_camera_params(env, B, device):
    mode = getattr(env, "camera_control_mode", "learned")
    if mode == "fixed":
        return (
            torch.full((B,), float(env.fixed_camera_exposure), device=device),
            torch.full((B,), float(env.fixed_camera_gain), device=device),
        )
    if mode == "fixed_random_static":
        e_lo, e_hi = env.fixed_random_exposure_range
        g_lo, g_hi = env.fixed_random_gain_range
        return (
            torch.empty((B,), device=device).uniform_(e_lo, e_hi),
            torch.empty((B,), device=device).uniform_(g_lo, g_hi),
        )
    return (
        torch.full((B,), 0.5, device=device),
        torch.full((B,), 0.5, device=device),
    )


def update_camera_params(cam_params, exposure, gain, env):
    if cam_params is None or cam_params.shape[-1] != 2:
        raise ValueError("camera policy must output [exposure, gain]")

    mode = getattr(env, "camera_control_mode", "learned")
    if mode == "fixed":
        e = torch.full_like(exposure, float(env.fixed_camera_exposure))
        g = torch.full_like(gain, float(env.fixed_camera_gain))
        return e, g, torch.stack([e, g], -1)
    if mode == "fixed_random_static":
        hist = torch.stack([exposure.detach(), gain.detach()], -1)
        return exposure.detach(), gain.detach(), hist

    alpha = float(getattr(env, "camera_ema_alpha", 0.7))
    e_target, g_target = cam_params.unbind(-1)
    e_target = e_target.clamp(0.0, 1.0)
    g_target = g_target.clamp(0.0, 1.0)
    exposure = alpha * exposure.detach() + (1.0 - alpha) * e_target
    gain = alpha * gain.detach() + (1.0 - alpha) * g_target
    return exposure, gain, cam_params


def _stack_history_or_tensor(values):
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return values
    if len(values) == 0:
        return None
    return torch.stack(values)


def compute_camera_param_stats(exposure_seq, gain_seq):
    exposure_seq = _stack_history_or_tensor(exposure_seq)
    gain_seq = _stack_history_or_tensor(gain_seq)
    if exposure_seq is None or gain_seq is None:
        return {}
    return {
        "exposure_mean": float(exposure_seq.detach().mean().item()),
        "gain_mean": float(gain_seq.detach().mean().item()),
    }

"""Rollout helpers for grayscale active sensing."""

import torch
import torch.nn.functional as F


def render_gray_sensor(env, exposure, gain, differentiable=True):
    if not hasattr(env, "gray_camera"):
        raise RuntimeError("env.gray_camera is not configured")
    ideal, render_aux = env.render_gray_ideal(return_aux=True)
    sensor_exposure = exposure if differentiable else exposure.detach()
    sensor_gain = gain if differentiable else gain.detach()

    # Approximate image-motion magnitude. Translational optical flow grows
    # roughly with speed / scene depth, so the same vehicle speed should blur
    # more strongly when approaching a nearby gate. Geometry is detached: this
    # term shapes the camera model only and does not create a hidden depth path
    # into the navigation policy.
    speed = env.v.norm(2, -1).detach()
    geometry_depth = (render_aux or {}).get("geometry_depth")
    if isinstance(geometry_depth, torch.Tensor):
        hit_mask = geometry_depth < 99.0
        valid_depth = torch.where(
            hit_mask,
            geometry_depth,
            torch.full_like(geometry_depth, float("nan")),
        )
        char_depth = torch.nanmedian(valid_depth.flatten(1), dim=1).values
        fallback = torch.full_like(char_depth, 3.0)
        char_depth = torch.where(torch.isfinite(char_depth), char_depth, fallback)
        char_depth = char_depth.clamp_min(
            float(getattr(env, "gray_motion_depth_floor", 0.35))
        )
        motion = speed / char_depth
    else:
        char_depth = torch.full_like(speed, 1.0)
        motion = speed

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
    aux["motion_proxy"] = motion.detach()
    aux["characteristic_depth"] = char_depth.detach()
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
    if mode in {"fixed", "mean_ae", "gradient_ae"}:
        exposure = torch.full((B,), float(env.fixed_camera_exposure), device=device)
        gain = torch.full((B,), float(env.fixed_camera_gain), device=device)
    elif mode == "fixed_random_static":
        e_lo, e_hi = env.fixed_random_exposure_range
        g_lo, g_hi = env.fixed_random_gain_range
        exposure = torch.empty((B,), device=device).uniform_(e_lo, e_hi)
        gain = torch.empty((B,), device=device).uniform_(g_lo, g_hi)
    else:
        exposure = torch.full((B,), 0.5, device=device)
        gain = torch.full((B,), 0.5, device=device)

    # Camera actuator state is episode-local. The measured delay comes from the
    # selected IMX900 calibration profile. The provisional profile uses zero.
    env._camera_command_queue = []
    env._camera_requested_state = torch.stack(
        [exposure.detach(), gain.detach()],
        -1,
    )

    calibration = getattr(getattr(env, "gray_camera", None), "calibration", None)
    nominal_delay = int(getattr(calibration, "command_delay_frames", 0))
    jitter = int(getattr(calibration, "command_delay_jitter_frames", 0))
    if jitter > 0:
        delta = int(torch.randint(
            low=-jitter,
            high=jitter + 1,
            size=(1,),
            device=device,
        ).item())
    else:
        delta = 0
    env._camera_effective_delay_frames = max(nominal_delay + delta, 0)
    return exposure, gain


def _camera_delay_frames(env):
    if hasattr(env, "_camera_effective_delay_frames"):
        return int(env._camera_effective_delay_frames)
    camera = getattr(env, "gray_camera", None)
    calibration = getattr(camera, "calibration", None)
    return int(getattr(calibration, "command_delay_frames", 0))


def _apply_camera_command_delay(env, requested, current_applied):
    delay = max(_camera_delay_frames(env), 0)
    queue = getattr(env, "_camera_command_queue", None)
    if queue is None:
        queue = []
        env._camera_command_queue = queue

    queue.append(requested)
    if len(queue) <= delay:
        return current_applied
    return queue.pop(0)


def update_camera_params(cam_params, exposure, gain, env):
    if cam_params is None or cam_params.shape[-1] != 2:
        raise ValueError("camera policy must output [exposure, gain]")

    mode = getattr(env, "camera_control_mode", "learned")
    current = torch.stack([exposure, gain], -1)

    if mode == "fixed":
        requested = torch.stack(
            [
                torch.full_like(exposure, float(env.fixed_camera_exposure)),
                torch.full_like(gain, float(env.fixed_camera_gain)),
            ],
            -1,
        )
    elif mode == "fixed_random_static":
        requested = current.detach()
    else:
        alpha = float(getattr(env, "camera_smoothing_alpha", 0.0))
        e_target, g_target = cam_params.unbind(-1)
        target = torch.stack(
            [e_target.clamp(0.0, 1.0), g_target.clamp(0.0, 1.0)],
            -1,
        )
        # This is policy-command smoothing only. Real camera latency/step
        # behavior is modeled separately from the calibration profile.
        requested = alpha * current.detach() + (1.0 - alpha) * target

    env._camera_requested_state = requested
    applied = _apply_camera_command_delay(env, requested, current.detach())
    return applied[:, 0], applied[:, 1], requested


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

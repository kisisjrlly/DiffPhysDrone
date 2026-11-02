"""Queue-aligned deterministic nominal motion forecast; no future disturbances."""
import math
import torch
from render.pose import dynamics_step, update_attitude, body_angular_velocity


def predict_exposure_motion(env, pending_commands, proposed_command, camera_delay_steps=0,
                            exposure_us=0., dt=None):
    dt = float(dt or env.control_period_s)
    count = int(camera_delay_steps) + 1
    if count < 1 or not math.isfinite(dt) or dt <= 0 or not math.isfinite(float(exposure_us)) or not 0 <= float(exposure_us) <= dt * 1e6:
        raise ValueError('invalid acquisition horizon')
    p, v, a, act, r = env.p, env.v, env.a, env.act, env.R
    commands = list(pending_commands)
    r_old = r
    # Unknown future innovations are zero; do not inspect env.dg or geometry.
    for k in range(count):
        command = commands[k] if k < len(commands) else proposed_command
        r_old = r
        target_direction = env.p_target - p
        act, p, v, a = dynamics_step(r, torch.zeros_like(p), env.z_drag_coef,
            env.drag_2, env.pitch_ctl_delay, command, act, p, v, env.v_wind, a,
            env.grad_decay, dt, .5)
        r = update_attitude(r, act, target_direction, torch.exp(-env.yaw_ctl_delay * dt), 5)
    omega = body_angular_velocity(r_old, r, dt)
    return dict(position=p, velocity=v, rotation=r, angular_velocity=omega,
                effective_time_s=count * dt, exposure_start_s=count * dt - float(exposure_us)*1e-6,
                exposure_end_s=count * dt, assumed_disturbance='zero future innovation and zero unknown external force')

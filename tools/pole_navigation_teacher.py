"""Explicit privileged flight-only warm start; never camera supervision.

The actor observes only its ordinary grayscale/state inputs. Geometry is used
by this teacher to label acceleration during the common pretraining stage.
Both sensor-gradient conditions must receive the same resulting checkpoint.
"""

import copy

import torch

from rollout_ops import build_local_frame, decode_action_direct
from tools.audit_sparse_poles import plan_diagnostic_path


def bounded_probe_command(env, args, step, amplitude, duration=.9):
    """Geometry-independent lateral measurement motion; no obstacle labels."""
    horizon = 2 / args.base_control_freq + 1 / env.fixed_pitch_ctl_delay
    u = min((step / args.base_control_freq + horizon) / duration, 1.)
    s = 10*u**3 - 15*u**4 + 6*u**5
    ds = (30*u**2 - 60*u**3 + 30*u**4) / duration
    dds = (60*u - 180*u**2 + 120*u**3) / duration**2
    p, v, a = (env.p.new_zeros(env.batch_size, 3) for _ in range(3))
    p[:, 0], p[:, 1], p[:, 2] = args.simple_start_x, amplitude*s, args.simple_slit_center_z
    v[:, 1], a[:, 1] = amplitude*ds, amplitude*dds
    p, v, a = ((env.R_scene @ x[:, :, None]).squeeze(-1) for x in (p, v, a))
    command = a + 6*(p - env.p - env.v*horizon - .5*env.a*horizon**2) + 4*(v - env.v - env.a*horizon) + env.fixed_drag_linear*env.v
    frame = build_local_frame(env)
    local = (command[:, None] @ frame).squeeze(1).clamp(-args.max_acc_cmd, args.max_acc_cmd)
    return local, decode_action_direct(local, frame, env, env.batch_size, args.max_acc_cmd)


class PoleNavigationTeacher:
    def __init__(self, env, args, *, clearance_margin=.14):
        self.dt = 1 / args.base_control_freq
        self.limit = args.max_acc_cmd
        points, durations = [], []
        for idx in range(env.batch_size):
            single = copy.copy(env)
            single.cyl = env.cyl[idx:idx + 1]
            p, d = plan_diagnostic_path(single, clearance_margin=clearance_margin)
            points.append(p[0])
            durations.append(d)
        self.local_points = torch.stack(points)
        self.points = (env.R_scene[:, None] @ self.local_points[..., None]).squeeze(-1)
        self.durations = env.p.new_tensor(durations)
        self.ends = self.durations.cumsum(-1)
        self.starts = self.ends - self.durations

    def reference(self, time):
        time = torch.as_tensor(time, device=self.points.device, dtype=self.points.dtype)
        index = (time > self.ends[:, :-1]).sum(-1)
        rows = torch.arange(len(self.points), device=self.points.device)
        duration = self.durations[rows, index]
        u = ((time - self.starts[rows, index]) / duration).clamp(0, 1)
        delta = self.points[rows, index + 1] - self.points[rows, index]
        s = 10 * u**3 - 15 * u**4 + 6 * u**5
        ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
        dds = (60 * u - 180 * u**2 + 120 * u**3) / duration.square()
        return (self.points[rows, index] + s[:, None] * delta,
                ds[:, None] * delta, dds[:, None] * delta)

    def command(self, env, time):
        horizon = 2 * self.dt + 1 / env.fixed_pitch_ctl_delay
        p, v, a = self.reference(time + horizon)
        predicted_p = env.p + env.v * horizon + .5 * env.a * horizon**2
        predicted_v = env.v + env.a * horizon
        world = a + 6 * (p - predicted_p) + 4 * (v - predicted_v) + env.fixed_drag_linear * env.v
        frame = build_local_frame(env)
        local = (world[:, None] @ frame).squeeze(1).clamp(-self.limit, self.limit)
        return local, decode_action_direct(local, frame, env, env.batch_size, self.limit)

"""Optional privileged flight teaching on the actor's own closed-loop rollout.

The initial measurement actions and all camera settings have no teacher
targets. Later flight targets use simulator geometry during training only.
They are detached: learning follows the actor's actual observation history,
including earlier actions through the differentiable sensor when enabled.
This is additional navigation supervision, not a pure navigation-loss result.
"""

import torch
import torch.nn.functional as F


class LiveFlightGuidance:
    def __init__(self, env, args):
        # Optional training path; the deployed actor never constructs a teacher.
        from tools.pole_navigation_teacher import PoleNavigationTeacher
        self.teacher = PoleNavigationTeacher(env, args, clearance_margin=args.live_flight_guidance_margin)
        self.start_step = args.live_flight_guidance_start_step
        self.limit = args.max_acc_cmd
        self.horizon = 2 / args.base_control_freq + 1 / env.fixed_pitch_ctl_delay
        with torch.no_grad():
            u = torch.linspace(0, 1, 65, device=env.p.device, dtype=env.p.dtype)
            s = 10 * u**3 - 15 * u**4 + 6 * u**5
            points = self.teacher.points
            self.reference_points = (points[:, :-1, None] +
                                     s[None, None, :, None] * (points[:, 1:, None] - points[:, :-1, None]))
            self.reference_points = self.reference_points.flatten(1, 2)
            self.reference_times = (self.teacher.starts[..., None] +
                                    self.teacher.durations[..., None] * u).flatten(1, 2)

    @torch.no_grad()
    def target(self, env):
        """Track progress along the safe route, rather than an external clock.

        The actor may already have moved forward during its unlabelled phase;
        the original offline teacher assumed a stationary initial probe.
        """
        index = (self.reference_points - env.p[:, None]).square().sum(-1).argmin(-1)
        rows = torch.arange(env.batch_size, device=env.p.device)
        phase = self.reference_times[rows, index]
        position, velocity, acceleration = self.teacher.reference(phase + self.horizon)
        predicted_p = env.p + env.v * self.horizon + .5 * env.a * self.horizon**2
        predicted_v = env.v + env.a * self.horizon
        world = (acceleration + 6 * (position - predicted_p) + 4 * (velocity - predicted_v)
                 + env.fixed_drag_linear * env.v)
        # Apply exactly the same local component limits as the actor.
        from rollout_ops import build_local_frame, decode_action_direct
        frame = build_local_frame(env)
        local = (world[:, None] @ frame).squeeze(1).clamp(-self.limit, self.limit)
        return decode_action_direct(local, frame, env, env.batch_size, self.limit)


def live_flight_guidance_loss(actions, targets, valid, start_step):
    """No labels for initial motion, terminal suffixes, or camera settings."""
    supervised = valid.clone()
    supervised[:start_step] = False
    weights = supervised.to(actions.dtype)
    error = F.smooth_l1_loss(actions, targets.detach(), reduction="none").mean(-1)
    return (error * weights).sum() / weights.sum().clamp_min(1)

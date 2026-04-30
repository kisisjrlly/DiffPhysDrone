"""Minimal training helpers for the active-sensing simulation branch."""
import os
from collections import defaultdict

import torch
import wandb

from env_cuda import Env


class MetricSmoother:
    """Accumulates scalar metrics and flushes averaged values to WandB."""

    ALLOWED = {
        'loss',
        'loss_v',
        'loss_obj_avoidance',
        'loss_collide',
        'loss_d_acc',
        'loss_d_jerk',
        'loss_cam_smooth',
        'loss_diff_depth_power',
        'collision_rate',
        'success_rate',
        'goal_dist/final',
        'cam/power_mean',
        'cam/exposure_mean',
        'cam/gain_mean',
        'iter_per_sec',
        'sim_fps',
    }

    def __init__(self, args):
        self._q: dict[str, list[float]] = defaultdict(list)
        self._args = args

    def add(self, d: dict):
        for k, v in d.items():
            if k in self.ALLOWED:
                self._q[k].append(float(v))

    def flush(self, step: int):
        if not self._q:
            return
        log = {k: sum(v) / len(v) for k, v in self._q.items() if v}
        wandb.log(log, step=step)
        self._q.clear()


def periodic_tail_ops(iter_idx: int, checkpoint_dir: str, model, smoother: MetricSmoother):
    if (iter_idx + 1) % 200 == 0:
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint{iter_idx // 200:04d}.pth')
        print('save checkpoint to:', ckpt_path)
        torch.save(model.state_dict(), ckpt_path)
        wandb.save(ckpt_path)
    if (iter_idx + 1) % 25 == 0:
        smoother.flush(iter_idx + 1)


def is_save_iter(i: int) -> bool:
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0


def estimate_optimizer_steps(args) -> int:
    return max(1, int(args.num_iters))


def build_env(batch_size: int, args, device, *, eval_mode: bool = False) -> Env:
    return Env(
        batch_size,
        int(args.depth_width),
        int(args.depth_height),
        args.grad_decay,
        device,
        eval_mode=eval_mode,
        fov_x_half_tan=args.fov_x_half_tan,
        cam_angle=args.cam_angle,
        ellipsoid_a=args.drone_a if args.ellipsoid_collision else 0.0,
        ellipsoid_c=args.drone_c if args.ellipsoid_collision else 0.0,
        cam_power_baseline=args.cam_power_baseline,
        camera_control_mode=args.camera_control_mode,
        sensor_grad_mode=args.sensor_grad_mode,
        fixed_camera_power=args.fixed_camera_power,
        fixed_camera_exposure=args.fixed_camera_exposure,
        fixed_camera_gain=args.fixed_camera_gain,
        fixed_random_power_min=args.fixed_random_power_min,
        fixed_random_power_max=args.fixed_random_power_max,
        fixed_random_exposure_min=args.fixed_random_exposure_min,
        fixed_random_exposure_max=args.fixed_random_exposure_max,
        fixed_random_gain_min=args.fixed_random_gain_min,
        fixed_random_gain_max=args.fixed_random_gain_max,
        cam_exposure_t_min=args.cam_exposure_t_min,
        cam_exposure_t_span=args.cam_exposure_t_span,
        cam_exposure_eff_min=args.cam_exposure_eff_min,
        cam_exposure_eff_max=args.cam_exposure_eff_max,
        cam_iso_gain_base=args.cam_iso_gain_base,
        cam_iso_gain_scale=args.cam_iso_gain_scale,
        cam_iso_gain_gamma=args.cam_iso_gain_gamma,
        cam_shot_noise_base=args.cam_shot_noise_base,
        depth_min_valid=args.depth_min_valid,
        depth_max_range=args.depth_max_range,
        scenarios=args.scenarios,
        sun_glare_eval_slot=getattr(args, 'sun_glare_eval_slot', None) if eval_mode else None,
        diff_sensor_impl=args.diff_sensor_impl,
    )

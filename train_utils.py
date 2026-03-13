"""
Training helper / utility functions extracted from main_cuda.py.

All functions that were previously module-level helpers in the monolithic
main_cuda.py live here.  They are imported by trainer.py and the new
slim main_cuda.py entry-point.
"""
from collections import defaultdict
import math
import os
from random import normalvariate
from typing import Optional

import torch
import torch.nn.functional as F
import wandb

from env_cuda import Env
from camera_semantics import CameraSemantics


# ── Smoothed metric logging ──────────────────────────────────────────────

class MetricSmoother:
    """Accumulates scalar metrics and flushes averaged values to WandB."""

    def __init__(self, sensor_flags: dict, args):
        self._q: dict[str, list[float]] = defaultdict(list)
        self._sf = sensor_flags
        self._args = args

    # -- public API used like the old ``smooth_dict`` ---
    def add(self, d: dict):
        for k, v in d.items():
            self._q[k].append(float(v))

    def flush(self, step: int):
        if not self._q:
            return
        log = {k: sum(v) / len(v) for k, v in self._q.items() if v}
        log = self._filter(log)
        wandb.log(log, step=step)
        self._q.clear()

    # -- internal filter (was ``_filter_metrics_by_mode``) --
    def _filter(self, log: dict) -> dict:
        out = dict(log)
        sf = self._sf
        args = self._args

        cam_keys = {'loss_cam_smooth', 'loss_fov_reg', 'loss_cam_range',
                     'speed_exposure_corr', 'fov_obstacle_corr', 'power_obstacle_corr'}
        optical_keys = {'loss_blur', 'loss_noise'}
        active_keys = {'loss_active_depth_power', 'loss_active_depth_blur'}
        wall_keys = {'slit_crossed', 'slit_pass_rate', 'roll_at_wall_deg'}

        if not sf['use_camera_control']:
            for k in cam_keys | optical_keys | active_keys:
                out.pop(k, None)
        else:
            if sf['use_active_depth']:
                for k in optical_keys:
                    out.pop(k, None)
                if 'fov_obstacle_corr' in out and 'power_obstacle_corr' not in out:
                    out['power_obstacle_corr'] = out.pop('fov_obstacle_corr')
            else:
                for k in active_keys:
                    out.pop(k, None)
                if not args.enable_camera_quality_loss:
                    for k in optical_keys:
                        out.pop(k, None)

        if not args.wall_slit:
            for k in wall_keys:
                out.pop(k, None)

        out['mode/is_passive_depth'] = float(sf['use_passive_depth'])
        out['mode/is_camera_luma'] = float(sf['use_camera_luma'])
        out['mode/has_depth_channel'] = float(sf['use_depth_channel'])
        out['mode/is_active_depth'] = float(sf['use_active_depth'])
        out['mode/use_camera_control'] = float(sf['use_camera_control'])
        return out


# ── Periodic tail operations ─────────────────────────────────────────────

def periodic_tail_ops(iter_idx: int, checkpoint_dir: str, model, smoother: MetricSmoother):
    """Checkpoint saving + smoothed-scalar flush (branch-agnostic)."""
    if (iter_idx + 1) % 1000 == 0:
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint{iter_idx // 1000:04d}.pth')
        print("save checkpoint to:", ckpt_path)
        torch.save(model.state_dict(), ckpt_path)
        wandb.save(ckpt_path)
    if (iter_idx + 1) % 25 == 0:
        smoother.flush(iter_idx + 1)


# ── Small pure helpers ───────────────────────────────────────────────────

def is_save_iter(i: int) -> bool:
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0


def detach_env_graph(env: Env):
    """Truncate computation graph at TBPTT chunk boundary while keeping values."""
    snap = env.save_state()
    dsnap = {k: (v.detach() if isinstance(v, torch.Tensor) else v)
             for k, v in snap.items()}
    env.restore_state(dsnap)


def distill_coef_at_iter(iter_idx: int, args) -> float:
    """Annealing distillation weight: high early, low late."""
    if args.num_iters <= 1:
        return float(args.distill_coef)
    final_ratio = float(min(max(args.distill_final_ratio, 0.0), 1.0))
    progress = float(iter_idx) / float(max(args.num_iters - 1, 1))
    ratio = 1.0 - (1.0 - final_ratio) * progress
    return float(args.distill_coef) * ratio


def teacher_dt_like_student(cam_exposure_mean: float, use_camera: bool,
                            base_control_freq: float,
                            cam_sem: Optional[CameraSemantics] = None) -> float:
    base_dt = normalvariate(1 / base_control_freq, 0.1 / base_control_freq)
    sem = cam_sem if cam_sem is not None else CameraSemantics()
    exposure_delay = (sem.exposure_to_time(cam_exposure_mean) * 0.01) if use_camera else 0.015
    return float(base_dt + exposure_delay)


def estimate_optimizer_steps(args) -> int:
    """Estimate total optimizer steps for LR scheduler."""
    if not args.tbptt_enable:
        return max(1, args.num_iters)
    n_chunks = max(1, math.ceil(args.timesteps / max(args.tbptt_chunk_steps, 1)))
    steps_per = max(1, math.ceil(n_chunks / max(args.tbptt_chunk_accum, 1)))
    if args.hybrid_full_bptt_every > 0:
        full_iters = args.num_iters // args.hybrid_full_bptt_every
    else:
        full_iters = 0
    tbptt_iters = args.num_iters - full_iters
    return max(1, tbptt_iters * steps_per + full_iters)


# ── Environment factory ──────────────────────────────────────────────────

def build_env(batch_size: int, args, sensor_flags: dict, device) -> Env:
    """Create an Env instance from parsed args + sensor flags."""
    sf = sensor_flags
    use_passive_depth = sf['use_passive_depth']

    tw = args.tof_width if args.tof_width is not None else max(args.imx_width // max(args.tof_downsample, 1), 1)
    th = args.tof_height if args.tof_height is not None else max(args.imx_height // max(args.tof_downsample, 1), 1)
    tw, th = int(tw), int(th)

    render_w = tw if use_passive_depth else args.imx_width
    render_h = th if use_passive_depth else args.imx_height

    return Env(
        batch_size, render_w, render_h, args.grad_decay, device,
        fov_x_half_tan=args.fov_x_half_tan, single=args.single,
        gate=args.gate, ground_voxels=args.ground_voxels,
        scaffold=args.scaffold, speed_mtp=args.speed_mtp,
        random_rotation=args.random_rotation, cam_angle=args.cam_angle,
        wall_slit=args.wall_slit,
        ellipsoid_a=args.drone_a if args.ellipsoid_collision else 0.0,
        ellipsoid_c=args.drone_c if args.ellipsoid_collision else 0.0,
        tof_downsample=args.tof_downsample, tof_width=tw, tof_height=th,
        camera_preset=args.cam_realism_preset,
        cam_enable_shadow=args.cam_enable_shadow,
        cam_enable_specular=args.cam_enable_specular,
        cam_enable_distortion=args.cam_enable_distortion,
        cam_enable_flare=args.cam_enable_flare,
        cam_enable_motion_blur=args.cam_enable_motion_blur,
        cam_enable_rolling=args.cam_enable_rolling,
        cam_noise_scale=args.cam_noise_scale,
        cam_blur_scale=args.cam_blur_scale,
        cam_fog_scale=args.cam_fog_scale,
        cam_lighting_scale=args.cam_lighting_scale,
        cam_ae_target=args.cam_ae_target,
        cam_exposure_t_min=args.cam_exposure_t_min,
        cam_exposure_t_span=args.cam_exposure_t_span,
        cam_exposure_eff_min=args.cam_exposure_eff_min,
        cam_exposure_eff_max=args.cam_exposure_eff_max,
        cam_iso_gain_base=args.cam_iso_gain_base,
        cam_iso_gain_scale=args.cam_iso_gain_scale,
        cam_iso_gain_gamma=args.cam_iso_gain_gamma,
        cam_shot_noise_base=args.cam_shot_noise_base,
        diff_sensor_impl=args.diff_sensor_impl,
    )


# ── Yaw-drift rotation matrix ───────────────────────────────────────────

def make_yaw_drift_R(B: int, device):
    """Build a small random yaw-drift rotation matrix (B, 3, 3)."""
    drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
    zeros = torch.zeros_like(drift_av)
    ones = torch.ones_like(drift_av)
    return torch.stack([
        torch.cos(drift_av), -torch.sin(drift_av), zeros,
        torch.sin(drift_av),  torch.cos(drift_av), zeros,
        zeros, zeros, ones,
    ], -1).reshape(B, 3, 3)

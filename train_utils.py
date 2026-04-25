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

import torch
import wandb

from config import OPENING_SCENES
from env_cuda import Env
from rollout_ops import diff_depth_exposure_to_time


# ── Smoothed metric logging ──────────────────────────────────────────────

class MetricSmoother:
    """Accumulates scalar metrics and flushes averaged values to WandB."""

    def __init__(self, args):
        self._q: dict[str, list[float]] = defaultdict(list)
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
        args = self._args

        wall_keys = {'slit_pass_rate'}
        has_opening_scene = any(name in OPENING_SCENES for name in getattr(args, 'scenarios', []))
        if not has_opening_scene:
            for k in wall_keys:
                out.pop(k, None)

        if len(getattr(args, 'scenarios', ['random_base'])) <= 1:
            for k in list(out.keys()):
                if k == 'scene/glare_level_id' or k.startswith('scene/is_sun_glare_l'):
                    continue
                if k.startswith('scene/'):
                    out.pop(k, None)

        active_term_names = {
            name for name, _, _ in active_loss_term_specs(
                args,
                distill_coef_iter=(
                    float(args.distill_coef)
                    if getattr(args, 'enable_teacher_student_training', False)
                    else None
                ),
            )
        }
        for k in list(out.keys()):
            if k.startswith('loss_contrib/'):
                name = k.split('/', 1)[1]
                if name not in active_term_names:
                    out.pop(k, None)
            elif k.startswith('loss_share/'):
                name = k.split('/', 1)[1]
                if name == 'physics_total':
                    if 'distill' not in active_term_names:
                        out.pop(k, None)
                elif name not in active_term_names:
                    out.pop(k, None)
        return out


def _coef_enabled(coef: float, eps: float = 1e-12) -> bool:
    return abs(float(coef)) > eps


def active_loss_term_specs(args, distill_coef_iter=None):
    """Return active loss terms for the current runtime mode and coefficient setup."""
    specs = []

    def add(name: str, raw_key: str, coef: float):
        if _coef_enabled(coef):
            specs.append((name, raw_key, float(coef)))

    add('v', 'loss_v', args.coef_v)
    add('obj_avoidance', 'loss_obj_avoidance', args.coef_obj_avoidance)
    add('d_acc', 'loss_d_acc', args.coef_d_acc)
    add('d_jerk', 'loss_d_jerk', args.coef_d_jerk)
    add('collide', 'loss_collide', args.coef_collide)

    if getattr(args, 'camera_control_mode', 'learned') != 'fixed':
        add('cam_smooth', 'loss_cam_smooth', args.coef_cam_smooth)
        add('power_reg', 'loss_power_reg', args.coef_power_reg)
    add('diff_depth_power', 'loss_diff_depth_power', args.coef_diff_depth_power)
    add('diff_depth_blur', 'loss_diff_depth_blur', args.coef_diff_depth_blur)
    add('diff_depth_noise', 'loss_diff_depth_noise', args.coef_diff_depth_noise)
    add('diff_depth_fill', 'loss_diff_depth_fill', args.coef_diff_depth_fill)

    if getattr(args, 'enable_teacher_student_training', False) and distill_coef_iter is not None:
        add('distill', 'loss_distill', distill_coef_iter)

    return specs


def filter_active_loss_scalars(loss_scalars: dict, args) -> dict:
    """Keep only the raw loss scalars that are active for the current mode."""
    if not getattr(args, 'wandb_log_raw_loss_terms', False):
        return {}
    out = {}
    for _, raw_key, _ in active_loss_term_specs(
        args,
        distill_coef_iter=(
            float(args.distill_coef)
            if getattr(args, 'enable_teacher_student_training', False)
            else None
        ),
        ):
        if raw_key not in loss_scalars:
            continue
        out[f'loss_raw/{raw_key.removeprefix("loss_")}'] = float(loss_scalars[raw_key])
    return out


# ── Periodic tail operations ─────────────────────────────────────────────

def periodic_tail_ops(iter_idx: int, checkpoint_dir: str, model, smoother: MetricSmoother):
    """Checkpoint saving + smoothed-scalar flush (branch-agnostic)."""
    if (iter_idx + 1) % 200 == 0:
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint{iter_idx // 200:04d}.pth')
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


def teacher_dt_like_student(cam_exposure_mean: float, base_control_freq: float, camera_semantics=None) -> float:
    base_dt = normalvariate(1 / base_control_freq, 0.1 / base_control_freq)
    exposure_delay = float(diff_depth_exposure_to_time(
        cam_exposure_mean,
        camera_semantics=camera_semantics,
    )) * 0.01
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

def build_env(batch_size: int, args, device, *, eval_mode: bool = False) -> Env:
    """Create a diff_depth-only Env instance from parsed args."""
    dw = int(args.depth_width)
    dh = int(args.depth_height)
    cam_model_randomize = bool(args.cam_model_randomize) and (not eval_mode)

    return Env(
        batch_size, dw, dh, args.grad_decay, device,
        eval_mode=eval_mode,
        fov_x_half_tan=args.fov_x_half_tan,
        cam_angle=args.cam_angle,
        ellipsoid_a=args.drone_a if args.ellipsoid_collision else 0.0,
        ellipsoid_c=args.drone_c if args.ellipsoid_collision else 0.0,
        camera_preset=args.cam_realism_preset,
        cam_enable_specular=args.cam_enable_specular,
        cam_enable_motion_blur=args.cam_enable_motion_blur,
        cam_noise_scale=args.cam_noise_scale,
        cam_blur_scale=args.cam_blur_scale,
        cam_fog_scale=args.cam_fog_scale,
        cam_lighting_scale=args.cam_lighting_scale,
        cam_model_randomize=cam_model_randomize,
        cam_model_randomize_scale=args.cam_model_randomize_scale,
        cam_power_nominal=args.cam_power_nominal,
        camera_control_mode=args.camera_control_mode,
        sensor_grad_mode=args.sensor_grad_mode,
        fixed_camera_power=args.fixed_camera_power,
        fixed_camera_exposure=args.fixed_camera_exposure,
        fixed_camera_gain=args.fixed_camera_gain,
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
        sun_glare_levels=args.sun_glare_levels,
        sun_glare_eval_level=args.sun_glare_eval_level if eval_mode else None,
        scene_fit_profiles_path=args.scene_fit_profiles_path,
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

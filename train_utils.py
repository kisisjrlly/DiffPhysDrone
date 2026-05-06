"""Minimal training helpers for the active-sensing simulation branch."""
import os
from collections import defaultdict

import torch
import wandb

from env_cuda import Env


class MetricSmoother:
    """Accumulates scalar metrics and flushes averaged values to WandB."""

    ALLOWED_EXACT = {
        'loss',
        'collision_rate',
        'success_rate',
        'charts/goal_dist',
        'cam/power_mean',
        'cam/exposure_mean',
        'cam/gain_mean',
        'iter_per_sec',
        'sim_fps',
    }
    ALLOWED_PREFIXES = (
        'loss_contrib/',
        'loss_share/',
    )

    def __init__(self, args):
        self._q: dict[str, list[float]] = defaultdict(list)
        self._args = args

    def add(self, d: dict):
        for k, v in d.items():
            if k in self.ALLOWED_EXACT or any(k.startswith(prefix) for prefix in self.ALLOWED_PREFIXES):
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
        cam_delta_max=args.cam_delta_max,
        cam_return_rate=args.cam_return_rate,
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
        scene_layout=args.scene_layout,
        corridor_scene_sequence=args.corridor_scene_sequence,
        corridor_wall_xs=args.corridor_wall_xs,
        corridor_wall_spacing=args.corridor_wall_spacing,
        corridor_stage_release_margin=args.corridor_stage_release_margin,
        corridor_shuffle_scene_order=args.corridor_shuffle_scene_order,
        sun_glare_eval_slot=getattr(args, 'sun_glare_eval_slot', None) if eval_mode else None,
        random_rotation=args.random_rotation,
        random_rotation_max_deg=args.random_rotation_max_deg,
        simple_start_x=args.simple_start_x,
        simple_goal_x=args.simple_goal_x,
        simple_wall_x=args.simple_wall_x,
        simple_slit_center_y_min=args.simple_slit_center_y_min,
        simple_slit_center_y_max=args.simple_slit_center_y_max,
        simple_slit_half_y=args.simple_slit_half_y,
        simple_slit_half_y_min=args.simple_slit_half_y_min,
        simple_slit_half_y_max=args.simple_slit_half_y_max,
        simple_slit_effect_half_z=args.simple_slit_effect_half_z,
        simple_slit_center_z=args.simple_slit_center_z,
        simple_slit_side_effect_width_y=args.simple_slit_side_effect_width_y,
        simple_slit_side_effect_half_z=args.simple_slit_side_effect_half_z,
        simple_glare_halo_width_y=args.simple_glare_halo_width_y,
        simple_glare_halo_extra_half_z=args.simple_glare_halo_extra_half_z,
        simple_glare_halo_strength=args.simple_glare_halo_strength,
        simple_back_wall_x_min=args.simple_back_wall_x_min,
        simple_back_wall_x_max=args.simple_back_wall_x_max,
        simple_slit_cue_halo_width_y=args.simple_slit_cue_halo_width_y,
        simple_slit_cue_extra_half_z=args.simple_slit_cue_extra_half_z,
        simple_key_cue_degrade_strength=args.simple_key_cue_degrade_strength,
        simple_specular_false_depth_strength=args.simple_specular_false_depth_strength,
        diff_sensor_impl=args.diff_sensor_impl,
    )

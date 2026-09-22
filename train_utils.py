"""Training helpers for the grayscale active-sensing branch."""

import os
from collections import defaultdict

import torch
import wandb

from env_cuda import Env
from sensors.differentiable_gray_camera import build_from_args as build_gray_camera


class MetricSmoother:
    ALLOWED_EXACT = {
        "loss",
        "collision_rate",
        "success_rate",
        "charts/goal_dist",
        "cam/exposure_mean",
        "cam/gain_mean",
        "cam/saturation_fraction",
        "cam/dark_fraction",
        "cam/blur_strength",
        "cam/light_scale",
        "cam/motion_proxy",
        "cam/characteristic_depth",
        "cam/grad_norm",
        "iter_per_sec",
        "sim_fps",
    }
    ALLOWED_PREFIXES = ("loss_contrib/", "loss_share/")

    def __init__(self, args):
        self._q = defaultdict(list)
        self._args = args

    def add(self, d):
        for key, value in d.items():
            if key in self.ALLOWED_EXACT or any(key.startswith(p) for p in self.ALLOWED_PREFIXES):
                self._q[key].append(float(value))

    def flush(self, step):
        if not self._q:
            return
        wandb.log({k: sum(v) / len(v) for k, v in self._q.items() if v}, step=step)
        self._q.clear()


def periodic_tail_ops(iter_idx, checkpoint_dir, model, smoother):
    if (iter_idx + 1) % 200 == 0:
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint{iter_idx // 200:04d}.pth")
        print("save checkpoint to:", ckpt_path)
        torch.save(model.state_dict(), ckpt_path)
        wandb.save(ckpt_path)
    if (iter_idx + 1) % 25 == 0:
        smoother.flush(iter_idx + 1)


def estimate_optimizer_steps(args):
    return max(1, int(args.num_iters))


def build_env(batch_size, args, device, *, eval_mode=False):
    env = Env(
        batch_size,
        int(args.gray_width),
        int(args.gray_height),
        args.grad_decay,
        device,
        eval_mode=eval_mode,
        fov_x_half_tan=args.fov_x_half_tan,
        cam_angle=args.cam_angle,
        ellipsoid_a=args.drone_a if args.ellipsoid_collision else 0.0,
        ellipsoid_c=args.drone_c if args.ellipsoid_collision else 0.0,
        camera_control_mode=args.camera_control_mode,
        sensor_grad_mode=args.sensor_grad_mode,
        camera_ema_alpha=args.camera_ema_alpha,
        fixed_camera_exposure=args.fixed_camera_exposure,
        fixed_camera_gain=args.fixed_camera_gain,
        fixed_random_exposure_min=args.fixed_random_exposure_min,
        fixed_random_exposure_max=args.fixed_random_exposure_max,
        fixed_random_gain_min=args.fixed_random_gain_min,
        fixed_random_gain_max=args.fixed_random_gain_max,
        scenarios=args.scenarios,
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
        simple_slit_center_z=args.simple_slit_center_z,
        simple_back_wall_x_min=args.simple_back_wall_x_min,
        simple_back_wall_x_max=args.simple_back_wall_x_max,
        gray_nominal_ambient=args.gray_nominal_ambient,
        gray_nominal_diffuse=args.gray_nominal_diffuse,
        gray_dark_scale=args.gray_dark_scale,
        gray_bright_scale=args.gray_bright_scale,
        gray_background_intensity=args.gray_background_intensity,
        gray_transition_x=args.gray_transition_x,
        gray_transition_width=args.gray_transition_width,
        gray_light_jitter=args.gray_light_jitter,
        gray_texture_strength=args.gray_texture_strength,
        gray_texture_scale=args.gray_texture_scale,
        gray_motion_depth_floor=args.gray_motion_depth_floor,
    )
    env.gray_camera = build_gray_camera(args).to(device)
    env.gray_enable_noise = bool(args.gray_enable_noise)
    return env

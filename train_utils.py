"""Training helpers for the grayscale active-sensing branch."""

import os
import hashlib
import json
import subprocess
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import torch
import wandb

from env_cuda import Env
from sensors.imx900_camera import build_from_args as build_imx900_camera
from render.scene_types import DIFFERENTIABLE_SCENES


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
    ALLOWED_PREFIXES = ("loss_contrib/", "loss_share/", "flight/")

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
        camera_smoothing_alpha=args.camera_smoothing_alpha,
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
        scene_layout=args.scene_layout,
        observation_grad_mode=args.observation_grad_mode,
        gray_temporal_samples=args.gray_temporal_samples,
        intent_gate_spacing=args.intent_gate_spacing,
        flight_max_speed=args.flight_max_speed,
        pole_count=args.pole_count,
        pole_radius_min=args.pole_radius_min,
        pole_radius_max=args.pole_radius_max,
        pole_spacing=args.pole_spacing,
        pole_x_jitter=args.pole_x_jitter,
        pole_lateral_range=args.pole_lateral_range,
        pole_lateral_jitter=args.pole_lateral_jitter,
        pole_arena_half_width=args.pole_arena_half_width,
        pole_layout=args.pole_layout,
        pole_pair_near_depth=args.pole_pair_near_depth,
        pole_pair_far_depth=args.pole_pair_far_depth,
        pole_pair_depth_jitter=args.pole_pair_depth_jitter,
        pole_pair_bearing=args.pole_pair_bearing,
        pole_pair_angular_radius=args.pole_pair_angular_radius,
    )
    env.gray_camera = build_imx900_camera(args).to(device)
    env.imx900_calibration = env.gray_camera.calibration
    env.gray_enable_noise = bool(args.gray_enable_noise)
    env.control_period_s = 1 / args.base_control_freq
    if args.scene_layout in DIFFERENTIABLE_SCENES:
        period_us = 1e6 / args.base_control_freq
        if env.imx900_calibration.exposure_us_max > period_us:
            raise ValueError("maximum exposure exceeds the simulated frame period")
    return env


def write_run_provenance(directory, args):
    """Bind a development run to the actual dirty source and calibration."""
    root = Path(__file__).resolve().parent
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()
    paths = [*root.glob("*.py")]
    for folder, patterns in {
        "render": ["*.py"], "sensors": ["*.py"], "controllers": ["*.py"],
        "tools": ["*.py"], "configs": ["*.args", "calibration/*.json"],
    }.items():
        for pattern in patterns:
            paths.extend((root / folder).glob(pattern))
    snapshot = out / "source_snapshot"
    for path in paths:
        target = snapshot / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    calibration = build_imx900_camera(args).calibration
    shutil.copy2(args.imx900_calibration, out / "calibration_snapshot.json")
    limitations = []
    if not calibration.calibrated:
        limitations.append("provisional sensor profile")
    if args.scene_layout in DIFFERENTIABLE_SCENES:
        limitations.extend(["piecewise hard visibility gradients", "constant twist during exposure"])
    manifest = {
        "git_revision": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
        "calibration_sha256": hashlib.sha256(Path(args.imx900_calibration).read_bytes()).hexdigest(),
        "calibrated": calibration.calibrated,
        "source_snapshot": "source_snapshot/",
        "configuration": vars(args),
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                    "deterministic_requested": args.deterministic,
                    "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled()},
        "scene_layout": args.scene_layout, "policy_arch": args.policy_arch,
        "observation_grad_mode": args.observation_grad_mode,
        "camera_delay_unit": "control_steps", "frame_period_s": 1 / args.base_control_freq,
        "temporal_integration_samples": args.gray_temporal_samples,
        "limitations": limitations,
    }
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")

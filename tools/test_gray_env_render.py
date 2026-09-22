#!/usr/bin/env python3
"""CUDA smoke test for geometric rays -> grayscale irradiance -> camera image."""

import argparse
import os

import cv2
import numpy as np
import torch

from config import build_parser, parse_scenarios, set_global_seed, validate_args
from train_utils import build_env


def _project_args(batch_size, scenario):
    parser = build_parser()
    args = parser.parse_args([
        "--batch_size", str(batch_size),
        "--gray_width", "96",
        "--gray_height", "72",
        "--gray_nn_width", "48",
        "--gray_nn_height", "36",
        "--scenarios", scenario,
        "--simple_start_x", "-1.5",
        "--simple_goal_x", "1.5",
        "--simple_wall_x", "0.0",
        "--simple_slit_center_y_min", "-0.2",
        "--simple_slit_center_y_max", "0.2",
        "--simple_slit_half_y", "0.15",
        "--simple_slit_half_y_min", "0.15",
        "--simple_slit_half_y_max", "0.15",
        "--simple_slit_center_z", "1.5",
        "--simple_back_wall_x_min", "2.0",
        "--simple_back_wall_x_max", "2.0",
        "--camera_control_mode", "fixed",
        "--fixed_camera_exposure", "0.35",
        "--fixed_camera_gain", "0.15",
        "--wandb_disabled",
    ])
    args.scenarios = parse_scenarios(args.scenarios)
    validate_args(args)
    set_global_seed(args.seed, deterministic=True)
    return args


def _save_gray(path, image):
    arr = image.detach().float().clamp(0.0, 1.0).cpu().numpy()
    arr = np.round(arr * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not cv2.imwrite(path, arr):
        raise RuntimeError("failed to write image: %s" % path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="logs/gray_smoke")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=["nominal", "dark", "bright"],
        choices=["nominal", "dark", "bright", "dark_to_bright", "bright_to_dark"],
    )
    parser.add_argument("--exposure", type=float, default=0.35)
    parser.add_argument("--gain", type=float, default=0.15)
    cli = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    for scenario in cli.scenarios:
        args = _project_args(cli.batch_size, scenario)
        env = build_env(cli.batch_size, args, device, eval_mode=True)
        env.reset(scene_name=scenario)

        ideal, render_aux = env.render_gray_ideal(return_aux=True)
        exposure = torch.full((cli.batch_size,), cli.exposure, device=device)
        gain = torch.full((cli.batch_size,), cli.gain, device=device)
        sensor, camera_aux = env.gray_camera(
            ideal,
            exposure,
            gain,
            motion=env.v.norm(2, -1).detach(),
            enable_noise=False,
        )

        assert ideal.shape == (cli.batch_size, 1, args.gray_height, args.gray_width)
        assert sensor.shape == ideal.shape
        assert torch.isfinite(ideal).all()
        assert torch.isfinite(sensor).all()

        for idx in range(min(cli.batch_size, 4)):
            prefix = f"{scenario}_{idx:02d}"
            _save_gray(os.path.join(cli.output_dir, f"ideal_{prefix}.png"), ideal[idx, 0])
            _save_gray(os.path.join(cli.output_dir, f"sensor_{prefix}.png"), sensor[idx, 0])

        print(
            f"[{scenario}] light={float(render_aux['light_scale'].mean()):.3f} "
            f"ideal_mean={float(ideal.mean()):.3f} sensor_mean={float(sensor.mean()):.3f} "
            f"sat={float(camera_aux['saturation_fraction'].mean()):.3f} "
            f"dark={float(camera_aux['dark_fraction'].mean()):.3f}"
        )

    print("gray smoke test: PASS")
    print("saved images to:", os.path.abspath(cli.output_dir))


if __name__ == "__main__":
    main()

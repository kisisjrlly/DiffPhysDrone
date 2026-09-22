#!/usr/bin/env python3
"""Smoke-test the grayscale renderer + differentiable camera on the CUDA env.

Run from the repository root in the mappo-mpc environment:

    python tools/test_gray_env_render.py

This does not train a policy. It verifies that the existing CUDA geometry
renderer, the new ideal grayscale appearance renderer, and the differentiable
camera can execute together on the local CUDA build.
"""

import argparse
import os

import cv2
import numpy as np
import torch

from config import (
    build_parser,
    parse_diff_sensor_impl,
    parse_scenarios,
    set_global_seed,
    validate_args,
)
from train_utils import build_env


def _project_args(batch_size):
    parser = build_parser()
    args = parser.parse_args([
        "--sensor_type", "gray",
        "--batch_size", str(batch_size),
        "--gray_width", "96",
        "--gray_height", "72",
        "--gray_nn_width", "48",
        "--gray_nn_height", "36",
        "--scenarios", "glare",
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
        "--sensor_grad_mode", "detached",
        "--fixed_camera_exposure", "0.35",
        "--fixed_camera_gain", "0.15",
        "--wandb_disabled",
    ])
    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
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
    parser.add_argument("--exposure", type=float, default=0.35)
    parser.add_argument("--gain", type=float, default=0.15)
    cli = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the DiffPhysDrone environment")

    device = torch.device("cuda")
    args = _project_args(cli.batch_size)
    env = build_env(cli.batch_size, args, device, eval_mode=True)
    env.reset()

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
        _save_gray(
            os.path.join(cli.output_dir, "ideal_%02d.png" % idx),
            ideal[idx, 0],
        )
        _save_gray(
            os.path.join(cli.output_dir, "sensor_%02d.png" % idx),
            sensor[idx, 0],
        )

    print("gray smoke test: PASS")
    print("ideal shape:", tuple(ideal.shape))
    print("ideal min/mean/max:",
          float(ideal.min()), float(ideal.mean()), float(ideal.max()))
    print("sensor min/mean/max:",
          float(sensor.min()), float(sensor.mean()), float(sensor.max()))
    print("exposure_us mean:", float(camera_aux["exposure_us"].mean()))
    print("gain_factor mean:", float(camera_aux["gain_factor"].mean()))
    print("saturation fraction:", float(camera_aux["saturation_fraction"].mean()))
    print("dark fraction:", float(camera_aux["dark_fraction"].mean()))
    print("hit fraction:", float(render_aux["hit_mask"].float().mean()))
    print("saved images to:", os.path.abspath(cli.output_dir))


if __name__ == "__main__":
    main()

"""Why is there no exposure-speed interaction? Measure, do not speculate.

The interaction probe reports no coupling. This tool answers *why*, because the
answer decides whether fixing the camera model is even possible.

Motion blur displaces the image by

    blur_px = v * T * (px_per_rad) / z

and px_per_rad is fixed by the optics, z is the wall distance. So blur needs a
long shutter. But the sensor saturates when T is long. If the exposures where
blur exceeds one pixel are all saturated, the two regimes never overlap and no
speed-dependent signal can survive - not because the probe is weak, but because
the imaging chain has no operating point that carries it.

This tool prints the feasible region: for each (speed, exposure) pair, the blur
in pixels and the measured saturation, then whether any cell has both usable
blur and usable dynamic range.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import parse_args  # noqa: E402
from train_utils import build_env  # noqa: E402
from rollout_ops import render_gray_sensor  # noqa: E402
from render.exposure_history import record_history_step  # noqa: E402
from render.pose import rotation_exp  # noqa: E402

EXPOSURES_MS = (0.1, 0.2, 0.4, 0.89, 2.0, 4.0, 8.0, 20.0, 50.0)
SPEEDS = (0.3, 0.6, 1.15)
#: Blur below this is not resolvable by the policy's 48x36 input.
BLUR_PX_USABLE = 1.0
#: Above this the image is mostly clipped, so contrast carries no information.
SATURATION_USABLE = 0.10
#: Wall distance the drone approaches to; sets the worst-case blur.
APPROACH_Z = 0.35


def blur_px(speed, exposure_ms, px_per_rad, z):
    return speed * (exposure_ms * 1e-3) * px_per_rad / z


def render_cell(args, device, speed, exposure_ms, count=32, seed=7):
    env = build_env(count, args, device)
    env.reset()
    env.exposure_history.prime_from_env(env)
    common = env.slit_center_y.new_full((count,), 0.0)
    env.voxels[:, :2, 1] += (common - env.slit_center_y)[:, None]
    env.slit_center_y = common
    env.p = env.p.new_tensor([[-1.2, 0.0, 1.5]]).expand(count, -1).clone()
    env.v = env.p.new_tensor([[1.0, 0.0, 0.0]]).expand(count, -1) * float(speed)
    env.R_old = env.R.clone()
    for _ in range(3):
        env.p = env.p + env.v * env.control_period_s
        record_history_step(env, env.control_period_s)
    generator = torch.Generator(device=device).manual_seed(seed)
    env.gray_texture_phase = torch.rand(count, device=device, generator=generator) * (2 * math.pi)
    angles = torch.zeros(count, 3, device=device)
    angles[:, 2] = (torch.rand(count, device=device, generator=generator) * 2 - 1) * math.radians(0.25)
    env.R = rotation_exp(angles)
    env.R_old = env.R.clone()

    us = torch.full((count,), float(exposure_ms) * 1e3, device=device)
    exposure = env.imx900_calibration.exposure_normalized(us)
    gain = env.imx900_calibration.gain_from_factor(torch.ones(count, device=device))
    gray, _ = render_gray_sensor(env, exposure, gain)
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        gray, (int(args.gray_nn_height), int(args.gray_nn_width)))
    flat = pooled.flatten(1)
    # Spatial contrast after removing per-sample brightness: the component that
    # carries structure and therefore the component blur destroys.
    centred = flat - flat.mean(1, keepdim=True)
    return {
        "mean": float(gray.mean()),
        "saturation": float((gray >= 0.999).float().mean()),
        "contrast": float(centred.pow(2).mean().sqrt()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/gray_joint_slit_residual.args")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--count", type=int, default=32)
    cli = parser.parse_args()

    tokens = []
    with open(cli.config) as handle:
        for line in handle:
            line = line.split("#")[0].strip()
            if line:
                tokens.extend(line.split())
    sys.argv = ["diagnose"] + tokens
    args = parse_args()
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")

    # Pixels per radian at the policy's input width.
    fov_half_tan = 0.82
    width = int(args.gray_nn_width)
    px_per_rad = (width / 2.0) / fov_half_tan
    period_ms = 1000.0 * float(args.base_control_freq and 1.0 / args.base_control_freq)

    rows = []
    for speed in SPEEDS:
        for exposure_ms in EXPOSURES_MS:
            stats = render_cell(args, device, speed, exposure_ms, count=cli.count)
            rows.append({
                "speed_m_s": speed,
                "exposure_ms": exposure_ms,
                "blur_px": blur_px(speed, exposure_ms, px_per_rad, APPROACH_Z),
                "shutter_over_period": exposure_ms / period_ms,
                **stats,
            })

    usable = [r for r in rows
              if r["blur_px"] >= BLUR_PX_USABLE
              and r["saturation"] <= SATURATION_USABLE]

    print(f"px per radian (width {width}) : {px_per_rad:.2f}")
    print(f"assumed approach distance     : {APPROACH_Z} m")
    print(f"usable needs blur >= {BLUR_PX_USABLE} px AND saturation <= {SATURATION_USABLE:.0%}")
    print()
    print(f"{'v':>5} {'T ms':>6} {'blur_px':>8} {'T/per':>6} {'mean':>6} {'sat':>6} {'contrast':>9} {'usable':>7}")
    for row in rows:
        ok = (row["blur_px"] >= BLUR_PX_USABLE
              and row["saturation"] <= SATURATION_USABLE)
        print(f"{row['speed_m_s']:>5.2f} {row['exposure_ms']:>6.2f} {row['blur_px']:>8.2f} "
              f"{row['shutter_over_period']:>6.2f} {row['mean']:>6.3f} "
              f"{row['saturation']:>6.3f} {row['contrast']:>9.4f} {'YES' if ok else '.':>7}")
    print()
    if usable:
        print(f"feasible cells: {len(usable)} -> interaction is reachable in principle")
    else:
        print("feasible cells: 0 -> no operating point has both resolvable blur and "
              "usable dynamic range.")
        print("  Blur needs a long shutter; a long shutter saturates. The two regimes "
              "never overlap, so no exposure-speed interaction can be measured here.")

    payload = {
        "px_per_rad": px_per_rad,
        "approach_z_m": APPROACH_Z,
        "blur_px_usable": BLUR_PX_USABLE,
        "saturation_usable": SATURATION_USABLE,
        "control_period_ms": period_ms,
        "rows": rows,
        "feasible_cells": len(usable),
        "interaction_reachable": bool(usable),
    }
    if cli.out_dir:
        os.makedirs(cli.out_dir, exist_ok=True)
        path = os.path.join(cli.out_dir, "interaction_reachability.json")
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

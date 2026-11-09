"""How much could a camera policy possibly help? Measure the perception error.

The bounded route learns a camera correction that feeds a fixed reader, and the
reader feeds a fixed controller. For that camera head to have anything to learn,
the reader's *continuous* estimate of the relative slit offset must be
noticeably worse than the offset the controller needs to resolve. If the reader
is already far more accurate than the crossing tolerance, no exposure change can
improve the task, and the C-vs-A comparison is decided in advance.

This tool measures the quantity that decides it:

    rmse(exposure, speed, distance) = error of a fixed non-privileged reader
                                      at estimating the lateral slit offset, in
                                      metres,

and compares it to the crossing tolerance, which for the single-wall slit is

    tolerance = simple_slit_half_y - drone_radius = 0.15 - 0.075 = 0.075 m.

The reader is deliberately the same ridge estimator used by the interaction
probe: fixed alpha, train-only normalisation, 48x36 input, no geometry, no
depth, no exposure value. It is a stand-in for "a competent fixed reader", not
a claim about the CNN the route will use. If a *linear* reader on the policy's
own input is already accurate to a few millimetres, a CNN will not be the
bottleneck and the camera head has no headroom either.
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

EXPOSURES_MS = (0.1, 0.2, 0.4, 0.89, 2.0, 8.0, 20.0, 50.0)
SPEEDS = (0.6, 1.15)
DISTANCES = (1.2, 0.8, 0.5)
#: Lateral offsets sampled uniformly in +/- this range, metres.
OFFSET_RANGE = 0.30


def frames_for(args, device, *, speed, exposure_ms, distance, count, seed, split):
    """Render one cell and return (features, true lateral offset)."""
    env = build_env(count, args, device)
    env.reset()
    env.exposure_history.prime_from_env(env)

    # Pin the slit centre so the label is the drone's own lateral offset.
    common = env.slit_center_y.new_full((count,), 0.0)
    env.voxels[:, :2, 1] += (common - env.slit_center_y)[:, None]
    env.slit_center_y = common

    generator = torch.Generator(device=device).manual_seed(seed + (7 if split == "test" else 0))
    offset = (torch.rand(count, device=device, generator=generator) * 2 - 1) * OFFSET_RANGE

    env.p = torch.stack([
        torch.full((count,), -float(distance), device=device),
        offset,
        torch.full((count,), 1.5, device=device),
    ], dim=-1)
    env.v = env.p.new_tensor([[1.0, 0.0, 0.0]]).expand(count, -1) * float(speed)
    env.R_old = env.R.clone()
    for _ in range(3):
        env.p = env.p + env.v * env.control_period_s
        record_history_step(env, env.control_period_s)

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
    return pooled.flatten(1), offset


def _ridge_fit_predict(train_x, train_y, test_x, generator):
    """Ridge regression solved in the dual, with alpha chosen on held-out data.

    The primal system is hopeless here: the input is 1728-dimensional and the
    sample count is in the hundreds, so ``X^T X + alpha I`` with a small alpha is
    rank deficient and the solution explodes - an earlier version of this tool
    produced RMSE of 60 cm and 190 cm on a target that only spans +/-30 cm, i.e.
    worse than predicting the mean. Solving the n-by-n dual instead, and
    selecting alpha by validation inside the training split, is stable and does
    not need the task-dependent tuning that made the first numbers meaningless.
    """
    mean = train_x.mean(0, keepdim=True)
    std = train_x.std(0, keepdim=True).clamp_min(1e-6)
    train_a = (train_x - mean) / std
    test_b = (test_x - mean) / std
    target = train_y.mean()
    centred = train_y - target

    count = train_a.shape[0]
    permutation = torch.randperm(count, generator=torch.Generator().manual_seed(
        int(generator.initial_seed() % (2**31))))
    split = int(count * 0.8)
    fit_idx, val_idx = permutation[:split], permutation[split:]

    k_fit = train_a[fit_idx] @ train_a[fit_idx].T
    k_val = train_a[val_idx] @ train_a[fit_idx].T
    eye = torch.eye(len(fit_idx), device=train_a.device)

    best_alpha, best_val = None, float("inf")
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        scale = alpha * k_fit.diagonal().mean().clamp_min(1e-9)
        coefficients = torch.linalg.solve(k_fit + scale * eye, centred[fit_idx])
        prediction = k_val @ coefficients
        error = float((prediction - centred[val_idx]).pow(2).mean().sqrt())
        if error < best_val:
            best_alpha, best_val = scale, error

    k_all = train_a @ train_a.T
    coefficients = torch.linalg.solve(
        k_all + best_alpha * torch.eye(count, device=train_a.device), centred)
    return (test_b @ train_a.T) @ coefficients + target


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="/tmp/route_single_wall.args")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distances", default=None,
                        help="comma separated metres; default is the module grid")
    parser.add_argument("--exposures", default=None,
                        help="comma separated milliseconds; default is the module grid")
    parser.add_argument("--speeds", default=None, help="comma separated m/s")
    cli = parser.parse_args()

    tokens = []
    with open(cli.config) as handle:
        for line in handle:
            line = line.split("#")[0].strip()
            if line:
                tokens.extend(line.split())
    sys.argv = ["headroom"] + tokens
    args = parse_args()
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")

    global DISTANCES, EXPOSURES_MS, SPEEDS
    if cli.distances:
        DISTANCES = tuple(float(v) for v in cli.distances.split(","))
    if cli.exposures:
        EXPOSURES_MS = tuple(float(v) for v in cli.exposures.split(","))
    if cli.speeds:
        SPEEDS = tuple(float(v) for v in cli.speeds.split(","))

    tolerance = float(args.simple_slit_half_y) - float(args.drone_c)
    # A reader that always answers the mean achieves exactly this.
    baseline_rmse = OFFSET_RANGE / math.sqrt(3.0)

    rows = []
    for speed in SPEEDS:
        for distance in DISTANCES:
            for exposure_ms in EXPOSURES_MS:
                train_x, train_y = frames_for(args, device, speed=speed,
                                              exposure_ms=exposure_ms, distance=distance,
                                              count=cli.count, seed=cli.seed, split="train")
                test_x, test_y = frames_for(args, device, speed=speed,
                                            exposure_ms=exposure_ms, distance=distance,
                                            count=cli.count, seed=cli.seed, split="test")
                pred = _ridge_fit_predict(train_x, train_y, test_x,
                                          torch.Generator(device=device).manual_seed(cli.seed + 11))
                rmse = float((pred - test_y).pow(2).mean().sqrt())
                rows.append({
                    "speed_m_s": speed,
                    "distance_m": distance,
                    "exposure_ms": exposure_ms,
                    "rmse_m": rmse,
                    "rmse_over_tolerance": rmse / tolerance,
                })

    print(f"crossing tolerance (0.15 half-width - 0.075 radius) : {tolerance*100:.1f} cm")
    print(f"predict-the-mean baseline RMSE                       : {baseline_rmse*100:.1f} cm")
    print()
    for speed in SPEEDS:
        print(f"--- speed {speed} m/s ---")
        print(f"{'dist m':>7} " + " ".join(f"{t:>7.2f}" for t in EXPOSURES_MS))
        for distance in DISTANCES:
            vals = [r for r in rows if r["speed_m_s"] == speed and r["distance_m"] == distance]
            print(f"{distance:>7.1f} " + " ".join(f"{r['rmse_m']*100:>7.2f}" for r in vals))
        print()

    worst = max(r["rmse_m"] for r in rows)
    best = min(r["rmse_m"] for r in rows)
    print(f"RMSE range over the whole grid : {best*100:.2f} cm .. {worst*100:.2f} cm")
    print(f"worst / tolerance              : {worst/tolerance:.2f}x  (want << 1)")

    # Does exposure change the error at all, holding speed and distance fixed?
    spread = []
    for speed in SPEEDS:
        for distance in DISTANCES:
            vals = [r["rmse_m"] for r in rows
                    if r["speed_m_s"] == speed and r["distance_m"] == distance]
            spread.append(max(vals) - min(vals))
    print(f"max RMSE variation across exposures (same pose) : {max(spread)*100:.2f} cm")

    payload = {
        "tolerance_m": tolerance,
        "baseline_rmse_m": baseline_rmse,
        "offset_range_m": OFFSET_RANGE,
        "exposures_ms": list(EXPOSURES_MS),
        "speeds_m_s": list(SPEEDS),
        "distances_m": list(DISTANCES),
        "worst_rmse_m": worst,
        "best_rmse_m": best,
        "worst_over_tolerance": worst / tolerance,
        "max_rmse_variation_across_exposures_m": max(spread),
        "rows": rows,
    }
    if cli.out_dir:
        os.makedirs(cli.out_dir, exist_ok=True)
        path = os.path.join(cli.out_dir, "perception_headroom.json")
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

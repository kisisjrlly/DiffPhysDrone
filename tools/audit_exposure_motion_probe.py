"""Exposure x speed readout grid, with no privileged information.

Why this tool exists
--------------------
Task loss in this simulator is built entirely from ground-truth geometry
(velocity tracking, clearance history, terminal position). Images enter the loss
only through the actions the policy chooses. So perturbing exposure on a *fixed*
flight controller changes nothing in the forward trajectory and nothing in the
loss - exactly zero, by construction. Measuring "does exposure matter" with task
loss therefore returns a guaranteed false negative, which is the failure mode
the plan's section 8 patch was written to remove.

This tool instead asks a question the images can actually answer: with a fixed
linear readout fitted on training data and evaluated on held-out data, can a
single frame tell which side of the slit the drone is on, and does that ability
depend on exposure in a way that interacts with speed?

The readout never sees the true slit centre, depth, exposure value, velocity or
any simulator state. It sees one flattened grayscale frame. That is what makes
it non-privileged.

Registered outcomes (plan section 9, patch 3)
---------------------------------------------
R1  interaction exists    peak exposure moves down as speed rises
R2  no interaction        peaks pinned near the shortest exposure for every
                          speed, or pinned to the ceiling, with no trend
R3  long exposure worse   readout degrades monotonically along the exposure axis

R2 and R3 are deliberately separate. R2 means there is nothing to learn; R3
means there is an interaction but its useful direction is blocked by noise or
saturation. Calling R3 "R2" would wrongly terminate the experiment; calling R2
"R3" would train a camera head that cannot matter.

Thresholds are part of the protocol lock, not chosen after seeing the numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import parse_args  # noqa: E402
from render.exposure_history import record_history_step  # noqa: E402
from render.intent_scene import render_exposure  # noqa: E402
from rollout_ops import render_gray_sensor  # noqa: E402
from train_utils import build_env  # noqa: E402

#: Exposures the plan registers, in milliseconds.
#:
#: The original grid was (0.89, 8, 20, 50). A run on that grid put the readout
#: peak at 0.89 ms - the *shortest* value tested - at every speed, so the grid
#: never bracketed the optimum and the probe could not answer the coupling
#: question (verdict R4_GRID_TRUNCATED). 0.89 ms was chosen as the equivalent of
#: the legacy E=0.10, not because it is physically special; the research profile
#: allows 0.1-50 ms. The grid is therefore extended down to the profile floor so
#: the optimum is interior, which is a precondition for reading a peak position
#: at all.
EXPOSURES_MS = (0.1, 0.2, 0.4, 0.89, 8.0, 20.0, 50.0)
#: Speeds the plan registers, in m/s.
SPEEDS = (0.3, 0.6, 1.15)
#: Class separation in slit centre, metres. Matched to the training task.
CENTER_DELTA = 0.05


def ridge_probe(train_x, train_y, test_x, test_y):
    """Fixed ridge strength, train-only normalization, no test-time selection.

    Same estimator as ``tools/audit_slit_precision.py`` so the two audits are
    directly comparable.
    """
    train_x, test_x = train_x.double(), test_x.double()
    mean = train_x.mean(0)
    scale = train_x.std(0).clamp_min(0.005)
    a = (train_x - mean) / scale
    b = (test_x - mean) / scale
    kernel = a @ a.T
    alpha = (0.1 * kernel.diagonal().mean()).clamp_min(1e-9)
    coeff = torch.linalg.solve(kernel + torch.eye(len(a), device=a.device) * alpha,
                              train_y.double())
    return b @ a.T @ coeff


def _accuracy(scores, labels):
    correct = ((scores > 0) == (labels > 0)).double()
    correct = torch.where(scores == 0, torch.full_like(correct, 0.5), correct)
    return float(correct.mean())


def _log_loss(scores, labels):
    probability = torch.sigmoid(scores).clamp(1e-6, 1 - 1e-6)
    target = (labels > 0).double()
    return float(-(target * probability.log() + (1 - target) * (1 - probability).log()).mean())


@torch.no_grad()
def build_frames(args, device, speed, exposure_ms, *, count, seed, split):
    """Render `count` frames at one (speed, exposure) cell.

    Class 0 / class 1 differ only in where the slit centre sits, plus the
    nuisance factors the policy must be robust to. Geometry is otherwise
    identical, so the readout cannot exploit a shortcut.
    """
    env = build_env(count, args, device)
    env.reset()
    env.exposure_history.prime_from_env(env)

    sign = torch.arange(count, device=device) % 2
    offset = (sign * 2 - 1) * (CENTER_DELTA / 2)

    # ``reset`` draws a random slit centre per sample; that draw would otherwise
    # be a nuisance far larger than the cue we want the readout to find. Pin the
    # centre to a common value first, then displace it by the class offset so
    # the only difference between the two classes is that displacement.
    common = env.slit_center_y.new_full((count,), 0.0)
    shift_to_common = common - env.slit_center_y
    env.voxels[:, :2, 1] += shift_to_common[:, None]
    env.voxels[:, :2, 1] += offset[:, None]
    env.slit_center_y = common + offset

    # Start at a fixed approach distance and fly straight at the wall, so speed
    # is the only motion variable.
    env.p = env.p.new_tensor([[-1.2, 0.0, 1.5]]).expand(count, -1).clone()
    direction = env.p.new_tensor([[1.0, 0.0, 0.0]])
    env.v = direction.expand(count, -1) * float(speed)
    env.R_old = env.R.clone()

    # Build a short consistent history so the shutter has real motion to
    # integrate; without this the history path would see a stationary world.
    for _ in range(3):
        env.p = env.p + env.v * env.control_period_s
        record_history_step(env, env.control_period_s)

    # Paired nuisance draws: identical across exposures and speeds so the grid
    # compares like with like.
    generator = torch.Generator(device=device).manual_seed(seed)
    env.gray_texture_phase = torch.rand(count, device=device, generator=generator) * (2 * math.pi)
    yaw = (torch.rand(count, device=device, generator=generator) * 2 - 1) * math.radians(0.25)
    angles = torch.zeros(count, 3, device=device)
    angles[:, 2] = yaw
    from render.pose import rotation_exp
    env.R = rotation_exp(angles)
    env.R_old = env.R.clone()

    us = torch.full((count,), float(exposure_ms) * 1e3, device=device)
    exposure = env.imx900_calibration.exposure_normalized(us)
    gain = env.imx900_calibration.gain_from_factor(torch.ones(count, device=device))
    gray, _ = render_gray_sensor(env, exposure, gain)

    # The policy's view: 48x36, exactly what a learned camera head would see.
    pooled = torch.nn.functional.adaptive_avg_pool2d(
        gray, (int(args.gray_nn_height), int(args.gray_nn_width))
    )
    labels = sign.float() * 2 - 1
    return pooled.flatten(1), labels, float(gray.mean()), float((gray >= 0.999).float().mean())


def audit(args, device, *, train_per_class, test_per_class, seed):
    cells = []
    for speed in SPEEDS:
        for exposure_ms in EXPOSURES_MS:
            train_x, train_y, mean_br, sat = build_frames(
                args, device, speed, exposure_ms,
                count=train_per_class, seed=seed, split="train",
            )
            test_x, test_y, _, _ = build_frames(
                args, device, speed, exposure_ms,
                count=test_per_class, seed=seed + 7919, split="test",
            )
            scores = ridge_probe(train_x, train_y, test_x, test_y)
            cells.append({
                "speed_m_s": speed,
                "exposure_ms": exposure_ms,
                "accuracy": _accuracy(scores, test_y),
                "log_loss": _log_loss(scores, test_y),
                "mean_brightness": mean_br,
                "saturation_fraction": sat,
                "frame_contrast": float(train_x.std(dim=1).mean()),
            })

    return cells


def classify(cells, *, slope_threshold, monotone_margin):
    """Apply the plan's registered R1/R2/R3 thresholds."""
    speeds = sorted({c["speed_m_s"] for c in cells})
    per_speed_peak = {}
    for speed in speeds:
        row = [c for c in cells if c["speed_m_s"] == speed]
        best = max(row, key=lambda c: c["accuracy"])
        per_speed_peak[speed] = {
            "peak_exposure_ms": best["exposure_ms"],
            "peak_accuracy": best["accuracy"],
            "exposures": {c["exposure_ms"]: c["accuracy"] for c in row},
            "log_losses": {c["exposure_ms"]: c["log_loss"] for c in row},
        }

    # Slope of peak exposure against speed (ms per m/s).
    xs = torch.tensor(speeds, dtype=torch.float64)
    ys = torch.tensor([per_speed_peak[s]["peak_exposure_ms"] for s in speeds],
                      dtype=torch.float64)
    x_centered = xs - xs.mean()
    denominator = (x_centered ** 2).sum().clamp_min(1e-12)
    slope = float(((x_centered) * (ys - ys.mean())).sum() / denominator)

    # Does the readout get monotonically worse with longer exposure at every speed?
    monotone_worse = True
    for speed in speeds:
        row = {c["exposure_ms"]: c["accuracy"]
               for c in sorted(cells, key=lambda c: c["exposure_ms"]) if c["speed_m_s"] == speed}
        ordered = [row[t] for t in sorted(row)]
        if any(ordered[i + 1] > ordered[i] + monotone_margin for i in range(len(ordered) - 1)):
            monotone_worse = False
            break

    # Pinned at the shortest exposure for every speed?
    pinned_short = all(per_speed_peak[s]["peak_exposure_ms"] == min(EXPOSURES_MS) for s in speeds)
    # Or pinned at the ceiling for every speed?
    pinned_long = all(per_speed_peak[s]["peak_exposure_ms"] == max(EXPOSURES_MS) for s in speeds)

    # ------------------------------------------------------------------ verdict
    # Two orthogonal axes, deliberately kept separate:
    #
    #   interaction  - does the *location* of the peak move with speed? This is
    #                  the only thing that says a camera head has something
    #                  speed-dependent to learn. It is read off the slope alone.
    #   penalty      - does the readout get worse as exposure lengthens? This is
    #                  the mechanism (saturation, blur) and says nothing about
    #                  coupling.
    #
    # An earlier version of this function labelled a short-pinned peak "R3: long
    # exposure worse" and then argued that interaction could still exist. That
    # was a logical error: "peak pinned at the shortest exposure for every
    # speed" is precisely the negation of "peak moves with speed". If the
    # optimum does not depend on speed, there is no exposure-motion interaction
    # to learn, whatever the mechanism for the penalty. R2 and R3 therefore both
    # mean *no interaction* and both block training.
    #
    # A third, distinct case matters for honesty: if the peak sits at the
    # shortest exposure *tested*, the true optimum may lie below the grid. That
    # is truncation, not decoupling, and it means the probe has not bracketed
    # the optimum and cannot answer the question either way.
    interaction_present = slope <= -slope_threshold

    if interaction_present:
        verdict = "R1_INTERACTION_EXISTS"
        blocks_training = False
        meaning = ("peak exposure falls as speed rises: exposure and motion are coupled, "
                   "so there is something for a camera head to learn."
                   + (" The readout also degrades with longer exposure, so the usable "
                      "exposure range sits at the short end." if monotone_worse else ""))
    elif pinned_short:
        verdict = "R4_GRID_TRUNCATED"
        blocks_training = True
        meaning = ("the peak sits at the shortest exposure tested at every speed, so the "
                   "true optimum is at or below 0.89 ms and the grid does not bracket it. "
                   "This is truncation, not evidence of decoupling: extend the grid down "
                   "to the profile floor (0.1 ms) before drawing any conclusion.")
    elif monotone_worse:
        verdict = "R2_NO_INTERACTION_SATURATION"
        blocks_training = True
        meaning = ("the peak position does not move with speed while the readout degrades "
                   "monotonically along the exposure axis: exposure and motion are "
                   "decoupled, and long exposure loses for a speed-independent reason "
                   "(saturation). There is no speed-dependent camera policy to learn.")
    elif pinned_long:
        verdict = "R3_NO_INTERACTION_FLAT"
        blocks_training = True
        meaning = ("peaks are pinned at the ceiling at every speed and the readout does not "
                   "degrade along the exposure axis: the grid is degenerate, it does not "
                   "separate the exposures at all.")
    else:
        verdict = "R3_NO_INTERACTION_FLAT"
        blocks_training = True
        meaning = ("peak position neither trends with speed nor degrades with exposure: the "
                   "grid is flat and carries no usable signal.")

    return {
        "per_speed": per_speed_peak,
        "peak_vs_speed_slope_ms_per_m_s": slope,
        "interaction_present": interaction_present,
        "long_exposure_penalty": monotone_worse,
        "monotone_worse": monotone_worse,
        "pinned_short": pinned_short,
        "pinned_long": pinned_long,
        "verdict": verdict,
        "blocks_training": blocks_training,
        "meaning": meaning,
        "thresholds": {"slope_threshold": slope_threshold,
                       "monotone_margin": monotone_margin},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train_per_class", type=int, default=64)
    parser.add_argument("--test_per_class", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slope_threshold", type=float, default=1.0,
                        help="ms of peak exposure per m/s of speed; part of the protocol lock")
    parser.add_argument("--monotone_margin", type=float, default=0.01)
    argv = sys.argv[1:]
    tool_argv, rest, i = [], [], 0
    tool_flags = {"--out_dir", "--device", "--train_per_class", "--test_per_class",
                  "--seed", "--slope_threshold", "--monotone_margin"}
    while i < len(argv):
        if argv[i] in tool_flags:
            tool_argv.extend(argv[i:i + 2])
            i += 2
        else:
            rest = argv[i:]
            break
    cli = parser.parse_args(tool_argv)
    sys.argv = ["audit_exposure_motion_probe"] + rest
    args = parse_args()
    device = torch.device(cli.device)
    if min(cli.train_per_class, cli.test_per_class) < 32:
        raise ValueError("at least 32 independent train and test samples per class")

    cells = audit(args, device, train_per_class=cli.train_per_class,
                  test_per_class=cli.test_per_class, seed=cli.seed)
    result = classify(cells, slope_threshold=cli.slope_threshold,
                      monotone_margin=cli.monotone_margin)

    report = {
        "protocol": {
            "exposures_ms": list(EXPOSURES_MS),
            "speeds_m_s": list(SPEEDS),
            "center_delta_m": CENTER_DELTA,
            "readout": "ridge_probe, fixed alpha, train-only normalization",
            "privileged_inputs": [],
            "task_loss_used": False,
            "seed": cli.seed,
            "train_per_class": cli.train_per_class,
            "test_per_class": cli.test_per_class,
        },
        "cells": cells,
        "classification": result,
    }
    os.makedirs(cli.out_dir, exist_ok=True)
    out = os.path.join(cli.out_dir, "exposure_motion_probe.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"{'speed':>6} {'T ms':>7} {'acc':>7} {'logloss':>9} {'mean':>7} {'sat':>6} {'contrast':>9}")
    for cell in cells:
        print(f"{cell['speed_m_s']:6.2f} {cell['exposure_ms']:7.2f} {cell['accuracy']:7.3f} "
              f"{cell['log_loss']:9.4f} {cell['mean_brightness']:7.4f} "
              f"{cell['saturation_fraction']:6.3f} {cell['frame_contrast']:9.5f}")
    print()
    print(f"peak vs speed slope : {result['peak_vs_speed_slope_ms_per_m_s']:+.3f} ms/(m/s) "
          f"(threshold {cli.slope_threshold})")
    print(f"monotone worse      : {result['monotone_worse']}")
    print(f"verdict             : {result['verdict']}")
    print(f"  {result['meaning']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

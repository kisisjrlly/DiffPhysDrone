"""Compare the two shutter-trajectory models on recorded motion.

The plan asks a specific question that a sampling-density check cannot answer:
does the *trajectory model* change the rendered image? Nine midpoints and
eighteen midpoints are both smooth approximations of the same assumed path, so
comparing them only measures integration error, never the assumption itself
(plan section 5.3, patch 1).

This tool therefore reconstructs the same frame twice from the same recorded
motion - once with end-velocity extrapolation, once with the recorded-history
integral - and reports three quantities:

``image_mae``        mean absolute difference between the two integrations, and
                     the same number as a fraction of mean scene brightness;
``shift_px``         an estimated global displacement between them, so the
                     pixel difference can be checked against the analytic
                     ``v*T/z*px_per_rad`` expectation;
``analytic_agreement`` on a synthetic strictly-constant-velocity segment the two
                     models must agree to ~0. This is the real consistency check:
                     it verifies the implementation, not the assumption.

It also reports ``shutter_over_period``, because a shutter shorter than one
control period cannot express a difference between the two models at all - the
recorded history is piecewise linear within that period and the trapezoidal
integral collapses to ``v_end * tau``. When that ratio is below one the tool
says so explicitly instead of reporting a vacuous zero.

Images are compared with ``enable_noise=False`` so noise cannot mask or fake a
difference, and the sample count is fixed at nine midpoints regardless of task
outcome.
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
from train_utils import build_env  # noqa: E402

#: Exposures the plan asks about, in milliseconds.
DEFAULT_EXPOSURES_MS = (0.89, 8.0, 20.0, 50.0)


def _drive(env, accelerations, dt):
    """Advance the env with a scripted world-frame velocity sequence.

    Position is updated *before* recording so the stored ``(p, v)`` pairs are
    mutually consistent - recording a velocity that has not yet been integrated
    produces a history that disagrees with the simulator's own pose.
    """
    for velocity in accelerations:
        env.v = env.v.new_tensor([velocity])
        env.p = env.p + env.v * dt
        record_history_step(env, dt)


def _constant_velocity_segment(env, speed=1.0, steps=4):
    """Strictly constant world-frame motion; both models must agree here."""
    return [[0.0, 0.0, -speed]] * steps


def _accelerating_segment(env, steps=4):
    """Velocity ramps so the shutter spans genuinely curved motion."""
    return [[0.0, 0.0, -1.0 * (k + 1)] for k in range(steps)]


def _shift_px(a, b):
    """Global displacement between two grayscale frames, in pixels.

    Uses a brightness-weighted centroid of the row/column profiles, which is
    enough to check order of magnitude against ``v*T/z*px_per_rad``. Phase
    correlation would be more precise but the images here are dominated by a
    large low-frequency background, so a centroid is the more robust estimate
    of a *global* shift.
    """
    def profile(image, axis):
        weight = image.sum(dim=axis)
        weight = weight.clamp_min(0.0)
        total = weight.sum(-1, keepdim=True).clamp_min(1e-9)
        coordinate = torch.arange(weight.shape[-1], device=weight.device, dtype=weight.dtype)
        return (weight * coordinate).sum(-1) / total.squeeze(-1)

    dx = (profile(b, 2) - profile(a, 2)).abs()
    dy = (profile(b, 3) - profile(a, 3)).abs()
    return math.hypot(float(dx.mean()), float(dy.mean()))


def _render_pair(env, exposure_ms, ):
    """Render one frame under both trajectory models, noise disabled."""
    previous = getattr(env, "gray_enable_noise", True)
    env.gray_enable_noise = False
    try:
        us = torch.full((env.batch_size,), float(exposure_ms) * 1e3, device=env.device)
        exposure = env.imx900_calibration.exposure_normalized(us)
        history = render_exposure(env, exposure, return_aux=True, trajectory="history")[0]
        extrapolated = render_exposure(env, exposure, return_aux=True, trajectory="extrapolate")[0]
    finally:
        env.gray_enable_noise = previous
    return history, extrapolated


def audit(args, device):
    env = build_env(1, args, device)
    env.reset()
    env.exposure_history.prime_from_env(env)
    dt = env.control_period_s
    period_ms = dt * 1e3
    px_per_rad = 0.5 * args.gray_width / math.atan(args.fov_x_half_tan)
    # ~34.9 px/rad at the policy's 48-wide view, ~69.9 px/rad at the 96-wide
    # render this tool compares. The plan quotes 29.27 for 48 px; the yardstick
    # here follows the actual rendered width, so ``analytic_shift_px`` is in the
    # units of the images being differenced.
    depth_floor = float(getattr(args, "gray_motion_depth_floor", 0.35))

    records = []
    analytic = {}

    # ---- analytic agreement: constant velocity must give ~0 for both models.
    _drive(env, _constant_velocity_segment(env), dt)
    for exposure_ms in DEFAULT_EXPOSURES_MS:
        history, extrapolated = _render_pair(env, exposure_ms)
        analytic[f"{exposure_ms:g}"] = float((history - extrapolated).abs().mean())

    # ---- accelerating segment: this is where the models can differ.
    _drive(env, _accelerating_segment(env), dt)
    for exposure_ms in DEFAULT_EXPOSURES_MS:
        history, extrapolated = _render_pair(env, exposure_ms)
        mean_brightness = float(history.mean())
        mae = float((history - extrapolated).abs().mean())
        speed = float(env.v.norm())
        analytic_offset_m = speed * exposure_ms * 1e-3
        records.append({
            "exposure_ms": exposure_ms,
            "shutter_over_period": exposure_ms * 1e-3 / dt,
            "spans_multiple_intervals": bool(exposure_ms * 1e-3 / dt > 1.0),
            "image_mae": mae,
            "mean_brightness": mean_brightness,
            "image_mae_relative": mae / max(mean_brightness, 1e-9),
            "shift_px": _shift_px(history, extrapolated),
            "analytic_shift_px": analytic_offset_m / max(depth_floor, 1e-6) * px_per_rad,
            "speed_m_s": speed,
        })

    # ---- verdict, stated in the plan's own terms.
    distinguishable = [r for r in records if r["image_mae_relative"] >= 0.05]
    constant_ok = all(v < 1e-6 for v in analytic.values())
    max_ratio = max(r["shutter_over_period"] for r in records)
    spans = any(r["spans_multiple_intervals"] for r in records)

    if not spans:
        verdict = "TRAJECTORY_MODELS_IDENTICAL_BY_CONSTRUCTION"
        explanation = (
            f"the longest shutter spans {max_ratio:.3f} of one control period "
            f"({period_ms:.2f} ms), so every shutter sample lies inside the most "
            "recent recorded interval, where the trapezoidal history integral is "
            "exactly v_end*tau and therefore equals extrapolation. The two models "
            "cannot be distinguished in this configuration; this is neither a pass "
            "nor a failure of the history implementation."
        )
    elif not distinguishable:
        verdict = "TRAJECTORY_MODELS_INDISTINGUISHABLE_IN_IMAGE"
        explanation = (
            "the shutter does span multiple recorded intervals yet the rendered "
            "images differ by less than 5% of mean brightness, so the recorded "
            "motion is close to uniform along the shutter for this scene."
        )
    else:
        verdict = "TRAJECTORY_MODELS_DISTINGUISHABLE"
        explanation = "at least one exposure shows a >=5% relative image difference."

    return {
        "control_period_ms": period_ms,
        "px_per_rad": px_per_rad,
        "depth_floor_m": depth_floor,
        "temporal_samples": int(getattr(args, "gray_temporal_samples", 1)),
        "noise_enabled": False,
        "analytic_agreement": analytic,
        "analytic_agreement_ok": constant_ok,
        "constant_velocity_max_mae": max(analytic.values()),
        "measurements": records,
        "verdict": verdict,
        "explanation": explanation,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    argv = sys.argv[1:]
    # Everything the config parser knows about comes after the tool's own flags;
    # split on the first token the tool does not define.
    tool_argv = []
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] in ("--out_dir", "--device"):
            tool_argv.extend(argv[i:i + 2])
            i += 2
        else:
            rest = argv[i:]
            break
    cli = parser.parse_args(tool_argv)
    sys.argv = ["audit_exposure_trajectory"] + rest
    args = parse_args()
    device = torch.device(cli.device)

    report = audit(args, device)
    os.makedirs(cli.out_dir, exist_ok=True)
    out = os.path.join(cli.out_dir, "exposure_trajectory_audit.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"control period      : {report['control_period_ms']:.2f} ms")
    print(f"px per radian       : {report['px_per_rad']:.2f}")
    print(f"analytic agreement  : max mae {report['constant_velocity_max_mae']:.3e} "
          f"({'ok' if report['analytic_agreement_ok'] else 'FAILED'})")
    print()
    header = f"{'T ms':>7} {'T/period':>9} {'image_mae':>12} {'relative':>9} {'shift_px':>10} {'expected':>10}"
    print(header)
    for row in report["measurements"]:
        print(f"{row['exposure_ms']:7.2f} {row['shutter_over_period']:9.3f} "
              f"{row['image_mae']:12.8f} {row['image_mae_relative']:8.4%} "
              f"{row['shift_px']:10.4f} {row['analytic_shift_px']:10.4f}")
    print()
    print(f"verdict: {report['verdict']}")
    print(f"  {report['explanation']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

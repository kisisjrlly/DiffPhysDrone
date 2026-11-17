#!/usr/bin/env python3
"""Closed-loop proof-camera screen for the differentiable single-wall slit.

This is a bounded mechanism check, not a trained-policy result.  Each sample
uses ``Env.run`` for the motion and lets the readout see only one noisy 48x36
gray frame.  The slit centre is used only to create the binary score label.
The same episode seed is replayed across exposure/speed cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import set_global_seed
from config import build_parser, parse_scenarios, validate_args
from eval import load_checkpoint_training_args, run_one_episode
from intent_model import build_policy
from render.intent_scene import render_scene_irradiance
from rollout_ops import render_gray_sensor
from tools.intent_common import load_intent_args
from train_utils import build_env

EXPOSURES_MS = (0.89, 8.0, 20.0, 50.0)
TARGET_SPEEDS = (0.35, 1.15)
HISTORICAL_CHECKPOINT = Path(
    "/home/zhaoguodong/work/code/DiffPhysDrone/checkpoint/2026-09-24-11-35-14/final.pth"
)


def _ridge(train_x, train_y, test_x):
    train_x, test_x = train_x.double(), test_x.double()
    mean = train_x.mean(0)
    scale = train_x.std(0).clamp_min(0.005)
    a, b = (train_x - mean) / scale, (test_x - mean) / scale
    kernel = a @ a.T
    alpha = (0.1 * kernel.diagonal().mean()).clamp_min(1e-9)
    coef = torch.linalg.solve(
        kernel + alpha * torch.eye(len(a), device=a.device), train_y.double()
    )
    return b @ a.T @ coef


def _shift_slit(env, offset):
    """Move the two historical wall slabs together; no truth enters policy."""
    env.voxels[:, :2, 1] += float(offset)
    env.slit_center_y = env.slit_center_y + float(offset)


def _capture(args, device, *, seed, target_speed, exposure_ms, offset, capture_step):
    set_global_seed(seed, deterministic=True)
    env = build_env(1, args, device, eval_mode=True)
    env.reset(scene_name="nominal")
    _shift_slit(env, offset)
    exposure = env.imx900_calibration.exposure_normalized(
        env.p.new_tensor([exposure_ms * 1000.0])
    )
    gain = env.imx900_calibration.gain_from_factor(env.p.new_tensor([1.0]))
    frames = None
    stats = []
    for step in range(int(args.timesteps)):
        # The action is a fixed image-independent approach controller.  It is
        # deliberately the same controller for all camera settings; the probe
        # measures whether the camera sees a different task cue under motion.
        local_v = (env.v[:, None] @ env.R).squeeze(1)
        a_local = torch.zeros_like(env.v)
        a_local[:, 0] = (float(target_speed) - local_v[:, 0]) * 3.0
        a_local[:, 0].clamp_(-float(args.max_acc_cmd), float(args.max_acc_cmd))
        action = torch.squeeze(a_local[:, None] @ env.R.transpose(1, 2), 1)
        gray, aux = render_gray_sensor(env, exposure, gain, differentiable=False)
        if step == int(capture_step):
            frames = F.adaptive_avg_pool2d(
                gray, (int(args.gray_nn_height), int(args.gray_nn_width))
            ).flatten(1).detach()
        stats.append({
            "step": step,
            "speed_m_s": float(env.v.norm().item()),
            "blur_strength": float(aux["blur_strength"].mean().item()),
            "temporal_blur_mae": float(aux.get("temporal_blur_mae", env.p.new_zeros(1)).mean().item()),
            "saturation_fraction": float(aux["saturation_fraction"].mean().item()),
            "exposure_us": float(aux["exposure_us"].mean().item()),
        })
        env.run(action, env.control_period_s, env.p_target - env.p)
        if bool(env.collision_since_last_step().any()):
            # Keep the requested capture step deterministic; after contact the
            # remainder is still a valid image/physics trace, but is flagged.
            pass
    if frames is None:
        raise RuntimeError("capture_step exceeds episode length")
    label = torch.tensor([1.0 if offset > 0 else -1.0], device=device)
    # Render a noiseless image at the final pose only for a diagnostic quality
    # reference.  It is never passed to the readout or used as a label.
    sharp_irr, _ = render_scene_irradiance(
        env, (env.R_scene_T @ env.p[:, :, None]).squeeze(-1),
        (env.R_scene_T @ env.R) @ env.R_cam,
    )
    sharp, _ = env.gray_camera(sharp_irr, exposure, gain, enable_noise=False)
    clean_gap = float((gray - sharp).abs().mean().item())
    capture_stats = stats[min(int(capture_step), len(stats) - 1)]
    return frames, label, capture_stats, clean_gap


def _cell(args, device, *, target_speed, exposure_ms, per_class, seed, capture_step):
    xs, ys, traces, gaps = [], [], [], []
    # Pair positive/negative offsets and use a different seed for every episode.
    for i in range(int(per_class) * 2):
        offset = 0.05 if i % 2 else -0.05
        x, y, trace, gap = _capture(
            args, device, seed=seed + i, target_speed=target_speed,
            exposure_ms=exposure_ms, offset=offset, capture_step=capture_step,
        )
        xs.append(x[0]); ys.append(y[0]); traces.append(trace); gaps.append(gap)
    x, y = torch.stack(xs), torch.stack(ys)
    half = len(x) // 2
    # Fixed split by episode order; no test-time exposure or speed selection.
    train_x, test_x, train_y, test_y = x[:half], x[half:], y[:half], y[half:]
    score = _ridge(train_x, train_y, test_x)
    accuracy = float((((score > 0) == (test_y > 0)).double()).mean())
    return {
        "speed_m_s": float(target_speed), "exposure_ms": float(exposure_ms),
        "accuracy": accuracy, "n_train": half, "n_test": len(test_x),
        "mean_abs_noisy_clean_gap": float(sum(gaps) / len(gaps)),
        "trace_summary": {
            "mean_speed_m_s": float(sum(r["speed_m_s"] for r in traces) / max(len(traces), 1)),
            "mean_temporal_blur_mae": float(sum(r["temporal_blur_mae"] for r in traces) / max(len(traces), 1)),
            "mean_blur_strength": float(sum(r["blur_strength"] for r in traces) / max(len(traces), 1)),
            "mean_saturation_fraction": float(sum(r["saturation_fraction"] for r in traces) / max(len(traces), 1)),
        },
    }


def _historical_args(scene):
    parser = build_parser()
    args = parser.parse_args([])
    args.resume = str(HISTORICAL_CHECKPOINT)
    load_checkpoint_training_args(args)
    values = dict(
        scene_layout=scene,
        observation_grad_mode="full" if scene == "single_wall_slit_diff" else "detached",
        pose_gradient_estimator="analytic",
        gray_temporal_samples=3 if scene == "single_wall_slit_diff" else 1,
        camera_control_mode="mean_ae",
        sensor_grad_mode="detached",
        imx900_calibration="configs/calibration/imx900_provisional.json",
        fixed_camera_exposure=0.10,
        fixed_camera_gain=0.0,
        simple_start_x=-1.5,
        timesteps=60,
        control_dt_jitter=0.1,
        batch_size=1,
        deterministic=True,
        amp=False,
        wandb_disabled=True,
    )
    for name, value in values.items():
        setattr(args, name, value)
    args.scenarios = parse_scenarios(["dark", "bright", "dark_to_bright", "bright_to_dark"])
    validate_args(args)
    return args


def run_baseline(out, device, episodes=8):
    """Compare old and differentiable-equivalent geometry on paired episodes."""
    if not HISTORICAL_CHECKPOINT.is_file():
        raise FileNotFoundError(HISTORICAL_CHECKPOINT)
    rows = []
    for scene in ("single_wall_slit", "single_wall_slit_diff"):
        args = _historical_args(scene)
        env = build_env(1, args, device, eval_mode=True)
        model = build_policy(args).to(device)
        model.load_state_dict(torch.load(HISTORICAL_CHECKPOINT, map_location=device), strict=True)
        model.eval()
        for episode in range(int(episodes)):
            scene_name = args.scenarios[episode % len(args.scenarios)]
            row, _ = run_one_episode(
                episode, scene_name, args, model, env, None, device,
                episode_seed=10000 + episode,
            )
            rows.append({"geometry": scene, **row})
    old = [r for r in rows if r["geometry"] == "single_wall_slit"]
    new = [r for r in rows if r["geometry"] == "single_wall_slit_diff"]
    summary = {
        "checkpoint": str(HISTORICAL_CHECKPOINT),
        "checkpoint_sha256": hashlib.sha256(HISTORICAL_CHECKPOINT.read_bytes()).hexdigest(),
        "episodes": int(episodes),
        "old_successes": int(sum(r["success_rate"] for r in old)),
        "diff_successes": int(sum(r["success_rate"] for r in new)),
        "paired_same_outcome": int(sum(
            (a["success_rate"] == b["success_rate"] and a["stop_reason"] == b["stop_reason"])
            for a, b in zip(old, new)
        )),
        "rows": rows,
        "pass": int(sum(r["success_rate"] for r in new)) >= 4,
    }
    (out / "historical_equivalent_baseline.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def interaction_summary(rows):
    by = {(r["speed_m_s"], r["exposure_ms"]): r for r in rows}
    best = {}
    for speed in TARGET_SPEEDS:
        candidates = [by[(speed, exposure)] for exposure in EXPOSURES_MS]
        best[speed] = max(candidates, key=lambda r: (r["accuracy"], -r["exposure_ms"]))
    slow, fast = best[TARGET_SPEEDS[0]], best[TARGET_SPEEDS[1]]
    short, long_ = EXPOSURES_MS[0], EXPOSURES_MS[-1]
    contrast = (
        by[(TARGET_SPEEDS[1], short)]["accuracy"]
        - by[(TARGET_SPEEDS[1], long_)]["accuracy"]
        - by[(TARGET_SPEEDS[0], short)]["accuracy"]
        + by[(TARGET_SPEEDS[0], long_)]["accuracy"]
    )
    return {
        "best_by_speed": {
            str(TARGET_SPEEDS[0]): {"exposure_ms": slow["exposure_ms"], "accuracy": slow["accuracy"]},
            str(TARGET_SPEEDS[1]): {"exposure_ms": fast["exposure_ms"], "accuracy": fast["accuracy"]},
        },
        "best_exposure_changes_with_speed": bool(slow["exposure_ms"] != fast["exposure_ms"]),
        "short_long_difference_in_differences": float(contrast),
        "no_saturation_at_probe_cells": all(
            r["trace_summary"]["mean_saturation_fraction"] < 0.1 for r in rows
        ),
        "physical_interaction_candidate": bool(
            slow["exposure_ms"] != fast["exposure_ms"] and
            abs(contrast) >= 0.10 and
            all(r["trace_summary"]["mean_saturation_fraction"] < 0.1 for r in rows)
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("baseline", "interaction", "summary", "all"), default="all")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--per_class", type=int, default=8)
    p.add_argument("--capture_step", type=int, default=20)
    cli = p.parse_args()
    out = Path(cli.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(cli.device)
    baseline = None
    if cli.stage == "all":
        baseline = run_baseline(out, device)
    args = load_intent_args(
        "configs/gray_single_wall_slit_diff.args", batch_size=1,
        gray_enable_noise=True, simple_slit_center_y_min=0.0,
        simple_slit_center_y_max=0.0, simple_back_wall_x_min=2.2,
        simple_back_wall_x_max=2.2,
    )
    if cli.stage == "baseline":
        print(json.dumps(run_baseline(out, device), indent=2))
        return
    rows = []
    for speed in TARGET_SPEEDS:
        for exposure in EXPOSURES_MS:
            print(f"speed={speed:g} exposure_ms={exposure:g}", flush=True)
            rows.append(_cell(args, device, target_speed=speed, exposure_ms=exposure,
                              per_class=cli.per_class,
                              seed=731000 + int(speed * 1000) * 100 + int(exposure * 10),
                              capture_step=cli.capture_step))
    payload = {
        "kind": "proof_camera_closed_loop_diagnostic",
        "calibrated": False,
        "camera_profile": str(args.imx900_calibration),
        "scene_layout": args.scene_layout,
        "real_env_run": True,
        "readout_uses_truth": False,
        "label_use": "slit offset only for held-out score",
        "rows": rows,
        "interaction_summary": interaction_summary(rows),
        "source_sha256": {
            "env_cuda.py": hashlib.sha256((ROOT / "env_cuda.py").read_bytes()).hexdigest(),
            "render/intent_scene.py": hashlib.sha256((ROOT / "render/intent_scene.py").read_bytes()).hexdigest(),
            "sensors/imx900_camera.py": hashlib.sha256((ROOT / "sensors/imx900_camera.py").read_bytes()).hexdigest(),
        },
        "stop_rule": "This screen authorizes training only if speed changes the best exposure/readout tradeoff; otherwise stop without training.",
    }
    if baseline is not None:
        payload["historical_equivalent_baseline"] = {
            "old_successes": baseline["old_successes"],
            "diff_successes": baseline["diff_successes"],
            "paired_same_outcome": baseline["paired_same_outcome"],
            "pass": baseline["pass"],
        }
    (out / "proof_camera_interaction.json").write_text(json.dumps(payload, indent=2) + "\n")
    summary = payload["interaction_summary"]
    (out / "decision.md").write_text(
        "# Proof camera interaction decision\n\n"
        f"- slow best: {summary['best_by_speed'][str(TARGET_SPEEDS[0])] }\n"
        f"- fast best: {summary['best_by_speed'][str(TARGET_SPEEDS[1])] }\n"
        f"- difference-in-differences: {summary['short_long_difference_in_differences']:.4f}\n"
        f"- physical interaction candidate: **{summary['physical_interaction_candidate']}**\n\n"
        "This is a software mechanism diagnostic. It is not a hardware calibration or a trained-policy result.\n"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

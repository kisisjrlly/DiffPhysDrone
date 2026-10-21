#!/usr/bin/env python3
"""Paired navigation/behavior diagnostics; interventions are not new baselines.

Removing motion integration keeps the exposure-dependent photon/noise model
but samples only the current pose. It is an idealized counterfactual, not a
physically achievable shutter mode and not proof of learned coordination.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from config import build_parser, parse_scenarios, validate_args
from eval import run_one_episode
from intent_model import build_policy
from tools.run_intent_pilot import bootstrap_difference
from tools.intent_common import load_intent_args
from train_utils import build_env, write_run_provenance
from rollout_ops import render_gray_sensor
from tools.pole_navigation_teacher import bounded_probe_command


def initial_probe_override(args, amplitude=.12, steps=24):
    def override(step, env, proposed):
        if step >= steps:
            return proposed, False
        local, _ = bounded_probe_command(env, args, step, env.p.new_full((env.batch_size,), amplitude))
        return local, True
    return override


def frozen_view_renderer():
    """Remove view changes while preserving actual vehicle/control state."""
    initial = None
    def render(env, exposure, gain, differentiable=False):
        nonlocal initial
        if initial is None:
            initial = {key: getattr(env, key).clone() for key in ("p", "R", "R_old")}
        proxy = copy.copy(env)
        for key, value in initial.items():
            setattr(proxy, key, value)
        proxy.v = torch.zeros_like(env.v)
        return render_gray_sensor(proxy, exposure, gain, differentiable=differentiable)
    return render


def evaluate(checkpoint, conditions, scenarios, per_scenario, seed, device, out, config=None):
    checkpoint = Path(checkpoint).resolve()
    args = build_parser().parse_args([])
    vars(args).update(json.loads((checkpoint.parent / "args.json").read_text()))
    if config:
        vars(args).update(vars(load_intent_args(config)))
    args.resume, args.batch_size, args.vis_enable = str(checkpoint), 1, False
    args.vis_episode_idx = -1
    scenarios = parse_scenarios(args.scenarios if scenarios is None else scenarios)
    args.scenarios = scenarios
    if not config:
        args.imx900_calibration = str(checkpoint.parent / "imx900_calibration.json")
    validate_args(args)
    if args.scene_layout != "sparse_poles":
        raise ValueError("this diagnostic is restricted to the sparse-pole scene")
    model = build_policy(args).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.eval()
    base_camera_mode = args.camera_control_mode
    out.mkdir(parents=True)
    write_run_provenance(out, args)
    record = {"checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "calibrated": json.loads(Path(args.imx900_calibration).read_text())["calibrated"],
              "evaluation_config_override": str(Path(config).resolve()) if config else None,
              "episodes_per_scenario": per_scenario,
              "scenarios": scenarios, "conditions": {},
              "claim": "Paired development diagnostics, not by itself evidence of flight-for-measurement.",
              "no_motion_integration_semantics": "Idealized current-pose image with exposure signal/noise retained; not real hardware."}
    record["frozen_view_semantics"] = "Hold the camera at its initial pose with zero exposure motion; actual vehicle dynamics and odometry continue. Removes view-change information; not a physical baseline."
    record["forced_probe_semantics"] = "Fixed camera; first 24 flight commands are an external geometry-independent lateral probe, then the actor flies. Conditional navigation diagnostic only; never learned sensing motion."
    for condition in conditions:
        args.policy_gray_mode = "zero" if condition == "zero_image" else "gray"
        args.camera_control_mode = "fixed" if condition == "forced_probe" else base_camera_mode
        env = build_env(1, args, device, eval_mode=True)
        if condition == "no_motion_integration":
            env.gray_temporal_samples = 1
        rows, traces, layouts = [], [], []
        with torch.no_grad():
            for idx in range(per_scenario * len(scenarios)):
                scenario = scenarios[idx % len(scenarios)]
                row, trace = run_one_episode(idx, scenario, args, model, env, None, device,
                                             collect_trace=True, episode_seed=seed + idx,
                                             sensor_renderer=frozen_view_renderer() if condition == "frozen_view" else None,
                                             action_override=initial_probe_override(args, .12 if idx % 2 else -.12) if condition == "forced_probe" else None)
                rows.append(row)
                traces.extend(trace)
                layouts.append({"episode_idx": idx, "episode_seed": seed + idx,
                                "poles_local": env.cyl[0].cpu().tolist()})
        (out / f"{condition}_traces.json").write_text(json.dumps(traces) + "\n")
        (out / f"{condition}_episodes.json").write_text(json.dumps(rows, indent=2) + "\n")
        (out / f"{condition}_layouts.json").write_text(json.dumps(layouts) + "\n")
        value = {"episodes": rows,
                 **{key: float(np.mean([r[key] for r in rows]))
                    for key in ("success_rate", "collision_rate", "elapsed_s", "avg_speed", "exposure_mean", "gain_mean", "min_clearance")},
                 "by_scenario": {s: {"n": per_scenario,
                                      "success_rate": float(np.mean([r["success_rate"] for r in rows if r["scenario"] == s])),
                                      "collision_rate": float(np.mean([r["collision_rate"] for r in rows if r["scenario"] == s]))}
                                 for s in scenarios}}
        record["conditions"][condition] = value
        if "nominal" in record["conditions"] and condition != "nominal":
            base = record["conditions"]["nominal"]["episodes"]
            value["success_minus_nominal"] = bootstrap_difference([r["success_rate"] for r in rows],
                                                                  [r["success_rate"] for r in base])
        (out / "evaluation_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"[{checkpoint.parent.name}/{condition}] success {value['success_rate']:.3f}, collisions {value['collision_rate']:.3f}", flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", help="Explicitly evaluate a checkpoint in a different, fully recorded scene configuration.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--conditions", nargs="+", choices=["nominal", "zero_image", "no_motion_integration", "frozen_view", "forced_probe"], default=["nominal"])
    p.add_argument("--scenarios", nargs="+", help="Default: exact order from the effective checkpoint/config, preserving seed-to-scene pairing.")
    p.add_argument("--episodes_per_scenario", type=int, default=10)
    p.add_argument("--eval_seed", type=int, default=752000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu_threads", type=int, default=4)
    cli = p.parse_args()
    if cli.episodes_per_scenario < 1:
        raise ValueError("positive episode budget required")
    out = Path(cli.out_dir).resolve()
    if out.exists():
        raise ValueError("refusing to overwrite a diagnostic")
    torch.set_num_threads(cli.cpu_threads)
    evaluate(cli.checkpoint, cli.conditions, cli.scenarios, cli.episodes_per_scenario,
             cli.eval_seed, torch.device(cli.device), out, config=cli.config)


if __name__ == "__main__":
    main()

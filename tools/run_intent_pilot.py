#!/usr/bin/env python3
"""Bounded matched development runs; explicitly not a method-superiority study."""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from config import build_parser, parse_scenarios, validate_args
from eval import run_one_episode
from intent_model import build_policy
from train_utils import build_env


VARIANTS = {
    "joint": [],
    "no_intent": ["--no-camera_use_flight_intent"],
    "motion_gradient_cut": ["--observation_grad_mode", "detached"],
    "camera_gradient_cut": ["--sensor_grad_mode", "detached"],
    "nondiff_sensing": ["--sensor_grad_mode", "detached", "--observation_grad_mode", "detached"],
    "blind": ["--policy_gray_mode", "zero"],
    "fixed": ["--camera_control_mode", "fixed"],
    "mean_ae": ["--camera_control_mode", "mean_ae"],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap_difference(a, b):
    values = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(311)
    means = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "episode_bootstrap_95": np.quantile(means, [.025, .975]).tolist(),
            "interpretation": "Exploratory episode interval from a short single-training-seed development run."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--config", default="configs/gray_joint_intent.args")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--training_seed", type=int, default=101)
    parser.add_argument("--resume", help="Common actor checkpoint supplied unchanged to every condition.")
    parser.add_argument("--camera_lr_multiplier", type=float)
    parser.add_argument("--eval_seed", type=int, default=721000)
    parser.add_argument("--eval_episodes", type=int, default=8)
    parser.add_argument("--wandb_disabled", action="store_true", help="Online logging is enabled unless explicitly disabled.")
    cli = parser.parse_args()
    if min(cli.iterations, cli.batch_size, cli.eval_episodes) < 1:
        raise ValueError("budgets must be positive")
    out = Path(cli.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"refusing to overwrite existing pilot directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    tokens = shlex.split(Path(cli.config).read_text(), comments=True)
    summary = {"purpose": "development pilot", "training_seed": cli.training_seed,
               "iterations": cli.iterations, "batch_size": cli.batch_size,
               "evaluation_episode_seeds": list(range(cli.eval_seed, cli.eval_seed + cli.eval_episodes)),
               "calibrated": False, "methods": {},
               "common_checkpoint": str(Path(cli.resume).resolve()) if cli.resume else None,
               "common_checkpoint_sha256": sha256(cli.resume) if cli.resume else None,
               "camera_lr_multiplier_override": cli.camera_lr_multiplier,
               "claim": "Checks train/eval integration only; does not establish learned coordination, superiority or real deployment readiness."}
    initials = []
    device = torch.device("cuda")
    for variant in cli.variants:
        directory = out / variant
        directory.mkdir()
        command = [sys.executable, "main_cuda.py", *tokens,
                   "--num_iters", str(cli.iterations), "--batch_size", str(cli.batch_size),
                   "--seed", str(cli.training_seed), "--checkpoint_dir", str(directory), *VARIANTS[variant]]
        if cli.resume:
            command.extend(["--resume", str(Path(cli.resume).resolve())])
        if cli.camera_lr_multiplier is not None:
            command.extend(["--camera_lr_multiplier", str(cli.camera_lr_multiplier)])
        if cli.wandb_disabled:
            command.append("--wandb_disabled")
        (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        print(f"[pilot] training {variant}; log={directory / 'train.log'}", flush=True)
        with (directory / "train.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        initial = json.loads((directory / "initialization_manifest.json").read_text())
        initials.append((initial["flight_parameter_sha256"], initial["camera_parameter_sha256"]))
        if len(set(initials)) != 1:
            raise RuntimeError("matched initialization hashes differ")
        args = build_parser().parse_args([])
        vars(args).update(json.loads((directory / "args.json").read_text()))
        args.batch_size, args.vis_enable = 1, False
        args.vis_episode_idx = -1
        args.imx900_calibration = str(directory / "imx900_calibration.json")
        args.scenarios = parse_scenarios(args.scenarios)
        validate_args(args)
        model = build_policy(args).to(device)
        checkpoint = directory / "final.pth"
        model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
        model.eval()
        env = build_env(1, args, device, eval_mode=True)
        rows, traces = [], []
        with torch.no_grad():
            for episode in range(cli.eval_episodes):
                scene = args.scenarios[episode % len(args.scenarios)]
                row, trace = run_one_episode(episode, scene, args, model, env, None, device,
                                             collect_trace=True, episode_seed=cli.eval_seed + episode)
                rows.append(row)
                traces.extend(trace)
        (directory / "episodes.json").write_text(json.dumps(rows, indent=2) + "\n")
        (directory / "traces.json").write_text(json.dumps(traces) + "\n")
        method = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                  "initialization": initial,
                  "success_rate": float(np.mean([r["success_rate"] for r in rows])),
                  "collision_rate": float(np.mean([r["collision_rate"] for r in rows])),
                  "mean_elapsed_s": float(np.mean([r["elapsed_s"] for r in rows])),
                  "by_scenario": {scene: {"n": sum(r["scenario"] == scene for r in rows),
                                             "success_rate": float(np.mean([r["success_rate"] for r in rows if r["scenario"] == scene]))}
                                  for scene in args.scenarios if any(r["scenario"] == scene for r in rows)},
                  "episode_rows": rows}
        summary["methods"][variant] = method
        (out / "pilot_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[pilot] {variant}: success={method['success_rate']:.3f}, collision={method['collision_rate']:.3f}", flush=True)
        del env, model
        torch.cuda.empty_cache()
    summary["matched_initialization_verified"] = len(set(initials)) == 1
    if "joint" in summary["methods"]:
        reference = [r["success_rate"] for r in summary["methods"]["joint"]["episode_rows"]]
        summary["joint_minus_controls"] = {
            key: bootstrap_difference(reference, [r["success_rate"] for r in value["episode_rows"]])
            for key, value in summary["methods"].items() if key != "joint"
        }
    (out / "pilot_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[pilot] completed: {out / 'pilot_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

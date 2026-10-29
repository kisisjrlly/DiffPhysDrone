#!/usr/bin/env python3
"""Paired camera-control interventions on the original narrow-slit task."""

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import subprocess

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import build_parser, parse_scenarios, set_global_seed, validate_args
from eval import run_one_episode
from intent_model import build_policy
from train_utils import build_env


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flight_parameter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if model._is_camera_parameter(name):
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def summarize_trace(rows):
    if not rows:
        return {"steps": 0}
    result = {"steps": len(rows)}
    for key in ("exposure", "gain", "saturation_fraction", "dark_fraction", "light_scale"):
        values = [float(row[key]) for row in rows]
        result[key + "_mean"] = float(np.mean(values))
        result[key + "_std"] = float(np.std(values))
        result[key + "_range"] = float(max(values) - min(values))
    return result


def bootstrap_paired_by_scenario(full_rows, control_rows, rounds=5000, seed=20261029):
    full = {row["episode_seed"]: row for row in full_rows}
    control = {row["episode_seed"]: row for row in control_rows}
    if full.keys() != control.keys():
        raise ValueError("camera conditions do not use the same episode seeds")
    scenarios = sorted({row["scenario"] for row in full_rows})
    differences = {
        scenario: np.asarray([
            full[row["episode_seed"]]["success_rate"] - row["success_rate"]
            for row in control_rows if row["scenario"] == scenario
        ], dtype=np.float64)
        for scenario in scenarios
    }
    observed = float(np.mean([values.mean() for values in differences.values()]))
    rng = np.random.default_rng(seed)
    draws = np.zeros(rounds, dtype=np.float64)
    for values in differences.values():
        indices = rng.integers(0, len(values), size=(rounds, len(values)))
        draws += values[indices].mean(axis=1) / len(differences)
    discordant = {
        "learned_success_control_failure": sum(
            full[row["episode_seed"]]["success_rate"] > row["success_rate"]
            for row in control_rows),
        "control_success_learned_failure": sum(
            full[row["episode_seed"]]["success_rate"] < row["success_rate"]
            for row in control_rows),
    }
    return {
        "learned_minus_control_success_rate": observed,
        "stratified_episode_bootstrap_95_ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        "discordant_paired_episodes": discordant,
        "interpretation": "Conditional on this trained checkpoint; not uncertainty across training runs.",
    }


def atomic_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Seed-101 full camera-policy checkpoint from the frozen-flight slit experiment.")
    parser.add_argument("--detached_checkpoint", default=None,
                        help="Optional matched detached camera-policy checkpoint for the same training seed.")
    parser.add_argument("--common_flight_checkpoint", required=True,
                        help="Mean-AE flight checkpoint used to initialize the full camera run.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--episodes_per_scenario", type=int, default=100)
    parser.add_argument("--first_episode_seed", type=int, default=900000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--method_filter", nargs="*", default=None)
    cli = parser.parse_args()
    if cli.episodes_per_scenario < 1:
        raise ValueError("episodes_per_scenario must be positive")
    checkpoint = Path(cli.checkpoint).resolve()
    detached_checkpoint = Path(cli.detached_checkpoint).resolve() if cli.detached_checkpoint else None
    common_checkpoint = Path(cli.common_flight_checkpoint).resolve()
    if not checkpoint.is_file() or not common_checkpoint.is_file():
        raise FileNotFoundError("both camera-policy and common-flight checkpoints must exist")
    if detached_checkpoint is not None and not detached_checkpoint.is_file():
        raise FileNotFoundError(f"detached checkpoint does not exist: {detached_checkpoint}")
    out = Path(cli.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)

    args_path = checkpoint.parent / "args.json"
    calibration_path = checkpoint.parent / "imx900_calibration.json"
    if not args_path.is_file() or not calibration_path.is_file():
        raise FileNotFoundError("checkpoint args.json and calibration snapshot are required")
    detached_init = None
    if detached_checkpoint is not None:
        detached_args_path = detached_checkpoint.parent / "args.json"
        detached_calibration_path = detached_checkpoint.parent / "imx900_calibration.json"
        full_init_path = checkpoint.parent / "initialization_manifest.json"
        detached_init_path = detached_checkpoint.parent / "initialization_manifest.json"
        required = (detached_args_path, detached_calibration_path, full_init_path, detached_init_path)
        if not all(path.is_file() for path in required):
            raise FileNotFoundError("matched detached args, calibration, and initialization manifests are required")
        full_training_args = json.loads(args_path.read_text())
        detached_training_args = json.loads(detached_args_path.read_text())
        ignored = {"checkpoint_dir", "sensor_grad_mode"}
        if {k: v for k, v in full_training_args.items() if k not in ignored} != {
            k: v for k, v in detached_training_args.items() if k not in ignored
        }:
            raise ValueError("full and detached training configurations differ beyond sensor_grad_mode")
        if sha256(calibration_path) != sha256(detached_calibration_path):
            raise ValueError("full and detached calibration snapshots differ")
        full_init = json.loads(full_init_path.read_text())
        detached_init = json.loads(detached_init_path.read_text())
        for key in ("flight_parameter_sha256", "camera_parameter_sha256"):
            if full_init.get(key) != detached_init.get(key):
                raise ValueError(f"full and detached initialization mismatch: {key}")
    args = build_parser().parse_args([])
    vars(args).update(json.loads(args_path.read_text()))
    args.resume = str(checkpoint)
    args.imx900_calibration = str(calibration_path)
    args.batch_size = 1
    args.seed = cli.first_episode_seed
    args.deterministic = True
    args.amp = False
    args.vis_enable = False
    args.vis_episode_idx = -1
    args.scenarios = parse_scenarios(["dark", "bright", "dark_to_bright", "bright_to_dark"])
    args.policy_gray_mode = "gray"
    args.include_camera_state_in_obs = False
    args.sensor_grad_mode = "detached"
    validate_args(args)
    if not args.train_camera_only:
        raise ValueError("this intervention design expects a frozen-flight camera-only checkpoint")
    if args.scene_layout not in (None, "single_wall_slit"):
        raise ValueError("checkpoint is not from the original narrow-slit scene")

    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(4)
    template = build_policy(args).to(device).eval()
    template.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    common = build_policy(args).to(device).eval()
    common.load_state_dict(torch.load(common_checkpoint, map_location=device), strict=True)
    learned_flight_hash = flight_parameter_hash(template)
    common_flight_hash = flight_parameter_hash(common)
    if learned_flight_hash != common_flight_hash:
        raise RuntimeError("camera checkpoint changed flight weights; intervention is not isolated")
    detached_flight_hash = None
    if detached_checkpoint is not None:
        detached_model = build_policy(args).to(device).eval()
        detached_model.load_state_dict(torch.load(detached_checkpoint, map_location=device), strict=True)
        detached_flight_hash = flight_parameter_hash(detached_model)
        if detached_flight_hash != common_flight_hash:
            raise RuntimeError("detached checkpoint changed flight weights; comparison is not matched")
        del detached_model

    # The fixed settings were specified before this new held-out evaluation.
    methods = [
        {"name": "learned_full", "mode": "learned", "checkpoint_variant": "full", "exposure": 0.10, "gain": 0.02},
        {"name": "mean_ae", "mode": "mean_ae", "checkpoint_variant": "full", "exposure": 0.10, "gain": 0.02},
        {"name": "fixed_nominal", "mode": "fixed", "checkpoint_variant": "full", "exposure": 0.10, "gain": 0.02},
        {"name": "fixed_low", "mode": "fixed", "checkpoint_variant": "full", "exposure": 0.03, "gain": 0.00},
        {"name": "fixed_high", "mode": "fixed", "checkpoint_variant": "full", "exposure": 0.35, "gain": 0.15},
        {"name": "fixed_prior_response_best", "mode": "fixed", "checkpoint_variant": "full", "exposure": 0.00, "gain": 0.80},
    ]
    if detached_checkpoint is not None:
        methods.insert(1, {"name": "learned_detached", "mode": "learned",
                           "checkpoint_variant": "detached", "exposure": 0.10, "gain": 0.02})
    if cli.method_filter:
        requested = set(cli.method_filter)
        unknown = requested - {item["name"] for item in methods}
        if unknown:
            raise ValueError(f"unknown method names: {sorted(unknown)}")
        methods = [item for item in methods if item["name"] in requested]

    seeds = {}
    next_seed = cli.first_episode_seed
    for scenario in args.scenarios:
        seeds[scenario] = list(range(next_seed, next_seed + cli.episodes_per_scenario))
        next_seed += cli.episodes_per_scenario
    manifest = {
        "status": "running",
        "claim_scope": "Provisional simulator, original single-wall narrow slit, conditional on one frozen flight checkpoint.",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "detached_checkpoint": str(detached_checkpoint) if detached_checkpoint else None,
        "detached_checkpoint_sha256": sha256(detached_checkpoint) if detached_checkpoint else None,
        "common_flight_checkpoint": str(common_checkpoint),
        "common_flight_checkpoint_sha256": sha256(common_checkpoint),
        "flight_weights_identical": True,
        "flight_parameter_sha256": learned_flight_hash,
        "detached_flight_parameter_sha256": detached_flight_hash,
        "matched_initialization": bool(detached_checkpoint is not None),
        "initial_camera_parameter_sha256": detached_init.get("camera_parameter_sha256") if detached_init else None,
        "calibration_snapshot": str(calibration_path),
        "calibration_sha256": sha256(calibration_path),
        "calibrated": False,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): sha256(path)
                           for path in (Path(__file__), Path(__file__).resolve().parents[1] / "eval.py",
                                        Path(__file__).resolve().parents[1] / "model.py",
                                        Path(__file__).resolve().parents[1] / "rollout_ops.py")},
        "device": str(device),
        "scenarios": args.scenarios,
        "episodes_per_scenario": cli.episodes_per_scenario,
        "episode_seeds_by_scenario": seeds,
        "configuration": vars(args),
        "methods": methods,
    }
    atomic_write(out / "manifest.json", manifest)
    episode_results = {}
    summaries = {}
    for method in methods:
        method_args = copy.deepcopy(args)
        method_args.camera_control_mode = method["mode"]
        method_args.fixed_camera_exposure = method["exposure"]
        method_args.fixed_camera_gain = method["gain"]
        validate_args(method_args)
        model = build_policy(method_args).to(device).eval()
        method_checkpoint = detached_checkpoint if method["checkpoint_variant"] == "detached" else checkpoint
        if method_checkpoint is None:
            raise RuntimeError("detached method requested without a detached checkpoint")
        model.load_state_dict(torch.load(method_checkpoint, map_location=device), strict=True)
        env = build_env(1, method_args, device, eval_mode=True)
        rows = []
        started = time.perf_counter()
        with torch.no_grad():
            for scenario_index, scenario in enumerate(method_args.scenarios):
                for local_index, episode_seed in enumerate(seeds[scenario]):
                    episode_index = scenario_index * cli.episodes_per_scenario + local_index
                    row, trace = run_one_episode(
                        episode_index, scenario, method_args, model, env, None, device,
                        collect_trace=True, episode_seed=episode_seed)
                    row["method"] = method["name"]
                    row["camera_mode"] = method["mode"]
                    row["trace_summary"] = summarize_trace(trace)
                    rows.append(row)
        elapsed = time.perf_counter() - started
        episode_results[method["name"]] = rows
        per_scenario = {}
        for scenario in method_args.scenarios:
            group = [row for row in rows if row["scenario"] == scenario]
            per_scenario[scenario] = {
                "episodes": len(group),
                "successes": int(sum(row["success_rate"] for row in group)),
                "success_rate": float(np.mean([row["success_rate"] for row in group])),
                "collision_rate": float(np.mean([row["collision_rate"] for row in group])),
                "timeout_rate": float(np.mean([row["stop_reason"] == "timeout" for row in group])),
                "mean_goal_distance": float(np.mean([row["final_goal_dist"] for row in group])),
                "mean_exposure": float(np.mean([row["trace_summary"]["exposure_mean"] for row in group])),
                "mean_gain": float(np.mean([row["trace_summary"]["gain_mean"] for row in group])),
                "mean_saturation": float(np.mean([row["saturation_fraction"] for row in group])),
                "mean_dark_fraction": float(np.mean([row["dark_fraction"] for row in group])),
                "mean_exposure_range": float(np.mean([row["trace_summary"]["exposure_range"] for row in group])),
                "mean_gain_range": float(np.mean([row["trace_summary"]["gain_range"] for row in group])),
            }
        summaries[method["name"]] = {
            "mode": method["mode"],
            "checkpoint_variant": method["checkpoint_variant"],
            "fixed_exposure": method["exposure"],
            "fixed_gain": method["gain"],
            "episodes": len(rows),
            "successes": int(sum(row["success_rate"] for row in rows)),
            "success_rate": float(np.mean([row["success_rate"] for row in rows])),
            "collision_rate": float(np.mean([row["collision_rate"] for row in rows])),
            "timeout_rate": float(np.mean([row["stop_reason"] == "timeout" for row in rows])),
            "mean_goal_distance": float(np.mean([row["final_goal_dist"] for row in rows])),
            "mean_saturation": float(np.mean([row["saturation_fraction"] for row in rows])),
            "mean_dark_fraction": float(np.mean([row["dark_fraction"] for row in rows])),
            "mean_exposure": float(np.mean([row["trace_summary"]["exposure_mean"] for row in rows])),
            "mean_gain": float(np.mean([row["trace_summary"]["gain_mean"] for row in rows])),
            "mean_exposure_range": float(np.mean([row["trace_summary"]["exposure_range"] for row in rows])),
            "mean_gain_range": float(np.mean([row["trace_summary"]["gain_range"] for row in rows])),
            "wall_time_s": elapsed,
            "by_scenario": per_scenario,
        }
        atomic_write(out / "episodes.json", episode_results)
        atomic_write(out / "summary_partial.json", summaries)
        print(method["name"], summaries[method["name"]]["successes"], "/", len(rows), flush=True)
        del model, env
        if device.type == "cuda":
            torch.cuda.empty_cache()

    learned_rows = episode_results["learned_full"]
    comparisons = {
        name: bootstrap_paired_by_scenario(learned_rows, rows)
        for name, rows in episode_results.items() if name != "learned_full"
    }
    summary = {
        "scope": manifest["claim_scope"],
        "manifest_sha256": sha256(out / "manifest.json"),
        "methods": summaries,
        "paired_learned_full_differences": comparisons,
        "interpretation_limit": "Single training checkpoint and provisional camera model; this is a gate for camera relevance, not a training-method ranking or hardware claim.",
    }
    atomic_write(out / "summary.json", summary)
    manifest["status"] = "complete"
    manifest["results_summary_sha256"] = sha256(out / "summary.json")
    atomic_write(out / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

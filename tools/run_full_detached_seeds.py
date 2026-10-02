"""Run matched full/detached camera-training pairs and paired evaluations.

This is an explicit experiment driver; it is intentionally not a scheduler.
"""

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
import math
import statistics
from pathlib import Path


def _run(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config_tokens(mode):
    path = Path("configs") / f"gray_camera_{mode}.args"
    return shlex.split(path.read_text(), comments=True)


def _assert_pair(full_dir, detached_dir):
    full_init = json.loads((full_dir / "initialization_manifest.json").read_text())
    detached_init = json.loads((detached_dir / "initialization_manifest.json").read_text())
    for key in ("flight_parameter_sha256", "camera_parameter_sha256"):
        if full_init[key] != detached_init[key]:
            raise RuntimeError(f"matched initialization failed: {key}")
    full_args = json.loads((full_dir / "args.json").read_text())
    detached_args = json.loads((detached_dir / "args.json").read_text())
    ignored = {"sensor_grad_mode", "checkpoint_dir"}
    if {k: v for k, v in full_args.items() if k not in ignored} != {
        k: v for k, v in detached_args.items() if k not in ignored
    }:
        raise RuntimeError("resolved full/detached args differ outside sensor_grad_mode")
    for name in ("imx900_calibration.json",):
        if _sha256(full_dir / name) != _sha256(detached_dir / name):
            raise RuntimeError(f"matched calibration failed: {name}")


def _success_rate(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sum(float(row["success_rate"]) for row in rows) / max(len(rows), 1)


def _read_episode_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _bootstrap_ci(values, seed=20261002, rounds=4000):
    """Deterministic percentile bootstrap interval for paired episode deltas."""
    values = [float(value) for value in values]
    if not values:
        return [None, None]
    import random
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return [means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]]


def _git_metadata():
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return {"revision": revision, "working_tree_status": status}


def _run_metadata(run_dir, checkpoint, eval_seed):
    init = json.loads((run_dir / "initialization_manifest.json").read_text())
    resolved_args = json.loads((run_dir / "args.json").read_text())
    calibration_path = run_dir / "imx900_calibration.json"
    calibration = json.loads(calibration_path.read_text())
    rows = _read_episode_rows(run_dir / "episodes.csv")
    by_scenario = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], []).append(float(row["success_rate"]))
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "initialization": init,
        "resolved_args": resolved_args,
        "calibration_snapshot": {
            "path": str(calibration_path.resolve()),
            "sha256": _sha256(calibration_path),
            "calibrated": bool(calibration.get("calibrated", False)),
            "profile_name": calibration.get("profile_name", ""),
        },
        "evaluation": {
            "episode_seed_start": min(int(row["episode_seed"]) for row in rows),
            "episode_seed_end": max(int(row["episode_seed"]) for row in rows),
            "episode_seed_base": int(eval_seed),
            "episodes": len(rows),
            "episodes_per_scenario": {key: len(value) for key, value in by_scenario.items()},
            "success_by_scenario": {
                key: sum(value) / len(value) for key, value in by_scenario.items()
            },
        },
    }


def _summarize_existing(root, seeds, scenarios, args):
    records = []
    for seed in seeds:
        full_dir = root / f"seed_{seed}" / "full"
        detached_dir = root / f"seed_{seed}" / "detached"
        _assert_pair(full_dir, detached_dir)
        full_rows = _read_episode_rows(full_dir / "episodes.csv")
        detached_rows = _read_episode_rows(detached_dir / "episodes.csv")
        if [(r["episode_seed"], r["scenario"]) for r in full_rows] != [(r["episode_seed"], r["scenario"]) for r in detached_rows]:
            raise RuntimeError(f"paired evaluation seeds differ for seed {seed}")
        deltas = [float(f["success_rate"]) - float(d["success_rate"]) for f, d in zip(full_rows, detached_rows)]
        per_scenario = {}
        for scenario in scenarios:
            indices = [i for i, row in enumerate(full_rows) if row["scenario"] == scenario]
            scenario_deltas = [deltas[i] for i in indices]
            per_scenario[scenario] = {
                "full_success_rate": sum(float(full_rows[i]["success_rate"]) for i in indices) / len(indices),
                "detached_success_rate": sum(float(detached_rows[i]["success_rate"]) for i in indices) / len(indices),
                "delta_success_rate": sum(scenario_deltas) / len(scenario_deltas),
                "paired_bootstrap_95_ci": _bootstrap_ci(scenario_deltas, seed=20261002 + seed),
                "episodes": len(indices),
            }
        records.append({
            "seed": seed,
            "full": _run_metadata(full_dir, full_dir / "final.pth", args.eval_seed),
            "detached": _run_metadata(detached_dir, detached_dir / "final.pth", args.eval_seed),
            "full_success_rate": _success_rate(full_dir / "episodes.csv"),
            "detached_success_rate": _success_rate(detached_dir / "episodes.csv"),
            "delta_success_rate": sum(deltas) / len(deltas),
            "paired_bootstrap_95_ci": _bootstrap_ci(deltas, seed=20261002 + seed),
            "by_scenario": per_scenario,
        })
    deltas = [record["delta_success_rate"] for record in records]
    mean_delta = sum(deltas) / len(deltas)
    std_delta = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    t95 = 2.7764451051977987 if len(deltas) == 5 else 1.96
    half_width = t95 * std_delta / math.sqrt(len(deltas)) if deltas else None
    return {
        "scope": "matched full-vs-detached camera training",
        "flight_checkpoint": str(Path(args.flight_checkpoint).resolve()),
        "flight_checkpoint_sha256": _sha256(Path(args.flight_checkpoint)),
        "git": _git_metadata(),
        "seeds": seeds,
        "scenarios": scenarios,
        "num_iters": args.num_iters,
        "batch_size": args.batch_size,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "runs": records,
        "mean_delta_success_rate": mean_delta,
        "training_seed_delta_std": std_delta,
        "training_seed_delta_t95_ci": [mean_delta - half_width, mean_delta + half_width] if half_width is not None else [None, None],
        "training_seed_deltas": deltas,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--num_iters", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_episodes", type=int, default=400)
    parser.add_argument("--scenarios", default="dark,bright,dark_to_bright,bright_to_dark")
    parser.add_argument("--eval_seed", type=int, default=50000)
    parser.add_argument("--summary_only", action="store_true", help="summarize existing runs without retraining")
    parser.add_argument("--wandb_disabled", action="store_true")
    args = parser.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    scenarios = args.scenarios.split(",")
    if args.summary_only:
        manifest = _summarize_existing(root, seeds, scenarios, args)
        (root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return
    python = sys.executable
    records = []
    for seed in seeds:
        checkpoints = {}
        for mode in ("full", "detached"):
            run_dir = root / f"seed_{seed}" / mode
            config_tokens = _load_config_tokens(mode)
            train_cmd = [
                python, "main_cuda.py", *config_tokens,
                "--resume", args.flight_checkpoint,
                "--checkpoint_dir", str(run_dir),
                "--seed", str(seed),
                "--num_iters", str(args.num_iters),
                "--batch_size", str(args.batch_size),
            ]
            if args.wandb_disabled:
                train_cmd.append("--wandb_disabled")
            _run(train_cmd, run_dir / "train.log")
            checkpoint = run_dir / "final.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoints[mode] = str(checkpoint)
            eval_cmd = [
                python, "eval.py",
                "--resume", str(checkpoint),
                "--batch_size", "1",
                "--eval_episodes", str(args.eval_episodes),
                "--seed", str(args.eval_seed),
                "--scenarios", *scenarios,
                "--eval_episode_csv", str(run_dir / "episodes.csv"),
            ]
            _run(eval_cmd, run_dir / "eval.log")
        _assert_pair(root / f"seed_{seed}" / "full", root / f"seed_{seed}" / "detached")
        full_success = _success_rate(root / f"seed_{seed}" / "full" / "episodes.csv")
        detached_success = _success_rate(root / f"seed_{seed}" / "detached" / "episodes.csv")
        records.append({
            "seed": seed,
            "full": checkpoints["full"],
            "detached": checkpoints["detached"],
            "full_success_rate": full_success,
            "detached_success_rate": detached_success,
            "delta_success_rate": full_success - detached_success,
        })
    manifest = _summarize_existing(root, seeds, scenarios, args)
    (root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

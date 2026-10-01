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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--num_iters", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_episodes", type=int, default=400)
    parser.add_argument("--scenarios", default="dark,bright,dark_to_bright,bright_to_dark")
    parser.add_argument("--wandb_disabled", action="store_true")
    args = parser.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    scenarios = args.scenarios.split(",")
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
                "--seed", "50000",
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
    manifest = {
        "scope": "matched full-vs-detached camera training",
        "flight_checkpoint": str(Path(args.flight_checkpoint).resolve()),
        "seeds": seeds,
        "scenarios": scenarios,
        "num_iters": args.num_iters,
        "batch_size": args.batch_size,
        "eval_episodes": args.eval_episodes,
        "runs": records,
        "mean_delta_success_rate": sum(r["delta_success_rate"] for r in records) / max(len(records), 1),
        "training_seed_deltas": [r["delta_success_rate"] for r in records],
    }
    (root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

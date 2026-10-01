"""Run matched full/detached camera-training pairs and paired evaluations.

This is an explicit experiment driver; it is intentionally not a scheduler.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


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
            train_cmd = [
                python, "main_cuda.py",
                "--resume", args.flight_checkpoint,
                "--checkpoint_dir", str(run_dir),
                "--train_camera_only",
                "--camera_control_mode", "learned",
                "--sensor_grad_mode", mode,
                "--seed", str(seed),
                "--num_iters", str(args.num_iters),
                "--batch_size", str(args.batch_size),
                "--scenarios", *scenarios,
                "--fixed_camera_exposure", "0.10",
                "--fixed_camera_gain", "0.02",
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
        records.append({"seed": seed, "full": checkpoints["full"], "detached": checkpoints["detached"]})
    manifest = {
        "scope": "matched full-vs-detached camera training",
        "flight_checkpoint": str(Path(args.flight_checkpoint).resolve()),
        "seeds": seeds,
        "scenarios": scenarios,
        "num_iters": args.num_iters,
        "batch_size": args.batch_size,
        "eval_episodes": args.eval_episodes,
        "runs": records,
    }
    (root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

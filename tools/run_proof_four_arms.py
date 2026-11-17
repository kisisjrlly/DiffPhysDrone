#!/usr/bin/env python3
"""Run the bounded four-arm proof-camera comparison.

The script refuses to start unless the historical-equivalent baseline and the
proof-camera interaction diagnostic have already been saved.  Every arm starts
from the same flight checkpoint and uses the same eight evaluation episodes.
This is a mechanism demonstration, not a publication-scale training sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python"
ARMS = {
    "A_flight_ae": "configs/gray_proof_arm_a.args",
    "B_camera_only": "configs/gray_proof_arm_b.args",
    "C_joint_full": "configs/gray_proof_arm_c.args",
    "D_joint_camera_gradient_cut": "configs/gray_proof_arm_d.args",
}


def _tokens(path):
    return shlex.split(Path(path).read_text(), comments=True)


def _run(cmd, log):
    with log.open("w") as handle:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log}")


def _evaluate(checkpoint, args_path, out_path, episodes, seed):
    # Import the runner in-process so all rows are written in one stable JSON
    # file and the same episode seeds can be checked directly.
    sys.path.insert(0, str(ROOT))
    import torch
    from config import build_parser, parse_scenarios, set_global_seed, validate_args
    from eval import run_one_episode
    from intent_model import build_policy
    from train_utils import build_env

    args = build_parser().parse_args(_tokens(args_path))
    args.resume = str(checkpoint)
    args.batch_size = 1
    args.eval_episodes = episodes
    args.seed = seed
    args.deterministic = True
    args.amp = False
    args.scenarios = parse_scenarios(args.scenarios)
    validate_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(seed, True)
    env = build_env(1, args, device, eval_mode=True)
    model = build_policy(args).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.eval()
    rows = []
    with torch.no_grad():
        for episode in range(episodes):
            scene = args.scenarios[episode % len(args.scenarios)]
            row, _ = run_one_episode(
                episode, scene, args, model, env, None, device,
                episode_seed=seed + episode,
            )
            rows.append(row)
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "successes": int(sum(r["success_rate"] for r in rows)),
        "episodes": episodes,
        "success_rate": float(sum(r["success_rate"] for r in rows) / episodes),
        "collision_rate": float(sum(r["collision_rate"] for r in rows) / episodes),
        "mean_elapsed_s": float(sum(r["elapsed_s"] for r in rows) / episodes),
        "rows": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_iters", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=10000)
    parser.add_argument("--skip_train", action="store_true")
    cli = parser.parse_args()
    out = Path(cli.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = Path(cli.base_checkpoint).resolve()
    if not base.is_file():
        raise FileNotFoundError(base)
    baseline = ROOT / "exports/proof_baseline_20261108/historical_equivalent_baseline.json"
    interaction = ROOT / "exports/proof_camera_diagnostic_20261108d/proof_camera_interaction.json"
    if not baseline.is_file() or not interaction.is_file():
        raise RuntimeError("run the baseline and proof-camera diagnostic before four arms")
    interaction_data = json.loads(interaction.read_text())
    if not interaction_data.get("interaction_summary", {}).get("physical_interaction_candidate", False):
        raise RuntimeError("proof-camera interaction gate did not pass; refusing to train")

    summaries = {}
    for name, config in ARMS.items():
        arm_dir = out / name
        arm_dir.mkdir()
        checkpoint = arm_dir / "final.pth"
        if not cli.skip_train:
            cmd = [PYTHON, "main_cuda.py", *_tokens(ROOT / config),
                   "--resume", str(base), "--num_iters", str(cli.train_iters),
                   "--checkpoint_dir", str(arm_dir), "--seed", str(cli.seed),
                   "--batch_size", "2"]
            _run(cmd, arm_dir / "train.log")
        if not checkpoint.is_file():
            candidates = sorted(arm_dir.glob("checkpoint*.pth"))
            if not candidates:
                raise FileNotFoundError(f"no checkpoint produced for {name}")
            checkpoint = candidates[-1]
        summaries[name] = _evaluate(
            checkpoint, ROOT / config, arm_dir / "evaluation.json",
            cli.episodes, cli.eval_seed,
        )
        print(name, summaries[name]["successes"], "/", cli.episodes, flush=True)

    report = {
        "base_checkpoint": str(base),
        "base_checkpoint_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "train_iters_per_arm": cli.train_iters,
        "paired_eval_episodes": cli.episodes,
        "physical_interaction_gate": interaction_data["interaction_summary"],
        "arms": summaries,
        "interpretation": (
            "C_joint_full is a candidate only if it exceeds both A_flight_ae and "
            "D_joint_camera_gradient_cut on the same episodes. This small run does "
            "not establish a publication claim."
        ),
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

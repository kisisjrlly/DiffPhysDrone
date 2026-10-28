#!/usr/bin/env python3
"""Audit a bounded pilot against its common initial actor and mirror layouts."""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from config import build_parser, parse_scenarios, set_global_seed, validate_args
from eval import run_one_episode
from intent_model import build_policy
from train_utils import build_env


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(pilot_dir, device):
    root = Path(pilot_dir).resolve()
    manifest = json.loads((root / "pilot_manifest.json").read_text())
    args = build_parser().parse_args([])
    vars(args).update(json.loads((root / "joint/args.json").read_text()))
    args.batch_size, args.vis_enable, args.vis_episode_idx = 1, False, -1
    args.scenarios = parse_scenarios(args.scenarios)
    args.imx900_calibration = str(root / "joint/imx900_calibration.json")
    validate_args(args)
    checkpoint = Path(args.resume)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    model = build_policy(args).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    env = build_env(1, args, device, eval_mode=True)
    initial, layouts = [], {}
    with torch.no_grad():
        for i, seed in enumerate(manifest["evaluation_episode_seeds"]):
            scene = args.scenarios[i % len(args.scenarios)]
            set_global_seed(seed, args.deterministic)
            env.reset(scene)
            if args.pole_layout == "parallax_pair":
                layouts[seed] = int(env.cyl[0, 0, 1].sign())
            row, _ = run_one_episode(i, scene, args, model, env, None, device, episode_seed=seed)
            initial.append(row)
    result = {
        "purpose": "bounded integration audit; no learned coordination or method-superiority claim",
        "pilot_manifest_sha256": digest(root / "pilot_manifest.json"),
        "configuration": json.loads((root / "joint/args.json").read_text()),
        "common_checkpoint_sha256": digest(checkpoint),
        "training_seed": manifest["training_seed"],
        "eval_seeds": manifest["evaluation_episode_seeds"],
        "matched_actor_initialization": manifest["matched_initialization_verified"],
        "matched_critic_initialization": manifest["matched_critic_initialization_verified"],
        "matched_source": manifest["matched_source_hashes_verified"],
        "matched_replay_targets": manifest["matched_replay_targets_verified"],
        "methods": {},
        "paired_difference": manifest.get("joint_minus_controls"),
    }
    initial_success = [r["success_rate"] for r in initial]
    rows_by_method = {"initial_common": initial}
    rows_by_method.update({name: method["episode_rows"] for name, method in manifest["methods"].items()})
    for name, rows in rows_by_method.items():
        item = {
            "episodes": len(rows), "successes": int(sum(r["success_rate"] for r in rows)),
            "collisions": int(sum(r["collision_rate"] for r in rows)),
            "identical_success_outcomes_to_initial": [r["success_rate"] for r in rows] == initial_success,
            "mean_exposure": statistics.mean(r["exposure_mean"] for r in rows),
            "mean_gain": statistics.mean(r["gain_mean"] for r in rows),
            "episode_rows": rows,
        }
        if layouts:
            item["by_near_pole_sign"] = {
                str(sign): {
                    "n": sum(layouts[r["episode_seed"]] == sign for r in rows),
                    "successes": int(sum(r["success_rate"] for r in rows if layouts[r["episode_seed"]] == sign)),
                } for sign in (-1, 1)
            }
        if name != "initial_common":
            directory = root / name
            source = json.loads((directory / "provenance.json").read_text())
            metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
            method = manifest["methods"][name]
            item.update({
                "checkpoint_sha256": digest(directory / "final.pth"),
                "critic_checkpoint_sha256": digest(directory / "critic_final.pth"),
                "initialization": method["initialization"],
                "source_provenance": source,
                "flight_replay": method["flight_replay"],
                "wandb": method["wandb"],
                "generated_transitions": method["generated_navigation_transitions"],
                "optimizer_steps": len(metrics),
                "training_process_wall_time_s": method["training_process_wall_time_s"],
                "optimizer_wall_time_s": sum(1 / r["iter_per_sec"] for r in metrics),
                "camera_gradient_norm_median": statistics.median(r["cam/grad_norm"] for r in metrics),
                "peak_cuda_allocated_mib": max(r.get("ac/peak_cuda_memory_mib", 0) for r in metrics),
            })
        result["methods"][name] = item
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    cli = parser.parse_args()
    result = audit(cli.pilot_dir, torch.device(cli.device))
    destination = Path(cli.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print({name: {k: v[k] for k in ("successes", "episodes", "by_near_pole_sign") if k in v}
           for name, v in result["methods"].items()})


if __name__ == "__main__":
    main()

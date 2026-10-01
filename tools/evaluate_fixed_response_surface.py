"""Paired fixed exposure/gain response surface evaluation."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import shutil
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import build_parser, parse_scenarios, set_global_seed, validate_args
from eval import load_checkpoint_training_args, run_one_episode
from model import Model
from rerun_vis import RerunVis
from train_utils import build_env


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_ci(values, seed=0, rounds=2000):
    values = torch.tensor(values, dtype=torch.float64)
    if values.numel() == 0:
        return [0.0, 0.0]
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randint(values.numel(), (rounds, values.numel()), generator=generator)
    means = values[idx].mean(dim=1)
    return [float(torch.quantile(means, 0.025)), float(torch.quantile(means, 0.975))]


def main():
    parser = build_parser()
    parser.set_defaults(batch_size=1)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--episodes_per_scenario", type=int, default=25)
    parser.add_argument("--exposures", default="0,.10,.20,.30,.40,.50,.60,.70,.80,.90,1.0")
    parser.add_argument("--gains", default="0,.10,.20,.30,.40,.50,.60,.70,.80,.90,1.0")
    args = parser.parse_args()
    cli_resume = args.resume
    cli_seed = args.seed
    cli_scenarios = list(args.scenarios)
    load_checkpoint_training_args(args)
    args.resume = cli_resume
    args.seed = cli_seed
    args.scenarios = cli_scenarios
    args.batch_size = 1
    args.scenarios = parse_scenarios(args.scenarios)
    set_global_seed(args.seed, args.deterministic)
    validate_args(args)
    if args.batch_size != 1:
        raise ValueError("response surface evaluation requires --batch_size 1")
    if not Path(args.resume).is_file():
        raise FileNotFoundError(args.resume)
    args.camera_control_mode = "fixed"
    args.policy_gray_mode = "gray"
    exposures = [float(x) for x in args.exposures.split(",") if x.strip()]
    gains = [float(x) for x in args.gains.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = Path(args.imx900_calibration)
    if calibration_path.is_file():
        shutil.copy2(calibration_path, out_dir / "imx900_calibration.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = build_env(1, args, device, eval_mode=True)
    model = Model(7 if args.no_odom else 10, 3,
                  include_camera_state_in_obs=args.include_camera_state_in_obs,
                  gray_nn_width=args.gray_nn_width,
                  gray_nn_height=args.gray_nn_height).to(device)
    model.load_state_dict(torch.load(args.resume, map_location=device), strict=True)
    model.eval()
    vis = RerunVis(enabled=False, app_id="DiffPhysDrone-ResponseSurface", spawn=False, show_aabb=False)

    rows = []
    # The same (scenario, local episode) seed is reused for every E/G setting.
    episode_seeds = {
        (scenario, local_idx): int(args.seed) + scenario_idx * args.episodes_per_scenario + local_idx
        for scenario_idx, scenario in enumerate(args.scenarios)
        for local_idx in range(args.episodes_per_scenario)
    }
    for exposure in exposures:
        for gain in gains:
            env.fixed_camera_exposure = exposure
            env.fixed_camera_gain = gain
            args.fixed_camera_exposure = exposure
            args.fixed_camera_gain = gain
            for scenario in args.scenarios:
                for local_idx in range(args.episodes_per_scenario):
                    ep_idx = episode_seeds[(scenario, local_idx)] - int(args.seed)
                    row, _ = run_one_episode(
                        ep_idx, scenario, args, model, env, vis, device,
                        episode_seed=episode_seeds[(scenario, local_idx)],
                    )
                    row.update({"exposure_setting": exposure, "gain_setting": gain})
                    rows.append(row)

    with (out_dir / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_setting = {}
    for row in rows:
        key = (row["exposure_setting"], row["gain_setting"])
        item = by_setting.setdefault(key, {"rows": [], "scenarios": {}})
        item["rows"].append(row)
        item["scenarios"].setdefault(row["scenario"], []).append(row)
    summaries = []
    for (exposure, gain), item in by_setting.items():
        scenario_summary = {}
        for scenario, scenario_rows in item["scenarios"].items():
            successes = [float(r["success_rate"]) for r in scenario_rows]
            scenario_summary[scenario] = {
                "success_rate": sum(successes) / len(successes),
                "success_ci95": _bootstrap_ci(successes),
                "collision_rate": sum(float(r["collision_rate"]) for r in scenario_rows) / len(scenario_rows),
            }
        all_success = [float(r["success_rate"]) for r in item["rows"]]
        summaries.append({
            "exposure": exposure,
            "gain": gain,
            "success_rate": sum(all_success) / len(all_success),
            "success_ci95": _bootstrap_ci(all_success),
            "worst_case_success_rate": min(v["success_rate"] for v in scenario_summary.values()),
            "scenarios": scenario_summary,
        })
    summaries.sort(key=lambda x: (-x["success_rate"], -x["worst_case_success_rate"]))
    manifest = {
        "scope": "provisional simulator; fixed-camera response surface",
        "checkpoint": str(Path(args.resume).resolve()),
        "checkpoint_sha256": _sha256(args.resume),
        "calibration": str(calibration_path.resolve()),
        "calibration_sha256": _sha256(calibration_path) if calibration_path.is_file() else None,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "scenarios": args.scenarios,
        "episodes_per_scenario": args.episodes_per_scenario,
        "exposures": exposures,
        "gains": gains,
        "episode_seed_rule": "seed + scenario_index * episodes_per_scenario + local_episode_index",
        "best_mean": summaries[0],
        "best_worst_case": max(summaries, key=lambda x: x["worst_case_success_rate"]),
        "settings": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"best_mean": summaries[0], "best_worst_case": manifest["best_worst_case"]}, indent=2))


if __name__ == "__main__":
    main()

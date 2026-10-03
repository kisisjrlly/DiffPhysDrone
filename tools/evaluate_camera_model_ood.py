"""Evaluate frozen checkpoints under controlled camera-model perturbations.

The tool creates an isolated evaluation directory for each variant, rewrites
only evaluation-time camera arguments/calibration fields, and invokes the
normal evaluator. Training checkpoints and the nominal experiment artifacts
are never modified.
"""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import random
from pathlib import Path


SCENARIOS = ("dark", "bright", "dark_to_bright", "bright_to_dark")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variant_spec(name):
    specs = {
        "nominal": {},
        "hard_saturation": {"gray_saturation_mode": "hard"},
        "ste_saturation": {"gray_saturation_mode": "ste"},
        "blur_half": {"blur_scale": 0.5},
        "blur_onehalf": {"blur_scale": 1.5},
        "noise_half": {"noise_scale": 0.5},
        "noise_std_double": {"noise_scale": 2.0},
        "control_step_delay_1": {"command_delay_frames": 1},
        "control_step_delay_2": {"command_delay_frames": 2},
        "exposure_scale_low": {"signal_scale": 0.9},
        "exposure_scale_high": {"signal_scale": 1.1},
        "exposure_mapping_slope_low": {"exposure_range_scale": 0.9},
        "exposure_mapping_slope_high": {"exposure_range_scale": 1.1},
        "exposure_mapping_offset": {"exposure_range_offset_us": 250.0},
        "gain_linear": {"gain_mapping": "linear"},
        "gain_log_perturb": {"gain_factor_max_scale": 1.1},
    }
    if name not in specs:
        raise ValueError(f"unknown camera-model variant: {name}")
    return dict(specs[name])


def make_variant_checkpoint(source_checkpoint, variant_dir, spec, method):
    source_checkpoint = Path(source_checkpoint).resolve()
    source_dir = source_checkpoint.parent
    variant_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = variant_dir / "checkpoint.pth"
    shutil.copy2(source_checkpoint, checkpoint)
    args_path = source_dir / "args.json"
    args = json.loads(args_path.read_text()) if args_path.is_file() else {}
    calibration_path = source_dir / "imx900_calibration.json"
    if not calibration_path.is_file():
        calibration_path = Path(args.get("imx900_calibration", "configs/calibration/imx900_provisional.json"))
    calibration = json.loads(calibration_path.read_text())

    if method == "fixed":
        args["camera_control_mode"] = "fixed"
        args["fixed_camera_exposure"] = 0.0
        args["fixed_camera_gain"] = 0.8
        args["sensor_grad_mode"] = "detached"
    if "gray_saturation_mode" in spec:
        args["gray_saturation_mode"] = spec["gray_saturation_mode"]
    if "blur_scale" in spec:
        calibration.setdefault("motion_blur", {})["scale"] = float(
            calibration.get("motion_blur", {}).get("scale", 0.08)
        ) * float(spec["blur_scale"])
    if "noise_scale" in spec:
        noise = calibration.setdefault("noise", {})
        for key in ("shot_alpha", "shot_beta", "read_std_base"):
            if key in noise:
                noise[key] = float(noise[key]) * float(spec["noise_scale"])
    if "command_delay_frames" in spec:
        calibration.setdefault("actuator", {})["command_delay_frames"] = int(
            spec["command_delay_frames"]
        )
    if "signal_scale" in spec:
        exposure = calibration.setdefault("exposure", {})
        exposure["signal_scale"] = float(exposure.get("signal_scale", 1.0)) * float(spec["signal_scale"])
    if "exposure_range_scale" in spec:
        exposure = calibration.setdefault("exposure", {})
        minimum = float(exposure.get("min_us", 100.0))
        maximum = float(exposure.get("max_us", 8000.0))
        reference = minimum + (maximum - minimum) * 0.5
        half_range = (maximum - minimum) * float(spec["exposure_range_scale"]) * 0.5
        exposure["min_us"] = reference - half_range
        exposure["max_us"] = reference + half_range
    if "exposure_range_offset_us" in spec:
        exposure = calibration.setdefault("exposure", {})
        offset = float(spec["exposure_range_offset_us"])
        exposure["min_us"] = float(exposure.get("min_us", 100.0)) + offset
        exposure["max_us"] = float(exposure.get("max_us", 8000.0)) + offset
    if "gain_mapping" in spec:
        calibration.setdefault("gain", {})["mapping"] = spec["gain_mapping"]
    if "gain_factor_max_scale" in spec:
        gain = calibration.setdefault("gain", {})
        gain["max_factor"] = float(gain.get("max_factor", 8.0)) * float(spec["gain_factor_max_scale"])

    variant_calibration = variant_dir / "imx900_calibration.json"
    variant_calibration.write_text(json.dumps(calibration, indent=2) + "\n")
    args["imx900_calibration"] = str(variant_calibration.resolve())
    args["checkpoint_dir"] = str(variant_dir.resolve())
    args_path_out = variant_dir / "args.json"
    args_path_out.write_text(json.dumps(args, indent=2) + "\n")
    return checkpoint, variant_calibration


def summarize(rows):
    by_scenario = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], []).append(float(row["success_rate"]))
    return {
        "episodes": len(rows),
        "success_rate": sum(float(row["success_rate"]) for row in rows) / len(rows),
        "collision_rate": sum(float(row["collision_rate"]) for row in rows) / len(rows),
        "success_by_scenario": {
            key: sum(values) / len(values) for key, values in by_scenario.items()
        },
    }


def bootstrap_ci(values, seed, rounds=4000):
    values = [float(value) for value in values]
    rng = random.Random(int(seed))
    samples = []
    for _ in range(rounds):
        draw = [values[rng.randrange(len(values))] for _ in values]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    return [samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]]


def git_metadata():
    def run(*command):
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    try:
        revision = run("git", "rev-parse", "HEAD")
        status = run("git", "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "worktree_dirty": None}
    return {"revision": revision, "worktree_dirty": bool(status)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_root", required=True)
    parser.add_argument("--flight_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--variants", default="nominal,hard_saturation,ste_saturation,blur_half,blur_onehalf,noise_half,noise_std_double,control_step_delay_1,control_step_delay_2,exposure_scale_low,exposure_scale_high,gain_linear,gain_log_perturb")
    parser.add_argument("--training_seeds", default="101")
    parser.add_argument("--methods", default="full,detached,fixed")
    parser.add_argument("--eval_seed", type=int, default=61000)
    parser.add_argument("--episodes_per_scenario", type=int, default=25)
    parser.add_argument("--control_frequency_hz", type=float, default=15.0)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    args = parser.parse_args()
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.training_seeds.split(",") if value.strip()]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    episodes = int(args.episodes_per_scenario) * len(args.scenarios)
    records = []
    row_cache = {}
    for variant in variants:
        spec = variant_spec(variant)
        for seed in seeds:
            for method in methods:
                if method == "fixed":
                    source = Path(args.flight_checkpoint)
                else:
                    source = Path(args.checkpoint_root) / f"seed_{seed}" / method / "final.pth"
                out = root / variant / f"seed_{seed}" / method
                checkpoint, calibration = make_variant_checkpoint(source, out, spec, method)
                csv_path = out / "episodes.csv"
                command = [
                    sys.executable, "eval.py", "--resume", str(checkpoint),
                    "--batch_size", "1", "--eval_episodes", str(episodes),
                    "--seed", str(args.eval_seed), "--scenarios", *args.scenarios,
                    "--eval_episode_csv", str(csv_path),
                ]
                with (out / "eval.log").open("w") as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                with csv_path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                row_cache[(variant, seed, method)] = rows
                records.append({
                    "variant": variant,
                    "variant_spec": spec,
                    "seed": seed,
                    "method": method,
                    "checkpoint": str(source.resolve()),
                    "checkpoint_sha256": sha256(source),
                    "calibration_sha256": sha256(calibration),
                    "eval_seed": int(args.eval_seed),
                    "episodes_per_scenario": int(args.episodes_per_scenario),
                    "summary": summarize(rows),
                })
    comparisons = []
    for variant in variants:
        for seed in seeds:
            full_rows = row_cache[(variant, seed, "full")]
            detached_rows = row_cache[(variant, seed, "detached")]
            paired_keys = [(row["episode_seed"], row["scenario"]) for row in full_rows]
            detached_keys = [(row["episode_seed"], row["scenario"]) for row in detached_rows]
            if paired_keys != detached_keys:
                raise RuntimeError(f"paired OOD evaluation seeds differ: {variant}/seed_{seed}")
            deltas = [
                float(full["success_rate"]) - float(detached["success_rate"])
                for full, detached in zip(full_rows, detached_rows)
            ]
            by_scenario = {}
            for index, row in enumerate(full_rows):
                by_scenario.setdefault(row["scenario"], []).append(deltas[index])
            comparisons.append({
                "variant": variant,
                "seed": seed,
                "full_detached_delta": sum(deltas) / len(deltas),
                "paired_bootstrap_95_ci": bootstrap_ci(deltas, 91000 + seed),
                "delta_by_scenario": {
                    scenario: {
                        "mean": sum(values) / len(values),
                        "bootstrap_95_ci": bootstrap_ci(values, 92000 + seed),
                        "episodes": len(values),
                    }
                    for scenario, values in by_scenario.items()
                },
            })
    variant_nominal = {}
    for variant in variants:
        for seed in seeds:
            for method in methods:
                if (variant, seed, method) not in row_cache:
                    continue
                if variant == "nominal":
                    variant_nominal[(seed, method)] = row_cache[(variant, seed, method)]
    variant_deltas = []
    for variant in variants:
        if variant == "nominal":
            continue
        for seed in seeds:
            for method in methods:
                nominal = variant_nominal.get((seed, method))
                current = row_cache.get((variant, seed, method))
                if nominal is None or current is None:
                    continue
                nominal_keys = [(row["episode_seed"], row["scenario"]) for row in nominal]
                current_keys = [(row["episode_seed"], row["scenario"]) for row in current]
                if nominal_keys != current_keys:
                    raise RuntimeError(f"paired OOD nominal seeds differ: {variant}/seed_{seed}/{method}")
                deltas = [float(row["success_rate"]) - float(base["success_rate"])
                          for row, base in zip(current, nominal)]
                variant_deltas.append({
                    "variant": variant,
                    "seed": seed,
                    "method": method,
                    "variant_minus_nominal": sum(deltas) / len(deltas),
                    "paired_bootstrap_95_ci": bootstrap_ci(deltas, 93000 + seed),
                })
    manifest = {
        "scope": "camera-model OOD evaluation of frozen checkpoints",
        "variants": variants,
        "training_seeds": seeds,
        "methods": methods,
        "scenarios": args.scenarios,
        "eval_seed": int(args.eval_seed),
        "episodes_per_scenario": int(args.episodes_per_scenario),
        "control_frequency_hz": float(args.control_frequency_hz),
        "delay_semantics": "command delay in control steps; convert to milliseconds using control_frequency_hz",
        "noise_semantics": "noise_std_scale multiplies configured standard-deviation coefficients",
        "git": git_metadata(),
        "calibration_profiles_are_measured": False,
        "runs": records,
        "full_detached_comparisons": comparisons,
        "variant_vs_nominal_comparisons": variant_deltas,
    }
    (root / "ood_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

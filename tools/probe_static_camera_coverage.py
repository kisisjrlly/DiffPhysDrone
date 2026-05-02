#!/usr/bin/env python3
"""
Probe whether one static camera setting can cover all shared-gate sensor regimes.

This is not a training script.  It renders fixed drone poses looking at the gate
opening while sweeping a power/exposure/gain grid, then compares:

- best global static camera setting over all scenes/slots/poses
- best per-scene static setting
- per-state oracle, which may choose a different setting for each state

If the best global static score is close to the oracle score, fixed-camera
baselines can likely learn the task after enough training.  If there is a clear
gap and the per-scene optima conflict, the benchmark has useful active sensing
pressure.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import canonicalize_sun_glare_slot, set_global_seed  # noqa: E402
from train_utils import build_env  # noqa: E402
from tools.probe_opening_depth_views import (  # noqa: E402
    SCENE_ORDER,
    SLOT_ORDER,
    ProbePose,
    _build_project_args,
    _local_mask_metrics,
    _make_poses,
    _opening_target,
    _parse_float_list,
    _parse_scenes,
    _parse_slots,
    _set_pose_look_at,
    _to_float,
    _write_csv,
)


@dataclass(frozen=True)
class GridSetting:
    name: str
    power: float
    exposure: float
    gain: float


def _parse_grid(text: str) -> list[float]:
    vals = _parse_float_list(text)
    out: list[float] = []
    for v in vals:
        v = float(v)
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"camera grid value must be in [0, 1], got {v}")
        if v not in out:
            out.append(v)
    return out


def _make_grid(powers: list[float], exposures: list[float], gains: list[float]) -> list[GridSetting]:
    settings: list[GridSetting] = []
    for p in powers:
        for e in exposures:
            for g in gains:
                settings.append(GridSetting(f"p{p:.2f}_e{e:.2f}_g{g:.2f}", p, e, g))
    return settings


def _scalar(value, default=0.0) -> float:
    return _to_float(value, default)


def _score_row(row: dict, weights: dict[str, float]) -> float:
    return (
        float(weights["fill"]) * float(row["local_fill"])
        + float(weights["quality"]) * float(row["local_quality_mean"])
        - float(weights["invalid"]) * float(row["glare_invalid_rate"])
        - float(weights["power"]) * max(0.0, float(row["power"]) - float(weights["power_baseline"])) ** 2
        - float(weights["blur"]) * float(row["exposure"]) ** 2
        - float(weights["gain"]) * float(row["gain"]) ** 2
    )


def _render_score(env, args, pose: ProbePose, target: torch.Tensor, setting: GridSetting,
                  weights: dict[str, float]) -> dict:
    _set_pose_look_at(env, pose, target)
    device = env.device
    power = torch.full((env.batch_size,), float(setting.power), device=device)
    exposure = torch.full((env.batch_size,), float(setting.exposure), device=device)
    gain = torch.full((env.batch_size,), float(setting.gain), device=device)
    depth, quality = env.render_diff_depth(power, exposure, gain)
    debug = env.export_last_diff_depth_debug(0)
    scalars = debug.get("scalars", {})
    images = debug.get("images", {})
    depth_np = depth[0].detach().cpu().numpy()
    quality_np = None if quality is None else quality[0].detach().cpu().numpy()
    local = _local_mask_metrics(
        depth_np,
        quality_np,
        images.get("scene_mask"),
        args.depth_min_valid,
        images.get("raw_depth_map"),
    )
    row = {
        "scene": env.current_scene_name,
        "sample": int(getattr(env, "_probe_sample_idx", 0)),
        "slot": str((env.current_scene_effects or {}).get("decision_open_slot_name", "")),
        "pose": pose.name,
        "x": pose.x,
        "y": pose.y,
        "z": pose.z,
        "power": float(setting.power),
        "exposure": float(setting.exposure),
        "gain": float(setting.gain),
        "setting": setting.name,
        "scene_effect_mean": _scalar(scalars.get("scene_effect_mean"), 0.0),
        "hazard_mask_mean": _scalar(scalars.get("hazard_mask_mean"), 0.0),
        "sun_mask_mean": _scalar(scalars.get("sun_mask_mean"), 0.0),
        "glare_invalid_rate": _scalar(scalars.get("glare_invalid_rate"), 0.0),
        "quality_mean": float(np.mean(quality_np)) if quality_np is not None else float("nan"),
        "invalid_rate": _scalar(scalars.get("invalid_rate"), 0.0),
        "ambient_ir_mean": _scalar(scalars.get("ambient_ir_mean"), 0.0),
        "signal_active_mean": _scalar(scalars.get("signal_active_mean"), 0.0),
        "signal_passive_mean": _scalar(scalars.get("signal_passive_mean"), 0.0),
        "spec_bloom_mean": _scalar(scalars.get("spec_bloom_mean"), 0.0),
        "washout_mean": _scalar(scalars.get("washout_mean"), 0.0),
    }
    row.update(local)
    row["score"] = _score_row(row, weights)
    return row


def _mean(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return sum(float(r[key]) for r in rows) / len(rows)


def _setting_key(row: dict) -> tuple[float, float, float]:
    return (float(row["power"]), float(row["exposure"]), float(row["gain"]))


def _setting_name(key: tuple[float, float, float]) -> str:
    return f"p={key[0]:.2f}, e={key[1]:.2f}, g={key[2]:.2f}"


def _fail_rate(rows: list[dict], threshold: float) -> float:
    if not rows:
        return float("nan")
    return sum(1.0 for r in rows if float(r["local_fill"]) < float(threshold)) / len(rows)


def _summarize(detail_rows: list[dict], min_fill_rate: float) -> tuple[list[dict], list[str]]:
    by_setting: dict[tuple[float, float, float], list[dict]] = defaultdict(list)
    by_scene_setting: dict[tuple[str, tuple[float, float, float]], list[dict]] = defaultdict(list)
    by_state: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for row in detail_rows:
        key = _setting_key(row)
        by_setting[key].append(row)
        by_scene_setting[(str(row["scene"]), key)].append(row)
        by_state[(str(row["scene"]), int(row["sample"]), str(row["slot"]), str(row["pose"]))].append(row)

    global_rank = sorted(
        ((key, _mean(rows, "score"), rows) for key, rows in by_setting.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    best_global_key, best_global_score, best_global_rows = global_rank[0]

    scene_best = {}
    for scene in SCENE_ORDER:
        candidates = [
            (key, _mean(rows, "score"), rows)
            for (scene_name, key), rows in by_scene_setting.items()
            if scene_name == scene
        ]
        if candidates:
            scene_best[scene] = sorted(candidates, key=lambda item: item[1], reverse=True)[0]

    oracle_rows = []
    for state, rows in by_state.items():
        oracle_rows.append(max(rows, key=lambda r: float(r["score"])))
    oracle_score = _mean(oracle_rows, "score")

    best_global_by_scene = {}
    for scene in SCENE_ORDER:
        rows = [r for r in best_global_rows if r["scene"] == scene]
        if rows:
            best_global_by_scene[scene] = _mean(rows, "score")

    summary_rows: list[dict] = []
    for rank, (key, score, rows) in enumerate(global_rank[:20], start=1):
        out = {
            "rank": rank,
            "scope": "global_static",
            "power": key[0],
            "exposure": key[1],
            "gain": key[2],
            "score": score,
            "local_fill": _mean(rows, "local_fill"),
            "local_quality_mean": _mean(rows, "local_quality_mean"),
            "glare_invalid_rate": _mean(rows, "glare_invalid_rate"),
            "fail_rate": _fail_rate(rows, min_fill_rate),
        }
        for scene in SCENE_ORDER:
            scene_rows = [r for r in rows if r["scene"] == scene]
            out[f"score_{scene}"] = _mean(scene_rows, "score") if scene_rows else float("nan")
        summary_rows.append(out)

    for scene, (key, score, rows) in scene_best.items():
        summary_rows.append({
            "rank": 1,
            "scope": f"best_static_for_{scene}",
            "power": key[0],
            "exposure": key[1],
            "gain": key[2],
            "score": score,
            "local_fill": _mean(rows, "local_fill"),
            "local_quality_mean": _mean(rows, "local_quality_mean"),
            "glare_invalid_rate": _mean(rows, "glare_invalid_rate"),
            "fail_rate": _fail_rate(rows, min_fill_rate),
            "score_glare": score if scene == "glare" else float("nan"),
            "score_specular": score if scene == "specular" else float("nan"),
            "score_dark": score if scene == "dark" else float("nan"),
        })

    summary_rows.append({
        "rank": 1,
        "scope": "per_state_oracle",
        "power": float("nan"),
        "exposure": float("nan"),
        "gain": float("nan"),
        "score": oracle_score,
        "local_fill": _mean(oracle_rows, "local_fill"),
        "local_quality_mean": _mean(oracle_rows, "local_quality_mean"),
        "glare_invalid_rate": _mean(oracle_rows, "glare_invalid_rate"),
        "fail_rate": _fail_rate(oracle_rows, min_fill_rate),
        "score_glare": _mean([r for r in oracle_rows if r["scene"] == "glare"], "score"),
        "score_specular": _mean([r for r in oracle_rows if r["scene"] == "specular"], "score"),
        "score_dark": _mean([r for r in oracle_rows if r["scene"] == "dark"], "score"),
    })

    oracle_gap = oracle_score - best_global_score
    global_fail = _fail_rate(best_global_rows, min_fill_rate)
    oracle_fail = _fail_rate(oracle_rows, min_fill_rate)
    scene_lines = []
    for scene in SCENE_ORDER:
        if scene not in scene_best:
            continue
        key, score, rows = scene_best[scene]
        fixed_scene_score = best_global_by_scene.get(scene, float("nan"))
        fixed_scene_rows = [r for r in best_global_rows if r["scene"] == scene]
        scene_lines.append(
            f"- {scene}: best {_setting_name(key)} score={score:.4f}; "
            f"global-fixed score={fixed_scene_score:.4f}; gap={score - fixed_scene_score:.4f}; "
            f"global-fixed fail={_fail_rate(fixed_scene_rows, min_fill_rate):.3f}; "
            f"scene-best fail={_fail_rate(rows, min_fill_rate):.3f}"
        )

    top_lines = []
    for rank, (key, score, rows) in enumerate(global_rank[:5], start=1):
        top_lines.append(
            f"{rank}. {_setting_name(key)} score={score:.4f} "
            f"fill={_mean(rows, 'local_fill'):.3f} "
            f"quality={_mean(rows, 'local_quality_mean'):.3f} "
            f"invalid={_mean(rows, 'glare_invalid_rate'):.3f} "
            f"fail={_fail_rate(rows, min_fill_rate):.3f}"
        )

    lines = [
        "# Static Camera Coverage Probe",
        "",
        f"- evaluated rows: {len(detail_rows)}",
        f"- static settings: {len(by_setting)}",
        f"- states: {len(by_state)}",
        f"- best global static: {_setting_name(best_global_key)} score={best_global_score:.4f}",
        f"- per-state oracle score: {oracle_score:.4f}",
        f"- oracle minus best-static gap: {oracle_gap:.4f}",
        f"- min local fill threshold: {float(min_fill_rate):.3f}",
        f"- best global static fail rate: {global_fail:.3f}",
        f"- per-state oracle fail rate: {oracle_fail:.3f}",
        "",
        "Top global static settings:",
        "",
        *top_lines,
        "",
        "Per-scene optima:",
        "",
        *scene_lines,
        "",
        "Interpretation:",
        "",
        "- A small oracle gap means a fixed camera can cover most sensor regimes.",
        "- A large oracle gap and conflicting per-scene optima indicate active camera control should matter.",
        "- This probe only measures perception pressure, not full flight success.",
        "",
    ]
    return summary_rows, lines


def _make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_final_full.args")
    parser.add_argument("--out_dir", default="paper/experiment/results/static_camera_coverage")
    parser.add_argument("--scenarios", nargs="*", default=list(SCENE_ORDER))
    parser.add_argument("--slots", nargs="*", default=list(SLOT_ORDER))
    parser.add_argument("--xs", default="-0.8,0.0,0.6,1.2",
                        help="Comma-separated x positions. Default focuses on opening-visible approach states.")
    parser.add_argument("--path_y_mode", default="slot", choices=["center", "blend", "slot"])
    parser.add_argument("--powers", default="0.25,0.40,0.55,0.70,0.85,0.95")
    parser.add_argument("--exposures", default="0.15,0.30,0.45,0.60,0.75,0.90")
    parser.add_argument("--gains", default="0.05,0.25,0.45,0.65")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sensor_impl", default="cuda", choices=["cuda", "python"],
                        help="Implementation used by env.render_diff_depth for the differentiable depth sensor.")
    parser.add_argument("--keep_scene_randomize", action="store_true")
    parser.add_argument("--keep_random_rotation", action="store_true",
                        help="Keep --random_rotation from config. Default disables it for geometry-readable probes.")
    parser.add_argument("--random_samples", type=int, default=1,
                        help="Number of reset samples per scene/slot. Useful with --keep_scene_randomize.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--score_fill_weight", type=float, default=1.0)
    parser.add_argument("--score_quality_weight", type=float, default=0.75)
    parser.add_argument("--score_invalid_weight", type=float, default=0.35)
    parser.add_argument("--score_power_weight", type=float, default=0.05)
    parser.add_argument("--score_blur_weight", type=float, default=0.02)
    parser.add_argument("--score_gain_weight", type=float, default=0.01)
    return parser


def main():
    parser = _make_arg_parser()
    script_args, project_overrides = parser.parse_known_args()
    out_dir = Path(script_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    project_args = _build_project_args(Path(script_args.config), project_overrides)
    project_args.diff_sensor_impl["diff_depth"] = str(script_args.sensor_impl)
    if not script_args.keep_scene_randomize:
        project_args.sun_glare_randomize = False
    if not script_args.keep_random_rotation:
        project_args.random_rotation = False
    if script_args.seed is not None:
        project_args.seed = int(script_args.seed)
    set_global_seed(project_args.seed, project_args.deterministic)

    device = torch.device(script_args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    scenes = _parse_scenes(script_args.scenarios)
    slots = _parse_slots(script_args.slots)
    xs = _parse_float_list(script_args.xs)
    settings = _make_grid(
        _parse_grid(script_args.powers),
        _parse_grid(script_args.exposures),
        _parse_grid(script_args.gains),
    )
    weights = {
        "fill": float(script_args.score_fill_weight),
        "quality": float(script_args.score_quality_weight),
        "invalid": float(script_args.score_invalid_weight),
        "power": float(script_args.score_power_weight),
        "blur": float(script_args.score_blur_weight),
        "gain": float(script_args.score_gain_weight),
        "power_baseline": float(project_args.cam_power_baseline),
    }

    detail_rows: list[dict] = []
    with torch.no_grad():
        for scene in scenes:
            for slot in slots:
                for sample_idx in range(max(1, int(script_args.random_samples))):
                    cond_args = copy.deepcopy(project_args)
                    cond_args.scenarios = [scene]
                    cond_args.sun_glare_eval_slot = canonicalize_sun_glare_slot(slot)
                    env = build_env(1, cond_args, device, eval_mode=True)
                    env._probe_sample_idx = int(sample_idx)
                    env.reset(scene_name=scene)
                    env._probe_sample_idx = int(sample_idx)
                    target = _opening_target(env)
                    poses = _make_poses(env, xs, script_args.path_y_mode)
                    for pose in poses:
                        for setting in settings:
                            detail_rows.append(_render_score(env, cond_args, pose, target, setting, weights))

    summary_rows, report_lines = _summarize(detail_rows, float(project_args.diff_depth_min_fill_rate))
    _write_csv(out_dir / "static_camera_coverage_detail.csv", detail_rows)
    _write_csv(out_dir / "static_camera_coverage_summary.csv", summary_rows)
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[static-camera-probe] scenes={scenes} slots={slots}")
    print(f"[static-camera-probe] settings={len(settings)} states={len(detail_rows) // max(len(settings), 1)} rows={len(detail_rows)}")
    print(f"[static-camera-probe] out_dir={out_dir}")
    for line in report_lines:
        if line.startswith("- best global") or line.startswith("- per-state") or line.startswith("- oracle"):
            print(f"[static-camera-probe] {line}")


if __name__ == "__main__":
    main()

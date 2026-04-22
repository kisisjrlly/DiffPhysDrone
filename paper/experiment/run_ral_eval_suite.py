#!/usr/bin/env python3
"""
Batch evaluation and plotting for the RAL experiment suite.

Usage example:
  python3 paper/experiment/run_ral_eval_suite.py \
    --ours_ckpt checkpoint/.../checkpoint0024.pth \
    --fixed_ckpt checkpoint/.../checkpoint0024.pth \
    --nondiff_ckpt checkpoint/.../checkpoint0024.pth
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = None
plt = None
build_parser = None
parse_diff_sensor_impl = None
parse_scenarios = None
parse_sun_glare_levels = None
canonicalize_sun_glare_level = None
set_global_seed = None
validate_args = None
run_one_episode = None
Model = None
build_env = None
format_results_dir = None


METHOD_SPECS = {
    "ours": {
        "label": "Ours",
        "args": {
            "camera_control_mode": "learned",
            "sensor_grad_mode": "full",
        },
        "color": "#d62728",
    },
    "fixed": {
        "label": "Fixed Camera",
        "args": {
            "camera_control_mode": "fixed",
            "sensor_grad_mode": "full",
        },
        "color": "#1f77b4",
    },
    "nondiff": {
        "label": "Non-Diff Active",
        "args": {
            "camera_control_mode": "learned",
            "sensor_grad_mode": "detached",
        },
        "color": "#2ca02c",
    },
}

GLARE_LEVEL_ORDER = ["l0", "l1", "l2", "l3"]


def _lazy_imports():
    global torch, plt
    global build_parser, parse_diff_sensor_impl, parse_scenarios
    global parse_sun_glare_levels, canonicalize_sun_glare_level
    global set_global_seed, validate_args, run_one_episode, Model, build_env
    global format_results_dir

    import torch as _torch
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as _plt

    from config import (
        build_parser as _build_parser,
        parse_diff_sensor_impl as _parse_diff_sensor_impl,
        parse_scenarios as _parse_scenarios,
        parse_sun_glare_levels as _parse_sun_glare_levels,
        canonicalize_sun_glare_level as _canonicalize_sun_glare_level,
        set_global_seed as _set_global_seed,
        validate_args as _validate_args,
    )
    from eval import run_one_episode as _run_one_episode
    from model import Model as _Model
    from format_ral_results import format_results_dir as _format_results_dir
    from train_utils import build_env as _build_env

    torch = _torch
    plt = _plt
    build_parser = _build_parser
    parse_diff_sensor_impl = _parse_diff_sensor_impl
    parse_scenarios = _parse_scenarios
    parse_sun_glare_levels = _parse_sun_glare_levels
    canonicalize_sun_glare_level = _canonicalize_sun_glare_level
    set_global_seed = _set_global_seed
    validate_args = _validate_args
    run_one_episode = _run_one_episode
    Model = _Model
    build_env = _build_env
    format_results_dir = _format_results_dir


def _read_args_tokens(config_path: Path) -> list[str]:
    tokens: list[str] = []
    with config_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            tokens.extend(line.split())
    return tokens


def _load_args_from_config(config_path: Path):
    parser = build_parser()
    args = parser.parse_args(_read_args_tokens(config_path))
    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.scenarios = parse_scenarios(args.scenarios)
    args.sun_glare_levels = parse_sun_glare_levels(args.sun_glare_levels)
    if args.sun_glare_eval_level is not None:
        args.sun_glare_eval_level = canonicalize_sun_glare_level(args.sun_glare_eval_level)
    validate_args(args)
    return args


def _build_model(args, device: torch.device) -> Model:
    obs_dim = 7 if args.no_odom else 10
    model = Model(
        obs_dim, 6,
        include_camera_state_in_obs=args.include_camera_state_in_obs,
        use_policy_intent=args.policy_output_intent,
        intent_dim=9,
        depth_nn_width=args.depth_nn_width,
        depth_nn_height=args.depth_nn_height,
        depth_use_pipeline=args.depth_use_pipeline,
        depth_min_valid=args.depth_min_valid,
        depth_max_range=args.depth_max_range,
    ).to(device)
    return model


def _load_checkpoint(model: Model, ckpt_path: Path, device: torch.device):
    state_dict = torch.load(str(ckpt_path), map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys for {ckpt_path.name}: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys for {ckpt_path.name}: {unexpected}")
    model.eval()


def _condition_label(scene_name: str, glare_level: str | None) -> str:
    if scene_name == "base":
        return "base"
    if scene_name == "sun_glare":
        return f"sun_glare_{glare_level}"
    return scene_name


def _compute_t_entry(trace_rows: list[dict]) -> int | None:
    if not trace_rows:
        return None
    zone_enter_x = float(trace_rows[0].get("zone_enter_x", 0.0))
    for row in trace_rows:
        if float(row.get("x", -1e9)) > zone_enter_x:
            return int(row["step_idx"])
    return None


def _summarize_rows(rows: list[dict], metric_keys: list[str]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row["method_key"]), str(row["condition"]))
        grouped.setdefault(key, []).append(row)

    out = []
    for (method_key, condition), items in grouped.items():
        record = {
            "method_key": method_key,
            "method_label": items[0]["method_label"],
            "condition": condition,
            "scene_name": items[0]["scene_name"],
            "glare_level": items[0]["glare_level"],
            "episodes": len(items),
        }
        for key in metric_keys:
            vals = [float(x.get(key, 0.0)) for x in items]
            record[key] = float(sum(vals) / max(len(vals), 1))
        out.append(record)
    return sorted(out, key=lambda x: (x["condition"], x["method_key"]))


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_success_vs_glare(summary_rows: list[dict], output_path: Path):
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=180)
    x = np.arange(len(GLARE_LEVEL_ORDER))
    for method_key, spec in METHOD_SPECS.items():
        ys = []
        for level in GLARE_LEVEL_ORDER:
            cond = f"sun_glare_{level}"
            row = next((r for r in summary_rows if r["method_key"] == method_key and r["condition"] == cond), None)
            ys.append(float(row["success_rate"]) if row is not None else np.nan)
        ax.plot(x, ys, marker="o", linewidth=2.2, label=spec["label"], color=spec["color"])
    ax.set_xticks(x, [lvl.upper() for lvl in GLARE_LEVEL_ORDER])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Sun Glare Severity")
    ax.set_ylabel("Success Rate")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_quality_and_stop(summary_rows: list[dict], output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=180, sharex=True)
    x = np.arange(len(GLARE_LEVEL_ORDER))
    for method_key, spec in METHOD_SPECS.items():
        q_vals = []
        s_vals = []
        for level in GLARE_LEVEL_ORDER:
            cond = f"sun_glare_{level}"
            row = next((r for r in summary_rows if r["method_key"] == method_key and r["condition"] == cond), None)
            q_vals.append(float(row["local_glare_quality"]) if row is not None else np.nan)
            s_vals.append(float(row["stop_before_glare_rate"]) if row is not None else np.nan)
        axes[0].plot(x, q_vals, marker="o", linewidth=2.0, label=spec["label"], color=spec["color"])
        axes[1].plot(x, s_vals, marker="o", linewidth=2.0, label=spec["label"], color=spec["color"])

    axes[0].set_ylabel("Local Glare Quality")
    axes[1].set_ylabel("Stop-Before-Glare Rate")
    for ax in axes:
        ax.set_xticks(x, [lvl.upper() for lvl in GLARE_LEVEL_ORDER])
        ax.set_xlabel("Sun Glare Severity")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _aggregate_aligned_trace(trace_rows: list[dict], method_key: str, condition: str, rel_min: int, rel_max: int):
    rel_buckets: dict[int, dict[str, list[float]]] = {}
    grouped: dict[int, list[dict]] = {}
    for row in trace_rows:
        if row["method_key"] != method_key or row["condition"] != condition:
            continue
        grouped.setdefault(int(row["episode_idx"]), []).append(row)

    for _, episode_rows in grouped.items():
        episode_rows = sorted(episode_rows, key=lambda x: int(x["step_idx"]))
        t_entry = _compute_t_entry(episode_rows)
        if t_entry is None:
            continue
        for row in episode_rows:
            rel_t = int(row["step_idx"]) - t_entry
            if rel_t < rel_min or rel_t > rel_max:
                continue
            bucket = rel_buckets.setdefault(rel_t, {
                "power": [],
                "exposure": [],
                "gain": [],
                "glare_quality_mean": [],
            })
            bucket["power"].append(float(row["power"]))
            bucket["exposure"].append(float(row["exposure"]))
            bucket["gain"].append(float(row["gain"]))
            bucket["glare_quality_mean"].append(float(row["glare_quality_mean"]))

    xs = list(range(rel_min, rel_max + 1))
    out = {"x": xs}
    for key in ("power", "exposure", "gain", "glare_quality_mean"):
        vals = []
        for rel_t in xs:
            bucket = rel_buckets.get(rel_t, {})
            series = bucket.get(key, [])
            vals.append(float(sum(series) / len(series)) if series else math.nan)
        out[key] = vals
    return out


def _plot_event_aligned(trace_rows: list[dict], plot_level: str, output_path: Path, rel_min: int = -12, rel_max: int = 24):
    condition = f"sun_glare_{plot_level}"
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 8.5), dpi=180, sharex=True)
    keys = [
        ("power", "Power"),
        ("exposure", "Exposure"),
        ("gain", "Gain"),
        ("glare_quality_mean", "Local Glare Quality"),
    ]
    for method_key, spec in METHOD_SPECS.items():
        agg = _aggregate_aligned_trace(trace_rows, method_key, condition, rel_min, rel_max)
        x = np.asarray(agg["x"], dtype=np.float32)
        for ax, (key, ylabel) in zip(axes, keys):
            y = np.asarray(agg[key], dtype=np.float32)
            ax.plot(x, y, linewidth=2.0, label=spec["label"], color=spec["color"])
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    axes[-1].axvline(0.0, color="k", linestyle="--", linewidth=1.2, alpha=0.65)
    axes[-1].set_xlabel("t - t_entry (steps)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_trajectory(trace_rows: list[dict], plot_level: str, output_path: Path):
    condition = f"sun_glare_{plot_level}"
    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=180)
    for method_key, spec in METHOD_SPECS.items():
        rows = [r for r in trace_rows if r["method_key"] == method_key and r["condition"] == condition and int(r["episode_idx"]) == 0]
        rows = sorted(rows, key=lambda x: int(x["step_idx"]))
        if not rows:
            continue
        xs = [float(r["x"]) for r in rows]
        ys = [float(r["y"]) for r in rows]
        ax.plot(xs, ys, linewidth=2.2, label=spec["label"], color=spec["color"])
        ax.scatter([xs[0]], [ys[0]], color=spec["color"], s=18)
        ax.scatter([xs[-1]], [ys[-1]], color=spec["color"], s=24, marker="x")
    ax.scatter([-3.0], [0.0], color="black", s=40, marker="o", label="Start")
    ax.scatter([2.0], [0.0], color="black", s=48, marker="*", label="Goal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Top-Down Trajectories ({plot_level.upper()})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _dump_metadata(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _evaluate_method(method_key: str, ckpt_path: Path, base_args, device: torch.device,
                     episodes_per_condition: int, include_base: bool, plot_level: str):
    spec = METHOD_SPECS[method_key]
    method_args = copy.deepcopy(base_args)
    for key, value in spec["args"].items():
        setattr(method_args, key, value)
    method_args.batch_size = 1
    method_args.vis_enable = False
    method_args.wandb_disabled = True
    validate_args(method_args)

    model = _build_model(method_args, device)
    _load_checkpoint(model, ckpt_path, device)
    vis = SimpleNamespace(enabled=False)

    episode_rows = []
    trace_rows = []
    conditions = []
    if include_base:
        conditions.append(("base", None))
    for level in GLARE_LEVEL_ORDER:
        conditions.append(("sun_glare", level))

    for cond_idx, (scene_name, glare_level) in enumerate(conditions):
        cond_args = copy.deepcopy(method_args)
        cond_args.scenarios = [scene_name]
        cond_args.sun_glare_eval_level = glare_level if scene_name == "sun_glare" else None
        cond_args.eval_episodes = int(episodes_per_condition)
        validate_args(cond_args)
        set_global_seed(int(cond_args.seed) + cond_idx, cond_args.deterministic)
        env = build_env(cond_args.batch_size, cond_args, device, eval_mode=True)
        cond_label = _condition_label(scene_name, glare_level)
        print(f"[suite] method={method_key} condition={cond_label} episodes={episodes_per_condition}")
        for ep_idx in range(episodes_per_condition):
            metrics, trace = run_one_episode(
                ep_idx, scene_name, glare_level, cond_args, model, env, vis, device, collect_trace=True)
            row = dict(metrics)
            row.update({
                "method_key": method_key,
                "method_label": spec["label"],
                "condition": cond_label,
                "episode_idx": int(ep_idx),
                "checkpoint": str(ckpt_path),
            })
            episode_rows.append(row)
            t_entry = _compute_t_entry(trace)
            for item in trace:
                item = dict(item)
                item.update({
                    "method_key": method_key,
                    "method_label": spec["label"],
                    "condition": cond_label,
                    "checkpoint": str(ckpt_path),
                    "t_entry": -1 if t_entry is None else int(t_entry),
                    "t_minus_entry": math.nan if t_entry is None else int(item["step_idx"]) - int(t_entry),
                })
                trace_rows.append(item)
        del env
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return episode_rows, trace_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/paper_final_full.args")
    parser.add_argument("--ours_ckpt", type=str, required=True)
    parser.add_argument("--fixed_ckpt", type=str, required=True)
    parser.add_argument("--nondiff_ckpt", type=str, required=True)
    parser.add_argument("--episodes_per_condition", type=int, default=12)
    parser.add_argument("--plot_level", type=str, default="l3")
    parser.add_argument("--skip_base", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    _lazy_imports()
    config_path = (ROOT / args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")

    ckpts = {
        "ours": Path(args.ours_ckpt).resolve(),
        "fixed": Path(args.fixed_ckpt).resolve(),
        "nondiff": Path(args.nondiff_ckpt).resolve(),
    }
    for name, path in ckpts.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")

    plot_level = canonicalize_sun_glare_level(args.plot_level)
    if plot_level not in GLARE_LEVEL_ORDER:
        raise ValueError(f"unsupported plot level: {plot_level}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (ROOT / "paper" / "experiment" / "results" / f"results_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_args = _load_args_from_config(config_path)
    base_args.seed = int(args.seed)
    base_args.vis_enable = False
    base_args.wandb_disabled = True

    all_episode_rows = []
    all_trace_rows = []
    for method_key, ckpt_path in ckpts.items():
        episode_rows, trace_rows = _evaluate_method(
            method_key=method_key,
            ckpt_path=ckpt_path,
            base_args=base_args,
            device=device,
            episodes_per_condition=int(args.episodes_per_condition),
            include_base=not args.skip_base,
            plot_level=plot_level,
        )
        all_episode_rows.extend(episode_rows)
        all_trace_rows.extend(trace_rows)

    summary_metric_keys = [
        "success_rate",
        "collision_rate",
        "stop_before_glare_rate",
        "time_to_goal",
        "avg_speed",
        "fill_rate",
        "local_glare_quality",
        "local_glare_invalid_rate",
        "power_mean",
        "exposure_mean",
        "gain_mean",
        "energy_proxy",
        "blur_proxy",
        "noise_proxy",
    ]
    summary_rows = _summarize_rows(all_episode_rows, summary_metric_keys)

    _write_csv(output_dir / "episode_metrics.csv", all_episode_rows)
    _write_csv(output_dir / "trace_metrics.csv", all_trace_rows)
    _write_csv(output_dir / "summary_metrics.csv", summary_rows)
    _dump_metadata(output_dir / "meta.json", {
        "config": str(config_path),
        "checkpoints": {k: str(v) for k, v in ckpts.items()},
        "episodes_per_condition": int(args.episodes_per_condition),
        "plot_level": plot_level,
        "device": str(device),
        "skip_base": bool(args.skip_base),
    })

    _plot_success_vs_glare(summary_rows, output_dir / "success_vs_glare.png")
    _plot_quality_and_stop(summary_rows, output_dir / "quality_and_stop_vs_glare.png")
    _plot_event_aligned(all_trace_rows, plot_level=plot_level, output_path=output_dir / f"event_aligned_{plot_level}.png")
    _plot_trajectory(all_trace_rows, plot_level=plot_level, output_path=output_dir / f"trajectory_{plot_level}.png")

    formatted_outputs = format_results_dir(output_dir)

    print("[suite] done.")
    print(f"[suite] output_dir: {output_dir}")
    print(f"[suite] summary csv: {output_dir / 'summary_metrics.csv'}")
    print(f"[suite] episode csv: {output_dir / 'episode_metrics.csv'}")
    print(f"[suite] trace csv  : {output_dir / 'trace_metrics.csv'}")
    for name, path in formatted_outputs.items():
        print(f"[suite] {name}: {path}")


if __name__ == "__main__":
    main()

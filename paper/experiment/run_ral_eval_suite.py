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
canonicalize_sun_glare_slot = None
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
    "blind": {
        "label": "Zero-Depth Trained",
        "args": {
            "camera_control_mode": "fixed",
            "sensor_grad_mode": "detached",
            "policy_depth_mode": "zero",
            "include_camera_state_in_obs": False,
        },
        "color": "#9467bd",
    },
    "ours_zero": {
        "label": "Ours w/ Zero Depth",
        "args": {
            "camera_control_mode": "learned",
            "sensor_grad_mode": "full",
            "policy_depth_mode": "zero",
        },
        "color": "#ff7f0e",
    },
    "fixed": {
        "label": "Fixed Camera",
        "args": {
            "camera_control_mode": "fixed",
            "sensor_grad_mode": "full",
        },
        "color": "#1f77b4",
    },
    "fixed_random": {
        "label": "Random Static Camera",
        "args": {
            "camera_control_mode": "fixed_random_static",
            "sensor_grad_mode": "full",
        },
        "color": "#17becf",
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
SUN_GLARE_SLOT_ORDER = ["far_left", "left", "right", "far_right"]
SUN_GLARE_SLOT_Y = {
    "far_left": -1.12,
    "left": -0.56,
    "right": 0.56,
    "far_right": 1.12,
}
ENTRY_PRE_STEPS = 5
ENTRY_POST_STEPS = 5
GATE_X = 1.82


def _lazy_imports():
    global torch, plt
    global build_parser, parse_diff_sensor_impl, parse_scenarios
    global parse_sun_glare_levels, canonicalize_sun_glare_level, canonicalize_sun_glare_slot
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
        canonicalize_sun_glare_slot as _canonicalize_sun_glare_slot,
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
    canonicalize_sun_glare_slot = _canonicalize_sun_glare_slot
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
    if getattr(args, "sun_glare_eval_slot", None) is not None:
        args.sun_glare_eval_slot = canonicalize_sun_glare_slot(args.sun_glare_eval_slot)
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


def _checkpoint_state_input_dim(ckpt_path: Path) -> int | None:
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    weight = state_dict.get("v_proj.weight")
    if weight is None or not hasattr(weight, "shape") or len(weight.shape) != 2:
        return None
    return int(weight.shape[1])


def _match_args_to_checkpoint(args, ckpt_path: Path):
    input_dim = _checkpoint_state_input_dim(ckpt_path)
    if input_dim is None:
        return args
    obs_dim = 7 if args.no_odom else 10
    if input_dim == obs_dim:
        expected_include_camera = False
    elif input_dim == obs_dim + 3:
        expected_include_camera = True
    else:
        raise ValueError(
            f"{ckpt_path} 的 v_proj.weight 输入维度为 {input_dim}，"
            f"与 obs_dim={obs_dim} 或 obs_dim+3={obs_dim + 3} 都不匹配"
        )
    if bool(args.include_camera_state_in_obs) != expected_include_camera:
        print(
            f"[suite][info] checkpoint={ckpt_path.name} requires "
            f"include_camera_state_in_obs={expected_include_camera}; "
            f"override config value {args.include_camera_state_in_obs}."
        )
        args.include_camera_state_in_obs = expected_include_camera
    return args


def _condition_label(scene_name: str, glare_level: str | None, slot_name: str | None = None) -> str:
    if scene_name == "base":
        return "base"
    if scene_name == "sun_glare":
        slot_suffix = f"_{slot_name}" if slot_name else ""
        return f"sun_glare_{glare_level}{slot_suffix}"
    return scene_name


def _compute_t_entry(trace_rows: list[dict]) -> int | None:
    if not trace_rows:
        return None
    for row in trace_rows:
        if float(row.get("scene_effect_mean", 0.0)) > 0.02:
            return int(row["step_idx"])
    zone_enter_x = float(trace_rows[0].get("zone_enter_x", 0.0))
    for row in trace_rows:
        if float(row.get("x", -1e9)) > zone_enter_x:
            return int(row["step_idx"])
    return None


def _window_mean(trace_rows: list[dict], key: str, step_lo: int, step_hi: int) -> float | None:
    vals: list[float] = []
    for row in trace_rows:
        step_idx = int(row["step_idx"])
        if step_idx < step_lo or step_idx > step_hi:
            continue
        raw = row.get(key, "")
        if raw in ("", None):
            continue
        val = float(raw)
        if math.isnan(val):
            continue
        vals.append(val)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _compute_post_entry_metrics(trace_rows: list[dict],
                                pre_steps: int = ENTRY_PRE_STEPS,
                                post_steps: int = ENTRY_POST_STEPS) -> dict[str, float]:
    metrics = {
        "post_entry_available": 0.0,
        "t_entry_step": -1.0,
        "post_entry_local_glare_quality": 0.0,
        "post_entry_local_glare_invalid_rate": 0.0,
        "post_entry_fill_rate": 0.0,
        "post_entry_scene_effect_mean": 0.0,
        "post_entry_power_mean": 0.0,
        "post_entry_exposure_mean": 0.0,
        "post_entry_gain_mean": 0.0,
        "post_entry_power_delta": 0.0,
        "post_entry_exposure_delta": 0.0,
        "post_entry_gain_delta": 0.0,
    }
    if not trace_rows:
        return metrics

    rows = sorted(trace_rows, key=lambda x: int(x["step_idx"]))
    t_entry = _compute_t_entry(rows)
    if t_entry is None:
        return metrics

    pre_lo = max(int(rows[0]["step_idx"]), int(t_entry) - int(pre_steps))
    pre_hi = int(t_entry) - 1
    post_lo = int(t_entry)
    post_hi = int(t_entry) + int(post_steps)

    pre_power = _window_mean(rows, "power", pre_lo, pre_hi)
    pre_exposure = _window_mean(rows, "exposure", pre_lo, pre_hi)
    pre_gain = _window_mean(rows, "gain", pre_lo, pre_hi)
    post_power = _window_mean(rows, "power", post_lo, post_hi)
    post_exposure = _window_mean(rows, "exposure", post_lo, post_hi)
    post_gain = _window_mean(rows, "gain", post_lo, post_hi)

    metrics["post_entry_available"] = 1.0
    metrics["t_entry_step"] = float(t_entry)
    metrics["post_entry_local_glare_quality"] = _window_mean(rows, "glare_quality_mean", post_lo, post_hi) or 0.0
    metrics["post_entry_local_glare_invalid_rate"] = _window_mean(rows, "glare_invalid_rate", post_lo, post_hi) or 0.0
    metrics["post_entry_fill_rate"] = _window_mean(rows, "fill_rate", post_lo, post_hi) or 0.0
    metrics["post_entry_scene_effect_mean"] = _window_mean(rows, "scene_effect_mean", post_lo, post_hi) or 0.0
    metrics["post_entry_power_mean"] = post_power or 0.0
    metrics["post_entry_exposure_mean"] = post_exposure or 0.0
    metrics["post_entry_gain_mean"] = post_gain or 0.0
    if pre_power is not None and post_power is not None:
        metrics["post_entry_power_delta"] = float(post_power - pre_power)
    if pre_exposure is not None and post_exposure is not None:
        metrics["post_entry_exposure_delta"] = float(post_exposure - pre_exposure)
    if pre_gain is not None and post_gain is not None:
        metrics["post_entry_gain_delta"] = float(post_gain - pre_gain)
    return metrics


def _safe_pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(ys) < 3 or len(xs) != len(ys):
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return 0.0
    x = x[valid]
    y = y[valid]
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def _compute_trace_diagnostics(trace_rows: list[dict], min_fill_rate: float,
                               opening_y_fallback: float | None = None) -> dict[str, float]:
    metrics = {
        "opening_y": 0.0,
        "opening_slot_id": 0.0,
        "y_at_gate": 0.0,
        "gate_y_error": 0.0,
        "abs_gate_y_error": 0.0,
        "final_y": 0.0,
        "corr_power_scene_effect": 0.0,
        "corr_exposure_scene_effect": 0.0,
        "corr_gain_scene_effect": 0.0,
        "corr_power_fill_gap": 0.0,
        "corr_exposure_fill_gap": 0.0,
        "corr_gain_fill_gap": 0.0,
    }
    if not trace_rows:
        return metrics

    rows = sorted(trace_rows, key=lambda x: int(x["step_idx"]))
    opening_y_raw = rows[0].get("decision_open_slot_y", None)
    if opening_y_raw in ("", None):
        opening_y = float(opening_y_fallback or 0.0)
    else:
        opening_y = float(opening_y_raw)
    opening_slot_id = float(rows[0].get("decision_open_side_id", 0.0) or 0.0)
    gate_row = min(rows, key=lambda r: abs(float(r.get("x", 0.0)) - GATE_X))
    y_at_gate = float(gate_row.get("y", 0.0))
    final_y = float(rows[-1].get("y", 0.0))

    power = [float(r.get("power", 0.0)) for r in rows]
    exposure = [float(r.get("exposure", 0.0)) for r in rows]
    gain = [float(r.get("gain", 0.0)) for r in rows]
    scene_effect = [float(r.get("scene_effect_mean", 0.0)) for r in rows]
    fill_gap = [max(0.0, float(min_fill_rate) - float(r.get("fill_rate", 0.0))) for r in rows]

    metrics.update({
        "opening_y": opening_y,
        "opening_slot_id": opening_slot_id,
        "y_at_gate": y_at_gate,
        "gate_y_error": y_at_gate - opening_y,
        "abs_gate_y_error": abs(y_at_gate - opening_y),
        "final_y": final_y,
        "corr_power_scene_effect": _safe_pearson(power, scene_effect),
        "corr_exposure_scene_effect": _safe_pearson(exposure, scene_effect),
        "corr_gain_scene_effect": _safe_pearson(gain, scene_effect),
        "corr_power_fill_gap": _safe_pearson(power, fill_gap),
        "corr_exposure_fill_gap": _safe_pearson(exposure, fill_gap),
        "corr_gain_fill_gap": _safe_pearson(gain, fill_gap),
    })
    return metrics


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
            "opening_slot": items[0].get("opening_slot", ""),
            "opening_y": items[0].get("opening_y", ""),
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


def _summary_metric(summary_rows: list[dict], method_key: str, level: str,
                    metric: str, slot_name: str | None = None) -> float:
    vals = []
    for row in summary_rows:
        if row.get("method_key") != method_key:
            continue
        if row.get("glare_level") != level:
            continue
        if slot_name is not None and row.get("opening_slot") != slot_name:
            continue
        if metric not in row:
            continue
        try:
            vals.append(float(row[metric]))
        except (TypeError, ValueError):
            pass
    if not vals:
        return math.nan
    return float(sum(vals) / len(vals))


def _plot_success_vs_glare(summary_rows: list[dict], output_path: Path):
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=180)
    x = np.arange(len(GLARE_LEVEL_ORDER))
    active_methods = [m for m in METHOD_SPECS if any(r.get("method_key") == m for r in summary_rows)]
    for method_key in active_methods:
        spec = METHOD_SPECS[method_key]
        ys = []
        for level in GLARE_LEVEL_ORDER:
            ys.append(_summary_metric(summary_rows, method_key, level, "success_rate"))
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


def _plot_success_by_slot(summary_rows: list[dict], output_path: Path):
    methods = [m for m in METHOD_SPECS if any(r.get("method_key") == m for r in summary_rows)]
    fig, axes = plt.subplots(
        len(methods), 1,
        figsize=(7.2, max(2.0, 2.1 * len(methods))),
        dpi=180,
        squeeze=False,
    )
    for row_idx, method_key in enumerate(methods):
        spec = METHOD_SPECS[method_key]
        mat = np.full((len(GLARE_LEVEL_ORDER), len(SUN_GLARE_SLOT_ORDER)), np.nan, dtype=np.float32)
        for i, level in enumerate(GLARE_LEVEL_ORDER):
            for j, slot in enumerate(SUN_GLARE_SLOT_ORDER):
                mat[i, j] = _summary_metric(summary_rows, method_key, level, "success_rate", slot)
        ax = axes[row_idx, 0]
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_title(spec["label"])
        ax.set_xticks(np.arange(len(SUN_GLARE_SLOT_ORDER)), SUN_GLARE_SLOT_ORDER, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(GLARE_LEVEL_ORDER)), [x.upper() for x in GLARE_LEVEL_ORDER])
        ax.set_ylabel("Glare")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    axes[-1, 0].set_xlabel("Opening Slot")
    fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.025, pad=0.02, label="Success Rate")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_entry_metrics(summary_rows: list[dict], output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=180, sharex=True)
    x = np.arange(len(GLARE_LEVEL_ORDER))
    active_methods = [m for m in METHOD_SPECS if any(r.get("method_key") == m for r in summary_rows)]
    for method_key in active_methods:
        spec = METHOD_SPECS[method_key]
        q_vals = []
        f_vals = []
        for level in GLARE_LEVEL_ORDER:
            q_vals.append(_summary_metric(summary_rows, method_key, level, "post_entry_local_glare_quality"))
            f_vals.append(_summary_metric(summary_rows, method_key, level, "post_entry_fill_rate"))
        axes[0].plot(x, q_vals, marker="o", linewidth=2.0, label=spec["label"], color=spec["color"])
        axes[1].plot(x, f_vals, marker="o", linewidth=2.0, label=spec["label"], color=spec["color"])

    axes[0].set_ylabel("Post-Entry LocalQ")
    axes[1].set_ylabel("Post-Entry Fill")
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


def _plot_event_aligned(trace_rows: list[dict], plot_level: str, plot_slot: str,
                        output_path: Path, rel_min: int = -12, rel_max: int = 24):
    condition = _condition_label("sun_glare", plot_level, plot_slot)
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 8.5), dpi=180, sharex=True)
    keys = [
        ("power", "Power"),
        ("exposure", "Exposure"),
        ("gain", "Gain"),
        ("glare_quality_mean", "Local Glare Quality"),
    ]
    active_methods = [m for m in METHOD_SPECS if any(r.get("method_key") == m for r in trace_rows)]
    for method_key in active_methods:
        spec = METHOD_SPECS[method_key]
        agg = _aggregate_aligned_trace(trace_rows, method_key, condition, rel_min, rel_max)
        x = np.asarray(agg["x"], dtype=np.float32)
        for ax, (key, ylabel) in zip(axes, keys):
            y = np.asarray(agg[key], dtype=np.float32)
            ax.plot(x, y, linewidth=2.0, label=spec["label"], color=spec["color"])
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].set_title(f"{plot_level.upper()} / {plot_slot}")
    axes[-1].axvline(0.0, color="k", linestyle="--", linewidth=1.2, alpha=0.65)
    axes[-1].set_xlabel("t - t_entry (steps)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_trajectory(trace_rows: list[dict], plot_level: str, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), dpi=180, sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, slot in zip(axes, SUN_GLARE_SLOT_ORDER):
        condition = _condition_label("sun_glare", plot_level, slot)
        opening_y = SUN_GLARE_SLOT_Y[slot]
        active_methods = [m for m in METHOD_SPECS if any(r.get("method_key") == m for r in trace_rows)]
        for method_key in active_methods:
            spec = METHOD_SPECS[method_key]
            rows = [
                r for r in trace_rows
                if r["method_key"] == method_key
                and r["condition"] == condition
                and int(r["episode_idx"]) == 0
            ]
            rows = sorted(rows, key=lambda x: int(x["step_idx"]))
            if not rows:
                continue
            xs = [float(r["x"]) for r in rows]
            ys = [float(r["y"]) for r in rows]
            ax.plot(xs, ys, linewidth=1.8, label=spec["label"], color=spec["color"])
            ax.scatter([xs[0]], [ys[0]], color=spec["color"], s=12)
            ax.scatter([xs[-1]], [ys[-1]], color=spec["color"], s=18, marker="x")
        ax.axhline(opening_y, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axvline(GATE_X, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
        ax.set_title(f"{slot} (opening y={opening_y:+.2f})")
        ax.grid(True, alpha=0.25)
    for ax in axes[2:]:
        ax.set_xlabel("x (m)")
    for ax in axes[::2]:
        ax.set_ylabel("y (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", ncol=min(4, len(handles)))
    fig.suptitle(f"Top-Down Trajectories ({plot_level.upper()})", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _dump_metadata(path: Path, payload: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _compute_trajectory_diffs(trace_rows: list[dict], method_a: str, method_b: str) -> list[dict]:
    grouped: dict[tuple[str, int, str], list[dict]] = {}
    for row in trace_rows:
        key = (str(row["condition"]), int(row["episode_idx"]), str(row["method_key"]))
        grouped.setdefault(key, []).append(row)

    out = []
    condition_eps = sorted({(cond, ep) for (cond, ep, _method) in grouped})
    for condition, ep_idx in condition_eps:
        rows_a = grouped.get((condition, ep_idx, method_a), [])
        rows_b = grouped.get((condition, ep_idx, method_b), [])
        if not rows_a or not rows_b:
            continue
        by_step_a = {int(r["step_idx"]): r for r in rows_a}
        by_step_b = {int(r["step_idx"]): r for r in rows_b}
        common_steps = sorted(set(by_step_a) & set(by_step_b))
        if not common_steps:
            continue
        y_diff = [
            abs(float(by_step_a[t].get("y", 0.0)) - float(by_step_b[t].get("y", 0.0)))
            for t in common_steps
        ]
        xy_diff = [
            math.hypot(
                float(by_step_a[t].get("x", 0.0)) - float(by_step_b[t].get("x", 0.0)),
                float(by_step_a[t].get("y", 0.0)) - float(by_step_b[t].get("y", 0.0)),
            )
            for t in common_steps
        ]
        last_t = common_steps[-1]
        first = by_step_a[common_steps[0]]
        out.append({
            "method_a": method_a,
            "method_b": method_b,
            "condition": condition,
            "glare_level": first.get("glare_level", ""),
            "opening_slot": first.get("opening_slot", ""),
            "episode_idx": ep_idx,
            "steps": len(common_steps),
            "mean_abs_y_diff": float(sum(y_diff) / len(y_diff)),
            "max_abs_y_diff": float(max(y_diff)),
            "mean_xy_diff": float(sum(xy_diff) / len(xy_diff)),
            "final_abs_y_diff": abs(
                float(by_step_a[last_t].get("y", 0.0)) - float(by_step_b[last_t].get("y", 0.0))
            ),
        })
    return out


def _evaluate_method(method_key: str, ckpt_path: Path, base_args, device: torch.device,
                     episodes_per_condition: int, include_base: bool, slots: list[str]):
    spec = METHOD_SPECS[method_key]
    method_args = copy.deepcopy(base_args)
    for key, value in spec["args"].items():
        setattr(method_args, key, value)
    method_args = _match_args_to_checkpoint(method_args, ckpt_path)
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
        conditions.append(("base", None, None))
    for level in GLARE_LEVEL_ORDER:
        for slot in slots:
            conditions.append(("sun_glare", level, slot))

    for cond_idx, (scene_name, glare_level, slot_name) in enumerate(conditions):
        cond_args = copy.deepcopy(method_args)
        cond_args.scenarios = [scene_name]
        cond_args.sun_glare_eval_level = glare_level if scene_name == "sun_glare" else None
        cond_args.sun_glare_eval_slot = slot_name if scene_name == "sun_glare" else None
        cond_args.eval_episodes = int(episodes_per_condition)
        validate_args(cond_args)
        set_global_seed(int(cond_args.seed) + cond_idx, cond_args.deterministic)
        env = build_env(cond_args.batch_size, cond_args, device, eval_mode=True)
        cond_label = _condition_label(scene_name, glare_level, slot_name)
        print(f"[suite] method={method_key} condition={cond_label} episodes={episodes_per_condition}")
        for ep_idx in range(episodes_per_condition):
            metrics, trace = run_one_episode(
                ep_idx, scene_name, glare_level, cond_args, model, env, vis, device, collect_trace=True)
            row = dict(metrics)
            # run_one_episode reports collision_rate over the internal continuous-collision
            # subdivision tensor. For this sweep we need episode-level binary outcomes.
            collided_ep = float(row.get("collided", 0.0)) > 0.5
            reached_ep = float(row.get("goal_reach_rate", 0.0)) > 0.5
            row["collision_rate"] = 1.0 if collided_ep else 0.0
            row["success_rate"] = 1.0 if (reached_ep and not collided_ep) else 0.0
            row.update(_compute_post_entry_metrics(trace))
            opening_y_fallback = SUN_GLARE_SLOT_Y.get(slot_name, 0.0) if slot_name else 0.0
            row.update(_compute_trace_diagnostics(
                trace,
                min_fill_rate=cond_args.diff_depth_min_fill_rate,
                opening_y_fallback=opening_y_fallback,
            ))
            row.update({
                "method_key": method_key,
                "method_label": spec["label"],
                "condition": cond_label,
                "scene_name": scene_name,
                "glare_level": glare_level or "",
                "opening_slot": slot_name or "",
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
                    "opening_slot": slot_name or "",
                    "opening_y": opening_y_fallback,
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
    parser.add_argument("--include_fixed_random", action="store_true")
    parser.add_argument("--fixed_random_ckpt", type=str, default=None)
    parser.add_argument("--nondiff_ckpt", type=str, required=True)
    parser.add_argument("--include_blind", action="store_true")
    parser.add_argument("--blind_ckpt", type=str, default=None)
    parser.add_argument("--include_ours_zero_ablation", action="store_true",
                        help="Use ours checkpoint with policy_depth_mode=zero to isolate depth-cue usage.")
    parser.add_argument("--slots", nargs="*", default=list(SUN_GLARE_SLOT_ORDER),
                        help="Sun-glare opening slots to sweep: far_left left right far_right")
    parser.add_argument("--episodes_per_condition", type=int, default=12)
    parser.add_argument("--plot_level", type=str, default="l3")
    parser.add_argument("--include_base", action="store_true")
    parser.add_argument("--skip_base", action="store_true", help=argparse.SUPPRESS)
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
    if args.include_fixed_random:
        if not args.fixed_random_ckpt:
            raise ValueError("--include_fixed_random 时必须提供 --fixed_random_ckpt")
        ckpts["fixed_random"] = Path(args.fixed_random_ckpt).resolve()
    if args.include_ours_zero_ablation:
        ckpts["ours_zero"] = Path(args.ours_ckpt).resolve()
    if args.include_blind:
        if not args.blind_ckpt:
            raise ValueError("--include_blind 时必须显式提供 --blind_ckpt；正式 blind baseline 需要先单独训练再评测")
        ckpts["blind"] = Path(args.blind_ckpt).resolve()
    for name, path in ckpts.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} checkpoint not found: {path}")

    plot_level = canonicalize_sun_glare_level(args.plot_level)
    if plot_level not in GLARE_LEVEL_ORDER:
        raise ValueError(f"unsupported plot level: {plot_level}")
    slots = []
    for raw_slot in args.slots:
        slot = canonicalize_sun_glare_slot(raw_slot)
        if slot not in SUN_GLARE_SLOT_ORDER:
            raise ValueError(f"unsupported slot: {raw_slot}")
        if slot not in slots:
            slots.append(slot)
    if not slots:
        raise ValueError("--slots 至少需要一个 opening slot")

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
    include_base = bool(args.include_base) and not bool(args.skip_base)
    for method_key, ckpt_path in ckpts.items():
        episode_rows, trace_rows = _evaluate_method(
            method_key=method_key,
            ckpt_path=ckpt_path,
            base_args=base_args,
            device=device,
            episodes_per_condition=int(args.episodes_per_condition),
            include_base=include_base,
            slots=slots,
        )
        all_episode_rows.extend(episode_rows)
        all_trace_rows.extend(trace_rows)

    summary_metric_keys = [
        "success_rate",
        "collision_rate",
        "goal_reach_rate",
        "stop_before_glare_rate",
        "time_to_goal",
        "final_goal_dist",
        "avg_speed",
        "fill_rate",
        "local_glare_quality",
        "local_glare_invalid_rate",
        "power_mean",
        "exposure_mean",
        "gain_mean",
        "opening_y",
        "opening_slot_id",
        "y_at_gate",
        "gate_y_error",
        "abs_gate_y_error",
        "final_y",
        "post_entry_available",
        "t_entry_step",
        "post_entry_local_glare_quality",
        "post_entry_local_glare_invalid_rate",
        "post_entry_fill_rate",
        "post_entry_scene_effect_mean",
        "post_entry_power_mean",
        "post_entry_exposure_mean",
        "post_entry_gain_mean",
        "post_entry_power_delta",
        "post_entry_exposure_delta",
        "post_entry_gain_delta",
        "energy_proxy",
        "blur_proxy",
        "noise_proxy",
        "corr_power_scene_effect",
        "corr_exposure_scene_effect",
        "corr_gain_scene_effect",
        "corr_power_fill_gap",
        "corr_exposure_fill_gap",
        "corr_gain_fill_gap",
    ]
    summary_rows = _summarize_rows(all_episode_rows, summary_metric_keys)
    trajectory_diff_rows = []
    if "ours_zero" in ckpts:
        trajectory_diff_rows.extend(_compute_trajectory_diffs(all_trace_rows, "ours", "ours_zero"))
    if "blind" in ckpts:
        trajectory_diff_rows.extend(_compute_trajectory_diffs(all_trace_rows, "ours", "blind"))
    for baseline in ("fixed", "fixed_random", "nondiff"):
        if baseline in ckpts:
            trajectory_diff_rows.extend(_compute_trajectory_diffs(all_trace_rows, "ours", baseline))

    _write_csv(output_dir / "episode_metrics.csv", all_episode_rows)
    _write_csv(output_dir / "trace_metrics.csv", all_trace_rows)
    _write_csv(output_dir / "summary_metrics.csv", summary_rows)
    _write_csv(output_dir / "trajectory_diffs.csv", trajectory_diff_rows)
    _dump_metadata(output_dir / "meta.json", {
        "config": str(config_path),
        "checkpoints": {k: str(v) for k, v in ckpts.items()},
        "episodes_per_condition": int(args.episodes_per_condition),
        "plot_level": plot_level,
        "slots": slots,
        "device": str(device),
        "include_base": bool(include_base),
    })

    _plot_success_vs_glare(summary_rows, output_dir / "success_vs_glare.png")
    _plot_success_by_slot(summary_rows, output_dir / "success_by_slot.png")
    _plot_post_entry_metrics(summary_rows, output_dir / "post_entry_vs_glare.png")
    for slot in slots:
        _plot_event_aligned(
            all_trace_rows,
            plot_level=plot_level,
            plot_slot=slot,
            output_path=output_dir / f"event_aligned_{plot_level}_{slot}.png",
        )
    _plot_trajectory(all_trace_rows, plot_level=plot_level, output_path=output_dir / f"trajectory_{plot_level}.png")

    formatted_outputs = format_results_dir(output_dir)

    print("[suite] done.")
    print(f"[suite] output_dir: {output_dir}")
    print(f"[suite] summary csv: {output_dir / 'summary_metrics.csv'}")
    print(f"[suite] episode csv: {output_dir / 'episode_metrics.csv'}")
    print(f"[suite] trace csv  : {output_dir / 'trace_metrics.csv'}")
    print(f"[suite] traj diff  : {output_dir / 'trajectory_diffs.csv'}")
    for name, path in formatted_outputs.items():
        print(f"[suite] {name}: {path}")


if __name__ == "__main__":
    main()

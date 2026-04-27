"""Shared utilities for D455 data collection and diff_depth scene calibration."""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np


SUPPORTED_CALIB_SCENES = (
    'glare',
    'specular',
    'dark',
)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def clamp01(x: float) -> float:
    return float(min(max(float(x), 0.0), 1.0))


def safe_mean(x: np.ndarray, default: float = 0.0) -> float:
    if x.size == 0:
        return float(default)
    return float(np.mean(x))


def safe_std(x: np.ndarray, default: float = 0.0) -> float:
    if x.size == 0:
        return float(default)
    return float(np.std(x))


def depth_valid_mask(depth_m: np.ndarray, min_valid_m: float, max_valid_m: float) -> np.ndarray:
    return np.isfinite(depth_m) & (depth_m >= float(min_valid_m)) & (depth_m <= float(max_valid_m))


def compute_depth_stats(
    depth_m: np.ndarray,
    min_valid_m: float = 0.3,
    max_valid_m: float = 6.0,
) -> dict[str, float]:
    mask = depth_valid_mask(depth_m, min_valid_m=min_valid_m, max_valid_m=max_valid_m)
    valid = depth_m[mask]
    invalid_ratio = float(1.0 - mask.mean())
    fill_rate = float(mask.mean())

    if valid.size == 0:
        return {
            'fill_rate': fill_rate,
            'invalid_ratio': invalid_ratio,
            'depth_mean_m': 0.0,
            'depth_std_m': 0.0,
            'depth_min_m': 0.0,
            'depth_max_m': 0.0,
            'depth_p10_m': 0.0,
            'depth_p50_m': 0.0,
            'depth_p90_m': 0.0,
            'depth_variance_m2': 0.0,
            'edge_std_m': 0.0,
        }

    gy, gx = np.gradient(depth_m.astype(np.float32), edge_order=1)
    edge = np.sqrt(gx * gx + gy * gy)
    edge_valid = edge[mask]

    return {
        'fill_rate': fill_rate,
        'invalid_ratio': invalid_ratio,
        'depth_mean_m': float(np.mean(valid)),
        'depth_std_m': float(np.std(valid)),
        'depth_min_m': float(np.min(valid)),
        'depth_max_m': float(np.max(valid)),
        'depth_p10_m': float(np.percentile(valid, 10)),
        'depth_p50_m': float(np.percentile(valid, 50)),
        'depth_p90_m': float(np.percentile(valid, 90)),
        'depth_variance_m2': float(np.var(valid)),
        'edge_std_m': float(np.std(edge_valid)) if edge_valid.size > 0 else 0.0,
    }


def normalize_with_range(value: float, lo: float, hi: float) -> float:
    hi = max(float(hi), float(lo) + 1e-6)
    return clamp01((float(value) - float(lo)) / (hi - lo))


def power_proxy_from_laser(laser_value: float, laser_max: float) -> float:
    laser_max = max(float(laser_max), 1e-6)
    return clamp01(float(laser_value) / laser_max)


def exposure_proxy_from_us(exposure_us: float, min_us: float, max_us: float) -> float:
    return normalize_with_range(float(exposure_us), float(min_us), float(max_us))


def gain_proxy_from_value(gain_value: float, min_gain: float, max_gain: float) -> float:
    return normalize_with_range(float(gain_value), float(min_gain), float(max_gain))


def row_writer_from_path(csv_path: str, fieldnames: list[str]):
    ensure_dir(os.path.dirname(csv_path) or '.')
    fh = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    return fh, writer


def write_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_csv_rows(csv_path: str) -> list[dict[str, str]]:
    with open(csv_path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def to_float_rows(rows: list[dict[str, str]], numeric_keys: list[str]) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    numeric_set = set(numeric_keys)
    for row in rows:
        item: dict[str, float | str] = {}
        for k, v in row.items():
            if k in numeric_set:
                try:
                    item[k] = float(v)
                except (TypeError, ValueError):
                    item[k] = math.nan
            else:
                item[k] = v
        out.append(item)
    return out


def aggregate_by_condition(rows: list[dict[str, float | str]], scene: str) -> dict[str, dict[str, float]]:
    scene_rows = [r for r in rows if r.get('scene') == scene]
    if not scene_rows:
        return {}

    buckets: dict[str, list[dict[str, float | str]]] = {}
    for row in scene_rows:
        key = str(row.get('condition_id', 'unknown'))
        buckets.setdefault(key, []).append(row)

    stats: dict[str, dict[str, float]] = {}
    metric_keys = [
        'fill_rate', 'invalid_ratio', 'depth_variance_m2', 'depth_std_m', 'edge_std_m',
        'power01', 'exposure01', 'gain01',
        'laser_power', 'exposure_us', 'gain_value',
        'brightness_proxy',
        'frame_delay_estimate',
    ]
    for key, bucket in buckets.items():
        out: dict[str, float] = {'num_rows': float(len(bucket))}
        for mk in metric_keys:
            vals = np.array([
                float(r[mk]) for r in bucket
                if mk in r and isinstance(r[mk], (int, float)) and np.isfinite(float(r[mk]))
            ], dtype=np.float64)
            out[f'{mk}_mean'] = safe_mean(vals)
            out[f'{mk}_std'] = safe_std(vals)
        stats[key] = out
    return stats


@dataclass(frozen=True)
class SceneFitResult:
    scene: str
    sim_profile: dict[str, float]
    fit_summary: dict[str, float]
    raw_condition_stats: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            'scene': self.scene,
            'sim_profile': self.sim_profile,
            'fit_summary': self.fit_summary,
            'raw_condition_stats': self.raw_condition_stats,
        }


def fit_scene_profile(scene: str, rows: list[dict[str, float | str]]) -> SceneFitResult:
    stats = aggregate_by_condition(rows, scene)
    if not stats:
        return SceneFitResult(
            scene=scene,
            sim_profile={},
            fit_summary={'num_conditions': 0.0},
            raw_condition_stats={},
        )

    def best_key(metric_name: str, reverse: bool = True) -> str:
        keys = list(stats.keys())
        keys.sort(key=lambda k: float(stats[k].get(metric_name, 0.0)), reverse=reverse)
        return keys[0]

    best_fill = stats[best_key('fill_rate_mean', reverse=True)]
    worst_fill = stats[best_key('fill_rate_mean', reverse=False)]
    highest_var = stats[best_key('depth_variance_m2_mean', reverse=True)]
    highest_invalid = stats[best_key('invalid_ratio_mean', reverse=True)]

    fit_summary = {
        'num_conditions': float(len(stats)),
        'best_fill_rate': float(best_fill.get('fill_rate_mean', 0.0)),
        'worst_fill_rate': float(worst_fill.get('fill_rate_mean', 0.0)),
        'highest_invalid_ratio': float(highest_invalid.get('invalid_ratio_mean', 0.0)),
        'highest_variance': float(highest_var.get('depth_variance_m2_mean', 0.0)),
    }

    sim_profile: dict[str, float] = {}

    sun_loss = max(0.0, best_fill.get('fill_rate_mean', 0.0) - worst_fill.get('fill_rate_mean', 0.0))
    sim_profile = {
        'ambient_add': float(np.clip(1.2 + 4.0 * sun_loss, 0.8, 4.5)),
        'active_drop': float(np.clip(0.15 + 1.1 * highest_invalid.get('invalid_ratio_mean', 0.0), 0.1, 0.9)),
        'quality_penalty': float(np.clip(0.8 + 3.0 * sun_loss, 0.5, 3.5)),
        'valid_bias_scale': float(np.clip(0.02 + 0.20 * highest_invalid.get('invalid_ratio_mean', 0.0), 0.0, 0.30)),
    }

    return SceneFitResult(
        scene=scene,
        sim_profile=sim_profile,
        fit_summary=fit_summary,
        raw_condition_stats=stats,
    )

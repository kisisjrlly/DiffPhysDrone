#!/usr/bin/env python3
"""Shared utilities for paper asset generation.

This module intentionally contains only data loading, statistical summaries and
small aggregation helpers. The public figure-generation entrypoint is
`tools/make_journal_assets.py`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


MM = 1.0 / 25.4
DOUBLE_COL = 183 * MM
SINGLE_COL = 89 * MM

MAIN_METHODS = ["flightonly", "fixed", "randfix", "nondiff", "zero"]
DIAG_METHODS = ["pretrained", "dagger", "flightonly"]
ALL_METHODS = ["flightonly", "fixed", "randfix", "nondiff", "zero", "pretrained", "dagger"]
SCENES = ["glare", "dark", "specular"]

METHOD_LABEL = {
    "flightonly": "Ours",
    "fixed": "Fixed",
    "randfix": "RandFix",
    "nondiff": "NonDiff",
    "zero": "Blind",
    "pretrained": "Pretrain",
    "dagger": "DAgger",
}
SCENE_LABEL = {"glare": "Glare", "dark": "Dark", "specular": "Specular"}

# Okabe-Ito inspired palette, kept deliberately restrained for publication use.
METHOD_COLOR = {
    "flightonly": "#0072B2",
    "fixed": "#7A7A7A",
    "randfix": "#CC79A7",
    "nondiff": "#E69F00",
    "zero": "#D55E00",
    "pretrained": "#9E9E9E",
    "dagger": "#009E73",
}
SCENE_COLOR = {"glare": "#D55E00", "dark": "#0072B2", "specular": "#CC79A7"}
PARAM_COLOR = {"power": "#0072B2", "exposure": "#009E73", "gain": "#D55E00"}


def read_eval(eval_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    episodes = []
    traces = []
    for method in ALL_METHODS:
        ep = eval_dir / "raw" / f"{method}_episodes.csv"
        tr = eval_dir / "raw" / f"{method}_trace.csv"
        if ep.exists():
            episodes.append(pd.read_csv(ep))
        if tr.exists():
            traces.append(pd.read_csv(tr))
    if not episodes or not traces:
        raise FileNotFoundError(f"raw episode/trace CSVs not found under {eval_dir / 'raw'}")
    summary = pd.read_csv(eval_dir / "summary_by_method_scene.csv")
    return summary, pd.concat(episodes, ignore_index=True), pd.concat(traces, ignore_index=True)


def read_diagnosis(eval_dir: Path) -> pd.DataFrame:
    path = eval_dir / "pretrain_online_offline_phase_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def wilson_ci(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 3000, seed: int = 42) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    return mean, min(lo, mean), max(hi, mean)


def metric_ci(episodes: pd.DataFrame, method: str, metric: str, scene: str | None = None) -> tuple[float, float, float]:
    df = episodes[episodes["method"] == method]
    if scene is not None:
        df = df[df["scene_name"] == scene]
    vals = df[metric].astype(float).to_numpy()
    if metric in {"success_rate", "collision_rate", "goal_reach_rate"}:
        mean = float(vals.mean())
        lo, hi = wilson_ci(float(vals.sum()), len(vals))
        return mean, lo, hi
    return bootstrap_mean_ci(vals, seed=100 + len(method) + len(metric) + (len(scene) if scene else 0))


def phase_from_local_x(x: float) -> str:
    if x < -0.25:
        return "before"
    if x <= 0.25:
        return "near"
    return "after"


def episode_phase_means(traces: pd.DataFrame) -> pd.DataFrame:
    df = traces.copy()
    df["phase"] = [phase_from_local_x(float(x)) for x in df["local_x"]]
    cols = ["power", "exposure", "gain", "scene_effect_mean", "local_x", "local_y", "goal_dist", "clearance"]
    return df.groupby(["method", "scene_name", "episode_idx", "phase"], as_index=False)[cols].mean()


def scene_param_ci(
    phase_ep: pd.DataFrame, method: str, scene: str, phase: str, param: str
) -> tuple[float, float, float]:
    vals = phase_ep[
        (phase_ep["method"] == method)
        & (phase_ep["scene_name"] == scene)
        & (phase_ep["phase"] == phase)
    ][param].to_numpy(float)
    return bootstrap_mean_ci(vals, seed=300 + len(method) + len(scene) + len(param))


def scene_param_vector(phase_ep: pd.DataFrame, method: str, scene: str, phase: str = "near") -> np.ndarray:
    return np.array([scene_param_ci(phase_ep, method, scene, phase, p)[0] for p in ["power", "exposure", "gain"]])


def l1_scene_separation_ci(
    phase_ep: pd.DataFrame, method: str, scene_a: str = "glare", scene_b: str = "dark", n_boot: int = 3000
) -> tuple[float, float, float]:
    a = phase_ep[
        (phase_ep["method"] == method) & (phase_ep["scene_name"] == scene_a) & (phase_ep["phase"] == "near")
    ][["power", "exposure", "gain"]].to_numpy(float)
    b = phase_ep[
        (phase_ep["method"] == method) & (phase_ep["scene_name"] == scene_b) & (phase_ep["phase"] == "near")
    ][["power", "exposure", "gain"]].to_numpy(float)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.abs(a.mean(axis=0) - b.mean(axis=0)).mean())
    rng = np.random.default_rng(700 + len(method))
    vals = []
    for _ in range(n_boot):
        aa = a[rng.integers(0, len(a), len(a))].mean(axis=0)
        bb = b[rng.integers(0, len(b), len(b))].mean(axis=0)
        vals.append(float(np.abs(aa - bb).mean()))
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return mean, min(lo, mean), max(hi, mean)


def binned_param(traces: pd.DataFrame, method: str, scene: str, param: str, bins: np.ndarray) -> pd.DataFrame:
    df = traces[(traces["method"] == method) & (traces["scene_name"] == scene)].copy()
    df["xbin"] = pd.cut(df["local_x"], bins=bins, labels=False, include_lowest=True)
    ep_bin = df.groupby(["episode_idx", "xbin"], as_index=False)[param].mean()
    rows = []
    for b in range(len(bins) - 1):
        vals = ep_bin[ep_bin["xbin"] == b][param].to_numpy(float)
        x_mid = 0.5 * (bins[b] + bins[b + 1])
        if len(vals) < 5:
            rows.append((x_mid, np.nan, np.nan, np.nan))
        else:
            mean, lo, hi = bootstrap_mean_ci(vals, n_boot=1200, seed=500 + b + len(param))
            rows.append((x_mid, mean, lo, hi))
    return pd.DataFrame(rows, columns=["x", "mean", "lo", "hi"])


def trajectory_envelope(traces: pd.DataFrame, episodes: pd.DataFrame, scene: str, bins: np.ndarray) -> pd.DataFrame:
    ep = episodes[(episodes["method"] == "flightonly") & (episodes["scene_name"] == scene)][
        ["episode_idx", "success_rate"]
    ]
    tr = traces[(traces["method"] == "flightonly") & (traces["scene_name"] == scene)].merge(
        ep, on="episode_idx", how="inner"
    )
    tr = tr[tr["success_rate"].astype(float) > 0.5].copy()
    tr["xbin"] = pd.cut(tr["local_x"], bins=bins, labels=False, include_lowest=True)
    ep_bin = tr.groupby(["episode_idx", "xbin"], as_index=False)["local_y"].mean()
    rows = []
    for b in range(len(bins) - 1):
        vals = ep_bin[ep_bin["xbin"] == b]["local_y"].to_numpy(float)
        x_mid = 0.5 * (bins[b] + bins[b + 1])
        if len(vals) < 5:
            rows.append((x_mid, np.nan, np.nan, np.nan))
        else:
            rows.append((x_mid, np.nanmedian(vals), np.nanpercentile(vals, 10), np.nanpercentile(vals, 90)))
    return pd.DataFrame(rows, columns=["x", "median", "lo", "hi"])


def ci_cell(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f} [{lo:.3f},{hi:.3f}]"

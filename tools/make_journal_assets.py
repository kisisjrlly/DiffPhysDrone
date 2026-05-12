#!/usr/bin/env python3
"""Create journal-ready figures and tables from the final evaluation suite.

The evaluation suite can emit dense diagnostic plots for debugging. This script
builds the smaller evidence-first figure/table set used by the manuscript:

Fig. 1  Study design and relabel-and-adapt protocol.
Fig. 2  Training convergence curves.
Fig. 3  Navigation performance with uncertainty and effect sizes.
Fig. 4  Active-camera mechanism in the final policy.
Fig. 6  DAgger relabel diagnosis.

All panels are regenerated from raw CSVs. No new evaluation is run here.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from paper_asset_utils import (  # noqa: E402
    ALL_METHODS,
    DOUBLE_COL,
    MAIN_METHODS,
    METHOD_COLOR,
    METHOD_LABEL,
    PARAM_COLOR,
    SCENE_COLOR,
    SCENE_LABEL,
    SCENES,
    SINGLE_COL,
    bootstrap_mean_ci,
    binned_param,
    ci_cell,
    episode_phase_means,
    l1_scene_separation_ci,
    metric_ci,
    read_diagnosis,
    read_eval,
    scene_param_ci,
    scene_param_vector,
    trajectory_envelope,
)


METHOD_LABEL_J = {
    **METHOD_LABEL,
    "flightonly": "Ours",
    "nondiff": "Non-diff.",
    "randfix": "Random fixed",
    "zero": "Blind",
}

METHOD_ORDER_MAIN = ["flightonly", "fixed", "randfix", "nondiff", "zero"]
METHOD_ORDER_NO_BLIND = ["flightonly", "fixed", "randfix", "nondiff"]
TRAINING_METHOD_ORDER = ["flightonly", "fixed", "randfix", "nondiff", "zero"]
DIAG_ORDER = ["pretrained", "dagger", "flightonly"]


def set_journal_style() -> None:
    """Conservative Nature/Science-like plot style."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 6.0,
            "axes.titlesize": 6.4,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.4,
            "axes.linewidth": 0.55,
            "lines.linewidth": 1.0,
            "patch.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )


def save_all(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            out_base.with_suffix(f".{ext}"),
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
            pad_inches=0.035,
            facecolor="white",
        )
    plt.close(fig)


def label_panel(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="bold")


def clean_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid == "x":
        ax.grid(axis="x", color="#E0E0E0", linewidth=0.45)
    elif grid == "y":
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.45)
    elif grid == "both":
        ax.grid(color="#E6E6E6", linewidth=0.42)
    ax.set_axisbelow(True)


def draw_round_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], text: str, fc: str) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=fc,
            edgecolor="#333333",
            linewidth=0.55,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=5.4, linespacing=1.08)


def draw_fig1_task_schematic(ax_task: plt.Axes) -> None:
    ax_task.set_aspect("equal")
    ax_task.set_xlim(-1.65, 1.65)
    ax_task.set_ylim(-0.92, 0.92)
    ax_task.axis("off")
    wall_color = "#4A4A4A"
    ax_task.add_patch(patches.Rectangle((-0.035, -0.92), 0.07, 0.66, facecolor=wall_color, linewidth=0))
    ax_task.add_patch(patches.Rectangle((-0.035, 0.26), 0.07, 0.66, facecolor=wall_color, linewidth=0))
    ax_task.add_patch(patches.Rectangle((-0.052, -0.16), 0.104, 0.32, facecolor="white", edgecolor="#222222", linewidth=0.55))
    ax_task.annotate("", xy=(1.38, 0.0), xytext=(-1.35, 0.0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#111111"))
    ax_task.scatter([-1.38], [0], s=18, color=METHOD_COLOR["flightonly"], zorder=4)
    ax_task.scatter([1.45], [0], s=44, marker="*", color="#009E73", zorder=4)
    for yy, scene in [(0.55, "glare"), (0.0, "specular"), (-0.55, "dark")]:
        ax_task.add_patch(
            patches.Ellipse(
                (0.0, yy),
                0.68,
                0.21,
                facecolor=SCENE_COLOR[scene],
                edgecolor=SCENE_COLOR[scene],
                alpha=0.16,
                linewidth=0.55,
            )
        )
        ax_task.text(0.41, yy, SCENE_LABEL[scene], color=SCENE_COLOR[scene], fontsize=5.5, va="center")
    ax_task.text(-1.55, 0.84, "Slit task", fontsize=6.1, fontweight="bold", va="top")
    ax_task.text(-1.39, -0.15, "start", ha="center", fontsize=5.2)
    ax_task.text(1.45, -0.15, "goal", ha="center", fontsize=5.2)
    ax_task.text(0.09, -0.19, "slit", ha="left", fontsize=5.0)


def draw_fig1_active_loop(ax_loop: plt.Axes) -> None:
    ax_loop.axis("off")
    ax_loop.set_xlim(0, 1)
    ax_loop.set_ylim(0, 1)
    boxes = [
        ((0.06, 0.62), (0.32, 0.17), "camera\n$p,e,g$", "#E8F1FA"),
        ((0.50, 0.62), (0.38, 0.17), "differentiable\nactive depth", "#FFF1D6"),
        ((0.50, 0.24), (0.38, 0.17), "policy GRU\nflight + camera", "#E8F5ED"),
        ((0.06, 0.24), (0.32, 0.17), "depth +\nstate", "#F2F2F2"),
    ]
    for xy, wh, text, fc in boxes:
        draw_round_box(ax_loop, xy, wh, text, fc)
    arrow = dict(arrowstyle="-|>", lw=0.65, color="#333333", shrinkA=1, shrinkB=1)
    ax_loop.annotate("", xy=(0.50, 0.705), xytext=(0.38, 0.705), arrowprops=arrow)
    ax_loop.annotate("", xy=(0.69, 0.41), xytext=(0.69, 0.62), arrowprops=arrow)
    ax_loop.annotate("", xy=(0.38, 0.325), xytext=(0.50, 0.325), arrowprops=dict(arrowstyle="<|-", lw=0.65, color="#333333"))
    ax_loop.annotate("", xy=(0.22, 0.62), xytext=(0.22, 0.41), arrowprops=dict(arrowstyle="<|-", lw=0.65, color="#333333"))
    ax_loop.text(0.04, 0.91, "Active-depth loop", fontsize=6.1, fontweight="bold", va="top")
    ax_loop.text(0.04, 0.09, "policy acts on vehicle state and sensor state", fontsize=5.1, color="#4A4A4A")


def draw_fig1_protocol(ax_train: plt.Axes) -> None:
    ax_train.axis("off")
    ax_train.set_xlim(0, 1)
    ax_train.set_ylim(0, 1)
    arrow = dict(arrowstyle="-|>", lw=0.65, color="#333333", shrinkA=1, shrinkB=1)
    stages = [
        (0.03, "online\nrollout", "#E8F1FA"),
        (0.27, "teacher\nrelabel", "#FFF1D6"),
        (0.51, "camera\npretrain", "#E8F5ED"),
        (0.75, "flight-only\nadapt", "#F3E8F2"),
    ]
    for x, text, fc in stages:
        draw_round_box(ax_train, (x, 0.57), (0.18, 0.20), text, fc)
    for x in [0.21, 0.45, 0.69]:
        ax_train.annotate("", xy=(x + 0.045, 0.67), xytext=(x + 0.01, 0.67), arrowprops=arrow)
    ax_train.add_patch(patches.Rectangle((0.15, 0.20), 0.68, 0.17, facecolor="#F7F7F7", edgecolor="#333333", linewidth=0.5))
    ax_train.text(0.49, 0.285, "closed-loop evaluation\n7 methods x 3 scenes\n300 episodes per scene", ha="center", va="center", fontsize=4.65, linespacing=1.0)
    ax_train.annotate("", xy=(0.48, 0.37), xytext=(0.84, 0.57), arrowprops=arrow)
    ax_train.text(0.02, 0.91, "Relabel and adapt", fontsize=6.1, fontweight="bold", va="top")
    ax_train.text(0.02, 0.09, "final adaptation freezes the camera branch", fontsize=5.1, color="#4A4A4A")


def bootstrap_delta_ci(
    episodes: pd.DataFrame,
    method: str,
    baseline: str,
    metric: str,
    scene: str | None = None,
    n_boot: int = 4000,
) -> tuple[float, float, float]:
    a = episodes[episodes["method"] == method]
    b = episodes[episodes["method"] == baseline]
    if scene is not None:
        a = a[a["scene_name"] == scene]
        b = b[b["scene_name"] == scene]
    av = a[metric].astype(float).to_numpy()
    bv = b[metric].astype(float).to_numpy()
    mean = float(av.mean() - bv.mean())
    if len(av) == 0 or len(bv) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(20260507 + len(method) + len(metric) + (0 if scene is None else len(scene)))
    vals = []
    for _ in range(n_boot):
        vals.append(
            float(
                av[rng.integers(0, len(av), len(av))].mean()
                - bv[rng.integers(0, len(bv), len(bv))].mean()
            )
        )
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return mean, min(lo, mean), max(hi, mean)


def format_vec(vals: np.ndarray) -> str:
    return "/".join(f"{v:.3f}" for v in vals)


def _method_from_wandb_column(col: str) -> str | None:
    name = col.lower()
    if "__min" in name or "__max" in name:
        return None
    if "flight_only" in name or "150703" in name:
        return "flightonly"
    if "fixed_random_static" in name or "012630" in name:
        return "randfix"
    if "depth-zero" in name or "100526" in name:
        return "zero"
    if "non-diffcam" in name or "cam-learned_grad-detached_depth-depth" in name or "015725" in name:
        return "nondiff"
    if "cam-fixed_grad-detached_depth-depth" in name:
        return "fixed"
    return None


def read_training_curves(eval_dir: Path) -> pd.DataFrame:
    rows = []
    for metric in ["loss", "success_rate", "collision_rate"]:
        paths = sorted((eval_dir / "raw").glob(f"wandb_export_*_{metric}.csv"))
        if not paths:
            continue
        # Use the newest export in this result directory. This prevents an
        # accidental mix of stale May-7 curves with the current May-8 eval set.
        path = paths[-1]
        df = pd.read_csv(path)
        if "Step" not in df:
            continue
        for col in df.columns:
            method = _method_from_wandb_column(col)
            if method is None or f" - {metric}" not in col:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            for step, value in zip(df["Step"], vals):
                if np.isfinite(value):
                    rows.append(
                        {
                            "step": float(step),
                            "method": method,
                            "metric": metric,
                            "value": float(value),
                        }
                    )
    return pd.DataFrame(rows)


def fig1_panel_exports(out_dir: Path) -> None:
    panel_dir = out_dir / "panels"
    fig = plt.figure(figsize=(SINGLE_COL, 2.15))
    ax = fig.add_subplot(111)
    draw_fig1_task_schematic(ax)
    save_all(fig, panel_dir / "fig1a_task_schematic")

    fig = plt.figure(figsize=(SINGLE_COL, 2.15))
    ax = fig.add_subplot(111)
    draw_fig1_active_loop(ax)
    save_all(fig, panel_dir / "fig1b_active_depth_loop")

    fig = plt.figure(figsize=(SINGLE_COL, 2.15))
    ax = fig.add_subplot(111)
    draw_fig1_protocol(ax)
    save_all(fig, panel_dir / "fig1c_relabeled_training_protocol")


def draw_training_metric(ax: plt.Axes, training: pd.DataFrame, metric: str, ylabel: str, scale: str) -> None:
    sub_metric = training[training["metric"] == metric]
    for method in TRAINING_METHOD_ORDER:
        sub = sub_metric[sub_metric["method"] == method].sort_values("step")
        if sub.empty:
            continue
        y = sub["value"].astype(float).rolling(window=3, min_periods=1, center=True).mean()
        ax.plot(
            sub["step"],
            y,
            color=METHOD_COLOR[method],
            label=METHOD_LABEL_J[method],
            linewidth=1.15,
        )
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    clean_axis(ax, "y")
    if scale == "log":
        ax.set_yscale("log")
    else:
        ax.set_ylim(-0.03, 1.03)


def fig2_training_convergence_panels(training: pd.DataFrame, out_dir: Path) -> None:
    if training.empty:
        return
    panel_dir = out_dir / "panels"
    specs = [
        ("fig2a_training_loss", "loss", "training loss", "log"),
        ("fig2b_training_success", "success_rate", "training success", "linear"),
        ("fig2c_training_collision", "collision_rate", "training collision", "linear"),
    ]
    for name, metric, ylabel, scale in specs:
        fig = plt.figure(figsize=(SINGLE_COL, 2.25))
        ax = fig.add_subplot(111)
        draw_training_metric(ax, training, metric, ylabel, scale)
        if metric == "loss":
            ax.legend(frameon=False, loc="upper right")
        save_all(fig, panel_dir / name)


def draw_overall_forest(ax: plt.Axes, episodes: pd.DataFrame) -> None:
    draw_metric_forest(ax, episodes, "success_rate", "success rate", (0.0, 1.0), show_ylabels=True)


def draw_metric_forest(
    ax: plt.Axes,
    episodes: pd.DataFrame,
    metric: str,
    xlabel: str,
    xlim: tuple[float, float],
    show_ylabels: bool = True,
) -> None:
    y_positions = np.arange(len(METHOD_ORDER_MAIN))[::-1]
    for yi, method in zip(y_positions, METHOD_ORDER_MAIN):
        mean, lo, hi = metric_ci(episodes, method, metric)
        ax.plot([lo, hi], [yi, yi], color="#222222", lw=0.7, zorder=1)
        ax.scatter([mean], [yi], s=30, color=METHOD_COLOR[method], edgecolor="#222222", linewidth=0.45, zorder=2)
        ax.text(xlim[1] + 0.018 * (xlim[1] - xlim[0]), yi, f"{mean:.2f}", va="center", fontsize=5.2)
    ax.set_yticks(y_positions)
    if show_ylabels:
        ax.set_yticklabels([METHOD_LABEL_J[m] for m in METHOD_ORDER_MAIN])
    else:
        ax.set_yticklabels([])
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.55, len(METHOD_ORDER_MAIN) - 0.45)
    ax.set_xlabel(xlabel)
    clean_axis(ax, "x")


def draw_delta_panel(ax: plt.Axes, episodes: pd.DataFrame) -> None:
    methods = ["flightonly", "randfix", "nondiff", "zero"]
    metrics = [("success_rate", "success"), ("fill_rate", "fill")]
    y = np.arange(len(methods))[::-1]
    offsets = [0.13, -0.13]
    for off, (metric, label) in zip(offsets, metrics):
        for yi, method in zip(y, methods):
            mean, lo, hi = bootstrap_delta_ci(episodes, method, "fixed", metric)
            color = "#0072B2" if metric == "success_rate" else "#009E73"
            ax.plot([lo, hi], [yi + off, yi + off], color=color, lw=0.8, alpha=0.85)
            ax.scatter([mean], [yi + off], s=20, color=color, edgecolor="#222222", linewidth=0.35, label=label if yi == y[0] else None)
    ax.axvline(0, color="#222222", lw=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABEL_J[m] for m in methods])
    ax.set_xlabel(r"$\Delta$ versus fixed camera")
    ax.set_xlim(-0.72, 0.22)
    ax.legend(frameon=False, loc="lower left", ncol=2, handletextpad=0.35)
    clean_axis(ax, "x")
    ax.set_title("Effect size", pad=5)


def draw_scene_delta_heatmap(ax: plt.Axes, episodes: pd.DataFrame) -> None:
    methods = ["flightonly", "randfix", "nondiff"]
    data = np.zeros((len(methods), len(SCENES)))
    for i, method in enumerate(methods):
        for j, scene in enumerate(SCENES):
            data[i, j] = bootstrap_delta_ci(episodes, method, "fixed", "success_rate", scene)[0]
    vmax = 0.18
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(SCENES)))
    ax.set_xticklabels([SCENE_LABEL[s] for s in SCENES])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([METHOD_LABEL_J[m] for m in methods])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:+.2f}", ha="center", va="center", fontsize=5.3, color="#111111")
    ax.set_title("Per-scene success gain", pad=5)
    return im


def draw_success_fill_plane(ax: plt.Axes, episodes: pd.DataFrame) -> None:
    for method in METHOD_ORDER_MAIN:
        succ = metric_ci(episodes, method, "success_rate")
        fill = metric_ci(episodes, method, "fill_rate")
        ax.errorbar(
            [fill[0]],
            [succ[0]],
            xerr=[[fill[0] - fill[1]], [fill[2] - fill[0]]],
            yerr=[[succ[0] - succ[1]], [succ[2] - succ[0]]],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.48,
            capsize=1.5,
            zorder=1,
        )
        ax.scatter([fill[0]], [succ[0]], s=34, color=METHOD_COLOR[method], edgecolor="#222222", linewidth=0.45, zorder=3)
        dx = 0.006
        dy = {
            "flightonly": 0.018,
            "fixed": -0.040,
            "randfix": 0.010,
            "nondiff": -0.015,
            "zero": 0.020,
        }[method]
        ax.text(fill[0] + dx, succ[0] + dy, METHOD_LABEL_J[method], fontsize=5.2)
    ax.set_xlabel("depth fill rate")
    ax.set_ylabel("success rate")
    ax.set_xlim(0.70, 0.99)
    ax.set_ylim(0.0, 0.84)
    clean_axis(ax, "both")
    ax.set_title("Performance-observation coupling", pad=5)


def draw_terminal_ecdf(ax: plt.Axes, episodes: pd.DataFrame) -> None:
    for method in METHOD_ORDER_NO_BLIND:
        vals = np.sort(episodes[episodes["method"] == method]["final_goal_dist"].astype(float).to_numpy())
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color=METHOD_COLOR[method], label=METHOD_LABEL_J[method], lw=1.05)
    ax.set_xlabel("terminal distance to goal (m)")
    ax.set_ylabel("episode fraction")
    ax.set_xlim(0, 1.9)
    ax.set_ylim(0, 1.0)
    clean_axis(ax, "both")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Terminal distance distribution", pad=5)


def fig3_navigation_panels(episodes: pd.DataFrame, out_dir: Path) -> None:
    panel_dir = out_dir / "panels"

    fig = plt.figure(figsize=(SINGLE_COL, 2.30))
    ax = fig.add_subplot(111)
    draw_metric_forest(ax, episodes, "success_rate", "success rate", (0.0, 1.0), show_ylabels=True)
    ax.set_title("Navigation success", pad=5)
    save_all(fig, panel_dir / "fig3a_navigation_success")

    fig = plt.figure(figsize=(SINGLE_COL, 2.30))
    ax = fig.add_subplot(111)
    draw_metric_forest(ax, episodes, "fill_rate", "depth fill rate", (0.65, 1.0), show_ylabels=True)
    ax.set_title("Observation quality", pad=5)
    save_all(fig, panel_dir / "fig3b_depth_fill")

    fig = plt.figure(figsize=(SINGLE_COL, 2.35))
    ax = fig.add_subplot(111)
    im = draw_scene_delta_heatmap(ax, episodes)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"$\Delta$ success")
    save_all(fig, panel_dir / "fig3c_scene_success_gain")

    fig = plt.figure(figsize=(SINGLE_COL, 2.35))
    ax = fig.add_subplot(111)
    draw_terminal_ecdf(ax, episodes)
    save_all(fig, panel_dir / "fig3d_terminal_distance_ecdf")


def draw_camera_heatmap(ax: plt.Axes, phase_ep: pd.DataFrame) -> None:
    params = ["power", "exposure", "gain"]
    data = np.array([[scene_param_ci(phase_ep, "flightonly", scene, "near", p)[0] for p in params] for scene in SCENES])
    im = ax.imshow(data, cmap="YlGnBu", vmin=0.0, vmax=0.85, aspect="auto")
    ax.set_xticks(np.arange(len(params)))
    ax.set_xticklabels(["power", "exposure", "gain"])
    ax.set_yticks(np.arange(len(SCENES)))
    ax.set_yticklabels([SCENE_LABEL[s] for s in SCENES])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=5.3, color="#111111")
    ax.set_title("Near-slit camera settings", pad=5)
    return im


def draw_exposure_gain_plane(ax: plt.Axes, phase_ep: pd.DataFrame) -> None:
    ax.axvline(0.5, color="#C7C7C7", lw=0.55, zorder=0)
    ax.axhline(0.5, color="#C7C7C7", lw=0.55, zorder=0)
    ax.scatter([0.5], [0.5], marker="+", s=32, color="#555555", linewidth=0.9, zorder=2)
    for scene in SCENES:
        p, e, g = scene_param_vector(phase_ep, "flightonly", scene)
        em = scene_param_ci(phase_ep, "flightonly", scene, "near", "exposure")
        gm = scene_param_ci(phase_ep, "flightonly", scene, "near", "gain")
        ax.errorbar(
            [e],
            [g],
            xerr=[[e - em[1]], [em[2] - e]],
            yerr=[[g - gm[1]], [gm[2] - g]],
            fmt="none",
            ecolor="#222222",
            elinewidth=0.45,
            capsize=1.4,
            zorder=1,
        )
        ax.scatter([e], [g], s=34 + 62 * p, color=SCENE_COLOR[scene], edgecolor="#222222", linewidth=0.45, zorder=3)
        ax.text(e + 0.020, g + 0.012, SCENE_LABEL[scene], color=SCENE_COLOR[scene], fontsize=5.3)
    ax.set_xlabel("exposure")
    ax.set_ylabel("gain")
    ax.set_xlim(0.0, 0.86)
    ax.set_ylim(0.0, 0.82)
    clean_axis(ax, "both")
    ax.set_title("Exposure-gain response plane", pad=5)


def draw_profile_pair(ax_top: plt.Axes, ax_bottom: plt.Axes, traces: pd.DataFrame) -> None:
    bins = np.linspace(-1.5, 1.5, 42)
    for scene in SCENES:
        for ax, param in [(ax_top, "exposure"), (ax_bottom, "gain")]:
            curve = binned_param(traces, "flightonly", scene, param, bins)
            ax.plot(curve["x"], curve["mean"], color=SCENE_COLOR[scene], label=SCENE_LABEL[scene])
            ax.fill_between(
                curve["x"].to_numpy(),
                curve["lo"].to_numpy(),
                curve["hi"].to_numpy(),
                color=SCENE_COLOR[scene],
                alpha=0.11,
                linewidth=0,
            )
    for ax, ylabel in [(ax_top, "exposure"), (ax_bottom, "gain")]:
        ax.axvspan(-0.25, 0.25, color="#BDBDBD", alpha=0.17, lw=0)
        ax.axvline(0, color="#333333", lw=0.55)
        ax.set_ylim(0, 0.84)
        ax.set_ylabel(ylabel)
        clean_axis(ax, "both")
    ax_top.legend(frameon=False, loc="upper right", ncol=3, handlelength=1.6)
    ax_top.tick_params(labelbottom=False)
    ax_bottom.set_xlabel("local x relative to wall (m)")


def draw_trajectory_envelope(ax: plt.Axes, episodes: pd.DataFrame, traces: pd.DataFrame) -> None:
    bins = np.linspace(-1.5, 1.5, 34)
    ax.axhspan(-0.15, 0.15, color="#F1F1F1", zorder=0)
    ax.add_patch(patches.Rectangle((-0.025, -0.82), 0.05, 0.66, facecolor="#666666", alpha=0.30, lw=0, zorder=1))
    ax.add_patch(patches.Rectangle((-0.025, 0.16), 0.05, 0.66, facecolor="#666666", alpha=0.30, lw=0, zorder=1))
    ax.axvline(0.0, color="#222222", lw=0.55, zorder=2)
    for scene in SCENES:
        env = trajectory_envelope(traces, episodes, scene, bins)
        ax.fill_between(env["x"].to_numpy(), env["lo"].to_numpy(), env["hi"].to_numpy(), color=SCENE_COLOR[scene], alpha=0.10, lw=0)
        ax.plot(env["x"], env["median"], color=SCENE_COLOR[scene], label=SCENE_LABEL[scene], lw=1.0)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-0.85, 0.85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("local x (m)")
    ax.set_ylabel("local y (m)")
    clean_axis(ax, "both")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Successful trajectory envelopes", pad=5)


def draw_near_slit_degradation(ax: plt.Axes, phase_ep: pd.DataFrame) -> None:
    phase_rows = []
    for scene in SCENES:
        vals = phase_ep[
            (phase_ep["method"] == "flightonly")
            & (phase_ep["scene_name"] == scene)
            & (phase_ep["phase"] == "near")
        ]["scene_effect_mean"].to_numpy(float)
        phase_rows.append(bootstrap_mean_ci(vals))
    y = np.arange(len(SCENES))
    means = [r[0] for r in phase_rows]
    lows = [r[1] for r in phase_rows]
    highs = [r[2] for r in phase_rows]
    ax.barh(y, means, color=[SCENE_COLOR[s] for s in SCENES], edgecolor="#222222", linewidth=0.45)
    ax.errorbar(
        means,
        y,
        xerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
        fmt="none",
        ecolor="#222222",
        elinewidth=0.55,
        capsize=1.4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([SCENE_LABEL[s] for s in SCENES])
    ax.set_xlabel("near-slit degradation proxy")
    ax.set_xlim(0, max(0.20, max(means) * 1.18))
    clean_axis(ax, "x")
    ax.set_title("Near-slit degradation encountered", pad=5)


def fig4_camera_mechanism_panels(
    episodes: pd.DataFrame, traces: pd.DataFrame, phase_ep: pd.DataFrame, out_dir: Path
) -> None:
    panel_dir = out_dir / "panels"

    fig = plt.figure(figsize=(SINGLE_COL, 2.35))
    ax = fig.add_subplot(111)
    draw_camera_heatmap(ax, phase_ep)
    save_all(fig, panel_dir / "fig4a_camera_fingerprint")

    fig = plt.figure(figsize=(SINGLE_COL, 2.35))
    ax = fig.add_subplot(111)
    draw_exposure_gain_plane(ax, phase_ep)
    save_all(fig, panel_dir / "fig4b_exposure_gain_plane")

    fig = plt.figure(figsize=(SINGLE_COL, 2.20))
    ax = fig.add_subplot(111)
    draw_near_slit_degradation(ax, phase_ep)
    save_all(fig, panel_dir / "fig4c_near_slit_degradation")

    fig = plt.figure(figsize=(DOUBLE_COL * 0.62, 2.65))
    sub = GridSpec(2, 1, figure=fig, hspace=0.12)
    ax_top = fig.add_subplot(sub[0, 0])
    ax_bottom = fig.add_subplot(sub[1, 0], sharex=ax_top)
    draw_profile_pair(ax_top, ax_bottom, traces)
    save_all(fig, panel_dir / "fig4d_exposure_gain_profiles")

    fig = plt.figure(figsize=(SINGLE_COL, 2.35))
    ax = fig.add_subplot(111)
    draw_trajectory_envelope(ax, episodes, traces)
    save_all(fig, panel_dir / "fig4e_trajectory_envelopes")


def offline_dagger_sep(diagnosis: pd.DataFrame) -> float:
    if diagnosis.empty:
        return np.nan
    sub = diagnosis[
        (diagnosis["source"] == "offline_teacher_dataset")
        & (diagnosis["method"] == "dagger")
        & (diagnosis["phase"] == "near")
    ]
    glare = sub[sub["scene"] == "glare"][["power", "exposure", "gain"]].mean().to_numpy(float)
    dark = sub[sub["scene"] == "dark"][["power", "exposure", "gain"]].mean().to_numpy(float)
    return float(np.abs(glare - dark).mean())


def delta_dark_minus_glare_ci(phase_ep: pd.DataFrame, method: str, param: str, n_boot: int = 3000) -> tuple[float, float, float]:
    d = phase_ep[
        (phase_ep["method"] == method) & (phase_ep["scene_name"] == "dark") & (phase_ep["phase"] == "near")
    ][param].to_numpy(float)
    g = phase_ep[
        (phase_ep["method"] == method) & (phase_ep["scene_name"] == "glare") & (phase_ep["phase"] == "near")
    ][param].to_numpy(float)
    mean = float(d.mean() - g.mean())
    rng = np.random.default_rng(8800 + len(method) + len(param))
    vals = [
        float(d[rng.integers(0, len(d), len(d))].mean() - g[rng.integers(0, len(g), len(g))].mean())
        for _ in range(n_boot)
    ]
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return mean, min(lo, mean), max(hi, mean)


def draw_dagger_semantics_progress(ax: plt.Axes, phase_ep: pd.DataFrame, diagnosis: pd.DataFrame) -> None:
    stages = ["offline\nteacher", "online\npretrain", "online\nDAgger", "final\npolicy"]
    vals = [offline_dagger_sep(diagnosis)]
    los = [np.nan]
    his = [np.nan]
    for method in DIAG_ORDER:
        mean, lo, hi = l1_scene_separation_ci(phase_ep, method)
        vals.append(mean)
        los.append(lo)
        his.append(hi)
    x = np.arange(len(stages))
    ax.plot(x, vals, color="#4A4A4A", lw=0.8, zorder=1)
    colors = ["#9E9E9E", METHOD_COLOR["pretrained"], METHOD_COLOR["dagger"], METHOD_COLOR["flightonly"]]
    for xi, mean, lo, hi, color in zip(x, vals, los, his, colors):
        if np.isfinite(lo):
            ax.errorbar([xi], [mean], yerr=[[mean - lo], [hi - mean]], fmt="none", ecolor="#222222", elinewidth=0.55, capsize=1.5)
        ax.scatter([xi], [mean], s=29, color=color, edgecolor="#222222", linewidth=0.42, zorder=3)
        ax.text(xi, mean + 0.025, f"{mean:.2f}", ha="center", fontsize=5.1)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=28, ha="right")
    ax.set_ylabel("glare-dark camera L1")
    finite_vals = [float(v) for v in vals if np.isfinite(v)]
    ax.set_ylim(0, max(0.42, max(finite_vals) * 1.20))
    clean_axis(ax, "y")
    ax.set_title("Online camera semantics", pad=5)


def draw_dark_glare_param_deltas(ax: plt.Axes, phase_ep: pd.DataFrame) -> None:
    params = ["power", "exposure", "gain"]
    methods = ["pretrained", "dagger", "flightonly"]
    x = np.arange(len(methods))
    width = 0.23
    for k, param in enumerate(params):
        means, lows, highs = [], [], []
        for method in methods:
            m, lo, hi = delta_dark_minus_glare_ci(phase_ep, method, param)
            means.append(m)
            lows.append(lo)
            highs.append(hi)
        xpos = x + (k - 1) * width
        ax.bar(xpos, means, width=width, color=PARAM_COLOR[param], edgecolor="#222222", linewidth=0.42, label=param)
        ax.errorbar(xpos, means, yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)], fmt="none", ecolor="#222222", elinewidth=0.5, capsize=1.4)
    ax.axhline(0, color="#222222", lw=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(["Pretrain", "DAgger", "Ours"])
    ax.set_ylabel("dark - glare near parameter")
    ax.set_ylim(-0.12, 0.72)
    ax.legend(frameon=False, ncol=3, loc="upper left", handlelength=1.2, columnspacing=0.9)
    clean_axis(ax, "y")
    ax.set_title("Recovered camera semantics", pad=5)


def draw_separation_success(ax: plt.Axes, episodes: pd.DataFrame, phase_ep: pd.DataFrame) -> None:
    scatter_methods = ["fixed", "nondiff", "pretrained", "dagger", "flightonly"]
    coords: dict[str, tuple[float, float]] = {}
    for method in scatter_methods:
        sep = l1_scene_separation_ci(phase_ep, method)[0]
        succ = metric_ci(episodes, method, "success_rate")[0]
        coords[method] = (sep, succ)
        ax.scatter([sep], [succ], s=34, color=METHOD_COLOR[method], edgecolor="#222222", linewidth=0.48, zorder=3)
    label_offsets = {
        "fixed": (0.010, -0.040),
        "nondiff": (0.010, 0.018),
        "pretrained": (-0.090, -0.032),
        "dagger": (-0.090, 0.022),
        "flightonly": (0.010, 0.018),
    }
    for method in scatter_methods:
        dx, dy = label_offsets[method]
        ax.text(coords[method][0] + dx, coords[method][1] + dy, METHOD_LABEL_J[method], fontsize=5.0)
    ax.annotate("", xy=coords["dagger"], xytext=coords["pretrained"], arrowprops=dict(arrowstyle="-|>", lw=0.65, color="#555555", shrinkA=4, shrinkB=4))
    ax.annotate("", xy=coords["flightonly"], xytext=coords["dagger"], arrowprops=dict(arrowstyle="-|>", lw=0.65, color="#555555", shrinkA=4, shrinkB=4))
    ax.set_xlabel("camera separation")
    ax.set_ylabel("success rate")
    ax.set_xlim(-0.02, 0.48)
    ax.set_ylim(0.0, 0.84)
    clean_axis(ax, "both")
    ax.set_title("Separation must be paired with flight adaptation", pad=5)


def fig6_dagger_diagnosis_panels(
    episodes: pd.DataFrame, phase_ep: pd.DataFrame, diagnosis: pd.DataFrame, out_dir: Path
) -> None:
    panel_dir = out_dir / "panels"

    fig = plt.figure(figsize=(SINGLE_COL, 2.35))
    ax = fig.add_subplot(111)
    draw_dagger_semantics_progress(ax, phase_ep, diagnosis)
    save_all(fig, panel_dir / "fig6a_camera_semantics_progress")

    fig = plt.figure(figsize=(SINGLE_COL, 2.35))
    ax = fig.add_subplot(111)
    draw_dark_glare_param_deltas(ax, phase_ep)
    save_all(fig, panel_dir / "fig6b_dark_glare_parameter_delta")

    fig = plt.figure(figsize=(SINGLE_COL, 2.35))
    ax = fig.add_subplot(111)
    draw_separation_success(ax, episodes, phase_ep)
    save_all(fig, panel_dir / "fig6c_separation_success")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_tables(episodes: pd.DataFrame, phase_ep: pd.DataFrame, out_dir: Path) -> None:
    table_dir = out_dir / "tables"
    fixed_success = metric_ci(episodes, "fixed", "success_rate")[0]
    fixed_fill = metric_ci(episodes, "fixed", "fill_rate")[0]
    main_counts = episodes[episodes["method"].isin(METHOD_ORDER_MAIN)].groupby("method").size()
    episodes_per_method = int(main_counts.min()) if len(main_counts) else 0
    scene_counts = episodes[episodes["method"].isin(METHOD_ORDER_MAIN)].groupby(["method", "scene_name"]).size()
    episodes_per_scene = int(scene_counts.min()) if len(scene_counts) else 0
    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Success & $\Delta$Success & Collision & Fill & $\Delta$Fill \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER_MAIN:
        success = metric_ci(episodes, method, "success_rate")
        collision = metric_ci(episodes, method, "collision_rate")
        fill = metric_ci(episodes, method, "fill_rate")
        cells = [
            METHOD_LABEL_J[method],
            ci_cell(*success),
            f"{success[0] - fixed_success:+.3f}",
            ci_cell(*collision),
            ci_cell(*fill),
            f"{fill[0] - fixed_fill:+.3f}",
        ]
        if method == "flightonly":
            cells[0] = r"\textbf{Ours}"
            cells[1] = rf"\textbf{{{cells[1]}}}"
            cells[2] = rf"\textbf{{{cells[2]}}}"
            cells[4] = rf"\textbf{{{cells[4]}}}"
            cells[5] = rf"\textbf{{{cells[5]}}}"
        rows.append(" & ".join(cells) + r" \\")
    rows += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Primary closed-loop navigation results over {episodes_per_method} episodes per method. Brackets denote 95\% confidence intervals: Wilson intervals for binary outcomes and bootstrap intervals over episodes for fill. Deltas are relative to the fixed-camera baseline.}}",
        r"\label{tab:journal_main_navigation}",
        r"\end{table}",
        "",
    ]
    write_text(table_dir / "table1_primary_navigation.tex", "\n".join(rows))

    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Glare & Dark & Specular \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER_MAIN:
        cells = [METHOD_LABEL_J[method]]
        for scene in SCENES:
            s = metric_ci(episodes, method, "success_rate", scene)[0]
            f = metric_ci(episodes, method, "fill_rate", scene)[0]
            cells.append(f"{s:.2f}/{f:.2f}")
        if method == "flightonly":
            cells[0] = r"\textbf{Ours}"
            cells = [cells[0], *(rf"\textbf{{{c}}}" for c in cells[1:])]
        rows.append(" & ".join(cells) + r" \\")
    rows += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Scene-level success/fill rates. Each scene contains {episodes_per_scene} closed-loop episodes per method.}}",
        r"\label{tab:journal_scene_breakdown}",
        r"\end{table}",
        "",
    ]
    write_text(table_dir / "table2_scene_breakdown.tex", "\n".join(rows))

    rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Glare p/e/g & Dark p/e/g & Specular p/e/g & Glare--dark L1 \\",
        r"\midrule",
    ]
    for method in ["flightonly", "fixed", "randfix", "nondiff", "pretrained", "dagger", "zero"]:
        cells = [METHOD_LABEL_J[method]]
        for scene in SCENES:
            cells.append(format_vec(scene_param_vector(phase_ep, method, scene)))
        cells.append(ci_cell(*l1_scene_separation_ci(phase_ep, method)))
        if method == "flightonly":
            cells[0] = r"\textbf{Ours}"
            cells = [cells[0], *(rf"\textbf{{{c}}}" for c in cells[1:])]
        rows.append(" & ".join(cells) + r" \\")
    rows += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Episode-aggregated near-slit camera behavior. Glare--dark L1 is the mean absolute camera-parameter difference between glare and dark near-slit phases, with bootstrap 95\% confidence intervals.}",
        r"\label{tab:journal_camera_response}",
        r"\end{table}",
        "",
    ]
    write_text(table_dir / "table3_camera_response.tex", "\n".join(rows))


def write_captions(out_dir: Path) -> None:
    text = """# Journal Figure Captions

Use the PDF files for manuscript layout. SVG files keep all text editable for
final artwork. The older `paper_assets` directory contains diagnostic plots and
should not be used in a submission.

## Figure 1 | Differentiable active depth sensing for slit navigation.

**a,** Single-wall slit navigation benchmark with randomized slit locations and
three sensor-degradation regimes: glare, low-reflectance dark material, and
specular false depth. **b,**
Closed-loop active-depth formulation. Camera power, exposure, and gain are
policy-controlled variables that change the next depth observation through a
differentiable sensor model. **c,** Training and evaluation protocol. Online
states are relabeled by a differentiable camera teacher, the camera head is
pretrained on these relabeled targets, and flight-control layers are then
adapted while the camera branch is fixed.

## Figure 2 | Training curves show that comparison policies reached stable regimes.

WandB training exports are plotted for the final active-camera policy, fixed
camera, random fixed camera, non-differentiable learned camera, and blind
zero-depth control. **a,** training loss. **b,** training success rate. **c,**
training collision rate. The curves are convergence diagnostics; all navigation
claims use the held-out closed-loop evaluations summarized in Figure 3 and
Tables 1--2.

## Figure 3 | Active camera control improves navigation while increasing valid depth.

All methods are evaluated for 300 episodes, with 100 episodes in each scene.
**a,b,** Overall success and depth-fill estimates with 95% confidence intervals.
**c,** Per-scene success change relative to fixed camera. **d,** Empirical
distribution of terminal goal distance. The proposed policy improves navigation
success while substantially increasing valid depth over fixed, random-fixed, and
non-differentiable camera baselines.

## Figure 4 | The learned camera policy implements scene-specific near-slit sensing.

**a,** Near-slit camera-parameter fingerprint for the final policy. **b,**
Exposure-gain response plane, where marker area scales with power and the grey
cross denotes the nominal camera setting. **c,** Near-slit degradation proxy
encountered by the final policy. **d,** Exposure and gain profiles as a function
of local distance to the wall; grey shading denotes the near-slit window. **e,**
Median successful trajectories with 10--90% episode envelopes. Low-reflectance
dark-material scenes keep exposure/gain high near the wall, whereas glare
suppresses both parameters.

## Figure 5 | Camera control changes what the policy observes near the slit.

Matched-pose qualitative depth sequences are rendered by
`tools/export_journal_depth_sequences.py` using a far-right slit. The first row
shows the local map, current pose, camera frustum and complete final-policy
trajectory from start through the slit toward the goal. The final policy
trajectory provides the reference poses. At each pose, raw geometric depth is
shown together with observed depth re-rendered using camera parameters from
fixed, random-fixed, and final active-camera policies. The comparison isolates
the sensor-parameter effect on the depth image at identical vehicle poses.
The manuscript uses the glare, dark, and specular composites as a compact
three-subfigure layout.

## Figure 6 | Camera relabeling and flight adaptation are complementary.

**a,** Glare-dark camera separation is measured in the relabeled teacher data
and in online closed-loop rollouts for the pretrained, DAgger-relabelled and
final policies. **b,** The separation is mainly carried by exposure and gain:
low-reflectance dark-material scenes require higher values than glare. **c,**
Camera semantics alone is insufficient for flight; final performance requires
combining DAgger camera relabeling with flight-only adaptation.
"""
    write_text(out_dir / "caption_drafts.md", text)


def write_readme(out_dir: Path, eval_dir: Path) -> None:
    text = f"""# Journal Assets

This directory contains submission-oriented figures and LaTeX tables generated
from:

`{eval_dir}`

The older `paper_assets` directory is diagnostic only and should not be used in a
paper submission. This `journal_assets` directory is the current recommended
figure/table set.

## Composite figures

- `figures/fig5_depth_observation_sequence_glare.pdf`
- `figures/fig5_depth_observation_sequence_dark.pdf`
- `figures/fig5_depth_observation_sequence_specular.pdf`

## Panel figures

The `figures/panels/` directory contains the standalone subfigure assets used by
the manuscript for Figures 1, 2, 3, 4, and 6.

- `figures/panels/fig2a_training_loss.pdf`
- `figures/panels/fig2b_training_success.pdf`
- `figures/panels/fig2c_training_collision.pdf`
- `figures/panels/fig1a_task_schematic.pdf`
- `figures/panels/fig1b_active_depth_loop.pdf`
- `figures/panels/fig1c_relabeled_training_protocol.pdf`
- `figures/panels/fig3a_navigation_success.pdf`
- `figures/panels/fig3b_depth_fill.pdf`
- `figures/panels/fig3c_scene_success_gain.pdf`
- `figures/panels/fig3d_terminal_distance_ecdf.pdf`
- `figures/panels/fig4a_camera_fingerprint.pdf`
- `figures/panels/fig4b_exposure_gain_plane.pdf`
- `figures/panels/fig4c_near_slit_degradation.pdf`
- `figures/panels/fig4d_exposure_gain_profiles.pdf`
- `figures/panels/fig4e_trajectory_envelopes.pdf`
- `figures/panels/fig6a_camera_semantics_progress.pdf`
- `figures/panels/fig6b_dark_glare_parameter_delta.pdf`
- `figures/panels/fig6c_separation_success.pdf`

## Tables

- `tables/table1_primary_navigation.tex`
- `tables/table2_scene_breakdown.tex`
- `tables/table3_camera_response.tex`

## Statistical conventions

- Binary outcomes use Wilson 95% confidence intervals in tables.
- Continuous episode metrics use bootstrap 95% confidence intervals over episodes.
- Camera behavior is averaged within episode and phase before summarizing across episodes.
- Phase windows: before `x < -0.25 m`, near `|x| <= 0.25 m`, after `x > 0.25 m`.

## Scope note

The assets are professionally formatted, but the evidence remains simulation-only
and single-training-seed. Claims in the manuscript should reflect that scope.
"""
    write_text(out_dir / "README.md", text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="paper/experiment/results/final_semantics_v3_eval_20260508")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    set_journal_style()
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "journal_assets"
    fig_dir = out_dir / "figures"

    _, episodes, traces = read_eval(eval_dir)
    phase_ep = episode_phase_means(traces)
    diagnosis = read_diagnosis(eval_dir)
    training = read_training_curves(eval_dir)

    fig1_panel_exports(fig_dir)
    fig2_training_convergence_panels(training, fig_dir)
    fig3_navigation_panels(episodes, fig_dir)
    fig4_camera_mechanism_panels(episodes, traces, phase_ep, fig_dir)
    fig6_dagger_diagnosis_panels(episodes, phase_ep, diagnosis, fig_dir)
    make_tables(episodes, phase_ep, out_dir)
    write_captions(out_dir)
    write_readme(out_dir, eval_dir)
    print(f"[journal-assets] wrote: {out_dir}")


if __name__ == "__main__":
    main()

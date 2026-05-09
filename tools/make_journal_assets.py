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


def fig1_system_protocol(out_dir: Path) -> None:
    fig = plt.figure(figsize=(DOUBLE_COL, 2.75))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.06, 1.18, 1.52], wspace=0.34)
    ax_task = fig.add_subplot(gs[0, 0])
    ax_loop = fig.add_subplot(gs[0, 1])
    ax_train = fig.add_subplot(gs[0, 2])

    # a. Task schematic.
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
    label_panel(ax_task, "a", -0.06, 1.02)

    # b. Closed-loop differentiable sensing graph.
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
    label_panel(ax_loop, "b", -0.07, 1.02)

    # c. Protocol as evidence-generating chain.
    ax_train.axis("off")
    ax_train.set_xlim(0, 1)
    ax_train.set_ylim(0, 1)
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
    ax_train.text(0.49, 0.285, "closed-loop evaluation\n7 methods x 3 scenes\n100 episodes per scene", ha="center", va="center", fontsize=4.65, linespacing=1.0)
    ax_train.annotate("", xy=(0.48, 0.37), xytext=(0.84, 0.57), arrowprops=arrow)
    ax_train.text(0.02, 0.91, "Relabel and adapt", fontsize=6.1, fontweight="bold", va="top")
    ax_train.text(0.02, 0.09, "final adaptation freezes the camera branch", fontsize=5.1, color="#4A4A4A")
    label_panel(ax_train, "c", -0.06, 1.02)
    save_all(fig, out_dir / "fig1_system_protocol")


def fig2_training_convergence(training: pd.DataFrame, out_dir: Path) -> None:
    if training.empty:
        return

    fig = plt.figure(figsize=(DOUBLE_COL, 2.55))
    gs = GridSpec(1, 3, figure=fig, wspace=0.34)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    specs = [
        ("loss", "training loss", "log"),
        ("success_rate", "training success", "linear"),
        ("collision_rate", "training collision", "linear"),
    ]
    for ax, (metric, ylabel, scale) in zip(axes, specs):
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
                linewidth=1.05,
            )
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        clean_axis(ax, "y")
        if scale == "log":
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.03, 1.03)
        label_panel(ax, chr(ord("a") + axes.index(ax)))
    axes[0].legend(frameon=False, loc="upper right")
    save_all(fig, out_dir / "fig2_training_convergence")


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


def fig2_navigation(episodes: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(DOUBLE_COL, 4.05))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.42, hspace=0.52)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    draw_metric_forest(ax0, episodes, "success_rate", "success rate", (0.0, 1.0), show_ylabels=True)
    ax0.set_title("Navigation success", pad=5)
    label_panel(ax0, "a", -0.12, 1.02)

    draw_metric_forest(ax1, episodes, "fill_rate", "depth fill rate", (0.65, 1.0), show_ylabels=False)
    ax1.set_title("Observation quality", pad=5)
    label_panel(ax1, "b", -0.12, 1.02)

    im = draw_scene_delta_heatmap(ax2, episodes)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.025)
    cbar.set_label(r"$\Delta$ success")
    label_panel(ax2, "c", -0.12, 1.05)

    draw_terminal_ecdf(ax3, episodes)
    label_panel(ax3, "d", -0.12, 1.05)
    save_all(fig, out_dir / "fig3_navigation_performance")


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


def fig3_camera_mechanism(episodes: pd.DataFrame, traces: pd.DataFrame, phase_ep: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(DOUBLE_COL, 4.75))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[0.92, 1.05, 1.22], height_ratios=[1.0, 1.0], wspace=0.42, hspace=0.48)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, :2], hspace=0.12)
    ax3a = fig.add_subplot(sub[0, 0])
    ax3b = fig.add_subplot(sub[1, 0], sharex=ax3a)
    ax4 = fig.add_subplot(gs[1, 2])

    draw_camera_heatmap(ax0, phase_ep)
    label_panel(ax0, "a", -0.22, 1.06)

    draw_exposure_gain_plane(ax1, phase_ep)
    label_panel(ax1, "b", -0.16, 1.06)

    # Scene effect confirms panels focus on degraded near-wall observations.
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
    ax2.barh(y, means, color=[SCENE_COLOR[s] for s in SCENES], edgecolor="#222222", linewidth=0.45)
    ax2.errorbar(
        means,
        y,
        xerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
        fmt="none",
        ecolor="#222222",
        elinewidth=0.55,
        capsize=1.4,
    )
    ax2.set_yticks(y)
    ax2.set_yticklabels([SCENE_LABEL[s] for s in SCENES])
    ax2.set_xlabel("near-slit degradation proxy")
    ax2.set_xlim(0, max(0.20, max(means) * 1.18))
    clean_axis(ax2, "x")
    ax2.set_title("Near-slit degradation encountered", pad=5)
    label_panel(ax2, "c", -0.18, 1.06)

    draw_profile_pair(ax3a, ax3b, traces)
    label_panel(ax3a, "d", -0.08, 1.08)

    draw_trajectory_envelope(ax4, episodes, traces)
    label_panel(ax4, "e", -0.17, 1.06)
    save_all(fig, out_dir / "fig4_active_camera_mechanism")


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


def fig6_dagger_diagnosis(episodes: pd.DataFrame, phase_ep: pd.DataFrame, diagnosis: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(DOUBLE_COL, 3.55))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.10, 1.15], wspace=0.43)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

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
    ax0.plot(x, vals, color="#4A4A4A", lw=0.8, zorder=1)
    colors = ["#9E9E9E", METHOD_COLOR["pretrained"], METHOD_COLOR["dagger"], METHOD_COLOR["flightonly"]]
    for xi, mean, lo, hi, color in zip(x, vals, los, his, colors):
        if np.isfinite(lo):
            ax0.errorbar([xi], [mean], yerr=[[mean - lo], [hi - mean]], fmt="none", ecolor="#222222", elinewidth=0.55, capsize=1.5)
        ax0.scatter([xi], [mean], s=29, color=color, edgecolor="#222222", linewidth=0.42, zorder=3)
        ax0.text(xi, mean + 0.025, f"{mean:.2f}", ha="center", fontsize=5.1)
    ax0.set_xticks(x)
    ax0.set_xticklabels(stages, rotation=28, ha="right")
    ax0.set_ylabel("glare-dark camera L1")
    ax0.set_ylim(0, 0.42)
    clean_axis(ax0, "y")
    ax0.set_title("Online camera semantics", pad=5)
    label_panel(ax0, "a", -0.18, 1.06)

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
        ax1.bar(xpos, means, width=width, color=PARAM_COLOR[param], edgecolor="#222222", linewidth=0.42, label=param)
        ax1.errorbar(xpos, means, yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)], fmt="none", ecolor="#222222", elinewidth=0.5, capsize=1.4)
    ax1.axhline(0, color="#222222", lw=0.55)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Pretrain", "DAgger", "Ours"])
    ax1.set_ylabel("dark - glare near parameter")
    ax1.set_ylim(-0.12, 0.72)
    ax1.legend(frameon=False, ncol=3, loc="upper left", handlelength=1.2, columnspacing=0.9)
    clean_axis(ax1, "y")
    ax1.set_title("Recovered camera semantics", pad=5)
    label_panel(ax1, "b", -0.16, 1.06)

    scatter_methods = ["fixed", "nondiff", "pretrained", "dagger", "flightonly"]
    coords: dict[str, tuple[float, float]] = {}
    for method in scatter_methods:
        sep = l1_scene_separation_ci(phase_ep, method)[0]
        succ = metric_ci(episodes, method, "success_rate")[0]
        coords[method] = (sep, succ)
        ax2.scatter([sep], [succ], s=34, color=METHOD_COLOR[method], edgecolor="#222222", linewidth=0.48, zorder=3)
    label_offsets = {
        "fixed": (0.010, -0.040),
        "nondiff": (0.010, 0.018),
        "pretrained": (0.010, -0.030),
        "dagger": (0.010, -0.020),
        "flightonly": (0.010, 0.018),
    }
    for method in scatter_methods:
        dx, dy = label_offsets[method]
        ax2.text(coords[method][0] + dx, coords[method][1] + dy, METHOD_LABEL_J[method], fontsize=5.2)
    for a, b in [("pretrained", "dagger"), ("dagger", "flightonly")]:
        ax2.annotate("", xy=coords[b], xytext=coords[a], arrowprops=dict(arrowstyle="-|>", lw=0.65, color="#555555", shrinkA=5, shrinkB=5))
    ax2.set_xlabel("camera separation")
    ax2.set_ylabel("success rate")
    ax2.set_xlim(-0.02, 0.40)
    ax2.set_ylim(0.0, 0.84)
    clean_axis(ax2, "both")
    ax2.set_title("Separation must be paired with flight adaptation", pad=5)
    label_panel(ax2, "c", -0.14, 1.06)
    save_all(fig, out_dir / "fig6_dagger_relabel_diagnosis")


def figS1_full_matrix(episodes: pd.DataFrame, phase_ep: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(DOUBLE_COL, 4.8))
    gs = GridSpec(2, 2, figure=fig, wspace=0.36, hspace=0.47)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    methods = ["flightonly", "fixed", "randfix", "nondiff", "pretrained", "dagger", "zero"]
    metrics = [
        ("success_rate", "success", "viridis", 0.0, 1.0),
        ("fill_rate", "depth fill", "magma", 0.55, 1.0),
        ("collision_rate", "collision", "inferno", 0.0, 1.0),
    ]
    for ax, (metric, title, cmap, vmin, vmax), label in zip(axes[:3], metrics, ["a", "b", "c"]):
        data = np.array([[metric_ci(episodes, m, metric, s)[0] for s in SCENES] for m in methods])
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(SCENES)))
        ax.set_xticklabels([SCENE_LABEL[s] for s in SCENES])
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels([METHOD_LABEL_J[m] for m in methods])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = "white" if data[i, j] < vmin + 0.33 * (vmax - vmin) else "#111111"
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=5.0, color=color)
        ax.set_title(title, pad=4)
        cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.025)
        cb.set_label(title)
        label_panel(ax, label, -0.12, 1.05)
    ax = axes[3]
    vals = np.array([l1_scene_separation_ci(phase_ep, m)[0] for m in methods])
    ax.bar(np.arange(len(methods)), vals, color=[METHOD_COLOR[m] for m in methods], edgecolor="#222222", linewidth=0.42)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([METHOD_LABEL_J[m] for m in methods], rotation=35, ha="right")
    ax.set_ylabel("glare-dark near L1")
    ax.set_ylim(0, 0.42)
    clean_axis(ax, "y")
    ax.set_title("camera separation", pad=4)
    label_panel(ax, "d", -0.12, 1.05)
    save_all(fig, out_dir / "extended_data_fig1_full_matrix")


def figS2_terminal_distance(episodes: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(SINGLE_COL, 2.65))
    ax = fig.add_subplot(111)
    box_data = [episodes[episodes["method"] == m]["final_goal_dist"].astype(float).to_numpy() for m in METHOD_ORDER_MAIN]
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, widths=0.58, medianprops=dict(color="#111111", lw=0.7))
    for patch, method in zip(bp["boxes"], METHOD_ORDER_MAIN):
        patch.set_facecolor(METHOD_COLOR[method])
        patch.set_alpha(0.62)
        patch.set_edgecolor("#222222")
    for item in bp["whiskers"] + bp["caps"]:
        item.set_linewidth(0.5)
    ax.set_xticks(np.arange(1, len(METHOD_ORDER_MAIN) + 1))
    ax.set_xticklabels([METHOD_LABEL_J[m] for m in METHOD_ORDER_MAIN], rotation=35, ha="right")
    ax.set_ylabel("terminal distance (m)")
    clean_axis(ax, "y")
    save_all(fig, out_dir / "extended_data_fig2_terminal_distance")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_tables(episodes: pd.DataFrame, phase_ep: pd.DataFrame, out_dir: Path) -> None:
    table_dir = out_dir / "tables"
    fixed_success = metric_ci(episodes, "fixed", "success_rate")[0]
    fixed_fill = metric_ci(episodes, "fixed", "fill_rate")[0]
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
        r"\caption{Primary closed-loop navigation results over 300 episodes per method. Brackets denote 95\% confidence intervals: Wilson intervals for binary outcomes and bootstrap intervals over episodes for fill. Deltas are relative to the fixed-camera baseline.}",
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
        r"\caption{Scene-level success/fill rates. Each scene contains 100 closed-loop episodes per method.}",
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
Separate glare, dark, and specular panels are provided.

## Figure 6 | Camera relabeling and flight adaptation are complementary.

**a,** Glare-dark camera separation is measured in the relabeled teacher data
and in online closed-loop rollouts for the pretrained, DAgger-relabelled and
final policies. **b,** The separation is mainly carried by exposure and gain:
low-reflectance dark-material scenes require higher values than glare. **c,**
Camera semantics alone is insufficient for flight; final performance requires
combining DAgger camera relabeling with flight-only adaptation.

## Extended Data Figure 1 | Complete method-by-scene evaluation matrix.

Full success, fill, collision, and glare-dark camera-separation matrices for all
main and diagnostic checkpoints.

## Extended Data Figure 2 | Terminal distance distributions.

Terminal goal-distance distributions for the main comparison methods.
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

## Main figures

- `figures/fig1_system_protocol.pdf`
- `figures/fig2_training_convergence.pdf`
- `figures/fig3_navigation_performance.pdf`
- `figures/fig4_active_camera_mechanism.pdf`
- `figures/fig5_depth_observation_sequence_glare.pdf`
- `figures/fig5_depth_observation_sequence_dark.pdf`
- `figures/fig5_depth_observation_sequence_specular.pdf`
- `figures/fig6_dagger_relabel_diagnosis.pdf`

## Extended data

- `figures/extended_data_fig1_full_matrix.pdf`
- `figures/extended_data_fig2_terminal_distance.pdf`
- `figures/extended_data_fig3_method_depth_sequences_glare.pdf`
- `figures/extended_data_fig3_method_depth_sequences_dark.pdf`
- `figures/extended_data_fig3_method_depth_sequences_specular.pdf`

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

    fig1_system_protocol(fig_dir)
    fig2_training_convergence(training, fig_dir)
    fig2_navigation(episodes, fig_dir)
    fig3_camera_mechanism(episodes, traces, phase_ep, fig_dir)
    fig6_dagger_diagnosis(episodes, phase_ep, diagnosis, fig_dir)
    figS1_full_matrix(episodes, phase_ep, fig_dir)
    figS2_terminal_distance(episodes, fig_dir)
    make_tables(episodes, phase_ep, out_dir)
    write_captions(out_dir)
    write_readme(out_dir, eval_dir)
    print(f"[journal-assets] wrote: {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create compact paper-style depth probe figures.

The full probe panels are useful for debugging, but they are too wide for a
paper.  This script rerenders a small, fixed set of scene/pose/camera conditions
and writes compact panels containing only:

- raw_depth
- depth observation
- quality
- invalid mask

The default choices intentionally show the active-sensing contrast:
glare needs low exposure/gain with high projector power, specular prefers a
low-power safe setting, and low-reflectance dark material benefits from a
stronger return setting.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import set_global_seed  # noqa: E402
from train_utils import build_env  # noqa: E402
from tools.probe_opening_depth_views import (  # noqa: E402
    CameraSetting,
    _build_project_args,
    _make_poses,
    _opening_target,
    _parse_camera_settings,
    _render_condition,
)


@dataclass(frozen=True)
class FigureSpec:
    scene: str
    slot: str
    pose_x: float
    settings: tuple[str, ...]


DEFAULT_SPECS = (
    FigureSpec("glare", "left", 0.45, ("baseline", "glare_expected", "overexposed")),
    FigureSpec("specular", "left", 0.45, ("baseline", "specular_safe", "high_power")),
    FigureSpec("dark", "left", -0.20, ("baseline", "dark_expected", "low_return_bad")),
)


def _masked_depth(depth: np.ndarray, min_valid: float) -> np.ndarray:
    out = depth.astype(np.float32).copy()
    out[out <= float(min_valid) + 1e-6] = np.nan
    return out


def _camera_setting_map(project_args) -> dict[str, CameraSetting]:
    return {setting.name: setting for setting in _parse_camera_settings(None, project_args)}


def _render_spec(spec: FigureSpec, project_args, device: torch.device):
    cond_args = copy.deepcopy(project_args)
    cond_args.scenarios = [spec.scene]
    cond_args.sun_glare_eval_slot = spec.slot
    env = build_env(1, cond_args, device, eval_mode=True)
    env.reset(scene_name=spec.scene)

    target = _opening_target(env)
    pose = _make_poses(env, [float(spec.pose_x)], "slot")[0]
    settings_by_name = _camera_setting_map(cond_args)

    rendered = []
    for name in spec.settings:
        if name not in settings_by_name:
            raise KeyError(f"unknown camera setting {name!r}; available={sorted(settings_by_name)}")
        row, maps = _render_condition(env, cond_args, pose, target, settings_by_name[name])
        row["paper_pose_x"] = float(spec.pose_x)
        row["paper_slot"] = spec.slot
        rendered.append((row, maps))
    return rendered


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_panel(path: Path, rendered: list[tuple[dict, dict]], project_args, *, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rendered)
    fig, axes = plt.subplots(n, 4, figsize=(10.8, max(1.85 * n, 2.4)), squeeze=False)
    depth_cmap = plt.cm.viridis.copy()
    depth_cmap.set_bad("black")

    for r, (row, maps) in enumerate(rendered):
        raw = maps["raw_depth"]
        depth = maps["depth"]
        quality = maps["quality"]
        invalid = maps["invalid"]
        raw_show = _masked_depth(raw, project_args.depth_min_valid)
        depth_show = _masked_depth(depth, project_args.depth_min_valid)

        axes[r, 0].imshow(raw_show, vmin=project_args.depth_min_valid,
                          vmax=project_args.depth_max_range, cmap=depth_cmap)
        axes[r, 1].imshow(depth_show, vmin=project_args.depth_min_valid,
                          vmax=project_args.depth_max_range, cmap=depth_cmap)
        axes[r, 2].imshow(np.zeros_like(depth) if quality is None else quality,
                          vmin=0.0, vmax=1.0, cmap="magma")
        axes[r, 3].imshow(np.zeros_like(depth) if invalid is None else invalid,
                          vmin=0.0, vmax=1.0, cmap="gray")

        labels = ("raw depth", "depth obs", "quality", "invalid")
        for c, label in enumerate(labels):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(label, fontsize=10)

        axes[r, 0].set_ylabel(
            f"{row['scene']} / {row['setting']}\n"
            f"p/e/g={row['power']:.2f}/{row['exposure']:.2f}/{row['gain']:.2f}\n"
            f"fill={row['local_fill']:.2f}, q={row['local_quality_mean']:.2f}",
            fontsize=8,
        )

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _raw_depth_variation(rendered: list[tuple[dict, dict]]) -> float:
    if not rendered:
        return float("nan")
    ref = rendered[0][1]["raw_depth"]
    return max(float(np.max(np.abs(maps["raw_depth"] - ref))) for _, maps in rendered)


def _write_report(path: Path, all_rows: list[dict], raw_variations: dict[str, float]):
    lines = [
        "# Compact Depth Probe Figures",
        "",
        "Generated compact paper-style panels from controlled opening-depth probes.",
        "",
        "Files:",
        "",
        "- `paper_depth_probe_compact_all.png`: all selected scenes/settings.",
        "- `paper_depth_probe_glare.png`: glare-only compact panel.",
        "- `paper_depth_probe_specular.png`: specular-only compact panel.",
        "- `paper_depth_probe_dark.png`: dark-only compact panel.",
        "- `paper_depth_probe_metrics.csv`: numeric metrics for the plotted rows.",
        "",
        "Sanity Checks:",
        "",
    ]
    for scene, raw_delta in raw_variations.items():
        lines.append(f"- {scene}: max raw-depth pixel delta across camera settings = `{raw_delta:.6g}`")

    lines.extend([
        "",
        "Plotted Conditions:",
        "",
        "| scene | slot | local x | setting | p/e/g | local fill | local quality | invalid |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ])
    for row in all_rows:
        lines.append(
            f"| {row['scene']} | {row['slot']} | {float(row['local_x']):.2f} | "
            f"{row['setting']} | {row['power']:.2f}/{row['exposure']:.2f}/{row['gain']:.2f} | "
            f"{float(row['local_fill']):.3f} | {float(row['local_quality_mean']):.3f} | "
            f"{float(row['invalid_rate']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/slit_active_sensing.args")
    parser.add_argument("--out_dir", default="paper/experiment/results/depth_probe_paper_figures_20260506")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sensor_impl", default="cuda", choices=["cuda", "python"])
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main():
    parser = _make_arg_parser()
    script_args, project_overrides = parser.parse_known_args()

    out_dir = Path(script_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    project_args = _build_project_args(Path(script_args.config), project_overrides)
    project_args.diff_sensor_impl["diff_depth"] = str(script_args.sensor_impl)
    if hasattr(project_args, "sun_glare_randomize"):
        project_args.sun_glare_randomize = False
    if hasattr(project_args, "random_rotation"):
        project_args.random_rotation = False
    if script_args.seed is not None:
        project_args.seed = int(script_args.seed)
    set_global_seed(project_args.seed, project_args.deterministic)

    device = torch.device(script_args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    all_rendered: list[tuple[dict, dict]] = []
    raw_variations: dict[str, float] = {}
    for spec in DEFAULT_SPECS:
        rendered = _render_spec(spec, project_args, device)
        all_rendered.extend(rendered)
        raw_variations[spec.scene] = _raw_depth_variation(rendered)
        _plot_panel(
            out_dir / f"paper_depth_probe_{spec.scene}.png",
            rendered,
            project_args,
            title=f"{spec.scene}: slot={spec.slot}, local x={spec.pose_x:+.2f}",
        )

    _plot_panel(
        out_dir / "paper_depth_probe_compact_all.png",
        all_rendered,
        project_args,
        title="Controlled depth observations under different camera parameters",
    )
    rows = [row for row, _ in all_rendered]
    _write_csv(out_dir / "paper_depth_probe_metrics.csv", rows)
    _write_report(out_dir / "summary.md", rows, raw_variations)

    print(f"[paper-fig] rows={len(rows)}")
    print(f"[paper-fig] out_dir={out_dir}")


if __name__ == "__main__":
    main()

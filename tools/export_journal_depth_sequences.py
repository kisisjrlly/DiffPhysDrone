#!/usr/bin/env python3
"""Export paper-quality qualitative depth observation sequences.

This script runs representative closed-loop rollouts and stores the actual
sensor observations needed for manuscript qualitative panels. It complements
`tools/make_journal_assets.py`: that script redraws aggregate CSV metrics,
whereas this script actually renders raw/geometric depth and camera-dependent
observed depth at selected rollout poses.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import canonicalize_sun_glare_slot, set_global_seed  # noqa: E402
from train_utils import build_env  # noqa: E402
from tools.probe_rollout_depth_views import (  # noqa: E402
    CameraSetting,
    RolloutCapture,
    _build_model,
    _load_checkpoint,
    _load_project_args,
    _plot_topdown_overview,
    _render_setting_at_capture,
    _rollout_captures,
)


METHOD_LABEL = {
    "fixed": "Fixed",
    "randfix": "Random fixed",
    "flightonly": "Ours",
}
METHOD_ORDER = ["fixed", "randfix", "flightonly"]
METHOD_COLOR = {
    "fixed": "#7A7A7A",
    "randfix": "#CC79A7",
    "flightonly": "#0072B2",
}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    checkpoint: Path
    camera_control_mode: str
    policy_depth_mode: str = "depth"
    train_flight_only: bool = False


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


def parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty local-x list")
    return vals


def method_specs(args: argparse.Namespace) -> dict[str, MethodSpec]:
    ckpts = {
        "fixed": args.fixed_ckpt,
        "randfix": args.randfix_ckpt,
        "flightonly": args.flightonly_ckpt,
    }
    templates = {
        "fixed": ("Fixed", "fixed", "depth", False),
        "randfix": ("Random fixed", "fixed_random_static", "depth", False),
        "flightonly": ("Ours", "learned", "depth", True),
    }
    out: dict[str, MethodSpec] = {}
    for key in METHOD_ORDER:
        ckpt = ckpts[key]
        if not ckpt:
            continue
        label, camera_mode, depth_mode, train_flight_only = templates[key]
        path = Path(ckpt)
        if not path.is_file():
            raise FileNotFoundError(f"{key} checkpoint not found: {path}")
        out[key] = MethodSpec(key, label, path, camera_mode, depth_mode, train_flight_only)
    missing = [m for m in METHOD_ORDER if m not in out]
    if missing:
        raise ValueError(f"missing checkpoints for methods: {missing}")
    return out


def configure_args(base_args, spec: MethodSpec, scene: str, slot: str, keep_random_rotation: bool):
    args = copy.deepcopy(base_args)
    args.batch_size = 1
    args.scenarios = [scene]
    args.sun_glare_eval_slot = canonicalize_sun_glare_slot(slot)
    args.camera_control_mode = spec.camera_control_mode
    args.policy_depth_mode = spec.policy_depth_mode
    args.sensor_grad_mode = "detached"
    args.train_flight_only = spec.train_flight_only
    args.vis_enable = False
    args.wandb_disabled = True
    if not keep_random_rotation and hasattr(args, "random_rotation"):
        args.random_rotation = False
    args.resume = str(spec.checkpoint)
    return args


def local_x_for_capture(env, cap: RolloutCapture) -> float:
    p = cap.snapshot["p"][0]
    local = torch.matmul(env.R_scene_T[0].to(p.device, p.dtype), p)
    return float(local[0].detach().cpu().item())


def capture_local_xy(env, cap: RolloutCapture) -> tuple[float, float]:
    p = cap.snapshot["p"][0]
    local = torch.matmul(env.R_scene_T[0].to(p.device, p.dtype), p)
    return float(local[0].detach().cpu().item()), float(local[1].detach().cpu().item())


def capture_trajectory_local_xy(env, captures: list[RolloutCapture]) -> np.ndarray:
    if not captures:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([capture_local_xy(env, cap) for cap in captures], dtype=np.float32)


def choose_captures(env, captures: list[RolloutCapture], targets: list[float]) -> list[RolloutCapture]:
    if not captures:
        raise RuntimeError("rollout produced no captures")
    xs = np.array([local_x_for_capture(env, cap) for cap in captures], dtype=np.float32)
    selected: list[RolloutCapture] = []
    used: set[int] = set()
    for target in targets:
        order = np.argsort(np.abs(xs - float(target)))
        idx = int(order[0])
        for candidate in order:
            if int(candidate) not in used:
                idx = int(candidate)
                break
        used.add(idx)
        selected.append(captures[idx])
    selected.sort(key=lambda cap: local_x_for_capture(env, cap))
    return selected


def build_method_rollout(
    base_args,
    spec: MethodSpec,
    scene: str,
    slot: str,
    targets: list[float],
    seed: int,
    device: torch.device,
    keep_random_rotation: bool,
) -> tuple:
    args = configure_args(base_args, spec, scene, slot, keep_random_rotation)
    set_global_seed(seed, getattr(args, "deterministic", False))
    model = _build_model(args, device)
    _load_checkpoint(model, spec.checkpoint, device)
    env = build_env(1, args, device, eval_mode=True)
    captures = _rollout_captures(
        env,
        args,
        model,
        scene,
        device,
        steps_filter=None,
        sample_every=1,
        max_samples=0,
    )
    selected = choose_captures(env, captures, targets)
    trajectory = capture_trajectory_local_xy(env, captures)
    return args, env, selected, trajectory


def masked_depth(arr: np.ndarray | None, min_valid: float) -> np.ndarray:
    if arr is None:
        return np.zeros((1, 1), dtype=np.float32)
    out = np.asarray(arr, dtype=np.float32).copy()
    out[out <= float(min_valid) + 1e-6] = np.nan
    return out


def render_same_pose_sequences(
    ref_env,
    ref_args,
    ref_caps: list[RolloutCapture],
    method_caps: dict[str, list[RolloutCapture]],
) -> tuple[list[dict], dict[tuple[str, int, str], dict[str, np.ndarray | None]]]:
    rows: list[dict] = []
    maps: dict[tuple[str, int, str], dict[str, np.ndarray | None]] = {}
    for col_idx, ref_cap in enumerate(ref_caps):
        for method in METHOD_ORDER:
            source_cap = method_caps[method][min(col_idx, len(method_caps[method]) - 1)]
            setting = CameraSetting(
                method,
                float(source_cap.power),
                float(source_cap.exposure),
                float(source_cap.gain),
            )
            row, rendered = _render_setting_at_capture(ref_env, ref_args, ref_cap, setting)
            row.update(
                {
                    "panel": "same_pose",
                    "method": method,
                    "method_label": METHOD_LABEL[method],
                    "column_idx": col_idx,
                    "reference_step": int(ref_cap.step),
                    "source_step": int(source_cap.step),
                    "reference_local_x": float(row["local_x"]),
                    "source_policy_power": float(source_cap.power),
                    "source_policy_exposure": float(source_cap.exposure),
                    "source_policy_gain": float(source_cap.gain),
                }
            )
            rows.append(row)
            maps[("same_pose", col_idx, method)] = rendered
            if method == "flightonly":
                maps[("same_pose", col_idx, "raw")] = rendered
    return rows, maps


def render_own_pose_sequences(
    env_by_method: dict[str, object],
    args_by_method: dict[str, object],
    caps_by_method: dict[str, list[RolloutCapture]],
) -> tuple[list[dict], dict[tuple[str, int, str], dict[str, np.ndarray | None]]]:
    rows: list[dict] = []
    maps: dict[tuple[str, int, str], dict[str, np.ndarray | None]] = {}
    for method in METHOD_ORDER:
        env = env_by_method[method]
        args = args_by_method[method]
        for col_idx, cap in enumerate(caps_by_method[method]):
            setting = CameraSetting(method, float(cap.power), float(cap.exposure), float(cap.gain))
            row, rendered = _render_setting_at_capture(env, args, cap, setting)
            row.update(
                {
                    "panel": "own_pose",
                    "method": method,
                    "method_label": METHOD_LABEL[method],
                    "column_idx": col_idx,
                    "reference_step": int(cap.step),
                    "source_step": int(cap.step),
                    "reference_local_x": float(row["local_x"]),
                }
            )
            rows.append(row)
            maps[("own_pose", col_idx, method)] = rendered
    return rows, maps


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: Path, maps: dict[tuple[str, int, str], dict[str, np.ndarray | None]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for key, rendered in maps.items():
        if len(key) == 4:
            scene, panel, col_idx, method = key
            prefix = f"{scene}_{panel}_col{int(col_idx):02d}_{method}"
        elif len(key) == 3:
            panel, col_idx, method = key
            prefix = f"{panel}_col{int(col_idx):02d}_{method}"
        else:
            prefix = "_".join(str(part) for part in key)
        for name in [
            "raw_depth",
            "depth",
            "quality",
            "invalid",
            "scene_effect",
            "scene_mask",
            "front_wall_hit",
            "back_wall_hit",
            "slit_cue",
            "key_cue_artifact",
        ]:
            arr = rendered.get(name)
            if arr is not None:
                arrays[f"{prefix}_{name}"] = np.asarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 6.0,
            "axes.titlesize": 6.2,
            "axes.labelsize": 5.8,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
            "axes.linewidth": 0.5,
        }
    )


def draw_same_pose_figure(
    out_base: Path,
    scene: str,
    slot: str,
    rows: list[dict],
    maps: dict[tuple[str, int, str], dict[str, np.ndarray | None]],
    min_valid: float,
    max_range: float,
    trajectory_xy: np.ndarray | None = None,
) -> None:
    setup_style()
    cols = sorted({int(r["column_idx"]) for r in rows if r["panel"] == "same_pose"})
    ncols = len(cols)
    row_names = ["map", "raw", *METHOD_ORDER]
    fig = plt.figure(figsize=(7.2, 1.06 * len(row_names) + 0.22))
    gs = GridSpec(len(row_names), ncols, figure=fig, hspace=0.12, wspace=0.04)
    depth_cmap = plt.cm.viridis.copy()
    depth_cmap.set_bad("black")
    row_lookup = {
        (int(r["column_idx"]), r["method"]): r
        for r in rows
        if r["panel"] == "same_pose"
    }
    for c, col in enumerate(cols):
        raw_maps = maps[("same_pose", col, "raw")]
        ref_row = row_lookup[(col, "flightonly")]
        ax = fig.add_subplot(gs[0, c])
        _plot_topdown_overview(ax, ref_row, type("Args", (), {"fov_x_half_tan": 0.82})())
        if trajectory_xy is not None and len(trajectory_xy) >= 2:
            ax.plot(
                trajectory_xy[:, 0],
                trajectory_xy[:, 1],
                color="#0072B2",
                linewidth=1.35,
                alpha=0.95,
                zorder=7,
            )
            ax.scatter(
                [trajectory_xy[0, 0]],
                [trajectory_xy[0, 1]],
                marker="o",
                s=22,
                color="#FFFFFF",
                edgecolors="#0072B2",
                linewidths=0.8,
                zorder=8,
            )
            ax.scatter(
                [trajectory_xy[-1, 0]],
                [trajectory_xy[-1, 1]],
                marker=">",
                s=30,
                color="#0072B2",
                edgecolors="#111111",
                linewidths=0.35,
                zorder=8,
            )
        ax.set_title(f"x={ref_row['local_x']:.2f}, step {int(ref_row['reference_step'])}")
        if c == 0:
            ax.set_ylabel("pose")

        ax = fig.add_subplot(gs[1, c])
        ax.imshow(
            masked_depth(raw_maps.get("raw_depth"), min_valid),
            vmin=min_valid,
            vmax=max_range,
            cmap=depth_cmap,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("raw\ngeometry")

        for r, method in enumerate(METHOD_ORDER, start=2):
            cell_maps = maps[("same_pose", col, method)]
            cell = row_lookup[(col, method)]
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(
                masked_depth(cell_maps.get("depth"), min_valid),
                vmin=min_valid,
                vmax=max_range,
                cmap=depth_cmap,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.03,
                0.96,
                f"p/e/g {cell['power']:.2f}/{cell['exposure']:.2f}/{cell['gain']:.2f}\nfill {cell['local_fill']:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=4.9,
                color="white",
                bbox=dict(facecolor=(0, 0, 0, 0.42), edgecolor="none", pad=1.0),
            )
            if c == 0:
                ax.set_ylabel(METHOD_LABEL[method], color=METHOD_COLOR[method])
    fig.suptitle(f"Observed depth at matched poses: {scene}, slot={slot}", y=0.995, fontsize=7.0)
    save_all(fig, out_base)


def draw_own_pose_figure(
    out_base: Path,
    scene: str,
    slot: str,
    rows: list[dict],
    maps: dict[tuple[str, int, str], dict[str, np.ndarray | None]],
    min_valid: float,
    max_range: float,
) -> None:
    setup_style()
    cols = sorted({int(r["column_idx"]) for r in rows if r["panel"] == "own_pose"})
    ncols = len(cols)
    fig = plt.figure(figsize=(7.2, 1.06 * len(METHOD_ORDER) + 0.28))
    gs = GridSpec(len(METHOD_ORDER), ncols, figure=fig, hspace=0.08, wspace=0.04)
    depth_cmap = plt.cm.viridis.copy()
    depth_cmap.set_bad("black")
    row_lookup = {
        (int(r["column_idx"]), r["method"]): r
        for r in rows
        if r["panel"] == "own_pose"
    }
    for r, method in enumerate(METHOD_ORDER):
        for c, col in enumerate(cols):
            cell_maps = maps[("own_pose", col, method)]
            cell = row_lookup[(col, method)]
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(
                masked_depth(cell_maps.get("depth"), min_valid),
                vmin=min_valid,
                vmax=max_range,
                cmap=depth_cmap,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.03,
                0.96,
                f"x {cell['local_x']:.2f}\np/e/g {cell['power']:.2f}/{cell['exposure']:.2f}/{cell['gain']:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=4.9,
                color="white",
                bbox=dict(facecolor=(0, 0, 0, 0.42), edgecolor="none", pad=1.0),
            )
            if r == 0:
                ax.set_title(f"sample {col + 1}")
            if c == 0:
                ax.set_ylabel(METHOD_LABEL[method], color=METHOD_COLOR[method])
    fig.suptitle(f"Observed depth along each method's own closed-loop trajectory: {scene}, slot={slot}", y=0.995, fontsize=7.0)
    save_all(fig, out_base)


def write_report(path: Path, rows: list[dict], scenes: list[str], slot: str) -> None:
    lines = [
        "# Journal Qualitative Depth Sequences",
        "",
        f"- scenes: `{', '.join(scenes)}`",
        f"- slit slot: `{slot}`",
        f"- rows: `{len(rows)}`",
        "",
        "Outputs:",
        "",
        "- `figures/fig5_depth_observation_sequence_<scene>.pdf`: matched-pose raw/depth comparison.",
        "- `qualitative_depth/depth_sequence_rows.csv`: per-panel camera parameters and local metrics.",
        "- `qualitative_depth/depth_sequence_arrays.npz`: raw depth, observed depth, quality, invalid and effect arrays.",
        "",
        "Interpretation:",
        "",
        "The matched-pose figure uses the final policy trajectory as the reference pose sequence and re-renders",
        "the sensor observation at those exact poses with camera settings taken from fixed, random-fixed,",
        "and active-camera policies.",
        "The first pose row overlays the complete final-policy local trajectory from start through the slit toward the goal.",
        "This isolates the camera-parameter effect on depth observations.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/slit_active_sensing.args")
    p.add_argument("--eval_dir", default="paper/experiment/results/final_dagger_flightonly_eval_20260507")
    p.add_argument("--out_dir", default=None, help="Default: <eval_dir>/journal_assets")
    p.add_argument("--scenarios", nargs="*", default=["glare", "dark", "specular"])
    p.add_argument("--slot", default="far_right")
    p.add_argument("--target_local_x", default="-1.20,-0.75,-0.35,-0.08,0.18")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--keep_random_rotation", action="store_true")
    p.add_argument("--flightonly_ckpt", default="checkpoint/2026-05-07-15-07-03/checkpoint0014.pth")
    p.add_argument("--fixed_ckpt", default="checkpoint/2026-05-06-12-04-58/checkpoint0014.pth")
    p.add_argument("--randfix_ckpt", default="checkpoint/2026-05-07-01-26-30/checkpoint0014.pth")
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    eval_dir = Path(cli.eval_dir)
    out_dir = Path(cli.out_dir) if cli.out_dir else eval_dir / "journal_assets"
    q_dir = out_dir / "qualitative_depth"
    fig_dir = out_dir / "figures"
    q_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    specs = method_specs(cli)
    base_args = _load_project_args(Path(cli.config), [])
    base_args.diff_sensor_impl["diff_depth"] = "cuda"
    targets = parse_float_list(cli.target_local_x)
    slot = canonicalize_sun_glare_slot(cli.slot)
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    all_rows: list[dict] = []
    all_maps: dict[tuple[str, int, str, str], dict[str, np.ndarray | None]] = {}
    for scene_idx, scene in enumerate(cli.scenarios):
        args_by_method = {}
        env_by_method = {}
        caps_by_method = {}
        trajectory_by_method = {}
        for method in METHOD_ORDER:
            args, env, caps, trajectory = build_method_rollout(
                base_args,
                specs[method],
                scene,
                slot,
                targets,
                int(cli.seed) + 1000 * scene_idx,
                device,
                bool(cli.keep_random_rotation),
            )
            args_by_method[method] = args
            env_by_method[method] = env
            caps_by_method[method] = caps
            trajectory_by_method[method] = trajectory

        ref_env = env_by_method["flightonly"]
        ref_args = args_by_method["flightonly"]
        same_rows, same_maps = render_same_pose_sequences(ref_env, ref_args, caps_by_method["flightonly"], caps_by_method)
        own_rows, own_maps = render_own_pose_sequences(env_by_method, args_by_method, caps_by_method)
        for row in same_rows + own_rows:
            row["scene_name"] = scene
            row["slot"] = slot
        scene_rows = same_rows + own_rows
        scene_maps = {**same_maps, **own_maps}
        all_rows.extend(scene_rows)
        for key, value in scene_maps.items():
            all_maps[(scene, *key)] = value

        draw_same_pose_figure(
            fig_dir / f"fig5_depth_observation_sequence_{scene}",
            scene,
            slot,
            same_rows,
            same_maps,
            float(ref_args.depth_min_valid),
            float(ref_args.depth_max_range),
            trajectory_by_method.get("flightonly"),
        )
    write_csv(q_dir / "depth_sequence_rows.csv", all_rows)
    save_npz(q_dir / "depth_sequence_arrays.npz", all_maps)
    write_report(q_dir / "README.md", all_rows, list(cli.scenarios), slot)
    print(f"[qual-depth] wrote: {q_dir}")
    print(f"[qual-depth] figures: {fig_dir}")


if __name__ == "__main__":
    main()

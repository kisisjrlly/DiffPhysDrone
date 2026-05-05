#!/usr/bin/env python3
"""
Render paper-friendly overlays for the local slit sensor/material region.

The shared-slit benchmark applies glare through the slit opening plus a local
halo, while specular/dark are material patches on the wall beside the slit.
This utility makes those masks visible on top of the rendered depth image and
in a simple top-down geometry sketch, so probed camera settings can be tied
back to the actual affected wall/slit region.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_utils import build_env  # noqa: E402
from tools.probe_opening_depth_views import (  # noqa: E402
    CameraSetting,
    SLOT_ORDER,
    _build_project_args,
    _make_poses,
    _opening_target,
    _parse_float_list,
    _parse_scenes,
    _parse_slots,
    _render_condition,
)


DEFAULT_SETTINGS = {
    "glare": CameraSetting("good_highP_lowE_lowG", 0.95, 0.30, 0.05),
    "specular": CameraSetting("good_lowP_lowE_lowG", 0.25, 0.30, 0.05),
    "dark": CameraSetting("good_highP_highE_highG", 0.95, 0.90, 0.65),
}

SCENE_COLORS = {
    "glare": (1.0, 0.75, 0.10),
    "specular": (0.95, 0.20, 0.10),
    "dark": (0.15, 0.45, 1.0),
}


def _camera_setting_for_scene(scene: str, text: str | None) -> CameraSetting:
    if text:
        parts = [float(x.strip()) for x in text.split(",") if x.strip()]
        if len(parts) != 3:
            raise ValueError("--camera must be 'power,exposure,gain'")
        return CameraSetting("custom", parts[0], parts[1], parts[2])
    return DEFAULT_SETTINGS.get(scene, DEFAULT_SETTINGS["glare"])


def _rgba_overlay(mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    alpha = np.clip(mask, 0.0, 1.0) * 0.48
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., 0] = float(color[0])
    rgba[..., 1] = float(color[1])
    rgba[..., 2] = float(color[2])
    rgba[..., 3] = alpha
    return rgba


def _plot_topdown(ax, fx: dict, row: dict, scene: str, color: tuple[float, float, float]):
    import matplotlib.patches as patches

    wall_x = float(fx.get("geometry_wall_x", row.get("wall_x", 1.82)))
    slit_y = float(fx.get("slit_center_y", row.get("slit_center_y", 0.0)))
    slit_half_y = float(fx.get("slit_half_y", 0.18))
    occluder_x = float(fx.get("geometry_occluder_x", 0.88))
    occluder_half_y = float(fx.get("geometry_occluder_half_y", 0.48))
    divider_x = float(fx.get("geometry_divider_x", 1.58))
    drone_x = float(row["x"])
    drone_y = float(row["y"])

    def rect(cx, cy, hx, hy, **kwargs):
        ax.add_patch(patches.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, **kwargs))

    rect(occluder_x, 0.0, 0.10, occluder_half_y, facecolor="0.28", edgecolor="black", lw=1.0)
    for y in (-0.84, 0.0, 0.84):
        rect(divider_x, y, 0.22, 0.05, facecolor="0.40", edgecolor="black", lw=0.8)

    y_min, y_max = -1.75, 1.75
    wall_half_x = 0.15
    if slit_y - slit_half_y > y_min:
        rect(wall_x, 0.5 * (y_min + slit_y - slit_half_y), wall_half_x,
             0.5 * (slit_y - slit_half_y - y_min), facecolor="0.18", edgecolor="black", lw=1.0)
    if y_max > slit_y + slit_half_y:
        rect(wall_x, 0.5 * (slit_y + slit_half_y + y_max), wall_half_x,
             0.5 * (y_max - slit_y - slit_half_y), facecolor="0.18", edgecolor="black", lw=1.0)

    region_kind = str(fx.get("hazard_region_kind", "opening"))
    if region_kind == "side_wall_patches":
        patch_half_y = float(fx.get("side_effect_half_y", 0.10))
        patch_half_z = float(fx.get("side_effect_half_z", 1.00))
        patch_centers_y = [
            slit_y - slit_half_y - patch_half_y,
            slit_y + slit_half_y + patch_half_y,
        ]
        for patch_y in patch_centers_y:
            rect(wall_x, patch_y, wall_half_x * 1.35, patch_half_y,
                 facecolor=(*color, 0.44), edgecolor=color, lw=2.0)
        ax.text(
            wall_x + 0.10,
            max(patch_centers_y),
            f"side patch z_half={patch_half_z:.2f}",
            color=color,
            fontsize=8,
            va="bottom",
        )
    else:
        core_half_y = float(fx.get("glare_core_half_y", slit_half_y))
        halo_half_y = float(fx.get("glare_halo_half_y", core_half_y))
        halo_strength = float(fx.get("glare_halo_strength", 0.0))
        if halo_strength > 0.0 and halo_half_y > core_half_y:
            rect(wall_x, slit_y, wall_half_x * 1.55, halo_half_y,
                 facecolor=(*color, 0.16 + 0.20 * min(halo_strength, 1.0)),
                 edgecolor=color, lw=1.2, alpha=0.75)
        rect(wall_x, slit_y, wall_half_x * 1.35, core_half_y,
             facecolor=(*color, 0.46), edgecolor=color, lw=2.0)

    ax.scatter([drone_x], [drone_y], c="white", edgecolors="black", s=42, zorder=4)
    ax.plot([drone_x, wall_x], [drone_y, slit_y], color=color, lw=1.2, alpha=0.85)
    ax.scatter([wall_x], [slit_y], marker="x", c=[color], s=50, zorder=5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, wall_x + 0.45)
    ax.set_ylim(-1.55, 1.55)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"top-down {scene} local region")
    ax.grid(True, alpha=0.25)


def _plot_overlay(path: Path, row: dict, maps: dict, fx: dict, args, scene: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = SCENE_COLORS.get(scene, (1.0, 0.2, 0.1))
    depth = np.asarray(maps["depth"], dtype=np.float32)
    mask = maps.get("scene_mask")
    if mask is None:
        mask = np.zeros_like(depth)
    mask = np.asarray(mask, dtype=np.float32)
    effect = maps.get("scene_effect")
    if effect is None:
        effect = np.zeros_like(depth)
    effect = np.asarray(effect, dtype=np.float32)

    depth_show = depth.copy()
    depth_show[depth <= float(args.depth_min_valid) + 1e-6] = np.nan
    depth_cmap = plt.cm.viridis.copy()
    depth_cmap.set_bad("black")

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.2), squeeze=False)
    ax0, ax1, ax2, ax3 = axes[0]
    ax0.imshow(depth_show, vmin=args.depth_min_valid, vmax=args.depth_max_range, cmap=depth_cmap)
    ax0.imshow(_rgba_overlay(mask, color))
    if np.nanmax(mask) > 0.05:
        ax0.contour(mask > 0.05, levels=[0.5], colors=[color], linewidths=1.2)
    ax0.set_title("depth + local material mask")

    ax1.imshow(mask, vmin=0, vmax=1, cmap="cividis")
    ax1.set_title(f"projected mask area={float(mask.mean()):.3f}")

    ax2.imshow(effect, vmin=0, vmax=1, cmap="inferno")
    ax2.set_title(f"sensor effect={float(row['scene_effect_mean']):.3f}")

    _plot_topdown(ax3, fx, row, scene, color)

    for ax in (ax0, ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"{scene} slot={row['slot']} pose={row['pose']} "
        f"pos=({row['x']:.2f},{row['y']:.2f},{row['z']:.2f}) "
        f"p/e/g={row['power']:.2f}/{row['exposure']:.2f}/{row['gain']:.2f}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/slit_active_sensing.args")
    parser.add_argument("--out_dir", default="paper/experiment/results/slit_material_overlays")
    parser.add_argument("--scenarios", nargs="*", default=["specular", "dark"])
    parser.add_argument("--slots", nargs="*", default=["right"])
    parser.add_argument("--xs", default="1.10,1.45")
    parser.add_argument("--path_y_mode", default="slot", choices=["center", "blend", "slot"])
    parser.add_argument("--camera", default=None, help="Optional shared power,exposure,gain override.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    script_args, overrides = parser.parse_known_args()

    if script_args.seed is not None:
        from config import set_global_seed
        set_global_seed(int(script_args.seed))

    project_args = _build_project_args(Path(script_args.config), overrides)
    project_args.random_rotation = False
    device = torch.device(script_args.device)
    scenes = _parse_scenes(script_args.scenarios)
    slots = _parse_slots(script_args.slots)
    xs = _parse_float_list(script_args.xs)
    out_dir = Path(script_args.out_dir)

    count = 0
    with torch.no_grad():
        for scene in scenes:
            if scene not in {"glare", "specular", "dark"}:
                continue
            for slot in slots:
                cond_args = copy.deepcopy(project_args)
                cond_args.scenarios = [scene]
                cond_args.sun_glare_eval_slot = slot
                env = build_env(1, cond_args, device, eval_mode=True)
                env.reset(scene_name=scene)
                target = _opening_target(env)
                setting = _camera_setting_for_scene(scene, script_args.camera)
                poses = _make_poses(env, xs, script_args.path_y_mode)
                for pose in poses:
                    row, maps = _render_condition(env, cond_args, pose, target, setting)
                    fx = dict(env.current_scene_effects or {})
                    path = out_dir / scene / slot / f"{pose.name}_material_overlay.png"
                    _plot_overlay(path, row, maps, fx, cond_args, scene)
                    count += 1

    print(f"[material-overlays] scenes={scenes} slots={slots} xs={xs}")
    print(f"[material-overlays] wrote {count} overlays under {out_dir}")


if __name__ == "__main__":
    main()

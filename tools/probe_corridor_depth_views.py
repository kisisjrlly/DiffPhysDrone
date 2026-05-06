#!/usr/bin/env python3
"""Probe the three-wall corridor layout and camera-dependent depth cues.

This is a geometry sanity check for the active-sensing corridor branch.  It
places the drone at deterministic local poses around every slit wall, points the
camera at the active wall's slit center, renders several camera settings, and
writes panels plus CSV/report summaries.
"""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    build_parser,
    canonicalize_sun_glare_slot,
    parse_diff_sensor_impl,
    parse_scenarios,
    parse_scene_sequence,
    set_global_seed,
    validate_args,
)
from rollout_ops import compute_depth_fill_rate  # noqa: E402
from train_utils import build_env  # noqa: E402


@dataclass(frozen=True)
class CameraSetting:
    name: str
    power: float
    exposure: float
    gain: float


DEFAULT_SETTINGS = (
    CameraSetting("fixed_mid", 0.50, 0.50, 0.50),
    CameraSetting("glare_expected", 0.92, 0.20, 0.05),
    CameraSetting("specular_safe", 0.35, 0.35, 0.20),
    CameraSetting("dark_expected", 0.55, 0.78, 0.60),
    CameraSetting("bad_low_light", 0.35, 0.18, 0.03),
    CameraSetting("bad_high_power", 0.95, 0.55, 0.45),
    CameraSetting("bad_overexposed", 0.55, 0.85, 0.70),
)


def _read_args_file(path: Path) -> list[str]:
    tokens: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            tokens.extend(shlex.split(line))
    return tokens


def _build_project_args(config_path: Path, overrides: list[str]):
    parser = build_parser()
    args = parser.parse_args(_read_args_file(config_path) + list(overrides))
    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.scenarios = parse_scenarios(args.scenarios)
    args.corridor_scene_sequence = parse_scene_sequence(args.corridor_scene_sequence)
    args.sun_glare_eval_slot = canonicalize_sun_glare_slot(args.sun_glare_eval_slot)
    args.batch_size = 1
    args.wandb_disabled = True
    args.vis_enable = False
    validate_args(args)
    return args


def _parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).replace(";", ",").split(",") if x.strip()]
    if not vals:
        raise ValueError("empty float list")
    return vals


def _parse_camera_settings(text: str | None) -> list[CameraSetting]:
    if not text:
        return list(DEFAULT_SETTINGS)
    out: list[CameraSetting] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        name, sep, values = item.partition(":")
        if not sep:
            raise ValueError(f"camera setting must be name:p,e,g, got {item!r}")
        parts = [float(x.strip()) for x in values.split(",") if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"camera setting needs three values: {item!r}")
        out.append(CameraSetting(name.strip(), parts[0], parts[1], parts[2]))
    return out or list(DEFAULT_SETTINGS)


def _safe_camera_frame(forward: torch.Tensor) -> torch.Tensor:
    forward = F.normalize(forward, dim=0)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=forward.device, dtype=forward.dtype)
    left = torch.cross(world_up, forward, dim=0)
    if float(left.norm().detach().cpu()) < 1e-5:
        alt_up = torch.tensor([0.0, 1.0, 0.0], device=forward.device, dtype=forward.dtype)
        left = torch.cross(alt_up, forward, dim=0)
    left = F.normalize(left, dim=0)
    up = F.normalize(torch.cross(forward, left, dim=0), dim=0)
    return torch.stack([forward, left, up], dim=-1)


def _set_local_pose_look_at(env, local_pos: torch.Tensor, local_target: torch.Tensor):
    device = env.device
    world_pos = torch.matmul(env.R_scene[0].to(device=device, dtype=torch.float32), local_pos)
    world_target = torch.matmul(env.R_scene[0].to(device=device, dtype=torch.float32), local_target)
    forward = world_target - world_pos
    if float(forward.norm().detach().cpu()) < 1e-4:
        forward = torch.tensor([1.0, 0.0, 0.0], device=device)
    R_cam_world = _safe_camera_frame(forward).unsqueeze(0).expand(env.batch_size, -1, -1)
    R_body = torch.matmul(R_cam_world, env.R_cam.transpose(1, 2))
    env.p = world_pos.unsqueeze(0).repeat(env.batch_size, 1).clone()
    env.p_old = env.p.clone()
    env.v.zero_()
    env.a.zero_()
    env.act.zero_()
    env.R = R_body.clone()
    env.R_old = env.R.clone()
    env.p_target = world_target.unsqueeze(0).repeat(env.batch_size, 1).clone()
    env._update_corridor_active_stage()


def _render(env, args, setting: CameraSetting):
    device = env.device
    p = torch.full((env.batch_size,), float(setting.power), device=device)
    e = torch.full((env.batch_size,), float(setting.exposure), device=device)
    g = torch.full((env.batch_size,), float(setting.gain), device=device)
    depth, quality = env.render_diff_depth(p, e, g)
    debug = env.export_last_diff_depth_debug(0)
    images = debug.get("images", {})
    scalars = debug.get("scalars", {})
    depth_np = depth[0].detach().cpu().numpy()
    quality_np = None if quality is None else quality[0].detach().cpu().numpy()
    fill = float(compute_depth_fill_rate(depth, min_valid_depth=args.depth_min_valid).reshape(-1)[0].detach().cpu())
    valid = depth_np > (float(args.depth_min_valid) + 1e-6)
    scene_mask = images.get("scene_mask")
    if scene_mask is not None and np.asarray(scene_mask).shape == depth_np.shape:
        local_mask = np.asarray(scene_mask, dtype=np.float32) > 0.05
    else:
        local_mask = np.ones_like(valid, dtype=bool)
    if not np.any(local_mask):
        local_mask = np.ones_like(valid, dtype=bool)
    return {
        "maps": {
            "raw_depth": images.get("raw_depth_map"),
            "depth": depth_np,
            "quality": quality_np,
            "invalid": images.get("invalid_mask"),
            "effect": images.get("scene_effect_map"),
            "scene_mask": scene_mask,
            "slit_cue": images.get("slit_cue_mask"),
            "key_cue": images.get("key_cue_artifact_map"),
        },
        "metrics": {
            "fill_rate": fill,
            "local_fill": float((valid & local_mask).sum() / max(int(local_mask.sum()), 1)),
            "quality_mean": float(np.mean(quality_np)) if quality_np is not None else float("nan"),
            "local_quality_mean": float(np.asarray(quality_np)[local_mask].mean()) if quality_np is not None else float("nan"),
            "invalid_rate": _scalar(scalars.get("invalid_rate"), 0.0),
            "scene_effect_mean": _scalar(scalars.get("scene_effect_mean"), 0.0),
            "scene_mask_mean": _scalar(scalars.get("scene_mask_mean"), 0.0),
            "slit_cue_mask_mean": _scalar(scalars.get("slit_cue_mask_mean"), 0.0),
            "key_cue_artifact_mean": _scalar(scalars.get("key_cue_artifact_mean"), 0.0),
            "glare_invalid_rate": _scalar(scalars.get("glare_invalid_rate"), 0.0),
            "glare_quality_mean": _scalar(scalars.get("glare_quality_mean"), 0.0),
        },
    }


def _scalar(value, default=0.0) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        return float(value.reshape(-1)[0].detach().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _stage_info(env, stage_idx: int) -> dict:
    old = env._corridor_stage_idx.clone()
    env._corridor_stage_idx.fill_(int(stage_idx))
    fx = env._select_batch_effects(env._corridor_stage_idx)
    env._corridor_stage_idx = old
    return env.get_scene_effects_for_env(0) if stage_idx == int(old[0]) else _effects_for_env(fx, env, 0)


def _effects_for_env(effects: dict, env, idx: int) -> dict:
    out = {}
    for key, value in effects.items():
        if torch.is_tensor(value):
            v = value.detach()
            if v.ndim >= 2 and v.shape[0] == env.batch_size:
                v = v[idx]
            elif v.ndim == 1 and v.shape[0] == env.batch_size:
                v = v[idx]
            out[key] = float(v.cpu().item()) if v.ndim == 0 else v.cpu().tolist()
        elif isinstance(value, list) and len(value) == env.batch_size:
            out[key] = value[idx]
        else:
            out[key] = value
    return out


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_panel(path: Path, rows_maps: list[tuple[dict, dict]], args, wall_xs: list[float]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    n = len(rows_maps)
    fig, axes = plt.subplots(n, 7, figsize=(22, max(2.2 * n, 3.0)), squeeze=False)
    depth_cmap = plt.cm.viridis.copy()
    depth_cmap.set_bad("black")
    first = rows_maps[0][0]

    for r, (row, maps) in enumerate(rows_maps):
        depth = maps["depth"].astype(np.float32)
        raw = maps.get("raw_depth")
        raw_show = None if raw is None else raw.astype(np.float32).copy()
        depth_show = depth.copy()
        if raw_show is not None:
            raw_show[raw_show <= float(args.depth_min_valid) + 1e-6] = np.nan
        depth_show[depth <= float(args.depth_min_valid) + 1e-6] = np.nan
        imgs = [
            (np.zeros_like(depth) if raw_show is None else raw_show, "raw", depth_cmap, args.depth_min_valid, args.depth_max_range),
            (depth_show, "depth", depth_cmap, args.depth_min_valid, args.depth_max_range),
            (np.zeros_like(depth) if maps["quality"] is None else maps["quality"], "quality", "magma", 0, 1),
            (np.zeros_like(depth) if maps["invalid"] is None else maps["invalid"], "invalid", "gray", 0, 1),
            (np.zeros_like(depth) if maps["effect"] is None else maps["effect"], "effect", "inferno", 0, 1),
            (np.zeros_like(depth) if maps["slit_cue"] is None else maps["slit_cue"], "slit cue", "cividis", 0, 1),
        ]
        for c, (img, title, cmap, vmin, vmax) in enumerate(imgs):
            axes[r, c].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[r, c].set_title(title)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        ax = axes[r, 6]
        for wx in wall_xs:
            ax.add_patch(patches.Rectangle((wx - 0.05, -1.45), 0.10, 2.90, facecolor="0.80", edgecolor="0.25", lw=0.8))
        ax.scatter([row["local_x"]], [row["local_y"]], s=48, c="white", edgecolors="black", zorder=3)
        ax.scatter([row["wall_x"]], [row["slit_y"]], marker="x", s=58, c="red", zorder=4)
        ax.plot([row["local_x"], row["wall_x"]], [row["local_y"], row["slit_y"]], color="#2b6cb0", lw=1.2)
        ax.set_xlim(min(-1.8, row["local_x"] - 0.4), max(wall_xs) + 0.65)
        ax.set_ylim(-1.55, 1.55)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title("local topdown")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        axes[r, 0].set_ylabel(
            f"{row['setting']}\n"
            f"p/e/g={row['power']:.2f}/{row['exposure']:.2f}/{row['gain']:.2f}\n"
            f"fill={row['local_fill']:.2f} q={row['local_quality_mean']:.2f}"
        )

    fig.suptitle(
        f"stage={first['stage']} scene={first['scene']} pose={first['pose']} "
        f"wall_x={first['wall_x']:.2f} slit_y={first['slit_y']:+.2f}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _analyze_expectations(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["stage"]), str(row["pose"])), []).append(row)
    out = []
    for (stage, pose), items in sorted(grouped.items()):
        by = {r["setting"]: r for r in items}
        scene = str(items[0]["scene"])
        record = {
            "stage": stage,
            "scene": scene,
            "pose": pose,
            "passed": 0.0,
            "delta": 0.0,
            "expectation": "",
        }
        if scene == "dark" and "dark_expected" in by and "bad_low_light" in by:
            good, bad = by["dark_expected"], by["bad_low_light"]
            delta = (float(good["local_fill"]) - float(bad["local_fill"])) + (
                float(good["local_quality_mean"]) - float(bad["local_quality_mean"]))
            record.update(expectation="dark_expected > bad_low_light", delta=delta, passed=1.0 if delta > 0.02 else 0.0)
        elif scene == "specular" and "specular_safe" in by and "bad_high_power" in by:
            good, bad = by["specular_safe"], by["bad_high_power"]
            delta = (float(good["local_fill"]) - float(bad["local_fill"])) + (
                float(good["local_quality_mean"]) - float(bad["local_quality_mean"]))
            record.update(expectation="specular_safe > bad_high_power", delta=delta, passed=1.0 if delta > 0.02 else 0.0)
        elif scene == "glare" and "glare_expected" in by and "bad_overexposed" in by:
            good, bad = by["glare_expected"], by["bad_overexposed"]
            delta = (float(good["local_fill"]) - float(bad["local_fill"])) + (
                float(good["local_quality_mean"]) - float(bad["local_quality_mean"])) + (
                float(bad["glare_invalid_rate"]) - float(good["glare_invalid_rate"]))
            record.update(expectation="glare_expected > bad_overexposed", delta=delta, passed=1.0 if delta > 0.02 else 0.0)
        out.append(record)
    return out


def _write_report(path: Path, args, rows: list[dict], checks: list[dict], wall_xs: list[float]):
    spacings = [wall_xs[i + 1] - wall_xs[i] for i in range(len(wall_xs) - 1)]
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["stage"]), str(row["pose"])), []).append(row)
    ranges = []
    for items in grouped.values():
        fills = [float(r["local_fill"]) for r in items]
        qualities = [float(r["local_quality_mean"]) for r in items]
        ranges.append((max(fills) - min(fills), max(qualities) - min(qualities)))
    passed = [c for c in checks if float(c.get("passed", 0.0)) > 0.5]
    lines = [
        "# Corridor Depth View Probe",
        "",
        f"- scene_layout: `{args.scene_layout}`",
        f"- sequence: `{'>'.join(args.corridor_scene_sequence)}`",
        f"- wall_xs: `{wall_xs}`",
        f"- wall spacings: `{spacings}` m",
        f"- start_x / goal_x: `{args.simple_start_x}` / `{args.simple_goal_x}`",
        f"- rendered rows: {len(rows)}",
        f"- probed states: {len(grouped)}",
        f"- expectation pass rate: {len(passed) / max(len(checks), 1):.3f} ({len(passed)}/{len(checks)})",
        f"- mean camera-setting local fill range: {np.mean([r[0] for r in ranges]) if ranges else 0.0:.4f}",
        f"- mean camera-setting local quality range: {np.mean([r[1] for r in ranges]) if ranges else 0.0:.4f}",
        "",
        "Spacing note:",
        "",
        f"- At max_speed `{getattr(args, 'fixed_max_speed', 'env_default')}`, 1.2 m corresponds to roughly 1.0 s if the drone flies near 1.15 m/s.",
        "- With 15 Hz control this is about 15 control steps between walls. That is usable for a toy benchmark, but short for learning a clean observe-adjust-recover pattern.",
        "- If the desired plot is clear camera recovery between walls, 1.6-2.0 m spacing is a safer next setting.",
        "",
        "Files:",
        "",
        "- `corridor_depth_probe_detail.csv`",
        "- `corridor_depth_probe_expectations.csv`",
        "- `panels/stage_*/...png`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/slit_active_sensing.args")
    parser.add_argument("--out_dir", default="paper/experiment/results/corridor_depth_views")
    parser.add_argument("--pose_offsets", default="-0.90,-0.55,-0.25,-0.08,0.18,0.42",
                        help="Local x offsets relative to each wall.")
    parser.add_argument("--camera_settings", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sensor_impl", default="cuda", choices=["cuda", "python"])
    parser.add_argument("--keep_random_rotation", action="store_true")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_panels", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    script_args, project_overrides = parser.parse_known_args()

    project_args = _build_project_args(Path(script_args.config), project_overrides)
    project_args.diff_sensor_impl["diff_depth"] = str(script_args.sensor_impl)
    if not script_args.keep_random_rotation:
        project_args.random_rotation = False
    if script_args.seed is not None:
        project_args.seed = int(script_args.seed)
    set_global_seed(project_args.seed, project_args.deterministic)

    if project_args.scene_layout != "three_wall_corridor":
        raise ValueError("probe_corridor_depth_views.py expects --scene_layout three_wall_corridor")
    device = torch.device(script_args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    env = build_env(1, project_args, device, eval_mode=True)
    env.reset(scene_name=None)
    wall_xs = [float(x) for x in project_args.corridor_wall_xs]
    offsets = _parse_float_list(script_args.pose_offsets)
    settings = _parse_camera_settings(script_args.camera_settings)
    out_dir = Path(script_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    panel_count = 0
    for stage, wall_x in enumerate(wall_xs):
        fx = _stage_info(env, stage)
        scene = str(fx.get("sensor_regime_name", "unknown"))
        slit_y = float(fx.get("slit_center_y", 0.0))
        slit_z = float(fx.get("slit_center_z", project_args.simple_slit_center_z))
        target = torch.tensor([wall_x, slit_y, slit_z], device=device, dtype=torch.float32)
        for offset in offsets:
            local_x = wall_x + float(offset)
            # before wall: align with current slit. after wall: keep same y so
            # top-down view remains readable.
            local_pos = torch.tensor([local_x, slit_y, slit_z], device=device, dtype=torch.float32)
            _set_local_pose_look_at(env, local_pos, target)
            rendered = []
            for setting in settings:
                result = _render(env, project_args, setting)
                fx_now = env.get_scene_effects_for_env(0)
                row = {
                    "stage": int(stage),
                    "scene": str(fx_now.get("sensor_regime_name", scene)),
                    "pose": f"dx{offset:+.2f}",
                    "wall_x": float(fx_now.get("geometry_wall_x", wall_x)),
                    "slit_y": float(fx_now.get("slit_center_y", slit_y)),
                    "slit_half_y": float(fx_now.get("slit_half_y", project_args.simple_slit_half_y)),
                    "local_x": float(local_x),
                    "local_y": float(slit_y),
                    "setting": setting.name,
                    "power": setting.power,
                    "exposure": setting.exposure,
                    "gain": setting.gain,
                }
                row.update(result["metrics"])
                rows.append(row)
                rendered.append((row, result["maps"]))
            if script_args.plots and (script_args.max_panels <= 0 or panel_count < script_args.max_panels):
                panel_path = out_dir / "panels" / f"stage_{stage}_{scene}" / f"dx_{offset:+.2f}.png"
                _plot_panel(panel_path, rendered, project_args, wall_xs)
                panel_count += 1

    checks = _analyze_expectations(rows)
    _write_csv(out_dir / "corridor_depth_probe_detail.csv", rows)
    _write_csv(out_dir / "corridor_depth_probe_expectations.csv", checks)
    _write_report(out_dir / "report.md", project_args, rows, checks, wall_xs)
    print(f"[corridor-probe] wrote {out_dir}")
    print(f"[corridor-probe] rows={len(rows)} checks={len(checks)} panels={panel_count}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Render depth observations while the drone explicitly looks at the gate opening.

This probe is meant for sanity-checking the simplified shared gate benchmark:

- scenarios: glare / specular / dark
- slots: far_left / left / right / far_right
- fixed drone positions along the approach path
- fixed camera settings at each identical pose

For every condition and pose, the script forces the camera optical axis to look
at the current opening center, then writes depth/quality/invalid/effect panels
plus CSV summaries.  This isolates whether the simulated depth image changes in
the expected way when camera parameters or sensor scenes change.
"""

from __future__ import annotations

import argparse
import copy
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
    set_global_seed,
    validate_args,
)
from rollout_ops import (  # noqa: E402
    compute_depth_fill_rate,
)
from train_utils import build_env  # noqa: E402


SLOT_ORDER = ("far_left", "left", "right", "far_right")
SCENE_ORDER = ("glare", "specular", "dark")


@dataclass(frozen=True)
class CameraSetting:
    name: str
    power: float
    exposure: float
    gain: float


@dataclass(frozen=True)
class ProbePose:
    name: str
    x: float
    y: float
    z: float


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
    if getattr(args, "sun_glare_eval_slot", None) is not None:
        args.sun_glare_eval_slot = canonicalize_sun_glare_slot(args.sun_glare_eval_slot)
    args.batch_size = 1
    args.wandb_disabled = True
    args.vis_enable = False
    validate_args(args)
    return args


def _parse_csv_tokens(text: str | None) -> list[str]:
    if text is None:
        return []
    out: list[str] = []
    for raw in str(text).replace(";", ",").split(","):
        token = raw.strip()
        if token:
            out.append(token)
    return out


def _parse_scenes(items: list[str]) -> list[str]:
    raw = items or list(SCENE_ORDER)
    scenes = parse_scenarios(raw)
    return [s for s in SCENE_ORDER if s in scenes] + [s for s in scenes if s not in SCENE_ORDER]


def _parse_slots(items: list[str]) -> list[str]:
    raw = items or list(SLOT_ORDER)
    slots = []
    for item in raw:
        for token in _parse_csv_tokens(item):
            slot = canonicalize_sun_glare_slot(token)
            if slot not in SLOT_ORDER:
                raise ValueError(f"unsupported slot: {token}")
            if slot not in slots:
                slots.append(slot)
    return slots or list(SLOT_ORDER)


def _parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty float list")
    return vals


def _parse_camera_settings(text: str | None, args) -> list[CameraSetting]:
    if not text:
        return [
            CameraSetting("baseline", float(args.cam_power_baseline), 0.45, 0.20),
            CameraSetting("fixed_mid", 0.75, 0.45, 0.45),
            CameraSetting("glare_expected", 0.92, 0.20, 0.05),
            CameraSetting("overexposed", 0.55, 0.85, 0.70),
            CameraSetting("specular_safe", 0.35, 0.35, 0.20),
            CameraSetting("high_power", 0.95, 0.45, 0.30),
            CameraSetting("dark_expected", 0.55, 0.78, 0.60),
            CameraSetting("low_light_bad", 0.45, 0.18, 0.03),
        ]

    settings: list[CameraSetting] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        name, sep, values = item.partition(":")
        if not sep:
            raise ValueError(f"--camera_settings entry must be name:p,e,g, got {item!r}")
        parts = [float(x.strip()) for x in values.split(",") if x.strip()]
        if len(parts) != 3:
            raise ValueError(f"--camera_settings entry must contain three values: {item!r}")
        settings.append(CameraSetting(name.strip(), parts[0], parts[1], parts[2]))
    if not settings:
        raise ValueError("--camera_settings produced no settings")
    return settings


def _opening_target(env) -> torch.Tensor:
    fx = env.current_scene_effects or {}
    center = fx.get("hazard_center", None)
    if torch.is_tensor(center):
        return center[0].detach().to(device=env.device, dtype=torch.float32)
    if center is not None:
        return torch.tensor(center, device=env.device, dtype=torch.float32).reshape(-1, 3)[0]
    gate_x = float(fx.get("geometry_gate_x", 1.82))
    slot_y = float(fx.get("decision_open_slot_y", 0.0))
    return torch.tensor([gate_x, slot_y, 1.50], device=env.device, dtype=torch.float32)


def _make_poses(env, xs: list[float], y_mode: str) -> list[ProbePose]:
    fx = env.current_scene_effects or {}
    start_y = float(fx.get("geometry_start_y", 0.0))
    slot_y = float(fx.get("decision_open_slot_y", 0.0))
    gate_x = float(fx.get("geometry_gate_x", 1.82))
    x_min = min(xs)
    denom = max(gate_x - x_min, 1e-6)
    R_scene = getattr(env, "R_scene", None)
    R0 = None if R_scene is None else R_scene[0].detach()
    poses: list[ProbePose] = []
    for idx, x in enumerate(xs):
        if y_mode == "slot":
            y = slot_y
        elif y_mode == "blend":
            alpha = float(np.clip((float(x) - x_min) / denom, 0.0, 1.0))
            y = (1.0 - alpha) * start_y + alpha * slot_y
        else:
            y = 0.0
        local = torch.tensor([float(x), float(y), 1.50], device=env.device, dtype=torch.float32)
        world = local if R0 is None else torch.matmul(R0, local)
        wx, wy, wz = [float(v) for v in world.detach().cpu().tolist()]
        poses.append(ProbePose(f"x{idx:02d}_{float(x):+.2f}", wx, wy, wz))
    return poses


def _safe_camera_frame(forward: torch.Tensor) -> torch.Tensor:
    forward = F.normalize(forward, dim=0)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=forward.device, dtype=forward.dtype)
    left = torch.cross(world_up, forward, dim=0)
    if float(left.norm().detach().cpu().item()) < 1e-5:
        alt_up = torch.tensor([0.0, 1.0, 0.0], device=forward.device, dtype=forward.dtype)
        left = torch.cross(alt_up, forward, dim=0)
    left = F.normalize(left, dim=0)
    up = F.normalize(torch.cross(forward, left, dim=0), dim=0)
    return torch.stack([forward, left, up], dim=-1)


def _set_pose_look_at(env, pose: ProbePose, target: torch.Tensor):
    device = env.device
    pos = torch.tensor([pose.x, pose.y, pose.z], device=device, dtype=torch.float32)
    target = target.to(device=device, dtype=torch.float32)
    forward = target - pos
    if float(forward.norm().detach().cpu().item()) < 1e-4:
        forward = torch.tensor([1.0, 0.0, 0.0], device=device)
    R_cam_world = _safe_camera_frame(forward).unsqueeze(0).expand(env.batch_size, -1, -1)
    R_body = torch.matmul(R_cam_world, env.R_cam.transpose(1, 2))

    env.p = pos.unsqueeze(0).repeat(env.batch_size, 1).clone()
    env.p_old = env.p.clone()
    env.v.zero_()
    env.a.zero_()
    env.act.zero_()
    env.R = R_body.clone()
    env.R_old = env.R.clone()
    env.p_target = target.unsqueeze(0).repeat(env.batch_size, 1).clone()


def _to_float(value, default: float = float("nan")) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return float(value.reshape(-1)[0].detach().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _local_mask_metrics(depth_np: np.ndarray, quality_np: np.ndarray | None,
                        mask_np: np.ndarray | None, min_valid: float) -> dict[str, float]:
    valid = depth_np > (float(min_valid) + 1e-6)
    if mask_np is not None and np.asarray(mask_np).size == depth_np.size:
        mask = np.asarray(mask_np, dtype=np.float32) > 0.05
    else:
        h, w = depth_np.shape
        mask = np.zeros_like(valid, dtype=bool)
        r0, r1 = int(0.35 * h), int(0.65 * h)
        c0, c1 = int(0.35 * w), int(0.65 * w)
        mask[r0:r1, c0:c1] = True
    if not np.any(mask):
        mask = np.ones_like(valid, dtype=bool)

    local_valid = valid & mask
    local_depth = depth_np[local_valid]
    out = {
        "local_mask_area": float(mask.mean()),
        "local_fill": float(local_valid.sum() / max(mask.sum(), 1)),
        "local_depth_mean": float(local_depth.mean()) if local_depth.size else 0.0,
        "local_depth_std": float(local_depth.std()) if local_depth.size else 0.0,
    }
    if quality_np is not None:
        out["local_quality_mean"] = float(np.asarray(quality_np)[mask].mean())
    else:
        out["local_quality_mean"] = float("nan")
    return out


def _render_condition(env, args, pose: ProbePose, target: torch.Tensor,
                      setting: CameraSetting) -> tuple[dict, dict[str, np.ndarray | None]]:
    _set_pose_look_at(env, pose, target)
    device = env.device
    power = torch.full((env.batch_size,), float(setting.power), device=device)
    exposure = torch.full((env.batch_size,), float(setting.exposure), device=device)
    gain = torch.full((env.batch_size,), float(setting.gain), device=device)

    depth, quality = env.render_diff_depth(power, exposure, gain)
    fill = compute_depth_fill_rate(depth, min_valid_depth=args.depth_min_valid)
    fill_soft = compute_depth_fill_rate(depth, min_valid_depth=args.depth_min_valid, softness=0.08)
    debug = env.export_last_diff_depth_debug(0)
    scalars = debug.get("scalars", {})
    images = debug.get("images", {})

    depth_np = depth[0].detach().cpu().numpy()
    quality_np = None if quality is None else quality[0].detach().cpu().numpy()
    invalid_np = images.get("invalid_mask")
    effect_np = images.get("scene_effect_map")
    scene_mask_np = images.get("scene_mask")
    valid = depth_np > (float(args.depth_min_valid) + 1e-6)
    valid_depth = depth_np[valid]

    row = {
        "scene": env.current_scene_name,
        "scene_tag": getattr(env, "current_scene_tag", env.current_scene_name),
        "slot": str((env.current_scene_effects or {}).get("decision_open_slot_name", "")),
        "opening_y": float((env.current_scene_effects or {}).get("decision_open_slot_y", 0.0)),
        "gate_x": float((env.current_scene_effects or {}).get("geometry_gate_x", 0.0)),
        "pose": pose.name,
        "x": pose.x,
        "y": pose.y,
        "z": pose.z,
        "look_target_x": float(target[0].detach().cpu().item()),
        "look_target_y": float(target[1].detach().cpu().item()),
        "look_target_z": float(target[2].detach().cpu().item()),
        "look_distance": float(torch.norm(target - torch.tensor([pose.x, pose.y, pose.z], device=target.device)).detach().cpu().item()),
        "setting": setting.name,
        "power": float(setting.power),
        "exposure": float(setting.exposure),
        "gain": float(setting.gain),
        "fill_rate": float(fill.reshape(-1)[0].detach().cpu().item()),
        "fill_rate_soft": float(fill_soft.reshape(-1)[0].detach().cpu().item()),
        "valid_depth_mean": float(valid_depth.mean()) if valid_depth.size else 0.0,
        "valid_depth_std": float(valid_depth.std()) if valid_depth.size else 0.0,
        "quality_mean": float(np.mean(quality_np)) if quality_np is not None else float("nan"),
        "invalid_rate": _to_float(scalars.get("invalid_rate"), 0.0),
        "scene_effect_mean": _to_float(scalars.get("scene_effect_mean"), 0.0),
        "sun_mask_mean": _to_float(scalars.get("sun_mask_mean"), 0.0),
        "hazard_mask_mean": _to_float(scalars.get("hazard_mask_mean"), _to_float(scalars.get("scene_mask_mean"), 0.0)),
        "sun_los_mean": _to_float(scalars.get("sun_los_mean"), 0.0),
        "hazard_los_mean": _to_float(scalars.get("hazard_los_mean"), 0.0),
        "glare_quality_mean": _to_float(scalars.get("glare_quality_mean"), 0.0),
        "glare_invalid_rate": _to_float(scalars.get("glare_invalid_rate"), 0.0),
        "washout_mean": _to_float(scalars.get("washout_mean"), 0.0),
        "ambient_ir_mean": _to_float(scalars.get("ambient_ir_mean"), 0.0),
        "signal_active_mean": _to_float(scalars.get("signal_active_mean"), 0.0),
        "signal_passive_mean": _to_float(scalars.get("signal_passive_mean"), 0.0),
        "spec_bloom_mean": _to_float(scalars.get("spec_bloom_mean"), 0.0),
        "sensor_regime_id": _to_float(scalars.get("sensor_regime_id"), -1.0),
    }
    row.update(_local_mask_metrics(depth_np, quality_np, scene_mask_np, args.depth_min_valid))

    maps = {
        "depth": depth_np,
        "quality": quality_np,
        "invalid": invalid_np,
        "scene_effect": effect_np,
        "scene_mask": scene_mask_np,
    }
    return row, maps


def _add_reference_diffs(rendered: list[tuple[dict, dict[str, np.ndarray | None]]],
                         min_valid: float):
    ref_row, ref_maps = rendered[0]
    ref_depth = ref_maps["depth"]
    ref_quality = ref_maps["quality"]
    ref_valid = ref_depth > (float(min_valid) + 1e-6)
    ref_mask = ref_maps.get("scene_mask")
    if ref_mask is not None and np.asarray(ref_mask).size == ref_depth.size:
        local_mask = np.asarray(ref_mask, dtype=np.float32) > 0.05
    else:
        local_mask = np.ones_like(ref_valid, dtype=bool)
    if not np.any(local_mask):
        local_mask = np.ones_like(ref_valid, dtype=bool)

    for row, maps in rendered:
        depth = maps["depth"]
        valid = depth > (float(min_valid) + 1e-6)
        union = np.logical_or(ref_valid, valid)
        abs_diff = np.abs(depth - ref_depth)
        row["reference_setting"] = ref_row["setting"]
        row["depth_mae_vs_ref"] = float(abs_diff[union].mean()) if np.any(union) else 0.0
        row["valid_xor_rate_vs_ref"] = float(np.logical_xor(ref_valid, valid).mean())
        row["local_depth_mae_vs_ref"] = float(abs_diff[local_mask].mean()) if np.any(local_mask) else 0.0
        row["local_valid_xor_vs_ref"] = float(np.logical_xor(ref_valid, valid)[local_mask].mean()) if np.any(local_mask) else 0.0
        quality = maps["quality"]
        if quality is not None and ref_quality is not None:
            row["quality_mae_vs_ref"] = float(np.abs(quality - ref_quality).mean())
            row["local_quality_mae_vs_ref"] = float(np.abs(quality - ref_quality)[local_mask].mean()) if np.any(local_mask) else 0.0
        else:
            row["quality_mae_vs_ref"] = float("nan")
            row["local_quality_mae_vs_ref"] = float("nan")


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


def _plot_panel(path: Path, rendered: list[tuple[dict, dict[str, np.ndarray | None]]], args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[probe][warn] matplotlib unavailable, skip plots: {exc}")
        return

    n = len(rendered)
    fig, axes = plt.subplots(n, 5, figsize=(17, max(2.25 * n, 3.0)), squeeze=False)
    depth_cmap = plt.cm.viridis.copy()
    depth_cmap.set_bad("black")
    first = rendered[0][0]
    for r, (row, maps) in enumerate(rendered):
        depth = maps["depth"].astype(np.float32)
        depth_show = depth.copy()
        depth_show[depth <= float(args.depth_min_valid) + 1e-6] = np.nan
        quality = maps["quality"]
        invalid = maps["invalid"]
        effect = maps["scene_effect"]
        mask = maps["scene_mask"]

        axes[r, 0].imshow(depth_show, vmin=args.depth_min_valid, vmax=args.depth_max_range, cmap=depth_cmap)
        axes[r, 0].set_title(f"{row['setting']} depth")
        axes[r, 1].imshow(np.zeros_like(depth) if quality is None else quality, vmin=0, vmax=1, cmap="magma")
        axes[r, 1].set_title(f"quality {row['quality_mean']:.2f}")
        axes[r, 2].imshow(np.zeros_like(depth) if invalid is None else invalid, vmin=0, vmax=1, cmap="gray")
        axes[r, 2].set_title(f"invalid {row['invalid_rate']:.2f}")
        axes[r, 3].imshow(np.zeros_like(depth) if effect is None else effect, vmin=0, vmax=1, cmap="inferno")
        axes[r, 3].set_title(f"effect {row['scene_effect_mean']:.2f}")
        axes[r, 4].imshow(np.zeros_like(depth) if mask is None else mask, vmin=0, vmax=1, cmap="cividis")
        axes[r, 4].set_title(f"local fill {row['local_fill']:.2f}")
        for c in range(5):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(
            f"p/e/g={row['power']:.2f}/{row['exposure']:.2f}/{row['gain']:.2f}\n"
            f"fill={row['fill_rate']:.2f}"
        )

    fig.suptitle(
        f"{first['scene']} {first['slot']} {first['pose']} "
        f"pos=({first['x']:.2f},{first['y']:.2f},{first['z']:.2f}) "
        f"look=({first['look_target_x']:.2f},{first['look_target_y']:.2f},{first['look_target_z']:.2f})"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _expectation_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (str(row["scene"]), str(row["slot"]), str(row["pose"]))
        grouped.setdefault(key, []).append(row)

    out: list[dict] = []
    for (scene, slot, pose), items in sorted(grouped.items()):
        by_name = {str(r["setting"]): r for r in items}
        visible = max(
            max(float(r.get("scene_effect_mean", 0.0)) for r in items),
            max(float(r.get("hazard_mask_mean", 0.0)) for r in items),
            max(float(r.get("sun_mask_mean", 0.0)) for r in items),
        ) > 0.005
        record = {
            "scene": scene,
            "slot": slot,
            "pose": pose,
            "checked": 0.0,
            "passed": 0.0,
            "expectation": "",
            "metric_delta": 0.0,
        }
        if not visible:
            record["expectation"] = "skipped: local sensor region not visible at this pose"
            out.append(record)
            continue
        if scene == "glare" and "glare_expected" in by_name and "overexposed" in by_name:
            good = by_name["glare_expected"]
            bad = by_name["overexposed"]
            delta = (
                float(good["local_fill"]) - float(bad["local_fill"])
                + float(good["local_quality_mean"]) - float(bad["local_quality_mean"])
                + float(bad["glare_invalid_rate"]) - float(good["glare_invalid_rate"])
            )
            record.update({
                "checked": 1.0,
                "passed": 1.0 if delta > 0.02 else 0.0,
                "expectation": "glare_expected should beat overexposed near opening",
                "metric_delta": float(delta),
            })
        elif scene == "specular" and "specular_safe" in by_name and "high_power" in by_name:
            good = by_name["specular_safe"]
            bad = by_name["high_power"]
            delta = (
                float(good["local_fill"]) - float(bad["local_fill"])
                + float(good["local_quality_mean"]) - float(bad["local_quality_mean"])
                + float(bad["local_valid_xor_vs_ref"]) * 0.0
            )
            record.update({
                "checked": 1.0,
                "passed": 1.0 if delta > 0.01 else 0.0,
                "expectation": "specular_safe should avoid high-power washout",
                "metric_delta": float(delta),
            })
        elif scene == "dark" and "dark_expected" in by_name and "low_light_bad" in by_name:
            good = by_name["dark_expected"]
            bad = by_name["low_light_bad"]
            delta = (
                float(good["local_fill"]) - float(bad["local_fill"])
                + float(good["local_quality_mean"]) - float(bad["local_quality_mean"])
            )
            record.update({
                "checked": 1.0,
                "passed": 1.0 if delta > 0.01 else 0.0,
                "expectation": "dark_expected should beat low exposure/gain",
                "metric_delta": float(delta),
            })
        out.append(record)
    return out


def _write_report(path: Path, detail_rows: list[dict], expectation_rows: list[dict]):
    checked = [r for r in expectation_rows if float(r.get("checked", 0.0)) > 0.5]
    passed = [r for r in checked if float(r.get("passed", 0.0)) > 0.5]
    fill_ranges = []
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in detail_rows:
        grouped.setdefault((row["scene"], row["slot"], row["pose"]), []).append(row)
    for items in grouped.values():
        vals = [float(r["local_fill"]) for r in items]
        fill_ranges.append(max(vals) - min(vals))

    lines = [
        "# Opening Depth View Probe",
        "",
        f"- rendered rows: {len(detail_rows)}",
        f"- probed states: {len(grouped)}",
        f"- expectation checks: {len(checked)}",
        f"- expectation pass rate: {len(passed) / max(len(checked), 1):.3f}",
        f"- mean local fill range across camera settings: {sum(fill_ranges) / max(len(fill_ranges), 1):.4f}",
        "",
        "Interpretation:",
        "",
        "- `glare`: `glare_expected` should generally show better local quality/fill than `overexposed` when the glare/hazard region is visible.",
        "- `specular`: `specular_safe` should avoid the local high-power washout induced by `high_power`.",
        "- `dark`: `dark_expected` should improve local edge visibility over `low_light_bad`.",
        "- If the panels look almost identical and local fill/quality ranges are near zero, the current depth model is not giving the policy a useful camera-control cue at that pose.",
        "",
        "Files:",
        "",
        "- `opening_depth_probe_detail.csv`: one row per condition/pose/camera setting.",
        "- `opening_depth_probe_expectations.csv`: coarse automatic expectation checks.",
        "- `panels/<scene>/<slot>/*.png`: visual panels.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_final_full.args")
    parser.add_argument("--out_dir", default="paper/experiment/results/opening_depth_probe")
    parser.add_argument("--scenarios", nargs="*", default=list(SCENE_ORDER))
    parser.add_argument("--slots", nargs="*", default=list(SLOT_ORDER))
    parser.add_argument("--xs", default="-2.8,-1.8,-0.8,0.0,0.55,1.05,1.45,1.70",
                        help="Comma-separated x positions before the gate.")
    parser.add_argument("--path_y_mode", default="slot", choices=["center", "blend", "slot"],
                        help="Position y coordinate; camera always looks at the opening center.")
    parser.add_argument("--camera_settings", default=None,
                        help="Optional semicolon list: name:p,e,g;name2:p,e,g")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--keep_scene_randomize", action="store_true",
                        help="Keep --sun_glare_randomize from config. Default disables it for controlled probes.")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_panels", type=int, default=0,
                        help="If >0, stop writing panels after this many images while still writing CSV rows.")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main():
    parser = _make_arg_parser()
    script_args, project_overrides = parser.parse_known_args()
    out_dir = Path(script_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    project_args = _build_project_args(Path(script_args.config), project_overrides)
    project_args.diff_sensor_impl["diff_depth"] = "cuda"
    if (not script_args.keep_scene_randomize) and hasattr(project_args, "sun_glare_randomize"):
        project_args.sun_glare_randomize = False
    if script_args.seed is not None:
        project_args.seed = int(script_args.seed)
    set_global_seed(project_args.seed, project_args.deterministic)

    device = torch.device(script_args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    scenes = _parse_scenes(script_args.scenarios)
    slots = _parse_slots(script_args.slots)
    xs = _parse_float_list(script_args.xs)
    settings = _parse_camera_settings(script_args.camera_settings, project_args)

    detail_rows: list[dict] = []
    panel_count = 0

    with torch.no_grad():
        for scene in scenes:
            for slot in slots:
                cond_args = copy.deepcopy(project_args)
                cond_args.scenarios = [scene]
                cond_args.sun_glare_eval_slot = slot
                env = build_env(1, cond_args, device, eval_mode=True)
                env.reset(scene_name=scene)
                target = _opening_target(env)
                poses = _make_poses(env, xs, script_args.path_y_mode)

                for pose in poses:
                    rendered = [_render_condition(env, cond_args, pose, target, setting)
                                for setting in settings]
                    _add_reference_diffs(rendered, cond_args.depth_min_valid)
                    detail_rows.extend(row for row, _ in rendered)
                    if script_args.plots and (
                        int(script_args.max_panels) <= 0 or panel_count < int(script_args.max_panels)
                    ):
                        panel_path = out_dir / "panels" / scene / slot / f"{pose.name}.png"
                        _plot_panel(panel_path, rendered, cond_args)
                        panel_count += 1

    expectation_rows = _expectation_rows(detail_rows)
    _write_csv(out_dir / "opening_depth_probe_detail.csv", detail_rows)
    _write_csv(out_dir / "opening_depth_probe_expectations.csv", expectation_rows)
    _write_report(out_dir / "report.md", detail_rows, expectation_rows)

    checked = [r for r in expectation_rows if float(r.get("checked", 0.0)) > 0.5]
    passed = [r for r in checked if float(r.get("passed", 0.0)) > 0.5]
    print(f"[opening-probe] scenes={scenes} slots={slots}")
    print(f"[opening-probe] wrote detail rows: {len(detail_rows)}")
    print(f"[opening-probe] wrote panels: {panel_count}")
    print(f"[opening-probe] expectation pass rate: {len(passed) / max(len(checked), 1):.3f} ({len(passed)}/{len(checked)})")
    print(f"[opening-probe] out_dir: {out_dir}")


if __name__ == "__main__":
    main()

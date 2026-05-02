#!/usr/bin/env python3
"""Compare camera settings on a fixed gate-approach trajectory.

This script is intentionally independent of a trained policy checkpoint.  It is
for validating the active-sensing environment itself:

- raw_depth geometry should match the top-down scene and should not depend on
  power/exposure/gain.
- depth_obs / quality / invalid should depend on scene and camera parameters.
- a differentiable camera-only optimizer should be able to improve gate-edge
  visibility from the same initial camera state.

Methods rendered per pose:

- fixed: config fixed_camera_power/exposure/gain.
- randfix_best: best candidate among K random static camera settings.  This is
  a generous randfix upper bound for this pose, not a learned policy.
- oracle_grid: best candidate from a dense p/e/g grid.
- diffopt: direct differentiable optimization of p/e/g at the current pose.

A trained learned-camera policy can be added later, but without a checkpoint the
honest substitute is diffopt: it proves whether the differentiable sensor model
contains a useful local optimization signal.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import set_global_seed  # noqa: E402
from train_utils import build_env  # noqa: E402
from tools.probe_opening_depth_views import (  # noqa: E402
    CameraSetting,
    _add_reference_diffs,
    _build_project_args,
    _local_mask_metrics,
    _make_poses,
    _opening_target,
    _parse_float_list,
    _parse_scenes,
    _parse_slots,
    _plot_topdown_overview,
    _render_condition,
    _set_pose_look_at,
    _to_float,
    _write_csv,
)


@dataclass(frozen=True)
class MethodResult:
    row: dict
    maps: dict[str, np.ndarray | None]


def _parse_triplet(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if text is None:
        return default
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError(f"expected p,e,g triplet, got {text!r}")
    return float(vals[0]), float(vals[1]), float(vals[2])


def _score_row(row: dict) -> float:
    edge_area = float(row.get("local_edge_area", 0.0))
    edge_fill = float(row.get("local_edge_fill", 0.0))
    edge_q = row.get("local_edge_quality_mean", float("nan"))
    if not math.isfinite(float(edge_q)):
        edge_q = float(row.get("local_quality_mean", 0.0))
    local_fill = float(row.get("local_fill", 0.0))
    local_q = float(row.get("local_quality_mean", 0.0))
    invalid = float(row.get("glare_invalid_rate", row.get("invalid_rate", 0.0)))
    if edge_area > 1e-5:
        return 0.58 * edge_fill + 0.34 * float(edge_q) + 0.08 * local_fill - 0.10 * invalid
    return 0.62 * local_fill + 0.38 * local_q - 0.10 * invalid


def _render_setting(env, cond_args, pose, target, setting: CameraSetting, method: str) -> MethodResult:
    row, maps = _render_condition(env, cond_args, pose, target, setting)
    row["method"] = method
    row["score"] = _score_row(row)
    return MethodResult(row=row, maps=maps)


def _candidate_grid(powers: list[float], exposures: list[float], gains: list[float], prefix: str) -> list[CameraSetting]:
    return [
        CameraSetting(f"{prefix}_p{p:.2f}_e{e:.2f}_g{g:.2f}", p, e, g)
        for p in powers for e in exposures for g in gains
    ]


def _best_candidate(env, cond_args, pose, target, candidates: list[CameraSetting],
                    method: str, write_rows: bool = False) -> tuple[MethodResult, list[dict]]:
    best: MethodResult | None = None
    rows: list[dict] = []
    with torch.no_grad():
        for setting in candidates:
            result = _render_setting(env, cond_args, pose, target, setting, method)
            result.row["candidate_setting"] = setting.name
            if write_rows:
                rows.append(dict(result.row))
            if best is None or float(result.row["score"]) > float(best.row["score"]):
                best = result
    if best is None:
        raise RuntimeError(f"no candidates for method {method}")
    best.row["setting"] = method
    best.row["candidate_setting"] = best.row.get("candidate_setting", "")
    return best, rows


def _raw_edge_weight(raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d4 = raw[:, None]
    d_far = F.max_pool2d(d4, 3, stride=1, padding=1)[:, 0]
    d_near = -F.max_pool2d(-d4, 3, stride=1, padding=1)[:, 0]
    edge = ((d_far - d_near) / (raw + 0.18)).clamp(0.0, 1.0)
    weight = (edge.detach() * mask.detach()).clamp_min(0.0)
    if float(weight.sum().detach().cpu().item()) < 1e-6:
        weight = mask.detach().clamp_min(0.0)
    if float(weight.sum().detach().cpu().item()) < 1e-6:
        weight = torch.ones_like(raw)
    return weight


def _logit01(value: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    value = value.clamp(eps, 1.0 - eps)
    return torch.log(value / (1.0 - value))


def _diffopt_camera(env, cond_args, pose, target, init_peg: tuple[float, float, float],
                    steps: int, lr: float, trace_prefix: dict) -> tuple[MethodResult, list[dict]]:
    device = env.device
    init = torch.tensor(init_peg, device=device, dtype=torch.float32)
    logits = _logit01(init).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=float(lr))
    trace_rows: list[dict] = []

    for step in range(max(0, int(steps))):
        opt.zero_grad(set_to_none=True)
        _set_pose_look_at(env, pose, target)
        peg = torch.sigmoid(logits).clamp(0.02, 0.98)
        power = peg[0:1]
        exposure = peg[1:2]
        gain = peg[2:3]
        env.render_diff_depth(power, exposure, gain)
        debug = env.last_diff_depth_debug or {}
        aux = env.get_last_diff_depth_train_aux()
        raw = debug.get("raw_depth_map", None)
        mask = debug.get("scene_mask", None)
        valid_prob = aux.get("valid_prob_map", None)
        quality = aux.get("quality_pre_valid", None)
        if raw is None or mask is None or valid_prob is None or quality is None:
            raise RuntimeError("diffopt requires raw_depth_map, scene_mask, valid_prob_map, quality_pre_valid")
        weight = _raw_edge_weight(raw, mask)
        denom = weight.sum(dim=(-2, -1)).clamp_min(1e-6)
        edge_valid = (valid_prob * weight).sum(dim=(-2, -1)) / denom
        edge_quality = (quality * weight).sum(dim=(-2, -1)) / denom
        # Small center prior prevents the optimizer from using extreme values
        # when several settings are visually equivalent at a pose.
        center_prior = (peg - init).pow(2).mean()
        objective = (0.68 * edge_valid + 0.32 * edge_quality).mean()
        loss = -objective + 0.002 * center_prior
        loss.backward()
        opt.step()

        trace_rows.append({
            **trace_prefix,
            "step": step,
            "power": float(peg[0].detach().cpu().item()),
            "exposure": float(peg[1].detach().cpu().item()),
            "gain": float(peg[2].detach().cpu().item()),
            "objective": float(objective.detach().cpu().item()),
            "loss": float(loss.detach().cpu().item()),
        })

    with torch.no_grad():
        final = torch.sigmoid(logits).clamp(0.02, 0.98).detach().cpu().numpy().tolist()
        setting = CameraSetting("diffopt", float(final[0]), float(final[1]), float(final[2]))
        result = _render_setting(env, cond_args, pose, target, setting, "diffopt")
        result.row["candidate_setting"] = "gradient_optimized"
        result.row["diffopt_steps"] = int(steps)
        result.row["diffopt_lr"] = float(lr)
    return result, trace_rows


def _write_report(path: Path, rows: list[dict]):
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["scene"]), []).append(row)

    lines = [
        "# Fixed Trajectory Camera Comparison",
        "",
        "This is a checkpoint-free environment validation experiment.",
        "",
        "Methods:",
        "",
        "- `fixed`: the configured fixed camera parameters.",
        "- `randfix_best`: best of random static samples at the same pose; this is a generous randfix upper bound.",
        "- `oracle_grid`: best p/e/g from a dense hand grid.",
        "- `diffopt`: p/e/g optimized directly through the differentiable sensor at the same pose.",
        "",
        "Important reading rule:",
        "",
        "- `raw_depth` should be geometrically consistent with the top-down map and nearly identical across methods.",
        "- Differences between methods should appear in `depth obs`, `quality`, and `invalid`, not in raw geometry.",
        "",
        "Scene summaries:",
        "",
    ]
    for scene, items in sorted(grouped.items()):
        by_method: dict[str, list[float]] = {}
        by_power: dict[str, list[float]] = {}
        by_exposure: dict[str, list[float]] = {}
        by_gain: dict[str, list[float]] = {}
        for row in items:
            method = str(row.get("method", ""))
            by_method.setdefault(method, []).append(float(row.get("score", 0.0)))
            by_power.setdefault(method, []).append(float(row.get("power", 0.0)))
            by_exposure.setdefault(method, []).append(float(row.get("exposure", 0.0)))
            by_gain.setdefault(method, []).append(float(row.get("gain", 0.0)))
        lines.append(f"## {scene}")
        for method in sorted(by_method):
            scores = by_method[method]
            lines.append(
                f"- `{method}` score={np.mean(scores):.3f} "
                f"p/e/g={np.mean(by_power[method]):.2f}/"
                f"{np.mean(by_exposure[method]):.2f}/"
                f"{np.mean(by_gain[method]):.2f}"
            )
        ranked = sorted(
            ((method, float(np.mean(scores))) for method, scores in by_method.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        lines.append("")
        lines.append("Edge-visibility ranking:")
        for idx, (method, score) in enumerate(ranked, start=1):
            lines.append(f"{idx}. `{method}` score={score:.3f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_rankings(path: Path, rows: list[dict]):
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["scene"]), str(row["slot"]), str(row["pose"])), []).append(row)
    out: list[dict] = []
    for (scene, slot, pose), items in sorted(grouped.items()):
        ranked = sorted(items, key=lambda r: float(r.get("score", 0.0)), reverse=True)
        for rank, row in enumerate(ranked, start=1):
            out.append({
                "scene": scene,
                "slot": slot,
                "pose": pose,
                "rank": rank,
                "method": row.get("method", ""),
                "score": row.get("score", 0.0),
                "power": row.get("power", 0.0),
                "exposure": row.get("exposure", 0.0),
                "gain": row.get("gain", 0.0),
                "local_edge_fill": row.get("local_edge_fill", 0.0),
                "local_edge_quality_mean": row.get("local_edge_quality_mean", 0.0),
                "invalid_rate": row.get("invalid_rate", 0.0),
            })
    _write_csv(path, out)


def _zoom_crop(image: np.ndarray | None, mask: np.ndarray | None, pad: int = 4) -> np.ndarray | None:
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim != 2:
        return None
    if mask is not None and np.asarray(mask).shape == arr.shape and np.any(np.asarray(mask) > 0.05):
        yy, xx = np.where(np.asarray(mask) > 0.05)
        y0 = max(0, int(yy.min()) - int(pad))
        y1 = min(arr.shape[0], int(yy.max()) + int(pad) + 1)
        x0 = max(0, int(xx.min()) - int(pad))
        x1 = min(arr.shape[1], int(xx.max()) + int(pad) + 1)
    else:
        h, w = arr.shape
        ch, cw = max(10, h // 3), max(10, w // 3)
        cy, cx = h // 2, w // 2
        y0, y1 = max(0, cy - ch // 2), min(h, cy + ch // 2)
        x0, x1 = max(0, cx - cw // 2), min(w, cx + cw // 2)
    if y1 <= y0 or x1 <= x0:
        return None
    return arr[y0:y1, x0:x1]


def _write_compare_panel(path: Path, rendered: list[MethodResult], cond_args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[fixed-trajectory][warn] matplotlib unavailable, skip plots: {exc}")
        return

    n = len(rendered)
    fig, axes = plt.subplots(n, 9, figsize=(30.0, max(2.45 * n, 3.0)), squeeze=False)
    depth_cmap = plt.cm.viridis.copy()
    depth_cmap.set_bad("black")

    first = rendered[0].row
    for r, result in enumerate(rendered):
        row = result.row
        maps = result.maps
        depth = maps["depth"].astype(np.float32)
        raw = maps.get("raw_depth")
        raw_show = None if raw is None else raw.astype(np.float32).copy()
        obs_show = depth.copy()
        if raw_show is not None:
            raw_show[raw_show <= float(cond_args.depth_min_valid) + 1e-6] = np.nan
        obs_show[depth <= float(cond_args.depth_min_valid) + 1e-6] = np.nan
        quality = maps.get("quality")
        invalid = maps.get("invalid")
        mask = maps.get("scene_mask")
        raw_crop = _zoom_crop(raw_show, mask)
        obs_crop = _zoom_crop(obs_show, mask)

        _plot_topdown_overview(axes[r, 0], row, cond_args)
        axes[r, 1].imshow(np.zeros_like(depth) if raw_show is None else raw_show,
                          vmin=cond_args.depth_min_valid, vmax=cond_args.depth_max_range, cmap=depth_cmap)
        axes[r, 1].set_title("raw depth")
        axes[r, 2].imshow(obs_show, vmin=cond_args.depth_min_valid, vmax=cond_args.depth_max_range, cmap=depth_cmap)
        axes[r, 2].set_title("depth obs")
        axes[r, 3].imshow(np.zeros((8, 8)) if raw_crop is None else raw_crop,
                          vmin=cond_args.depth_min_valid, vmax=cond_args.depth_max_range, cmap=depth_cmap)
        axes[r, 3].set_title("gate raw crop")
        axes[r, 4].imshow(np.zeros((8, 8)) if obs_crop is None else obs_crop,
                          vmin=cond_args.depth_min_valid, vmax=cond_args.depth_max_range, cmap=depth_cmap)
        axes[r, 4].set_title("gate obs crop")
        axes[r, 5].imshow(np.zeros_like(depth) if quality is None else quality, vmin=0, vmax=1, cmap="magma")
        axes[r, 5].set_title(f"quality {float(row.get('local_quality_mean', 0.0)):.2f}")
        axes[r, 6].imshow(np.zeros_like(depth) if invalid is None else invalid, vmin=0, vmax=1, cmap="gray")
        axes[r, 6].set_title(f"invalid {float(row.get('invalid_rate', 0.0)):.2f}")
        axes[r, 7].imshow(np.zeros_like(depth) if mask is None else mask, vmin=0, vmax=1, cmap="cividis")
        axes[r, 7].set_title(f"edge fill {float(row.get('local_edge_fill', 0.0)):.2f}")

        if raw is not None:
            h = raw.shape[0]
            center = h // 2
            lo = max(0, center - 1)
            hi = min(h, center + 2)
            raw_profile = np.nanmean(raw[lo:hi], axis=0)
            obs_valid = np.where(depth > cond_args.depth_min_valid + 1e-6, depth, np.nan)
            if np.isfinite(obs_valid).any():
                finite = np.isfinite(obs_valid)
                counts = finite.sum(axis=0)
                sums = np.where(finite, obs_valid, 0.0).sum(axis=0)
                obs_profile = np.divide(
                    sums,
                    counts,
                    out=np.full_like(sums, np.nan, dtype=np.float32),
                    where=counts > 0,
                )
            else:
                obs_profile = np.full_like(raw_profile, np.nan, dtype=np.float32)
            axes[r, 8].plot(raw_profile, label="raw", lw=1.8)
            axes[r, 8].plot(obs_profile, label="obs", lw=1.4)
            axes[r, 8].set_ylim(cond_args.depth_min_valid, cond_args.depth_max_range)
            axes[r, 8].grid(True, alpha=0.25)
            axes[r, 8].legend(loc="upper right", fontsize=7)
        axes[r, 8].set_title("center depth profile")

        for c in range(1, 8):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(
            f"{row['method']}\n"
            f"p/e/g={row['power']:.2f}/{row['exposure']:.2f}/{row['gain']:.2f}\n"
            f"score={row['score']:.2f}",
            fontsize=9,
        )

    fig.suptitle(
        f"{first['scene']} {first['slot']} {first['pose']} "
        f"local=({first['local_x']:.2f},{first['local_y']:.2f},{first['local_z']:.2f})"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_final_full.args")
    parser.add_argument("--out_dir", default="paper/experiment/results/fixed_trajectory_camera_compare")
    parser.add_argument("--scenarios", nargs="*", default=["glare", "specular", "dark"])
    parser.add_argument("--slots", nargs="*", default=["left", "right"])
    parser.add_argument("--xs", default="-0.85,-0.45,-0.05,0.30,0.58",
                        help="Comma-separated local x positions along a reasonable gate-approach trajectory.")
    parser.add_argument("--path_y_mode", default="slot", choices=["center", "blend", "slot"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sensor_impl", default="cuda", choices=["cuda", "python"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_random_rotation", action="store_true")
    parser.add_argument("--fixed_setting", default=None,
                        help="Override fixed p,e,g. Default uses config fixed_camera_*.")
    parser.add_argument("--diffopt_init", default="0.50,0.35,0.25")
    parser.add_argument("--diffopt_steps", type=int, default=40)
    parser.add_argument("--diffopt_lr", type=float, default=0.18)
    parser.add_argument("--randfix_k", type=int, default=24)
    parser.add_argument("--oracle_powers", default="0.18,0.32,0.50,0.70,0.90,0.96")
    parser.add_argument("--oracle_exposures", default="0.14,0.24,0.34,0.50,0.66,0.82,0.92")
    parser.add_argument("--oracle_gains", default="0.03,0.15,0.32,0.55,0.78,0.92")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_panels", type=int, default=0)
    parser.add_argument("--write_candidates", action="store_true")
    return parser


def main():
    parser = _make_arg_parser()
    script_args, project_overrides = parser.parse_known_args()
    out_dir = Path(script_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    project_args = _build_project_args(Path(script_args.config), project_overrides)
    project_args.diff_sensor_impl["diff_depth"] = str(script_args.sensor_impl)
    if not script_args.keep_random_rotation and hasattr(project_args, "random_rotation"):
        project_args.random_rotation = False
    set_global_seed(int(script_args.seed), getattr(project_args, "deterministic", False))

    device = torch.device(script_args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    scenes = _parse_scenes(script_args.scenarios)
    slots = _parse_slots(script_args.slots)
    xs = _parse_float_list(script_args.xs)
    fixed_peg = _parse_triplet(script_args.fixed_setting, (
        float(project_args.fixed_camera_power),
        float(project_args.fixed_camera_exposure),
        float(project_args.fixed_camera_gain),
    ))
    diffopt_init = _parse_triplet(script_args.diffopt_init, (0.50, 0.35, 0.25))
    oracle = _candidate_grid(
        _parse_float_list(script_args.oracle_powers),
        _parse_float_list(script_args.oracle_exposures),
        _parse_float_list(script_args.oracle_gains),
        "grid",
    )

    rng = np.random.default_rng(int(script_args.seed))
    detail_rows: list[dict] = []
    candidate_rows: list[dict] = []
    trace_rows: list[dict] = []
    panel_count = 0

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
                rendered: list[MethodResult] = []

                fixed_setting = CameraSetting("fixed", *fixed_peg)
                with torch.no_grad():
                    rendered.append(_render_setting(env, cond_args, pose, target, fixed_setting, "fixed"))

                randfix = [
                    CameraSetting(
                        f"rand_{i:03d}",
                        float(rng.uniform(*env.fixed_random_power_range)),
                        float(rng.uniform(*env.fixed_random_exposure_range)),
                        float(rng.uniform(*env.fixed_random_gain_range)),
                    )
                    for i in range(max(1, int(script_args.randfix_k)))
                ]
                rand_best, rand_rows = _best_candidate(
                    env, cond_args, pose, target, randfix, "randfix_best",
                    write_rows=bool(script_args.write_candidates),
                )
                rendered.append(rand_best)
                candidate_rows.extend(rand_rows)

                oracle_best, oracle_rows = _best_candidate(
                    env, cond_args, pose, target, oracle, "oracle_grid",
                    write_rows=bool(script_args.write_candidates),
                )
                rendered.append(oracle_best)
                candidate_rows.extend(oracle_rows)

                diffopt, trace = _diffopt_camera(
                    env,
                    cond_args,
                    pose,
                    target,
                    diffopt_init,
                    steps=int(script_args.diffopt_steps),
                    lr=float(script_args.diffopt_lr),
                    trace_prefix={
                        "scene": scene,
                        "slot": slot,
                        "pose": pose.name,
                        "local_x": float(rendered[0].row.get("local_x", 0.0)),
                        "local_y": float(rendered[0].row.get("local_y", 0.0)),
                    },
                )
                rendered.append(diffopt)
                trace_rows.extend(trace)

                _add_reference_diffs([(r.row, r.maps) for r in rendered], cond_args.depth_min_valid)
                detail_rows.extend(dict(r.row) for r in rendered)

                if script_args.plots and (
                    int(script_args.max_panels) <= 0 or panel_count < int(script_args.max_panels)
                ):
                    panel_path = out_dir / "panels" / scene / slot / f"{pose.name}.png"
                    _write_compare_panel(panel_path, rendered, cond_args)
                    panel_count += 1

    _write_csv(out_dir / "trajectory_camera_compare_detail.csv", detail_rows)
    _write_csv(out_dir / "trajectory_camera_compare_diffopt_trace.csv", trace_rows)
    _write_rankings(out_dir / "trajectory_camera_compare_rankings.csv", detail_rows)
    if script_args.write_candidates:
        _write_csv(out_dir / "trajectory_camera_compare_candidates.csv", candidate_rows)
    _write_report(out_dir / "report.md", detail_rows)

    print(f"[fixed-trajectory] scenes={scenes} slots={slots}")
    print(f"[fixed-trajectory] rows={len(detail_rows)} panels={panel_count}")
    print(f"[fixed-trajectory] out_dir={out_dir}")
    print("[fixed-trajectory] note: diffopt is checkpoint-free differentiable camera optimization, not a trained policy.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Trajectory-level differentiable camera optimization benchmark.

This script implements the "no camera policy training" experiment:

1. Use a fixed, hand-defined trajectory through the wall slit.
2. Compare static camera baselines against a whole-trajectory differentiable
   optimizer over p/e/g camera parameters.
3. Optimize a generic sensor-health objective, not scene labels.

The method is checkpoint-free.  It is meant to make the core differentiable
perception claim explicit before reintroducing learned policies.
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

from rollout_ops import compute_depth_fill_health, diff_depth_exposure_to_time  # noqa: E402
from train_utils import build_env  # noqa: E402
from config import set_global_seed  # noqa: E402
from tools.probe_opening_depth_views import (  # noqa: E402
    CameraSetting,
    ProbePose,
    _build_project_args,
    _local_mask_metrics,
    _parse_float_list,
    _parse_scenes,
    _parse_slots,
    _plot_topdown_overview,
    _render_condition,
    _set_pose_look_at,
    _write_csv,
)


@dataclass(frozen=True)
class TrajectoryState:
    pose: ProbePose
    target: torch.Tensor
    local_x: float
    local_y: float
    local_z: float
    phase: str


def _parse_triplet(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if text is None:
        return default
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != 3:
        raise ValueError(f"expected p,e,g triplet, got {text!r}")
    return float(vals[0]), float(vals[1]), float(vals[2])


def _logit01(value: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    value = value.clamp(eps, 1.0 - eps)
    return torch.log(value / (1.0 - value))


def _local_to_world(env, local_xyz: tuple[float, float, float]) -> torch.Tensor:
    local = torch.tensor(local_xyz, device=env.device, dtype=torch.float32)
    R_scene = getattr(env, "R_scene", None)
    if torch.is_tensor(R_scene) and R_scene.ndim == 3 and R_scene.shape[0] > 0:
        return torch.matmul(R_scene[0].to(device=env.device, dtype=torch.float32), local)
    return local


def _build_trajectory(env, xs: list[float], *, y_mode: str, target_mode: str,
                      after_wall_margin: float, lookahead_x: float) -> list[TrajectoryState]:
    fx = env.current_scene_effects or {}
    start_y = float(fx.get("geometry_start_y", 0.0))
    slot_y = float(fx.get("decision_open_slot_y", 0.0))
    wall_x = float(fx.get("geometry_wall_x", 0.0))
    goal_x = float(fx.get("geometry_goal_x", getattr(env, "simple_goal_x", 1.5)))
    slit_z = float(fx.get("slit_center_z", getattr(env, "simple_slit_center_z", 1.5)))
    x_min = min(xs)
    denom = max(wall_x - x_min, 1e-6)
    states: list[TrajectoryState] = []
    for idx, x in enumerate(xs):
        if y_mode == "slot":
            y = slot_y
        elif y_mode == "blend":
            alpha = float(np.clip((float(x) - x_min) / denom, 0.0, 1.0))
            y = (1.0 - alpha) * start_y + alpha * slot_y
        else:
            y = 0.0
        if float(x) < wall_x - 0.20:
            phase = "before"
        elif float(x) <= wall_x + float(after_wall_margin):
            phase = "near"
        else:
            phase = "after"

        pos_world = _local_to_world(env, (float(x), float(y), slit_z))
        if target_mode == "opening":
            target_local_x = wall_x
        elif target_mode == "goal":
            target_local_x = goal_x
        else:
            target_local_x = wall_x if phase != "after" else min(goal_x, max(float(x) + lookahead_x, wall_x + 0.25))
        target_world = _local_to_world(env, (target_local_x, slot_y, slit_z))
        states.append(TrajectoryState(
            pose=ProbePose(f"t{idx:02d}_x{float(x):+.2f}", float(pos_world[0]), float(pos_world[1]), float(pos_world[2])),
            target=target_world.detach(),
            local_x=float(x),
            local_y=float(y),
            local_z=gate_z,
            phase=phase,
        ))
    return states


def _set_speed_toward_target(env, state: TrajectoryState, speed_mps: float) -> None:
    if float(speed_mps) <= 0.0:
        env.v.zero_()
        return
    pos = torch.tensor([state.pose.x, state.pose.y, state.pose.z], device=env.device, dtype=torch.float32)
    direction = F.normalize(state.target.to(env.device) - pos, dim=0)
    env.v = direction.reshape(1, 3).repeat(env.batch_size, 1) * float(speed_mps)


def _raw_edge_weight(raw: torch.Tensor, region: torch.Tensor | None) -> torch.Tensor:
    d4 = raw[:, None]
    d_far = F.max_pool2d(d4, 3, stride=1, padding=1)[:, 0]
    d_near = -F.max_pool2d(-d4, 3, stride=1, padding=1)[:, 0]
    edge = ((d_far - d_near) / (raw + 0.18)).clamp(0.0, 1.0)
    if region is not None:
        weight = (edge.detach() * region.detach()).clamp_min(0.0)
    else:
        weight = edge.detach().clamp_min(0.0)
    if float(weight.sum().detach().cpu().item()) < 1e-6 and region is not None:
        weight = region.detach().clamp_min(0.0)
    if float(weight.sum().detach().cpu().item()) < 1e-6:
        weight = torch.ones_like(raw)
    return weight


def _render_health_terms(env, args, state: TrajectoryState, peg: torch.Tensor,
                         speed_mps: float, health_patch_rows: int,
                         health_patch_cols: int, health_cvar_frac: float) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    _set_pose_look_at(env, state.pose, state.target)
    _set_speed_toward_target(env, state, speed_mps)
    power = peg[0:1]
    exposure = peg[1:2]
    gain = peg[2:3]
    depth, _quality_obs = env.render_diff_depth(power, exposure, gain)
    aux = env.get_last_diff_depth_train_aux()
    debug = env.last_diff_depth_debug or {}
    valid_prob = aux.get("valid_prob_map", None)
    quality = aux.get("quality_pre_valid", None)
    raw = debug.get("raw_depth_map", None)
    region = debug.get("scene_mask", None)
    if valid_prob is None or quality is None or raw is None:
        raise RuntimeError("trajectory diffopt requires valid_prob_map, quality_pre_valid, raw_depth_map")
    fill_health = compute_depth_fill_health(
        env,
        depth,
        min_valid_depth=float(args.depth_min_valid),
        patch_rows=int(health_patch_rows),
        patch_cols=int(health_patch_cols),
        cvar_frac=float(health_cvar_frac),
    )
    if region is not None:
        region = region.clamp(0.0, 1.0)
        region_mass = region.mean(dim=(-2, -1))
        region_weight = region / region.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        region_valid = (valid_prob * region_weight).sum(dim=(-2, -1))
        region_quality = (quality * region_weight).sum(dim=(-2, -1))
    else:
        region_mass = torch.zeros_like(fill_health)
        region_valid = fill_health
        region_quality = quality.mean(dim=(-2, -1))
    edge_weight = _raw_edge_weight(raw, region)
    edge_weight = edge_weight / edge_weight.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    edge_valid = (valid_prob * edge_weight).sum(dim=(-2, -1))
    edge_quality = (quality * edge_weight).sum(dim=(-2, -1))
    terms = {
        "fill_health": fill_health,
        "region_mass": region_mass,
        "region_valid": region_valid,
        "region_quality": region_quality,
        "edge_valid": edge_valid,
        "edge_quality": edge_quality,
    }
    scalars = {
        "fill_health": float(fill_health.detach().reshape(-1)[0].cpu()),
        "region_mass": float(region_mass.detach().reshape(-1)[0].cpu()),
        "region_valid": float(region_valid.detach().reshape(-1)[0].cpu()),
        "region_quality": float(region_quality.detach().reshape(-1)[0].cpu()),
        "edge_valid": float(edge_valid.detach().reshape(-1)[0].cpu()),
        "edge_quality": float(edge_quality.detach().reshape(-1)[0].cpu()),
    }
    return terms, scalars


def _trajectory_loss(env, args, states: list[TrajectoryState], cam_seq: torch.Tensor,
                     nominal: torch.Tensor, init_cam: torch.Tensor, speed_mps: float,
                     cfg) -> tuple[torch.Tensor, list[dict[str, float]], dict[str, torch.Tensor]]:
    loss_terms: list[torch.Tensor] = []
    scalar_rows: list[dict[str, float]] = []
    fill_values = []
    region_values = []
    for t, state in enumerate(states):
        peg = cam_seq[t].clamp(0.001, 0.999)
        terms, scalars = _render_health_terms(
            env, args, state, peg, speed_mps,
            cfg.health_patch_rows, cfg.health_patch_cols, cfg.health_cvar_frac,
        )
        visible = (terms["region_mass"] / max(float(cfg.region_visible_mass), 1e-6)).clamp(0.0, 1.0).detach()
        fill_gap = F.relu(float(cfg.fill_target) - terms["fill_health"]).pow(2)
        region_gap = F.relu(float(cfg.region_target) - terms["region_valid"]).pow(2) * visible
        edge_gap = F.relu(float(cfg.edge_target) - terms["edge_valid"]).pow(2) * visible
        quality_gap = F.relu(float(cfg.quality_target) - terms["region_quality"]).pow(2) * visible
        healthy = ((terms["fill_health"] - float(cfg.recovery_fill_target)) / max(float(cfg.recovery_margin), 1e-6)).clamp(0.0, 1.0).detach()
        nominal_loss = (peg - nominal).pow(2).mean() * healthy
        power_loss = F.relu(peg[0] - float(cfg.power_baseline)).pow(2)
        exp_phys = diff_depth_exposure_to_time(peg[1], camera_semantics=env.cam_sem)
        blur_loss = (float(speed_mps) * exp_phys).pow(2)
        noise_loss = peg[2].pow(2)
        step_loss = (
            float(cfg.coef_fill) * fill_gap.mean()
            + float(cfg.coef_region) * region_gap.mean()
            + float(cfg.coef_edge) * edge_gap.mean()
            + float(cfg.coef_quality) * quality_gap.mean()
            + float(cfg.coef_nominal) * nominal_loss.mean()
            + float(cfg.coef_power) * power_loss.mean()
            + float(cfg.coef_blur) * blur_loss.mean()
            + float(cfg.coef_noise) * noise_loss.mean()
        )
        loss_terms.append(step_loss)
        scalars.update({
            "step_loss": float(step_loss.detach().cpu()),
            "fill_gap": float(fill_gap.detach().reshape(-1)[0].cpu()),
            "region_gap": float(region_gap.detach().reshape(-1)[0].cpu()),
            "edge_gap": float(edge_gap.detach().reshape(-1)[0].cpu()),
            "quality_gap": float(quality_gap.detach().reshape(-1)[0].cpu()),
            "healthy_recovery_weight": float(healthy.detach().reshape(-1)[0].cpu()),
        })
        scalar_rows.append(scalars)
        fill_values.append(terms["fill_health"])
        region_values.append(terms["region_valid"])

    cam_for_smooth = torch.cat([init_cam.reshape(1, 3), cam_seq], dim=0)
    smooth_loss = cam_for_smooth.diff(dim=0).pow(2).mean()
    total = torch.stack(loss_terms).mean() + float(cfg.coef_smooth) * smooth_loss
    aux = {
        "smooth_loss": smooth_loss.detach(),
        "fill_mean": torch.stack(fill_values).mean().detach(),
        "region_mean": torch.stack(region_values).mean().detach(),
    }
    return total, scalar_rows, aux


def _optimize_trajectory(env, args, states: list[TrajectoryState], init_peg: tuple[float, float, float],
                         nominal_peg: tuple[float, float, float], speed_mps: float, cfg,
                         trace_prefix: dict) -> tuple[torch.Tensor, list[dict]]:
    device = env.device
    init = torch.tensor(init_peg, device=device, dtype=torch.float32)
    nominal = torch.tensor(nominal_peg, device=device, dtype=torch.float32)
    logits = _logit01(init).reshape(1, 3).repeat(len(states), 1).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=float(cfg.diffopt_lr))
    trace_rows: list[dict] = []

    for step in range(max(1, int(cfg.diffopt_steps))):
        opt.zero_grad(set_to_none=True)
        cam_seq = torch.sigmoid(logits).clamp(0.001, 0.999)
        total, _rows, aux = _trajectory_loss(env, args, states, cam_seq, nominal, init, speed_mps, cfg)
        total.backward()
        torch.nn.utils.clip_grad_norm_([logits], 10.0)
        opt.step()
        if step % max(1, int(cfg.trace_every)) == 0 or step == int(cfg.diffopt_steps) - 1:
            with torch.no_grad():
                cam = torch.sigmoid(logits).clamp(0.001, 0.999)
                trace_rows.append({
                    **trace_prefix,
                    "opt_step": step,
                    "loss": float(total.detach().cpu()),
                    "smooth_loss": float(aux["smooth_loss"].cpu()),
                    "fill_mean": float(aux["fill_mean"].cpu()),
                    "region_mean": float(aux["region_mean"].cpu()),
                    "power_mean": float(cam[:, 0].mean().cpu()),
                    "exposure_mean": float(cam[:, 1].mean().cpu()),
                    "gain_mean": float(cam[:, 2].mean().cpu()),
                })
    return torch.sigmoid(logits).clamp(0.001, 0.999).detach(), trace_rows


def _optimize_trajectory_multistart(env, args, states: list[TrajectoryState],
                                    init_candidates: list[tuple[str, tuple[float, float, float]]],
                                    nominal_peg: tuple[float, float, float], speed_mps: float,
                                    cfg, trace_prefix: dict) -> tuple[torch.Tensor, list[dict], dict]:
    best_seq = None
    best_trace: list[dict] = []
    best_info: dict = {}
    nominal = torch.tensor(nominal_peg, device=env.device, dtype=torch.float32)
    init_for_eval = nominal.clone()
    for name, init in init_candidates:
        seq, trace = _optimize_trajectory(
            env,
            args,
            states,
            init,
            nominal_peg,
            speed_mps,
            cfg,
            trace_prefix={**trace_prefix, "restart": name},
        )
        with torch.no_grad():
            loss, _rows, aux = _trajectory_loss(
                env, args, states, seq.to(env.device), nominal, init_for_eval, speed_mps, cfg)
        value = float(loss.detach().cpu())
        if not best_info or value < float(best_info["loss"]):
            best_seq = seq
            best_trace = trace
            best_info = {
                "restart": name,
                "loss": value,
                "fill_mean": float(aux["fill_mean"].cpu()),
                "region_mean": float(aux["region_mean"].cpu()),
            }
    if best_seq is None:
        raise RuntimeError("no trajectory diffopt restarts were run")
    return best_seq, best_trace, best_info


def _evaluate_sequence(env, args, states: list[TrajectoryState], cam_seq: torch.Tensor,
                       method: str, nominal_peg: tuple[float, float, float],
                       speed_mps: float, cfg) -> tuple[list[dict], list[tuple[dict, dict]]]:
    rows: list[dict] = []
    rendered: list[tuple[dict, dict]] = []
    nominal = torch.tensor(nominal_peg, device=env.device, dtype=torch.float32)
    init = nominal.clone()
    with torch.no_grad():
        total, health_rows, aux = _trajectory_loss(env, args, states, cam_seq.to(env.device), nominal, init, speed_mps, cfg)
        for t, state in enumerate(states):
            setting = CameraSetting(
                method,
                float(cam_seq[t, 0].detach().cpu()),
                float(cam_seq[t, 1].detach().cpu()),
                float(cam_seq[t, 2].detach().cpu()),
            )
            row, maps = _render_condition(env, args, state.pose, state.target, setting)
            row.update({
                "method": method,
                "step": t,
                "phase": state.phase,
                "traj_loss_total": float(total.detach().cpu()),
                "traj_smooth_loss": float(aux["smooth_loss"].cpu()),
                "traj_fill_mean": float(aux["fill_mean"].cpu()),
                "traj_region_mean": float(aux["region_mean"].cpu()),
                **health_rows[t],
            })
            rows.append(row)
            rendered.append((row, maps))
    return rows, rendered


def _best_static_sequence(env, args, states: list[TrajectoryState], candidates: list[CameraSetting],
                          nominal_peg: tuple[float, float, float], speed_mps: float, cfg,
                          method: str) -> tuple[CameraSetting, list[dict]]:
    best_setting = candidates[0]
    best_rows: list[dict] = []
    best_loss = float("inf")
    for setting in candidates:
        cam_seq = torch.tensor([[setting.power, setting.exposure, setting.gain]] * len(states), device=env.device)
        rows, _ = _evaluate_sequence(env, args, states, cam_seq, method, nominal_peg, speed_mps, cfg)
        loss = float(rows[0]["traj_loss_total"]) if rows else float("inf")
        if loss < best_loss:
            best_loss = loss
            best_setting = setting
            best_rows = rows
    return best_setting, best_rows


def _random_settings(env, k: int, rng: np.random.Generator) -> list[CameraSetting]:
    out = []
    for i in range(max(1, int(k))):
        out.append(CameraSetting(
            f"rand_{i:03d}",
            float(rng.uniform(*env.fixed_random_power_range)),
            float(rng.uniform(*env.fixed_random_exposure_range)),
            float(rng.uniform(*env.fixed_random_gain_range)),
        ))
    return out


def _grid_settings(powers: list[float], exposures: list[float], gains: list[float]) -> list[CameraSetting]:
    return [
        CameraSetting(f"p{p:.2f}_e{e:.2f}_g{g:.2f}", float(p), float(e), float(g))
        for p in powers for e in exposures for g in gains
    ]


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and str(r[key]) != "" and math.isfinite(float(r[key]))]
    return float(np.mean(vals)) if vals else float("nan")


def _write_summary_report(path: Path, rows: list[dict], trace_rows: list[dict], cfg) -> None:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((str(row["scene"]), str(row["slot"]), str(row["method"])), []).append(row)
    lines = [
        "# Trajectory-Level Diffopt Camera Benchmark",
        "",
        "This benchmark does not train a camera policy.  It directly optimizes the full camera-parameter trajectory through the differentiable depth sensor.",
        "",
        "Generic sensor loss:",
        "",
        f"- patch-CVaR fill target: `{cfg.fill_target}`",
        f"- visible affected-region target: `{cfg.region_target}`",
        f"- edge validity target: `{cfg.edge_target}`",
        f"- healthy nominal recovery: target `{cfg.recovery_fill_target}`, margin `{cfg.recovery_margin}`, coef `{cfg.coef_nominal}`",
        f"- temporal smooth coef: `{cfg.coef_smooth}`",
        "",
        "## Method Summary",
        "",
        "| scene | slot | method | loss | fill | region | p/e/g | before p/e/g | near p/e/g | after p/e/g |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, items in sorted(groups.items()):
        scene, slot, method = key
        def phase_mean(phase: str, cam_key: str) -> float:
            phase_rows = [r for r in items if str(r.get("phase")) == phase]
            return _mean(phase_rows, cam_key)
        lines.append(
            f"| {scene} | {slot} | {method} | "
            f"{_mean(items, 'traj_loss_total'):.4f} | "
            f"{_mean(items, 'fill_health'):.3f} | "
            f"{_mean(items, 'region_valid'):.3f} | "
            f"{_mean(items, 'power'):.3f}/{_mean(items, 'exposure'):.3f}/{_mean(items, 'gain'):.3f} | "
            f"{phase_mean('before', 'power'):.3f}/{phase_mean('before', 'exposure'):.3f}/{phase_mean('before', 'gain'):.3f} | "
            f"{phase_mean('near', 'power'):.3f}/{phase_mean('near', 'exposure'):.3f}/{phase_mean('near', 'gain'):.3f} | "
            f"{phase_mean('after', 'power'):.3f}/{phase_mean('after', 'exposure'):.3f}/{phase_mean('after', 'gain'):.3f} |"
        )
    lines.extend(["", "## Reading", ""])
    lines.append("- A useful result is not simply lower loss.  The `after` phase should move back toward nominal while `near` remains scene-dependent.")
    lines.append("- If `trajectory_diffopt` wins the loss but does not recover after the wall, increase healthy nominal recovery.")
    lines.append("- If it recovers everywhere and loses the near-wall effect, reduce recovery or increase region/edge health.")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_curve_plot(path: Path, rows: list[dict], title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[diffopt-benchmark][warn] matplotlib unavailable, skip curve plot: {exc}")
        return
    methods = sorted({str(r["method"]) for r in rows})
    fig, axes = plt.subplots(5, 1, figsize=(11.0, 11.5), sharex=True)
    for method in methods:
        items = sorted([r for r in rows if str(r["method"]) == method], key=lambda r: int(r["step"]))
        xs = [float(r["local_x"]) for r in items]
        axes[0].plot(xs, [float(r["power"]) for r in items], marker="o", label=method)
        axes[1].plot(xs, [float(r["exposure"]) for r in items], marker="o", label=method)
        axes[2].plot(xs, [float(r["gain"]) for r in items], marker="o", label=method)
        axes[3].plot(xs, [float(r["fill_health"]) for r in items], marker="o", label=method)
        axes[4].plot(xs, [float(r["region_valid"]) for r in items], marker="o", label=method)
    labels = ["power", "exposure", "gain", "patch-CVaR fill", "region validity"]
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.03, 1.03)
    axes[0].legend(ncol=min(4, len(methods)), fontsize=8)
    axes[-1].set_xlabel("local x")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_depth_panel(path: Path, rendered_by_method: dict[str, list[tuple[dict, dict]]],
                       selected_steps: list[int], args) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[diffopt-benchmark][warn] matplotlib unavailable, skip depth panel: {exc}")
        return
    methods = list(rendered_by_method.keys())
    nrows = len(methods)
    ncols = max(1, len(selected_steps))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("black")
    for r, method in enumerate(methods):
        rendered = rendered_by_method[method]
        by_step = {int(row["step"]): (row, maps) for row, maps in rendered}
        for c, step in enumerate(selected_steps):
            ax = axes[r, c]
            if step not in by_step:
                ax.axis("off")
                continue
            row, maps = by_step[step]
            depth = maps["depth"].astype(np.float32).copy()
            depth[depth <= float(args.depth_min_valid) + 1e-6] = np.nan
            ax.imshow(depth, vmin=float(args.depth_min_valid), vmax=float(args.depth_max_range), cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{method} t={step} x={float(row['local_x']):.2f}\n"
                f"p/e/g={float(row['power']):.2f}/{float(row['exposure']):.2f}/{float(row['gain']):.2f}"
            )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/slit_active_sensing.args")
    parser.add_argument("--out_dir", default="paper/experiment/results/diffopt_camera_benchmark")
    parser.add_argument("--scenarios", nargs="*", default=["glare", "specular", "dark"])
    parser.add_argument("--slots", nargs="*", default=["left", "right"])
    parser.add_argument("--xs", default="-1.20,-0.90,-0.60,-0.35,-0.18,-0.05,0.10,0.35,0.70,1.05,1.35")
    parser.add_argument("--path_y_mode", default="slot", choices=["center", "blend", "slot"])
    parser.add_argument("--target_mode", default="opening_then_goal", choices=["opening_then_goal", "opening", "goal"])
    parser.add_argument("--after_wall_margin", type=float, default=0.18)
    parser.add_argument("--lookahead_x", type=float, default=0.80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sensor_impl", default="cuda", choices=["cuda", "python"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_random_rotation", action="store_true")
    parser.add_argument("--speed_mps", type=float, default=1.0)
    parser.add_argument("--fixed_setting", default=None)
    parser.add_argument("--nominal_setting", default="0.50,0.50,0.50")
    parser.add_argument("--diffopt_init", default=None)
    parser.add_argument("--diffopt_steps", type=int, default=160)
    parser.add_argument("--diffopt_lr", type=float, default=0.08)
    parser.add_argument("--diffopt_random_restarts", type=int, default=4)
    parser.add_argument("--trace_every", type=int, default=10)
    parser.add_argument("--randfix_k", type=int, default=48)
    parser.add_argument("--oracle_grid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--oracle_powers", default="0.18,0.32,0.50,0.70,0.90")
    parser.add_argument("--oracle_exposures", default="0.12,0.24,0.38,0.50,0.68,0.84")
    parser.add_argument("--oracle_gains", default="0.03,0.15,0.32,0.55,0.78")
    parser.add_argument("--health_patch_rows", type=int, default=6)
    parser.add_argument("--health_patch_cols", type=int, default=8)
    parser.add_argument("--health_cvar_frac", type=float, default=0.25)
    parser.add_argument("--fill_target", type=float, default=0.90)
    parser.add_argument("--region_target", type=float, default=0.88)
    parser.add_argument("--edge_target", type=float, default=0.86)
    parser.add_argument("--quality_target", type=float, default=0.58)
    parser.add_argument("--recovery_fill_target", type=float, default=0.92)
    parser.add_argument("--recovery_margin", type=float, default=0.08)
    parser.add_argument("--region_visible_mass", type=float, default=0.006)
    parser.add_argument("--coef_fill", type=float, default=12.0)
    parser.add_argument("--coef_region", type=float, default=8.0)
    parser.add_argument("--coef_edge", type=float, default=5.0)
    parser.add_argument("--coef_quality", type=float, default=1.0)
    parser.add_argument("--coef_nominal", type=float, default=1.5)
    parser.add_argument("--coef_smooth", type=float, default=2.0)
    parser.add_argument("--coef_power", type=float, default=0.05)
    parser.add_argument("--coef_blur", type=float, default=0.01)
    parser.add_argument("--coef_noise", type=float, default=0.03)
    parser.add_argument("--power_baseline", type=float, default=None)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth_panel_steps", default="0,4,6,8,10")
    return parser


def main() -> None:
    parser = _make_parser()
    script_args, project_overrides = parser.parse_known_args()
    out_dir = Path(script_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_args = _build_project_args(Path(script_args.config), project_overrides)
    project_args.diff_sensor_impl["diff_depth"] = str(script_args.sensor_impl)
    if not bool(script_args.keep_random_rotation):
        project_args.random_rotation = False
    set_global_seed(int(script_args.seed), getattr(project_args, "deterministic", False))

    device = torch.device(script_args.device)
    scenes = _parse_scenes(script_args.scenarios)
    slots = _parse_slots(script_args.slots)
    xs = _parse_float_list(script_args.xs)
    nominal = _parse_triplet(script_args.nominal_setting, (0.5, 0.5, 0.5))
    fixed = _parse_triplet(script_args.fixed_setting, (
        float(project_args.fixed_camera_power),
        float(project_args.fixed_camera_exposure),
        float(project_args.fixed_camera_gain),
    ))
    diffopt_init = _parse_triplet(script_args.diffopt_init, nominal)
    cfg = argparse.Namespace(**vars(script_args))
    cfg.power_baseline = float(project_args.cam_power_baseline if script_args.power_baseline is None else script_args.power_baseline)

    rng = np.random.default_rng(int(script_args.seed))
    all_rows: list[dict] = []
    all_trace: list[dict] = []
    selected_steps = [int(x) for x in _parse_float_list(script_args.depth_panel_steps)]

    for scene in scenes:
        for slot in slots:
            cond_args = copy.deepcopy(project_args)
            cond_args.scenarios = [scene]
            cond_args.sun_glare_eval_slot = slot
            env = build_env(1, cond_args, device, eval_mode=True)
            env.reset(scene_name=scene)
            states = _build_trajectory(
                env,
                xs,
                y_mode=str(script_args.path_y_mode),
                target_mode=str(script_args.target_mode),
                after_wall_margin=float(script_args.after_wall_margin),
                lookahead_x=float(script_args.lookahead_x),
            )
            fixed_seq = torch.tensor([fixed] * len(states), device=device, dtype=torch.float32)
            fixed_rows, fixed_rendered = _evaluate_sequence(
                env, cond_args, states, fixed_seq, "fixed", nominal, float(script_args.speed_mps), cfg)
            all_rows.extend(fixed_rows)

            rand_candidates = _random_settings(env, int(script_args.randfix_k), rng)
            rand_setting, _rand_probe_rows = _best_static_sequence(
                env, cond_args, states, rand_candidates, nominal, float(script_args.speed_mps), cfg, "randfix_best")
            rand_seq = torch.tensor([[rand_setting.power, rand_setting.exposure, rand_setting.gain]] * len(states), device=device)
            rand_rows, rand_rendered = _evaluate_sequence(
                env, cond_args, states, rand_seq, "randfix_best", nominal, float(script_args.speed_mps), cfg)
            all_rows.extend(rand_rows)

            oracle_rendered = None
            if bool(script_args.oracle_grid):
                grid = _grid_settings(
                    _parse_float_list(script_args.oracle_powers),
                    _parse_float_list(script_args.oracle_exposures),
                    _parse_float_list(script_args.oracle_gains),
                )
                oracle_setting, _ = _best_static_sequence(
                    env, cond_args, states, grid, nominal, float(script_args.speed_mps), cfg, "oracle_static")
                oracle_seq = torch.tensor([[oracle_setting.power, oracle_setting.exposure, oracle_setting.gain]] * len(states), device=device)
                oracle_rows, oracle_rendered = _evaluate_sequence(
                    env, cond_args, states, oracle_seq, "oracle_static", nominal, float(script_args.speed_mps), cfg)
                all_rows.extend(oracle_rows)

            init_candidates: list[tuple[str, tuple[float, float, float]]] = [
                ("nominal", diffopt_init),
                ("fixed", fixed),
                ("randfix_best", (rand_setting.power, rand_setting.exposure, rand_setting.gain)),
            ]
            for ridx in range(max(0, int(script_args.diffopt_random_restarts))):
                init_candidates.append((
                    f"random_{ridx:02d}",
                    (
                        float(rng.uniform(*env.fixed_random_power_range)),
                        float(rng.uniform(*env.fixed_random_exposure_range)),
                        float(rng.uniform(*env.fixed_random_gain_range)),
                    ),
                ))
            diff_seq, trace, diff_info = _optimize_trajectory_multistart(
                env, cond_args, states, init_candidates, nominal, float(script_args.speed_mps), cfg,
                trace_prefix={"scene": scene, "slot": slot},
            )
            diff_rows, diff_rendered = _evaluate_sequence(
                env, cond_args, states, diff_seq, "trajectory_diffopt", nominal, float(script_args.speed_mps), cfg)
            for row in diff_rows:
                row["diffopt_restart"] = diff_info.get("restart", "")
            all_rows.extend(diff_rows)
            all_trace.extend(trace)

            if bool(script_args.plots):
                rows_for_plot = fixed_rows + rand_rows + diff_rows
                if oracle_rendered is not None:
                    rows_for_plot += oracle_rows
                _write_curve_plot(
                    out_dir / "curves" / f"{scene}_{slot}_camera_curves.png",
                    rows_for_plot,
                    title=f"{scene} {slot} trajectory camera optimization",
                )
                rendered_by_method = {
                    "fixed": fixed_rendered,
                    "randfix_best": rand_rendered,
                    "trajectory_diffopt": diff_rendered,
                }
                if oracle_rendered is not None:
                    rendered_by_method["oracle_static"] = oracle_rendered
                _write_depth_panel(
                    out_dir / "depth_panels" / f"{scene}_{slot}_depth_panel.png",
                    rendered_by_method,
                    selected_steps,
                    cond_args,
                )
            print(
                f"[diffopt-benchmark] {scene}/{slot} "
                f"fixed_loss={fixed_rows[0]['traj_loss_total']:.4f} "
                f"randfix_loss={rand_rows[0]['traj_loss_total']:.4f} "
                f"diffopt_loss={diff_rows[0]['traj_loss_total']:.4f} "
                f"restart={diff_info.get('restart', '')}"
            )

    _write_csv(out_dir / "trajectory_diffopt_detail.csv", all_rows)
    _write_csv(out_dir / "trajectory_diffopt_trace.csv", all_trace)
    _write_summary_report(out_dir / "report.md", all_rows, all_trace, cfg)
    print(f"[diffopt-benchmark] rows={len(all_rows)} trace_rows={len(all_trace)}")
    print(f"[diffopt-benchmark] out_dir={out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnostic closed-loop experiment for active sensing.

This is intentionally not a trained policy evaluation.  It uses one simple
waypoint controller for every method and changes only the camera-parameter
selection rule.

The controller commits to the true slit lane only after the current depth
observation has enough local slit-edge visibility.  If the camera setting makes
the slit edge unreadable, the controller keeps flying the center route and tends
to hit the wall.  This makes the experiment useful as a quick,
checkpoint-free demonstration that perception quality changes closed-loop
behavior.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import set_global_seed  # noqa: E402
from train_utils import build_env  # noqa: E402
from tools.probe_fixed_trajectory_camera_compare import (  # noqa: E402
    _candidate_grid,
    _raw_edge_weight,
    _score_row,
)
from tools.probe_opening_depth_views import (  # noqa: E402
    CameraSetting,
    ProbePose,
    _build_project_args,
    _local_mask_metrics,
    _parse_float_list,
    _parse_scenes,
    _parse_slots,
    _to_float,
    _write_csv,
)


def _min_clearance_from_vec(vec_now, batch_size: int):
    dist = torch.norm(vec_now, 2, -1)
    if dist.ndim == 1:
        return dist
    if dist.ndim == 2:
        if dist.shape[0] == batch_size and dist.shape[1] != batch_size:
            return dist.min(dim=1).values
        if dist.shape[1] == batch_size:
            return dist.min(dim=0).values
        return dist.min(dim=0).values
    return dist.reshape(-1, batch_size).min(dim=0).values


def _world_from_local(env, local_xyz: torch.Tensor) -> torch.Tensor:
    return torch.bmm(env.R_scene, local_xyz[None, :, None])[:, :, 0][0]


def _local_from_world(env, world_xyz: torch.Tensor) -> torch.Tensor:
    return torch.bmm(env.R_scene_T, world_xyz[None, :, None])[:, :, 0][0]


def _render_current(env, args, setting: CameraSetting) -> tuple[dict, torch.Tensor]:
    device = env.device
    power = torch.full((1,), float(setting.power), device=device)
    exposure = torch.full((1,), float(setting.exposure), device=device)
    gain = torch.full((1,), float(setting.gain), device=device)
    depth, quality = env.render_diff_depth(power, exposure, gain)
    debug = env.export_last_diff_depth_debug(0)
    images = debug.get("images", {})
    scalars = debug.get("scalars", {})
    depth_np = depth[0].detach().cpu().numpy()
    quality_np = None if quality is None else quality[0].detach().cpu().numpy()
    raw_np = images.get("raw_depth_map")
    mask_np = images.get("scene_mask")
    local = _local_from_world(env, env.p[0]).detach().cpu().numpy()
    valid = depth_np > (float(args.depth_min_valid) + 1e-6)
    valid_depth = depth_np[valid]
    row = {
        "scene": env.current_scene_name,
        "slot": str((env.current_scene_effects or {}).get("slit_slot_name", "")),
        "x": float(env.p[0, 0].detach().cpu()),
        "y": float(env.p[0, 1].detach().cpu()),
        "z": float(env.p[0, 2].detach().cpu()),
        "local_x": float(local[0]),
        "local_y": float(local[1]),
        "local_z": float(local[2]),
        "setting": setting.name,
        "power": float(setting.power),
        "exposure": float(setting.exposure),
        "gain": float(setting.gain),
        "fill_rate": float(valid.mean()),
        "valid_depth_mean": float(valid_depth.mean()) if valid_depth.size else 0.0,
        "quality_mean": float(np.mean(quality_np)) if quality_np is not None else 0.0,
        "invalid_rate": _to_float(scalars.get("invalid_rate"), 0.0),
        "glare_invalid_rate": _to_float(scalars.get("glare_invalid_rate"), 0.0),
    }
    row.update(_local_mask_metrics(depth_np, quality_np, mask_np, args.depth_min_valid, raw_np))
    row["score"] = _score_row(row)
    return row, depth


def _best_grid_current(env, args, candidates: list[CameraSetting]) -> tuple[CameraSetting, dict]:
    best_setting = candidates[0]
    best_row = None
    with torch.no_grad():
        for setting in candidates:
            row, _ = _render_current(env, args, setting)
            if best_row is None or float(row["score"]) > float(best_row["score"]):
                best_row = row
                best_setting = setting
    return best_setting, dict(best_row)


def _diffopt_current(env, args, init_peg, steps: int, lr: float) -> tuple[CameraSetting, dict]:
    device = env.device
    init = torch.tensor(init_peg, device=device, dtype=torch.float32)
    eps = 1e-4
    logits = torch.log(init.clamp(eps, 1.0 - eps) / (1.0 - init.clamp(eps, 1.0 - eps))).detach().clone()
    logits.requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=float(lr))
    for _ in range(max(0, int(steps))):
        opt.zero_grad(set_to_none=True)
        peg = torch.sigmoid(logits).clamp(0.02, 0.98)
        env.render_diff_depth(peg[0:1], peg[1:2], peg[2:3])
        debug = env.last_diff_depth_debug or {}
        aux = env.get_last_diff_depth_train_aux()
        raw = debug.get("raw_depth_map", None)
        mask = debug.get("scene_mask", None)
        valid_prob = aux.get("valid_prob_map", None)
        quality = aux.get("quality_pre_valid", None)
        if raw is None or mask is None or valid_prob is None or quality is None:
            raise RuntimeError("diffopt current requires raw/mask/valid/quality maps")
        weight = _raw_edge_weight(raw, mask)
        denom = weight.sum(dim=(-2, -1)).clamp_min(1e-6)
        edge_valid = (valid_prob * weight).sum(dim=(-2, -1)) / denom
        edge_quality = (quality * weight).sum(dim=(-2, -1)) / denom
        center_prior = (peg - init).pow(2).mean()
        objective = (0.68 * edge_valid + 0.32 * edge_quality).mean()
        loss = -objective + 0.002 * center_prior
        loss.backward()
        opt.step()
    setting = CameraSetting(
        "diffopt",
        float(torch.sigmoid(logits)[0].detach().cpu()),
        float(torch.sigmoid(logits)[1].detach().cpu()),
        float(torch.sigmoid(logits)[2].detach().cpu()),
    )
    row, _ = _render_current(env, args, setting)
    return setting, row


def _choose_camera(method: str, env, args, rng, oracle_grid, diffopt_init, diffopt_steps, diffopt_lr,
                   rand_static: CameraSetting | None):
    if method == "fixed":
        setting = CameraSetting(
            "fixed",
            float(args.fixed_camera_power),
            float(args.fixed_camera_exposure),
            float(args.fixed_camera_gain),
        )
        row, _ = _render_current(env, args, setting)
        return setting, row, rand_static
    if method == "randfix":
        if rand_static is None:
            setting = CameraSetting(
                "randfix",
                float(rng.uniform(*env.fixed_random_power_range)),
                float(rng.uniform(*env.fixed_random_exposure_range)),
                float(rng.uniform(*env.fixed_random_gain_range)),
            )
        else:
            setting = rand_static
        row, _ = _render_current(env, args, setting)
        return setting, row, setting
    if method == "oracle_grid":
        setting, row = _best_grid_current(env, args, oracle_grid)
        setting = CameraSetting("oracle_grid", setting.power, setting.exposure, setting.gain)
        return setting, row, rand_static
    if method == "diffopt":
        setting, row = _diffopt_current(env, args, diffopt_init, diffopt_steps, diffopt_lr)
        return setting, row, rand_static
    raise ValueError(f"unsupported method {method!r}")


def _controller_step(env, target_world, speed: float, max_acc: float):
    pos = env.p[0]
    vel = env.v[0]
    err = target_world - pos
    dist = torch.norm(err).clamp_min(1e-6)
    v_des = err / dist * min(float(speed), float(dist) * 2.0)
    acc = 2.6 * (v_des - vel) + 1.2 * err
    acc_norm = torch.norm(acc).clamp_min(1e-6)
    acc = acc / acc_norm * torch.clamp(acc_norm, max=float(max_acc))
    return acc[None], err[None]


def run_episode(method: str, scene: str, slot: str, args, script_args, device, rng):
    cond_args = copy.deepcopy(args)
    cond_args.scenarios = [scene]
    cond_args.sun_glare_eval_slot = slot
    cond_args.random_rotation = False
    env = build_env(1, cond_args, device, eval_mode=True)
    env.reset(scene_name=scene)

    oracle_grid = _candidate_grid(
        _parse_float_list(script_args.oracle_powers),
        _parse_float_list(script_args.oracle_exposures),
        _parse_float_list(script_args.oracle_gains),
        "grid",
    )
    diffopt_init = tuple(float(x) for x in _parse_float_list(script_args.diffopt_init))
    rand_static = None
    committed = False
    trace = []
    collided = False
    reached = False
    stop_reason = "timeout"

    ctl_dt = 1.0 / float(script_args.control_freq)
    for t in range(int(script_args.timesteps)):
        vec = env.find_vec_to_nearest_pt()
        clearance = _min_clearance_from_vec(vec, 1)
        goal_dist = torch.norm(env.p_target - env.p, dim=-1)
        if bool((clearance <= float(args.collision_clearance)).any().item()):
            collided = True
            stop_reason = "collision"
            break
        if bool((goal_dist < float(script_args.goal_radius)).any().item()):
            reached = True
            stop_reason = "goal"
            break

        setting, cam_row, rand_static = _choose_camera(
            method, env, cond_args, rng, oracle_grid,
            diffopt_init, int(script_args.diffopt_steps), float(script_args.diffopt_lr),
            rand_static,
        )
        edge_fill = float(cam_row.get("local_edge_fill", 0.0))
        if edge_fill >= float(script_args.detect_edge_fill):
            committed = True

        fx = env.get_scene_effects_for_env(0)
        slot_y = float(fx.get("slit_center_y", 0.0))
        local_pos = _local_from_world(env, env.p[0])
        if committed:
            if float(local_pos[0]) < 0.55:
                target_local = torch.tensor([0.45, slot_y * 1.28, 1.5], device=device)
            elif float(local_pos[0]) < 1.35:
                target_local = torch.tensor([1.28, slot_y * 1.18, 1.5], device=device)
            elif float(local_pos[0]) < 1.95:
                target_local = torch.tensor([2.05, slot_y, 1.5], device=device)
            else:
                target_local = torch.tensor([3.0, 0.0, 1.5], device=device)
        else:
            target_local = torch.tensor([1.65, 0.0, 1.5], device=device)
        target_world = _world_from_local(env, target_local)
        act, v_pred = _controller_step(env, target_world, script_args.speed, script_args.max_acc)

        trace.append({
            "method": method,
            "scene": scene,
            "slot": slot,
            "step": t,
            "committed": float(committed),
            "edge_fill": edge_fill,
            "score": float(cam_row.get("score", 0.0)),
            "power": float(setting.power),
            "exposure": float(setting.exposure),
            "gain": float(setting.gain),
            "local_x": float(local_pos[0].detach().cpu()),
            "local_y": float(local_pos[1].detach().cpu()),
            "local_z": float(local_pos[2].detach().cpu()),
            "target_local_x": float(target_local[0].detach().cpu()),
            "target_local_y": float(target_local[1].detach().cpu()),
            "clearance": float(clearance[0].detach().cpu()),
            "goal_dist": float(goal_dist[0].detach().cpu()),
        })
        env.run(act, ctl_dt=ctl_dt, v_pred=v_pred)

    vec = env.find_vec_to_nearest_pt()
    clearance = _min_clearance_from_vec(vec, 1)
    goal_dist = torch.norm(env.p_target - env.p, dim=-1)
    if bool((clearance <= float(args.collision_clearance)).any().item()):
        collided = True
        stop_reason = "collision"
    if bool((goal_dist < float(script_args.goal_radius)).any().item()):
        reached = True
        if not collided:
            stop_reason = "goal"
    success = bool(reached and not collided)
    summary = {
        "method": method,
        "scene": scene,
        "slot": slot,
        "success": float(success),
        "reached": float(reached),
        "collided": float(collided),
        "stop_reason": stop_reason,
        "final_goal_dist": float(goal_dist[0].detach().cpu()),
        "final_clearance": float(clearance[0].detach().cpu()),
        "steps": len(trace),
        "committed": float(committed),
    }
    return summary, trace


def _make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/slit_active_sensing.args")
    parser.add_argument("--out_dir", default="paper/experiment/results/simple_active_sensing_closed_loop")
    parser.add_argument("--scenarios", nargs="*", default=["glare", "specular", "dark"])
    parser.add_argument("--slots", nargs="*", default=["left", "right"])
    parser.add_argument("--methods", nargs="*", default=["fixed", "randfix", "oracle_grid", "diffopt"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=90)
    parser.add_argument("--control_freq", type=float, default=15.0)
    parser.add_argument("--speed", type=float, default=1.25)
    parser.add_argument("--max_acc", type=float, default=2.8)
    parser.add_argument("--goal_radius", type=float, default=0.35)
    parser.add_argument("--detect_edge_fill", type=float, default=0.55)
    parser.add_argument("--diffopt_init", default="0.50,0.35,0.25")
    parser.add_argument("--diffopt_steps", type=int, default=12)
    parser.add_argument("--diffopt_lr", type=float, default=0.18)
    parser.add_argument("--oracle_powers", default="0.18,0.32,0.50,0.70,0.90,0.96")
    parser.add_argument("--oracle_exposures", default="0.14,0.24,0.34,0.50,0.66,0.82,0.92")
    parser.add_argument("--oracle_gains", default="0.03,0.15,0.32,0.55,0.78,0.92")
    return parser


def main():
    parser = _make_arg_parser()
    script_args, project_overrides = parser.parse_known_args()
    out_dir = Path(script_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = _build_project_args(Path(script_args.config), project_overrides)
    args.random_rotation = False
    args.wandb_disabled = True
    set_global_seed(int(script_args.seed), getattr(args, "deterministic", False))
    device = torch.device(script_args.device)
    if device.type != "cuda":
        raise RuntimeError("this environment uses the CUDA simulator; run with --device cuda")

    scenes = _parse_scenes(script_args.scenarios)
    slots = _parse_slots(script_args.slots)
    rng = np.random.default_rng(int(script_args.seed))
    summaries = []
    traces = []
    for scene in scenes:
        for slot in slots:
            for method in script_args.methods:
                summary, trace = run_episode(str(method), scene, slot, args, script_args, device, rng)
                summaries.append(summary)
                traces.extend(trace)
                print(
                    f"[closed-loop] scene={scene} slot={slot} method={method} "
                    f"success={summary['success']:.0f} collided={summary['collided']:.0f} "
                    f"goal_dist={summary['final_goal_dist']:.3f} committed={summary['committed']:.0f}"
                )

    _write_csv(out_dir / "simple_closed_loop_summary.csv", summaries)
    _write_csv(out_dir / "simple_closed_loop_trace.csv", traces)
    print(f"[closed-loop] out_dir={out_dir}")


if __name__ == "__main__":
    main()

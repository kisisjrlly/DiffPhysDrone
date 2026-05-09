#!/usr/bin/env python3
"""Render camera sweeps on actual closed-loop rollout states.

This complements probe_opening_depth_views.py.  The opening probe forces the
camera to look at the slit center; this script first runs a checkpoint in the
environment, samples the real policy states along that trajectory, then restores
each state and renders multiple camera settings from the actual pose/attitude.
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
from random import normalvariate

import numpy as np
import torch
from torch.cuda.amp.autocast_mode import autocast

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
from eval import _collision_from_clearance, _min_clearance_from_vec  # noqa: E402
from model import Model  # noqa: E402
from rollout_ops import (  # noqa: E402
    build_local_frame,
    build_state_vector,
    compute_depth_fill_rate,
    compute_target_velocity,
    decode_action_direct,
    diff_depth_exposure_to_time,
    init_camera_params,
    render_sensors,
    select_policy_depth_obs,
    update_camera_params,
)
from train_utils import build_env  # noqa: E402
from tools.probe_opening_depth_views import (  # noqa: E402
    CameraSetting,
    _local_mask_metrics,
    _parse_camera_settings,
    _parse_scenes,
    _plot_topdown_overview,
    _to_float,
    _write_csv,
)


@dataclass
class RolloutCapture:
    scene: str
    step: int
    snapshot: dict[str, torch.Tensor]
    power: float
    exposure: float
    gain: float
    ctl_dt: float
    goal_dist: float
    clearance: float
    collided: bool


def _read_args_file(path: Path) -> list[str]:
    tokens: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            tokens.extend(shlex.split(line))
    return tokens


def _load_project_args(config_path: Path, overrides: list[str]):
    parser = build_parser()
    args = parser.parse_args(_read_args_file(config_path) + list(overrides))
    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.scenarios = parse_scenarios(args.scenarios)
    args.sun_glare_eval_slot = canonicalize_sun_glare_slot(args.sun_glare_eval_slot)
    args.batch_size = 1
    args.wandb_disabled = True
    args.vis_enable = False
    validate_args(args)
    return args


def _parse_steps(text: str | None) -> set[int] | None:
    if text is None or not str(text).strip():
        return None
    out: set[int] = set()
    for raw in str(text).replace(";", ",").split(","):
        raw = raw.strip()
        if raw:
            out.add(int(raw))
    return out


def _parse_float_targets(text: str | None) -> list[float] | None:
    if text is None or not str(text).strip():
        return None
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("--target_local_x did not contain any values")
    return vals


def _build_model(args, device):
    obs_dim = 7 if args.no_odom else 10
    return Model(
        obs_dim,
        3,
        include_camera_state_in_obs=args.include_camera_state_in_obs,
        use_policy_intent=False,
        depth_nn_width=args.depth_nn_width,
        depth_nn_height=args.depth_nn_height,
        depth_use_pipeline=args.depth_use_pipeline,
        depth_min_valid=args.depth_min_valid,
        depth_max_range=args.depth_max_range,
    ).to(device)


def _load_checkpoint(model, checkpoint: Path, device):
    state = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state)
    model.eval()


def _local_vec(env, vec: torch.Tensor) -> torch.Tensor:
    return torch.matmul(env.R_scene_T[0].to(vec.device, vec.dtype), vec)


def _snapshot_row_base(env, cap: RolloutCapture, setting: CameraSetting) -> dict:
    pos = env.p[0].detach()
    target = env.p_target[0].detach()
    pos_local = _local_vec(env, pos)
    target_local = _local_vec(env, target)
    R_cam_world = (env.R[0] @ env.R_cam[0]).detach()
    forward = R_cam_world[:, 0]
    look = pos + forward * 1.25
    look_local = _local_vec(env, look)
    fx = env.current_scene_effects or {}
    slot = env.get_scene_effects_for_env(0).get("slit_slot_name", "")
    goal_local_fx = fx.get("geometry_goal_local", None)
    if torch.is_tensor(goal_local_fx):
        goal_vec = goal_local_fx[0] if goal_local_fx.ndim >= 2 else goal_local_fx
        goal_local_x = float(goal_vec[0].detach().cpu())
        goal_local_y = float(goal_vec[1].detach().cpu())
    else:
        goal_local_x = _to_float(fx.get("geometry_goal_x"), 0.0)
        goal_local_y = 0.0

    return {
        "scene": cap.scene,
        "slot": slot,
        "step": int(cap.step),
        "setting": setting.name,
        "power": float(setting.power),
        "exposure": float(setting.exposure),
        "gain": float(setting.gain),
        "policy_power": float(cap.power),
        "policy_exposure": float(cap.exposure),
        "policy_gain": float(cap.gain),
        "ctl_dt": float(cap.ctl_dt),
        "goal_dist": float(cap.goal_dist),
        "clearance": float(cap.clearance),
        "collided": 1.0 if cap.collided else 0.0,
        "x": float(pos[0].detach().cpu()),
        "y": float(pos[1].detach().cpu()),
        "z": float(pos[2].detach().cpu()),
        "local_x": float(pos_local[0].detach().cpu()),
        "local_y": float(pos_local[1].detach().cpu()),
        "local_z": float(pos_local[2].detach().cpu()),
        "look_target_x": float(look[0].detach().cpu()),
        "look_target_y": float(look[1].detach().cpu()),
        "look_target_z": float(look[2].detach().cpu()),
        "look_target_local_x": float(look_local[0].detach().cpu()),
        "look_target_local_y": float(look_local[1].detach().cpu()),
        "look_target_local_z": float(look_local[2].detach().cpu()),
        "geometry_kind": str(fx.get("geometry_kind", "single_wall_slit")),
        "wall_x": _to_float(fx.get("geometry_wall_x"), 0.0),
        "back_wall_x": _to_float(fx.get("geometry_back_wall_x"), 0.0),
        "slit_center_y": _to_float(fx.get("slit_center_y"), 0.0),
        "slit_half_y": _to_float(fx.get("slit_half_y"), 0.0),
        "wall_half_z": _to_float(fx.get("geometry_wall_half_z"), 1.0),
        "goal_local_x": goal_local_x,
        "goal_local_y": goal_local_y,
        "camera_forward_x": float(forward[0].detach().cpu()),
        "camera_forward_y": float(forward[1].detach().cpu()),
        "camera_forward_z": float(forward[2].detach().cpu()),
    }


def _render_setting_at_capture(env, args, cap: RolloutCapture, setting: CameraSetting):
    env.restore_state(cap.snapshot)
    device = env.device
    power = torch.full((1,), float(setting.power), device=device)
    exposure = torch.full((1,), float(setting.exposure), device=device)
    gain = torch.full((1,), float(setting.gain), device=device)
    depth, quality = env.render_diff_depth(power, exposure, gain)
    debug = env.export_last_diff_depth_debug(0)
    scalars = debug.get("scalars", {})
    images = debug.get("images", {})
    depth_np = depth[0].detach().cpu().numpy()
    quality_np = None if quality is None else quality[0].detach().cpu().numpy()
    raw_np = images.get("raw_depth_map")
    scene_mask_np = images.get("scene_mask")
    row = _snapshot_row_base(env, cap, setting)
    row.update({
        "fill_rate": float(compute_depth_fill_rate(depth, args.depth_min_valid).reshape(-1)[0].detach().cpu()),
        "quality_mean": float(np.mean(quality_np)) if quality_np is not None else float("nan"),
        "invalid_rate": _to_float(scalars.get("invalid_rate"), 0.0),
        "scene_effect_mean": _to_float(scalars.get("scene_effect_mean"), 0.0),
        "scene_mask_mean": _to_float(scalars.get("scene_mask_mean"), 0.0),
        "slit_cue_mask_mean": _to_float(scalars.get("slit_cue_mask_mean"), 0.0),
        "key_cue_artifact_mean": _to_float(scalars.get("key_cue_artifact_mean"), 0.0),
        "front_wall_hit_mean": _to_float(scalars.get("front_wall_hit_mean"), 0.0),
        "back_wall_hit_mean": _to_float(scalars.get("back_wall_hit_mean"), 0.0),
        "scene_mask_on_back_wall_mean": _to_float(scalars.get("scene_mask_on_back_wall_mean"), 0.0),
        "glare_invalid_rate": _to_float(scalars.get("glare_invalid_rate"), 0.0),
        "glare_quality_mean": _to_float(scalars.get("glare_quality_mean"), 0.0),
    })
    row.update(_local_mask_metrics(
        depth_np,
        quality_np,
        scene_mask_np,
        args.depth_min_valid,
        raw_np,
        images.get("slit_cue_mask"),
    ))
    maps = {
        "depth": depth_np,
        "raw_depth": raw_np,
        "quality": quality_np,
        "invalid": images.get("invalid_mask"),
        "scene_effect": images.get("scene_effect_map"),
        "scene_mask": scene_mask_np,
        "front_wall_hit": images.get("front_wall_hit_mask"),
        "back_wall_hit": images.get("back_wall_hit_mask"),
        "slit_cue": images.get("slit_cue_mask"),
        "key_cue_artifact": images.get("key_cue_artifact_map"),
        "aperture_artifact": images.get("aperture_artifact_map"),
    }
    return row, maps


def _rollout_captures(env, args, model, scene: str, device, steps_filter: set[int] | None,
                      sample_every: int, max_samples: int) -> list[RolloutCapture]:
    env.reset(scene_name=scene)
    model.reset()
    h = None
    cam_h = None
    act_buffer = [env.act] * 2
    power, exposure, gain = init_camera_params(env, env.batch_size, device)
    captures: list[RolloutCapture] = []
    collided_cum = torch.zeros((env.batch_size,), dtype=torch.bool, device=device)
    use_amp = bool(args.amp and device.type == "cuda")

    for t in range(int(args.timesteps)):
        base_dt = normalvariate(1 / args.base_control_freq, 0.1 / args.base_control_freq)
        exposure_delay = float(diff_depth_exposure_to_time(exposure.mean().detach(), camera_semantics=env.cam_sem)) * 0.01
        ctl_dt = float(base_dt + exposure_delay)
        vec_now = env.find_vec_to_nearest_pt()
        min_clearance = _min_clearance_from_vec(vec_now, env)
        goal_dist = torch.norm(env.p_target - env.p, dim=-1).detach()
        collided_cum |= _collision_from_clearance(min_clearance, args)

        render_power, render_exposure, render_gain = power, exposure, gain
        depth_obs, _ = render_sensors(env, ctl_dt, power, exposure, gain, differentiable=False)

        want = (t in steps_filter) if steps_filter is not None else (t % max(int(sample_every), 1) == 0)
        if want and (max_samples <= 0 or len(captures) < int(max_samples)):
            captures.append(RolloutCapture(
                scene=scene,
                step=t,
                snapshot=env.save_state(),
                power=float(render_power[0].detach().cpu()),
                exposure=float(render_exposure[0].detach().cpu()),
                gain=float(render_gain[0].detach().cpu()),
                ctl_dt=ctl_dt,
                goal_dist=float(goal_dist[0].detach().cpu()),
                clearance=float(min_clearance[0].detach().cpu()),
                collided=bool(collided_cum[0].detach().cpu().item()),
            ))

        if bool(collided_cum.any().item()):
            break

        policy_depth_obs = select_policy_depth_obs(depth_obs, args.policy_depth_mode)
        target_v_raw = env.p_target - env.p.detach()
        R = build_local_frame(env)
        target_v = compute_target_velocity(target_v_raw, env)
        state, _local_v, camera_state, camera_motion_state = build_state_vector(
            env, target_v, R, power, exposure, gain,
            args.no_odom, args.include_camera_state_in_obs,
        )
        with autocast(enabled=use_amp):
            act_raw, cam_params, h, cam_h = model(
                state,
                h,
                depth_obs=policy_depth_obs,
                add_noise=False,
                cam_hx=cam_h,
                camera_state=camera_state,
                camera_motion_state=camera_motion_state,
            )
        act_final = decode_action_direct(act_raw.float(), R, env, env.batch_size, args.max_acc_cmd)
        power, exposure, gain, _ = update_camera_params(cam_params.float(), power, exposure, gain, env)
        act_buffer.append(act_final)
        env.run(act_buffer[t], ctl_dt, target_v_raw)

        vec_after = env.find_vec_to_nearest_pt()
        min_clearance_after = _min_clearance_from_vec(vec_after, env)
        collided_cum |= _collision_from_clearance(min_clearance_after, args)
        if bool(collided_cum.any().item()):
            break

    return captures


def _capture_local_x(env, cap: RolloutCapture) -> float:
    p = cap.snapshot["p"][0]
    local = torch.matmul(env.R_scene_T[0].to(p.device, p.dtype), p)
    return float(local[0].detach().cpu().item())


def _select_captures_by_local_x(env, captures: list[RolloutCapture], targets: list[float]) -> list[RolloutCapture]:
    if not captures:
        return []
    xs = np.asarray([_capture_local_x(env, cap) for cap in captures], dtype=np.float32)
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
    selected.sort(key=lambda cap: _capture_local_x(env, cap))
    return selected


def _masked_depth(depth: np.ndarray, min_valid: float):
    out = depth.astype(np.float32).copy()
    out[out <= float(min_valid) + 1e-6] = np.nan
    return out


def _center_profile(depth: np.ndarray, min_valid: float | None = None):
    arr = depth.astype(np.float32)
    if min_valid is not None:
        arr = _masked_depth(arr, min_valid)
    h = arr.shape[0]
    lo = max(0, h // 2 - 1)
    hi = min(h, h // 2 + 2)
    band = arr[lo:hi]
    valid = np.isfinite(band)
    sums = np.where(valid, band, 0.0).sum(axis=0)
    counts = valid.sum(axis=0)
    out = np.full((arr.shape[1],), np.nan, dtype=np.float32)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def _plot_capture_panel(path: Path, rendered: list[tuple[dict, dict]], args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[rollout-probe][warn] matplotlib unavailable: {exc}")
        return

    n = len(rendered)
    fig, axes = plt.subplots(n, 9, figsize=(29.0, max(2.25 * n, 3.2)), squeeze=False)
    depth_cmap = plt.cm.viridis.copy()
    depth_cmap.set_bad("black")
    first = rendered[0][0]
    for r, (row, maps) in enumerate(rendered):
        depth = maps["depth"]
        raw = maps.get("raw_depth")
        raw_show = np.zeros_like(depth) if raw is None else _masked_depth(raw, args.depth_min_valid)
        depth_show = _masked_depth(depth, args.depth_min_valid)
        quality = maps.get("quality")
        invalid = maps.get("invalid")
        effect = maps.get("scene_effect")
        key_cue = maps.get("key_cue_artifact")
        aperture = maps.get("aperture_artifact")
        mask = maps.get("scene_mask")

        _plot_topdown_overview(axes[r, 0], row, args)
        axes[r, 0].set_title("actual pose")
        axes[r, 1].imshow(raw_show, vmin=args.depth_min_valid, vmax=args.depth_max_range, cmap=depth_cmap)
        axes[r, 1].set_title("raw")
        axes[r, 2].imshow(depth_show, vmin=args.depth_min_valid, vmax=args.depth_max_range, cmap=depth_cmap)
        axes[r, 2].set_title("depth obs")
        axes[r, 3].imshow(np.zeros_like(depth) if quality is None else quality, vmin=0, vmax=1, cmap="magma")
        axes[r, 3].set_title(f"q {float(row.get('local_quality_mean', 0.0)):.2f}")
        axes[r, 4].imshow(np.zeros_like(depth) if invalid is None else invalid, vmin=0, vmax=1, cmap="gray")
        axes[r, 4].set_title(f"invalid {float(row.get('invalid_rate', 0.0)):.2f}")
        axes[r, 5].imshow(np.zeros_like(depth) if effect is None else effect, vmin=0, vmax=1, cmap="inferno")
        axes[r, 5].set_title(f"effect {float(row.get('scene_effect_mean', 0.0)):.2f}")
        axes[r, 6].imshow(np.zeros_like(depth) if key_cue is None else key_cue, vmin=0, vmax=1, cmap="plasma")
        axes[r, 6].set_title(
            f"artifact {float(row.get('key_cue_artifact_mean', 0.0)):.2f}\n"
            f"back leak {float(row.get('scene_mask_on_back_wall_mean', 0.0)):.3f}"
        )
        axes[r, 7].imshow(np.zeros_like(depth) if aperture is None else aperture, vmin=0, vmax=1, cmap="plasma")
        axes[r, 7].set_title("aperture")
        if raw is not None:
            axes[r, 8].plot(_center_profile(raw), label="raw", lw=1.6)
        axes[r, 8].plot(_center_profile(depth, args.depth_min_valid), label="obs", lw=1.4)
        axes[r, 8].set_ylim(args.depth_min_valid, args.depth_max_range)
        axes[r, 8].grid(True, alpha=0.25)
        axes[r, 8].legend(fontsize=7)
        axes[r, 8].set_title("center profile")
        for c in range(1, 8):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(
            f"{row['setting']}\n"
            f"p/e/g={row['power']:.2f}/{row['exposure']:.2f}/{row['gain']:.2f}\n"
            f"fill={row['local_fill']:.2f}",
            fontsize=9,
        )

    fig.suptitle(
        f"{first['scene']} step={int(first['step'])} slot={first['slot']} "
        f"local=({first['local_x']:.2f},{first['local_y']:.2f},{first['local_z']:.2f}) "
        f"goal_dist={first['goal_dist']:.2f}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_report(path: Path, rows: list[dict], captures_by_scene: dict[str, list[RolloutCapture]]):
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((str(row["scene"]), int(row["step"])), []).append(row)
    lines = [
        "# Rollout Depth View Probe",
        "",
        f"- rendered rows: {len(rows)}",
        f"- sampled rollout states: {len(grouped)}",
        "",
        "Sampled states:",
        "",
    ]
    for scene, caps in captures_by_scene.items():
        steps = ", ".join(str(c.step) for c in caps)
        lines.append(f"- {scene}: {len(caps)} states; steps={steps}")
    lines.extend([
        "",
        "Mean metrics by scene/setting:",
        "",
    ])
    if rows:
        groups: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            groups.setdefault((str(row["scene"]), str(row["setting"])), []).append(row)
        for (scene, setting), vals in sorted(groups.items()):
            mean_fill = sum(float(v["local_fill"]) for v in vals) / len(vals)
            mean_artifact = sum(float(v["key_cue_artifact_mean"]) for v in vals) / len(vals)
            mean_invalid = sum(float(v["invalid_rate"]) for v in vals) / len(vals)
            lines.append(
                f"- {scene} / {setting}: local_fill={mean_fill:.3f}, "
                f"artifact={mean_artifact:.3f}, invalid={mean_invalid:.3f}"
            )
    lines.extend([
        "",
        "Files:",
        "",
        "- `rollout_depth_probe_detail.csv`: one row per sampled state and camera setting.",
        "- `rollout_depth_probe_arrays.npz`: raw/depth/quality/invalid/effect/mask/cue arrays.",
        "- `panels/<scene>/step_XXX.png`: rendered panels from actual rollout pose/attitude.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_npz(path: Path, maps: dict[tuple[str, int, str], dict[str, np.ndarray | None]]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for (scene, step, setting), rendered in maps.items():
        prefix = f"{scene}_step{int(step):03d}_{setting}"
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
            "aperture_artifact",
        ]:
            arr = rendered.get(name)
            if arr is not None:
                arrays[f"{prefix}_{name}"] = np.asarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/slit_active_sensing_auto_ours.args")
    parser.add_argument("--checkpoint", default="checkpoint/2026-05-04-23-55-46/checkpoint0006.pth")
    parser.add_argument("--out_dir", default="paper/experiment/results/rollout_depth_views")
    parser.add_argument("--scenarios", nargs="*", default=["glare", "specular", "dark"])
    parser.add_argument("--steps", default=None, help="Comma-separated rollout steps to sample. Overrides --sample_every.")
    parser.add_argument("--target_local_x", default=None,
                        help="Comma-separated local x positions. If set, the script samples the full rollout and keeps the closest actual states.")
    parser.add_argument("--sample_every", type=int, default=8)
    parser.add_argument("--max_samples_per_scene", type=int, default=8)
    parser.add_argument("--camera_settings", default=None)
    parser.add_argument("--sensor_impl", default="cuda", choices=["cuda", "python"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_panels", type=int, default=0)
    return parser


def main():
    parser = _make_arg_parser()
    script_args, project_overrides = parser.parse_known_args()
    out_dir = Path(script_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    project_args = _load_project_args(Path(script_args.config), project_overrides)
    project_args.diff_sensor_impl["diff_depth"] = str(script_args.sensor_impl)
    if script_args.seed is not None:
        project_args.seed = int(script_args.seed)
    set_global_seed(project_args.seed, project_args.deterministic)

    device = torch.device(script_args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    checkpoint = Path(script_args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model = _build_model(project_args, device)
    _load_checkpoint(model, checkpoint, device)
    scenes = _parse_scenes(script_args.scenarios)
    settings_base = _parse_camera_settings(script_args.camera_settings, project_args)
    steps_filter = _parse_steps(script_args.steps)
    target_local_x = _parse_float_targets(script_args.target_local_x)

    detail_rows: list[dict] = []
    detail_maps: dict[tuple[str, int, str], dict[str, np.ndarray | None]] = {}
    captures_by_scene: dict[str, list[RolloutCapture]] = {}
    panel_count = 0

    with torch.no_grad():
        for scene in scenes:
            cond_args = copy.deepcopy(project_args)
            cond_args.scenarios = [scene]
            env = build_env(1, cond_args, device, eval_mode=True)
            caps = _rollout_captures(
                env,
                cond_args,
                model,
                scene,
                device,
                steps_filter,
                1 if target_local_x is not None else int(script_args.sample_every),
                0 if target_local_x is not None else int(script_args.max_samples_per_scene),
            )
            if target_local_x is not None:
                caps = _select_captures_by_local_x(env, caps, target_local_x)
            captures_by_scene[scene] = caps
            for cap in caps:
                settings = [CameraSetting("policy_actual", cap.power, cap.exposure, cap.gain)] + settings_base
                rendered = [_render_setting_at_capture(env, cond_args, cap, setting) for setting in settings]
                detail_rows.extend(row for row, _ in rendered)
                for row, maps in rendered:
                    detail_maps[(str(row["scene"]), int(row["step"]), str(row["setting"]))] = maps
                if script_args.plots and (
                    int(script_args.max_panels) <= 0 or panel_count < int(script_args.max_panels)
                ):
                    panel_path = out_dir / "panels" / scene / f"step_{cap.step:03d}.png"
                    _plot_capture_panel(panel_path, rendered, cond_args)
                    panel_count += 1

    _write_csv(out_dir / "rollout_depth_probe_detail.csv", detail_rows)
    _save_npz(out_dir / "rollout_depth_probe_arrays.npz", detail_maps)
    _write_report(out_dir / "report.md", detail_rows, captures_by_scene)
    print(f"[rollout-probe] scenes={scenes}")
    print(f"[rollout-probe] sampled states={sum(len(v) for v in captures_by_scene.values())}")
    print(f"[rollout-probe] rows={len(detail_rows)} panels={panel_count}")
    print(f"[rollout-probe] out_dir={out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Probe whether diff_depth observations actually change when camera parameters change.

The script renders the same drone state multiple times with different
power/exposure/gain settings.  It can either sample states from a checkpoint
rollout, or use hand-picked positions in the current scene.
"""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.cuda.amp.autocast_mode import autocast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    build_parser,
    canonicalize_sun_glare_level,
    parse_diff_sensor_impl,
    parse_scenarios,
    parse_sun_glare_levels,
    set_global_seed,
    validate_args,
)
from lqr import build_velocity_tracking_linear_system, solve_batched_dlqr  # noqa: E402
from model import Model  # noqa: E402
from rollout_ops import (  # noqa: E402
    build_local_frame,
    build_state_vector,
    compute_depth_fill_rate,
    compute_target_velocity,
    decode_action_direct,
    decode_action_lqr,
    diff_depth_exposure_to_time,
    diff_depth_fill_softness,
    init_camera_params,
    render_sensors,
    select_policy_depth_obs,
    update_camera_params,
)
from train_utils import build_env, make_yaw_drift_R  # noqa: E402


@dataclass(frozen=True)
class CameraSetting:
    name: str
    power: float
    exposure: float
    gain: float
    use_policy: bool = False


def _read_args_file(path: Path) -> list[str]:
    tokens: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            tokens.extend(shlex.split(line))
    return tokens


def _build_project_args(config_path: Path, overrides: list[str]):
    parser = build_parser()
    tokens = _read_args_file(config_path) + list(overrides)
    args = parser.parse_args(tokens)
    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.scenarios = parse_scenarios(args.scenarios)
    args.sun_glare_levels = parse_sun_glare_levels(args.sun_glare_levels)
    if args.sun_glare_eval_level is not None:
        args.sun_glare_eval_level = canonicalize_sun_glare_level(args.sun_glare_eval_level)
    args.batch_size = 1
    args.wandb_disabled = True
    args.vis_enable = False
    validate_args(args)
    return args


def _checkpoint_state_input_dim(ckpt_path: Path) -> int | None:
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    weight = state_dict.get("v_proj.weight")
    if weight is None or not hasattr(weight, "shape") or len(weight.shape) != 2:
        return None
    return int(weight.shape[1])


def _match_args_to_checkpoint(args, ckpt_path: Path):
    input_dim = _checkpoint_state_input_dim(ckpt_path)
    if input_dim is None:
        return args
    obs_dim = 7 if args.no_odom else 10
    if input_dim == obs_dim:
        expected_include_camera = False
    elif input_dim == obs_dim + 3:
        expected_include_camera = True
    else:
        raise ValueError(
            f"{ckpt_path} has v_proj input dim={input_dim}, "
            f"but expected {obs_dim} or {obs_dim + 3}"
        )
    if bool(args.include_camera_state_in_obs) != expected_include_camera:
        print(
            f"[probe][info] checkpoint requires "
            f"include_camera_state_in_obs={expected_include_camera}; "
            f"override config value {args.include_camera_state_in_obs}."
        )
        args.include_camera_state_in_obs = expected_include_camera
    return args


def _make_model(args, device: torch.device) -> Model:
    dim_obs = 7 if args.no_odom else 10
    return Model(
        dim_obs=dim_obs,
        dim_action=9 if args.policy_output_intent else 6,
        include_camera_state_in_obs=args.include_camera_state_in_obs,
        use_policy_intent=args.policy_output_intent,
        intent_dim=9,
        depth_nn_width=args.depth_nn_width,
        depth_nn_height=args.depth_nn_height,
        depth_use_pipeline=args.depth_use_pipeline,
        depth_min_valid=args.depth_min_valid,
        depth_max_range=args.depth_max_range,
    ).to(device)


def _parse_float_triplet(text: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"camera setting must be p,e,g, got: {text}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _default_camera_settings(args, include_policy: bool) -> list[CameraSetting]:
    settings: list[CameraSetting] = []
    if include_policy:
        settings.append(CameraSetting("policy", 0.0, 0.0, 0.0, use_policy=True))
    settings.extend([
        CameraSetting(
            "baseline",
            float(args.cam_power_baseline),
            float(args.fixed_camera_exposure),
            float(args.fixed_camera_gain),
        ),
        CameraSetting("lowP_lowE_lowG", 0.55, 0.25, 0.05),
        CameraSetting("lowP_midE_lowG", 0.55, 0.45, 0.05),
        CameraSetting("highP_lowE_lowG", 0.90, 0.25, 0.05),
        CameraSetting("highP_midE_midG", 0.90, 0.45, 0.30),
        CameraSetting("midP_highE_midG", 0.70, 0.65, 0.30),
        CameraSetting("lowP_highE_highG", 0.55, 0.75, 0.60),
    ])
    return settings


def _parse_camera_settings(spec: str | None, args, include_policy: bool) -> list[CameraSetting]:
    if not spec:
        return _default_camera_settings(args, include_policy)
    out: list[CameraSetting] = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if item == "policy":
            out.append(CameraSetting("policy", 0.0, 0.0, 0.0, use_policy=True))
            continue
        name, sep, values = item.partition(":")
        if not sep:
            raise ValueError(
                "--camera_settings entries must look like name:p,e,g; "
                f"got {item!r}"
            )
        p, e, g = _parse_float_triplet(values)
        out.append(CameraSetting(name.strip(), p, e, g))
    if not out:
        raise ValueError("--camera_settings produced no settings")
    return out


def _parse_int_list(text: str | None) -> list[int] | None:
    if text is None or not text.strip():
        return None
    return sorted({int(x.strip()) for x in text.split(",") if x.strip()})


def _probe_steps(timesteps: int, explicit: str | None, count: int) -> set[int]:
    parsed = _parse_int_list(explicit)
    if parsed is not None:
        return {max(0, min(int(timesteps) - 1, x)) for x in parsed}
    count = max(1, min(int(count), int(timesteps)))
    vals = np.linspace(0, int(timesteps) - 1, count)
    return {int(round(x)) for x in vals}


def _parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("empty float list")
    return vals


def _capture_snapshot(env, episode: int, probe_idx: int, step: int,
                      power: torch.Tensor, exposure: torch.Tensor, gain: torch.Tensor):
    return {
        "episode": int(episode),
        "probe_idx": int(probe_idx),
        "step": int(step),
        "p": env.p.detach().clone(),
        "v": env.v.detach().clone(),
        "a": env.a.detach().clone(),
        "R": env.R.detach().clone(),
        "R_old": env.R_old.detach().clone(),
        "act": env.act.detach().clone(),
        "p_target": env.p_target.detach().clone(),
        "policy_power": power.detach().clone(),
        "policy_exposure": exposure.detach().clone(),
        "policy_gain": gain.detach().clone(),
    }


def _restore_snapshot(env, snap):
    env.p = snap["p"].clone()
    env.v = snap["v"].clone()
    env.a = snap["a"].clone()
    env.R = snap["R"].clone()
    env.R_old = snap["R_old"].clone()
    env.act = snap["act"].clone()
    env.p_target = snap["p_target"].clone()


def _collect_policy_snapshots(model, env, args, device: torch.device,
                              episode: int, scene_name: str | None,
                              scene_variant: str | None,
                              steps_to_probe: set[int]):
    B = env.batch_size
    use_amp = bool(args.amp and device.type == "cuda")
    env.reset(scene_name=scene_name, scene_variant=scene_variant)
    model.reset()

    h = None
    act_buffer = [env.act] * 2
    target_v_raw = env.p_target - env.p
    yaw_drift_R = make_yaw_drift_R(B, device) if args.yaw_drift else None
    power, exposure, gain = init_camera_params(env, B, device)
    A_lqr, B_lqr = build_velocity_tracking_linear_system(B, 1 / 15, device)
    depth_input_mode = str(getattr(args, "policy_depth_mode", "depth")).strip().lower()

    snapshots = []
    probe_idx = 0
    for t in range(args.timesteps):
        ctl_dt = 1.0 / max(float(args.base_control_freq), 1e-6)
        exposure_delay = float(diff_depth_exposure_to_time(
            exposure.mean().detach(),
            camera_semantics=env.cam_sem,
        )) * 0.01
        ctl_dt += exposure_delay

        depth_obs, _ = render_sensors(env, ctl_dt, power, exposure, gain, differentiable=False)
        if t in steps_to_probe:
            snapshots.append(_capture_snapshot(env, episode, probe_idx, t, power, exposure, gain))
            probe_idx += 1

        policy_depth_obs = select_policy_depth_obs(depth_obs, depth_input_mode)
        if args.yaw_drift and yaw_drift_R is not None:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ yaw_drift_R, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()

        env.run(act_buffer[t], ctl_dt, target_v_raw)
        R = build_local_frame(env)
        target_v = compute_target_velocity(target_v_raw, env)
        state, local_v = build_state_vector(
            env, target_v, R, power, exposure, gain,
            args.no_odom, args.include_camera_state_in_obs,
        )

        if args.policy_output_intent:
            with autocast(enabled=use_amp):
                act_raw, cam_params, h, intent = model(
                    state, h, return_intent=True, depth_obs=policy_depth_obs, add_noise=False)
            intent = intent.float()
        else:
            with autocast(enabled=use_amp):
                act_raw, cam_params, h = model(
                    state, h, depth_obs=policy_depth_obs, add_noise=False)
            intent = None
        act_raw = act_raw.float()
        cam_params = cam_params.float()
        power, exposure, gain, _ = update_camera_params(cam_params, power, exposure, gain, env)

        if args.use_dmpc and args.policy_output_intent and intent is not None:
            vec_now = env.find_vec_to_nearest_pt()
            act_final, _ = decode_action_lqr(
                intent, R, env, local_v, B,
                A_lqr, B_lqr,
                args.lqr_horizon, args.lqr_reg, args.max_acc_cmd,
                args.inject_depth_into_lqr, args.depth_safe_dist, args.depth_repel_gain,
                vec_now, solve_batched_dlqr,
            )
        else:
            act_final, _ = decode_action_direct(act_raw, R, env, B, args.max_acc_cmd)
        act_buffer.append(act_final)

    return snapshots


def _manual_snapshots(env, args, episode: int, scene_name: str | None,
                      scene_variant: str | None, xs: Iterable[float]):
    env.reset(scene_name=scene_name, scene_variant=scene_variant)
    B = env.batch_size
    device = env.device
    power = torch.full((B,), float(args.cam_power_baseline), device=device)
    exposure = torch.full((B,), float(args.fixed_camera_exposure), device=device)
    gain = torch.full((B,), float(args.fixed_camera_gain), device=device)

    start_y = float(env.p[0, 1].detach().cpu().item())
    slot_y = float((env.current_scene_effects or {}).get("decision_open_slot_y", start_y))
    x_values = list(xs)
    x_min = min(x_values)
    x_max = max(x_values)
    denom = max(x_max - x_min, 1e-6)

    snapshots = []
    for idx, x in enumerate(x_values):
        alpha = float(np.clip((x - x_min) / denom, 0.0, 1.0))
        env.p[:, 0] = float(x)
        env.p[:, 1] = (1.0 - alpha) * start_y + alpha * slot_y
        env.p[:, 2] = 1.5
        env.v.zero_()
        env.a.zero_()
        snapshots.append(_capture_snapshot(env, episode, idx, -1, power, exposure, gain))
    return snapshots


def _to_float(value, default: float = float("nan")) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return float(value.reshape(-1)[0].detach().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _render_at_snapshot(env, args, snap, setting: CameraSetting):
    _restore_snapshot(env, snap)
    device = env.device
    if setting.use_policy:
        power = snap["policy_power"].to(device)
        exposure = snap["policy_exposure"].to(device)
        gain = snap["policy_gain"].to(device)
        p_val = float(power[0].detach().cpu().item())
        e_val = float(exposure[0].detach().cpu().item())
        g_val = float(gain[0].detach().cpu().item())
    else:
        p_val, e_val, g_val = float(setting.power), float(setting.exposure), float(setting.gain)
        power = torch.full((env.batch_size,), p_val, device=device)
        exposure = torch.full((env.batch_size,), e_val, device=device)
        gain = torch.full((env.batch_size,), g_val, device=device)

    depth, quality = env.render_diff_depth(power, exposure, gain)
    fill = compute_depth_fill_rate(depth, min_valid_depth=args.depth_min_valid)
    fill_soft = compute_depth_fill_rate(
        depth,
        min_valid_depth=args.depth_min_valid,
        softness=diff_depth_fill_softness(args.depth_min_valid),
    )
    debug = env.export_last_diff_depth_debug(0)
    scalars = debug.get("scalars", {})
    images = debug.get("images", {})

    depth_np = depth[0].detach().cpu().numpy()
    quality_np = None if quality is None else quality[0].detach().cpu().numpy()
    invalid_np = images.get("invalid_mask")
    scene_effect_np = images.get("scene_effect_map")

    valid = depth_np > (float(args.depth_min_valid) + 1e-6)
    valid_depth = depth_np[valid]
    row = {
        "episode": snap["episode"],
        "probe_idx": snap["probe_idx"],
        "step": snap["step"],
        "x": float(snap["p"][0, 0].detach().cpu().item()),
        "y": float(snap["p"][0, 1].detach().cpu().item()),
        "z": float(snap["p"][0, 2].detach().cpu().item()),
        "policy_power": float(snap["policy_power"][0].detach().cpu().item()),
        "policy_exposure": float(snap["policy_exposure"][0].detach().cpu().item()),
        "policy_gain": float(snap["policy_gain"][0].detach().cpu().item()),
        "setting": setting.name,
        "power": p_val,
        "exposure": e_val,
        "gain": g_val,
        "fill_rate": float(fill.reshape(-1)[0].detach().cpu().item()),
        "fill_rate_soft": float(fill_soft.reshape(-1)[0].detach().cpu().item()),
        "valid_depth_mean": float(valid_depth.mean()) if valid_depth.size else 0.0,
        "valid_depth_std": float(valid_depth.std()) if valid_depth.size else 0.0,
        "quality_mean": float(np.mean(quality_np)) if quality_np is not None else float("nan"),
        "invalid_rate": _to_float(scalars.get("invalid_rate")),
        "scene_effect_mean": _to_float(scalars.get("scene_effect_mean"), 0.0),
        "sun_mask_mean": _to_float(scalars.get("sun_mask_mean"), 0.0),
        "sun_los_mean": _to_float(scalars.get("sun_los_mean"), 0.0),
        "hazard_los_mean": _to_float(scalars.get("hazard_los_mean"), 0.0),
        "glare_quality_mean": _to_float(scalars.get("glare_quality_mean"), 0.0),
        "glare_invalid_rate": _to_float(scalars.get("glare_invalid_rate"), 0.0),
        "washout_mean": _to_float(scalars.get("washout_mean"), 0.0),
        "ambient_ir_mean": _to_float(scalars.get("ambient_ir_mean"), 0.0),
        "signal_active_mean": _to_float(scalars.get("signal_active_mean"), 0.0),
        "signal_passive_mean": _to_float(scalars.get("signal_passive_mean"), 0.0),
        "spec_bloom_mean": _to_float(scalars.get("spec_bloom_mean"), 0.0),
        "motion_blur_mean": _to_float(scalars.get("motion_blur_mean"), 0.0),
        "decision_open_slot_id": _to_float(scalars.get("decision_open_slot_id"), 0.0),
        "glare_level_id": _to_float(scalars.get("glare_level_id"), -1.0),
        "sensor_regime_id": _to_float(scalars.get("sensor_regime_id"), -1.0),
    }
    maps = {
        "depth": depth_np,
        "quality": quality_np,
        "invalid": invalid_np,
        "scene_effect": scene_effect_np,
    }
    return row, maps


def _add_reference_diffs(rendered: list[tuple[dict, dict]], min_valid: float):
    ref_row, ref_maps = rendered[0]
    ref_depth = ref_maps["depth"]
    ref_quality = ref_maps["quality"]
    ref_valid = ref_depth > (float(min_valid) + 1e-6)
    for row, maps in rendered:
        depth = maps["depth"]
        valid = depth > (float(min_valid) + 1e-6)
        union = np.logical_or(ref_valid, valid)
        if np.any(union):
            abs_diff = np.abs(depth - ref_depth)
            row["depth_mae_vs_ref"] = float(abs_diff[union].mean())
            row["depth_changed_px_rate_5cm"] = float((abs_diff[union] > 0.05).mean())
        else:
            row["depth_mae_vs_ref"] = 0.0
            row["depth_changed_px_rate_5cm"] = 0.0
        row["valid_xor_rate_vs_ref"] = float(np.logical_xor(ref_valid, valid).mean())
        quality = maps["quality"]
        if quality is not None and ref_quality is not None:
            row["quality_mae_vs_ref"] = float(np.abs(quality - ref_quality).mean())
        else:
            row["quality_mae_vs_ref"] = float("nan")
        row["reference_setting"] = ref_row["setting"]


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_plots(out_dir: Path, snap, rendered: list[tuple[dict, dict]], args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[probe][warn] matplotlib unavailable, skip plots: {exc}")
        return

    n = len(rendered)
    fig, axes = plt.subplots(n, 4, figsize=(14, max(2.2 * n, 3.0)), squeeze=False)
    depth_cmap = plt.cm.viridis.copy()
    depth_cmap.set_bad("black")
    for r, (row, maps) in enumerate(rendered):
        depth = maps["depth"].astype(np.float32)
        depth_show = depth.copy()
        depth_show[depth <= float(args.depth_min_valid) + 1e-6] = np.nan
        quality = maps["quality"]
        invalid = maps["invalid"]
        scene_effect = maps["scene_effect"]

        axes[r, 0].imshow(depth_show, vmin=args.depth_min_valid, vmax=args.depth_max_range, cmap=depth_cmap)
        axes[r, 0].set_title(f"{row['setting']} depth")
        axes[r, 1].imshow(np.zeros_like(depth) if quality is None else quality, vmin=0, vmax=1, cmap="magma")
        axes[r, 1].set_title(f"quality fill={row['fill_rate']:.2f}")
        axes[r, 2].imshow(np.zeros_like(depth) if invalid is None else invalid, vmin=0, vmax=1, cmap="gray")
        axes[r, 2].set_title(f"invalid={row['invalid_rate']:.2f}")
        axes[r, 3].imshow(np.zeros_like(depth) if scene_effect is None else scene_effect, vmin=0, vmax=1, cmap="inferno")
        axes[r, 3].set_title(f"effect={row['scene_effect_mean']:.2f}")
        for c in range(4):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(f"p/e/g={row['power']:.2f}/{row['exposure']:.2f}/{row['gain']:.2f}")

    fig.suptitle(
        f"episode={snap['episode']} probe={snap['probe_idx']} step={snap['step']} "
        f"pos=({rendered[0][0]['x']:.2f},{rendered[0][0]['y']:.2f},{rendered[0][0]['z']:.2f})"
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"probe_ep{snap['episode']:02d}_idx{snap['probe_idx']:02d}_step{snap['step']:04d}.png", dpi=150)
    plt.close(fig)


def _state_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["episode"]), int(row["probe_idx"])), []).append(row)
    out = []
    for (episode, probe_idx), items in sorted(grouped.items()):
        fill_vals = [float(x["fill_rate"]) for x in items]
        quality_vals = [float(x["quality_mean"]) for x in items if math.isfinite(float(x["quality_mean"]))]
        depth_diff_vals = [float(x.get("depth_mae_vs_ref", 0.0)) for x in items]
        valid_xor_vals = [float(x.get("valid_xor_rate_vs_ref", 0.0)) for x in items]
        first = items[0]
        out.append({
            "episode": episode,
            "probe_idx": probe_idx,
            "step": first["step"],
            "x": first["x"],
            "y": first["y"],
            "z": first["z"],
            "setting_count": len(items),
            "fill_min": min(fill_vals),
            "fill_max": max(fill_vals),
            "fill_range": max(fill_vals) - min(fill_vals),
            "quality_min": min(quality_vals) if quality_vals else float("nan"),
            "quality_max": max(quality_vals) if quality_vals else float("nan"),
            "quality_range": (max(quality_vals) - min(quality_vals)) if quality_vals else float("nan"),
            "max_depth_mae_vs_ref": max(depth_diff_vals),
            "max_valid_xor_rate_vs_ref": max(valid_xor_vals),
            "scene_effect_mean_ref": first.get("scene_effect_mean", 0.0),
            "reference_setting": first.get("reference_setting", first.get("setting")),
        })
    return out


def _write_report(path: Path, detail_rows: list[dict], state_rows: list[dict], settings: list[CameraSetting]):
    if not detail_rows:
        path.write_text("# Camera Depth Probe\n\nNo rows collected.\n", encoding="utf-8")
        return
    fill_ranges = [float(r["fill_range"]) for r in state_rows]
    quality_ranges = [
        float(r["quality_range"]) for r in state_rows
        if math.isfinite(float(r["quality_range"]))
    ]
    depth_ranges = [float(r["max_depth_mae_vs_ref"]) for r in state_rows]
    valid_xor = [float(r["max_valid_xor_rate_vs_ref"]) for r in state_rows]

    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    lines = [
        "# Camera Depth Probe",
        "",
        f"- states: {len(state_rows)}",
        f"- settings: {', '.join(s.name for s in settings)}",
        f"- mean fill_range across settings: {mean(fill_ranges):.4f}",
        f"- mean quality_range across settings: {mean(quality_ranges):.4f}" if quality_ranges else "- mean quality_range across settings: n/a",
        f"- mean max depth MAE vs reference: {mean(depth_ranges):.4f} m",
        f"- mean max valid-mask XOR vs reference: {mean(valid_xor):.4f}",
        "",
        "Interpretation guide:",
        "",
        "- If fill_range, quality_range, depth MAE, and valid-mask XOR are all near zero, this state is insensitive to camera controls.",
        "- If only quality changes but valid depth/fill barely changes, the policy may still see nearly the same depth cue.",
        "- If fill/valid/depth maps change strongly near the opening, camera control has a real observable channel to exploit.",
        "",
        "Files:",
        "",
        "- `camera_probe_detail.csv`: one row per state and camera setting.",
        "- `camera_probe_state_summary.csv`: one row per sampled state.",
        "- `probe_ep*_idx*_step*.png`: depth/quality/invalid/effect panels.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_arg_parser():
    parser = argparse.ArgumentParser(
        description="Render identical drone states with multiple camera settings."
    )
    parser.add_argument("--config", default="configs/paper_final_full.args",
                        help="Project .args file to load before optional overrides.")
    parser.add_argument("--resume", default=None,
                        help="Optional checkpoint. If omitted, use manual probe positions.")
    parser.add_argument("--out_dir", default="paper/experiment/results/camera_param_probe",
                        help="Directory for CSV/PNG outputs.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--scene_name", default="sun_glare")
    parser.add_argument("--scene_variant", default=None,
                        help="Optional fixed scene variant, e.g. l2.")
    parser.add_argument("--probe_steps", default=None,
                        help="Comma-separated rollout steps to probe, e.g. 0,8,16,24.")
    parser.add_argument("--num_probe_states", type=int, default=8,
                        help="Used when --probe_steps is not provided.")
    parser.add_argument("--manual_xs", default="-2.8,-1.8,-0.8,0.0,0.55,0.9,1.2,1.45",
                        help="Manual x positions used when --resume is omitted.")
    parser.add_argument("--camera_settings", default=None,
                        help="Optional semicolon list: name:p,e,g;name2:p,e,g. Use 'policy' for rollout policy camera.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force_python_sensor", action=argparse.BooleanOptionalAction, default=True,
                        help="Force diff_depth=python so quality/debug maps are available.")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main():
    script_parser = _make_arg_parser()
    script_args, project_overrides = script_parser.parse_known_args()
    out_dir = Path(script_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    project_args = _build_project_args(Path(script_args.config), project_overrides)
    if script_args.force_python_sensor:
        project_args.diff_sensor_impl["diff_depth"] = "python"
    set_global_seed(project_args.seed, project_args.deterministic)

    device = torch.device(script_args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    ckpt_path = Path(script_args.resume).expanduser() if script_args.resume else None
    model = None
    if ckpt_path is not None:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        project_args = _match_args_to_checkpoint(project_args, ckpt_path)
        model = _make_model(project_args, device)
        state_dict = torch.load(str(ckpt_path), map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print("[probe][warn] missing keys:", missing)
        if unexpected:
            print("[probe][warn] unexpected keys:", unexpected)
        model.eval()

    env = build_env(1, project_args, device, eval_mode=True)
    settings = _parse_camera_settings(script_args.camera_settings, project_args, include_policy=model is not None)
    detail_rows: list[dict] = []

    with torch.no_grad():
        for ep in range(int(script_args.episodes)):
            if model is not None:
                steps = _probe_steps(project_args.timesteps, script_args.probe_steps, script_args.num_probe_states)
                snapshots = _collect_policy_snapshots(
                    model, env, project_args, device, ep,
                    script_args.scene_name, script_args.scene_variant, steps,
                )
            else:
                xs = _parse_float_list(script_args.manual_xs)
                snapshots = _manual_snapshots(
                    env, project_args, ep,
                    script_args.scene_name, script_args.scene_variant, xs,
                )

            for snap in snapshots:
                rendered = [_render_at_snapshot(env, project_args, snap, setting) for setting in settings]
                _add_reference_diffs(rendered, project_args.depth_min_valid)
                detail_rows.extend(row for row, _ in rendered)
                if script_args.plots:
                    _make_plots(out_dir, snap, rendered, project_args)

    state_rows = _state_summary(detail_rows)
    _write_csv(out_dir / "camera_probe_detail.csv", detail_rows)
    _write_csv(out_dir / "camera_probe_state_summary.csv", state_rows)
    _write_report(out_dir / "report.md", detail_rows, state_rows, settings)

    print(f"[probe] wrote {len(detail_rows)} detail rows")
    print(f"[probe] out_dir: {out_dir}")
    if state_rows:
        fill_range = np.mean([float(r["fill_range"]) for r in state_rows])
        depth_mae = np.mean([float(r["max_depth_mae_vs_ref"]) for r in state_rows])
        valid_xor = np.mean([float(r["max_valid_xor_rate_vs_ref"]) for r in state_rows])
        print(
            "[probe] mean sensitivity: "
            f"fill_range={fill_range:.4f}, "
            f"max_depth_mae={depth_mae:.4f}m, "
            f"valid_xor={valid_xor:.4f}"
        )


if __name__ == "__main__":
    main()

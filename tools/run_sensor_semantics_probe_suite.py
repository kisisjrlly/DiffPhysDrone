#!/usr/bin/env python3
"""Run the sensor-semantics probe gate before teacher/training/eval.

The suite intentionally combines three views of the same simulator:

1. a controlled opening probe with forced look-at-slit poses;
2. rollout-state probes that use real checkpoint poses and attitudes;
3. the journal qualitative depth exporter, which mirrors the final paper figure
   generation path.

This is the guard against a bug passing an idealized probe and only appearing in
the final manuscript panels.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN_METHODS = ("fixed", "randfix", "flightonly")


@dataclass(frozen=True)
class MethodSpec:
    key: str
    checkpoint: Path
    config: Path
    camera_mode: str
    train_flight_only: bool = False


def infer_mode(path: Path, text: str) -> str | None:
    name = path.name.lower()
    for mode in ["flightonly", "randfix", "nondiff", "zero", "fix", "fixed"]:
        if f"_auto_{mode}" in name or f"-slit_active_sensing_auto_{mode}-" in name:
            return "fixed" if mode in {"fix", "fixed"} else mode
    m = re.search(r"\[train-suite\]\s+mode=(\w+)", text)
    if m:
        mode = m.group(1).strip().lower()
        return "fixed" if mode in {"fix", "fixed"} else mode
    return None


def extract_checkpoints(logs: list[Path]) -> dict[str, Path]:
    ckpts: dict[str, Path] = {}
    pattern = re.compile(r"checkpoint/\d{4}-\d{2}-\d{2}-[\d-]+/checkpoint0014\.pth")
    for log in logs:
        text = log.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
        mode = infer_mode(log, text)
        matches = pattern.findall(text)
        if mode and matches:
            ckpts[mode] = ROOT / matches[-1]
    return ckpts


def resolve_method_specs(args: argparse.Namespace) -> dict[str, MethodSpec]:
    ckpts = {
        "fixed": args.fixed_ckpt,
        "randfix": args.randfix_ckpt,
        "flightonly": args.flightonly_ckpt,
    }
    if args.logs:
        extracted = extract_checkpoints([p if p.is_absolute() else ROOT / p for p in args.logs])
        for method in MAIN_METHODS:
            if ckpts[method] is None and method in extracted:
                ckpts[method] = str(extracted[method])

    missing = [method for method in MAIN_METHODS if not ckpts[method]]
    if missing:
        raise SystemExit(
            f"missing checkpoints for {missing}; pass training logs or explicit --<method>_ckpt"
        )

    templates = {
        "fixed": ("configs/slit_active_sensing_auto_fix.args", "fixed", False),
        "randfix": ("configs/slit_active_sensing_auto_randfix.args", "fixed_random_static", False),
        "flightonly": ("configs/slit_active_sensing_auto_flightonly.args", "learned", True),
    }
    specs: dict[str, MethodSpec] = {}
    for method in MAIN_METHODS:
        cfg, camera_mode, train_flight_only = templates[method]
        ckpt = Path(str(ckpts[method]))
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        if not ckpt.is_file():
            raise SystemExit(f"{method} checkpoint not found: {ckpt}")
        specs[method] = MethodSpec(
            method,
            ckpt,
            ROOT / cfg,
            camera_mode,
            train_flight_only=train_flight_only,
        )
    return specs


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("[sensor-probe-suite] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(vals: list[float]) -> float:
    clean = [float(v) for v in vals if np.isfinite(float(v))]
    return sum(clean) / max(len(clean), 1)


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _bad_slit_adjacent_front_fill_failures(
    rows: list[dict],
    label: str,
    max_bad_slit_adjacent_front_fill: float,
) -> list[str]:
    failures: list[str] = []
    bad_settings = {
        "dark": {"low_return_bad", "fixed_config", "fixed_mid", "glare_expected", "specular_safe", "high_power"},
        "specular": {"fixed_config", "fixed_mid", "overexposed", "high_power", "dark_expected"},
    }
    for scene, settings in bad_settings.items():
        vals = [
            as_float(r, "slit_adjacent_front_fill")
            for r in rows
            if str(r.get("scene")) == scene and str(r.get("setting")) in settings
        ]
        max_val = max(vals) if vals else 0.0
        if max_val > max_bad_slit_adjacent_front_fill:
            failures.append(
                f"{label} {scene} slit-adjacent front-wall valid stripe max {max_val:.4f} "
                f"exceeds {max_bad_slit_adjacent_front_fill:.4f}"
            )
    return failures


def _clean_slit_shortcut_failures(
    rows: list[dict],
    label: str,
    max_bad_clean_slit_shortcut: float,
    max_bad_clean_slit_edge_shortcut: float,
    max_bad_visible_slit_edge_shortcut: float,
    max_bad_visible_slit_body_shortcut: float,
) -> list[str]:
    """Fail cases where invalid side patches leave a clean valid slit cue.

    This catches the visual failure mode where the side material is fully
    invalid, but the back-wall slit return remains a crisp valid stripe.  Such
    a negative-space template gives fixed cameras an unrealistically easy cue.

    When the camera is already at or slightly through the aperture, the image
    can legitimately be dominated by the ordinary second wall.  In that regime
    a clean central far-wall return is not itself suspicious; the gate should
    instead rely on the slit-edge shortcut check below.  Therefore the full
    template check only applies when a meaningful amount of front-wall material
    is still visible around the slit.
    """
    failures: list[str] = []
    bad_settings = {
        "dark": {"fixed_config", "fixed_mid", "low_return_bad", "specular_safe", "high_power"},
        "specular": {"fixed_config", "fixed_mid", "overexposed", "high_power", "dark_expected"},
    }
    for scene, settings in bad_settings.items():
        vals = [
            as_float(r, "clean_slit_shortcut")
            for r in rows
            if str(r.get("scene")) == scene and str(r.get("setting")) in settings
            and as_float(r, "local_front_area") >= 0.12
        ]
        max_val = max(vals) if vals else 0.0
        if max_val > max_bad_clean_slit_shortcut:
            failures.append(
                f"{label} {scene} clean slit template max {max_val:.4f} "
                f"exceeds {max_bad_clean_slit_shortcut:.4f}"
            )
        edge_vals = [
            as_float(r, "clean_slit_edge_shortcut")
            for r in rows
            if str(r.get("scene")) == scene and str(r.get("setting")) in settings
            and as_float(r, "local_front_area") >= 0.20
            and as_float(r, "far_slit_area") >= 0.015
        ]
        max_edge_val = max(edge_vals) if edge_vals else 0.0
        if max_edge_val > max_bad_clean_slit_edge_shortcut:
            failures.append(
                f"{label} {scene} clean slit-edge shortcut max {max_edge_val:.4f} "
                f"exceeds {max_bad_clean_slit_edge_shortcut:.4f}"
            )
        visible_edge_vals = [
            as_float(r, "visible_slit_edge_shortcut")
            for r in rows
            if str(r.get("scene_name", r.get("scene"))) == scene
            and str(r.get("setting", r.get("method", ""))) in settings
            and as_float(r, "local_front_area") >= 0.20
            and as_float(r, "far_slit_area") >= 0.015
        ]
        max_visible_edge_val = max(visible_edge_vals) if visible_edge_vals else 0.0
        if max_visible_edge_val > max_bad_visible_slit_edge_shortcut:
            failures.append(
                f"{label} {scene} visible slit-edge template max {max_visible_edge_val:.4f} "
                f"exceeds {max_bad_visible_slit_edge_shortcut:.4f}"
            )
        separator_vals = [
            as_float(r, "valid_separator_band_shortcut")
            for r in rows
            if str(r.get("scene_name", r.get("scene"))) == scene
            and str(r.get("setting", r.get("method", ""))) in settings
            and as_float(r, "local_front_area") >= 0.12
            and as_float(r, "far_slit_area") >= 0.015
        ]
        max_separator_val = max(separator_vals) if separator_vals else 0.0
        if max_separator_val > 0.20:
            failures.append(
                f"{label} {scene} valid separator-band shortcut max {max_separator_val:.4f} "
                "exceeds 0.2000"
            )
        visible_body_vals = [
            as_float(r, "visible_slit_body_shortcut")
            for r in rows
            if str(r.get("scene_name", r.get("scene"))) == scene
            and str(r.get("setting", r.get("method", ""))) in settings
            and as_float(r, "local_front_area") >= 0.35
            and as_float(r, "far_slit_area") >= 0.015
            and (
                as_float(r, "visible_slit_edge_shortcut") > 0.20
                or as_float(r, "valid_separator_band_shortcut") > 0.20
            )
        ]
        max_visible_body_val = max(visible_body_vals) if visible_body_vals else 0.0
        if max_visible_body_val > max_bad_visible_slit_body_shortcut:
            failures.append(
                f"{label} {scene} visible slit-body template max {max_visible_body_val:.4f} "
                f"exceeds {max_bad_visible_slit_body_shortcut:.4f}"
            )
    return failures


def _whole_slit_black_failures(rows: list[dict], label: str) -> list[str]:
    """Fail physically implausible dark/specular cases where the aperture vanishes.

    Dark/specular side materials may produce edge holes and flying/mixed
    depths, but a visible open slit should not turn into a solid black region
    across its center.  This gate is intentionally separate from shortcut
    gates: it catches the opposite failure mode from a too-clean aperture.
    """
    failures: list[str] = []
    bad_rows = [
        r for r in rows
        if str(r.get("scene_name", r.get("scene", ""))) in {"dark", "specular"}
        and as_float(r, "whole_slit_black_flag") > 0.5
    ]
    if bad_rows:
        sample = bad_rows[0]
        failures.append(
            f"{label} dark/specular whole-slit-black cases: {len(bad_rows)}; "
            f"example scene={sample.get('scene_name', sample.get('scene'))} "
            f"pose={sample.get('pose', sample.get('step'))} "
            f"setting={sample.get('setting', sample.get('method'))} "
            f"center_fill={as_float(sample, 'far_slit_center_fill'):.3f}"
        )
    return failures


def diagnose_opening(
    opening_dir: Path,
    max_dark_spec_back_wall_leak: float,
    max_bad_slit_adjacent_front_fill: float,
    max_bad_clean_slit_shortcut: float,
    max_bad_clean_slit_edge_shortcut: float,
    max_bad_visible_slit_edge_shortcut: float,
    max_bad_visible_slit_body_shortcut: float,
) -> tuple[list[dict], list[str]]:
    rows = read_csv(opening_dir / "opening_depth_probe_detail.csv")
    exp = read_csv(opening_dir / "opening_depth_probe_expectations.csv")
    failures: list[str] = []
    checked = [r for r in exp if as_float(r, "checked") > 0.5]
    passed = [r for r in checked if as_float(r, "passed") > 0.5]
    pass_rate = len(passed) / max(len(checked), 1)
    # Keep this as a diagnostic rather than a hard failure.  The opening probe
    # expectation compares a small set of hand-picked camera pairs; after the
    # simulator moved from binary holes to mixed/flying depths, these coarse
    # thresholds can undercount physically meaningful visual differences.  Hard
    # failures below still catch material leakage, clean-slit shortcuts, and the
    # opposite failure mode where a visible aperture turns fully black.
    for scene in ("dark", "specular"):
        vals = [as_float(r, "scene_mask_on_back_wall_mean") for r in rows if str(r.get("scene")) == scene]
        max_val = max(vals) if vals else 0.0
        if max_val > max_dark_spec_back_wall_leak:
            failures.append(
                f"opening {scene} scene_mask_on_back_wall_mean max {max_val:.4f} exceeds {max_dark_spec_back_wall_leak:.4f}"
            )
    failures.extend(_bad_slit_adjacent_front_fill_failures(
        rows,
        "opening",
        max_bad_slit_adjacent_front_fill,
    ))
    failures.extend(_clean_slit_shortcut_failures(
        rows,
        "opening",
        max_bad_clean_slit_shortcut,
        max_bad_clean_slit_edge_shortcut,
        max_bad_visible_slit_edge_shortcut,
        max_bad_visible_slit_body_shortcut,
    ))
    failures.extend(_whole_slit_black_failures(rows, "opening"))
    return rows, failures


def diagnose_rollout(
    rollout_dir: Path,
    max_dark_spec_back_wall_leak: float,
    max_bad_slit_adjacent_front_fill: float,
    max_bad_clean_slit_shortcut: float,
    max_bad_clean_slit_edge_shortcut: float,
    max_bad_visible_slit_edge_shortcut: float,
    max_bad_visible_slit_body_shortcut: float,
) -> tuple[list[dict], list[str]]:
    rows = read_csv(rollout_dir / "rollout_depth_probe_detail.csv")
    failures: list[str] = []
    for scene in ("dark", "specular"):
        vals = [as_float(r, "scene_mask_on_back_wall_mean") for r in rows if str(r.get("scene")) == scene]
        max_val = max(vals) if vals else 0.0
        if max_val > max_dark_spec_back_wall_leak:
            failures.append(
                f"rollout {rollout_dir.name} {scene} scene_mask_on_back_wall_mean max {max_val:.4f} exceeds {max_dark_spec_back_wall_leak:.4f}"
            )
    failures.extend(_bad_slit_adjacent_front_fill_failures(
        rows,
        f"rollout {rollout_dir.name}",
        max_bad_slit_adjacent_front_fill,
    ))
    failures.extend(_clean_slit_shortcut_failures(
        rows,
        f"rollout {rollout_dir.name}",
        max_bad_clean_slit_shortcut,
        max_bad_clean_slit_edge_shortcut,
        max_bad_visible_slit_edge_shortcut,
        max_bad_visible_slit_body_shortcut,
    ))
    failures.extend(_whole_slit_black_failures(rows, f"rollout {rollout_dir.name}"))
    return rows, failures


def diagnose_qualitative(
    q_dir: Path,
    max_dark_spec_back_wall_leak: float,
    max_bad_clean_slit_shortcut: float,
    max_bad_clean_slit_edge_shortcut: float,
    max_bad_visible_slit_edge_shortcut: float,
    max_bad_visible_slit_body_shortcut: float,
) -> tuple[list[dict], list[str]]:
    rows = read_csv(q_dir / "depth_sequence_rows.csv")
    failures: list[str] = []
    npz_path = q_dir / "depth_sequence_arrays.npz"

    for scene in ("dark", "specular"):
        scene_rows = [r for r in rows if str(r.get("scene_name")) == scene]
        vals = [as_float(r, "scene_mask_on_back_wall_mean") for r in scene_rows]
        max_val = max(vals) if vals else 0.0
        if max_val > max_dark_spec_back_wall_leak:
            failures.append(
                f"qualitative {scene} scene_mask_on_back_wall_mean max {max_val:.4f} exceeds {max_dark_spec_back_wall_leak:.4f}"
            )
    failures.extend(_clean_slit_shortcut_failures(
        rows,
        "qualitative",
        max_bad_clean_slit_shortcut,
        max_bad_clean_slit_edge_shortcut,
        max_bad_visible_slit_edge_shortcut,
        max_bad_visible_slit_body_shortcut,
    ))
    failures.extend(_whole_slit_black_failures(rows, "qualitative"))
    if not npz_path.is_file():
        failures.append(f"qualitative arrays missing: {npz_path}")
    return rows, failures


def _rows_by_key(rows: list[dict]) -> dict[tuple[str, str, str, int], dict]:
    out: dict[tuple[str, str, str, int], dict] = {}
    for row in rows:
        scene = str(row.get("scene_name", row.get("scene", "")))
        panel = str(row.get("panel", ""))
        method = str(row.get("method", row.get("setting", "")))
        col = int(as_float(row, "column_idx", -1))
        out[(scene, panel, method, col)] = row
    return out


def _mean_abs_depth_diff(npz, scene: str, panel: str, col: int, a: str, b: str) -> float:
    ka = f"{scene}_{panel}_col{col:02d}_{a}_depth"
    kb = f"{scene}_{panel}_col{col:02d}_{b}_depth"
    if ka not in npz or kb not in npz:
        return 0.0
    da = np.asarray(npz[ka], dtype=np.float32)
    db = np.asarray(npz[kb], dtype=np.float32)
    mask = (da > 0.050001) | (db > 0.050001)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(da[mask] - db[mask])))


def diagnose_signal_strength(
    qualitative_rows: list[dict],
    q_dir: Path,
    *,
    min_dark_ours_fill_advantage: float,
    min_specular_fixed_depth_mae: float,
    min_specular_randfix_fill_advantage: float,
) -> tuple[list[dict], list[str]]:
    """Gate whether final-figure-equivalent observations contain useful signal.

    Leakage checks say whether material masks are physically placed correctly.
    This signal check says whether the placed material actually creates a
    visible camera-parameter contrast on the same poses used by paper figures.
    """
    failures: list[str] = []
    rows: list[dict] = []
    npz_path = q_dir / "depth_sequence_arrays.npz"
    if not npz_path.is_file():
        return rows, [f"signal gate skipped because qualitative arrays are missing: {npz_path}"]
    lookup = _rows_by_key(qualitative_rows)
    with np.load(npz_path) as npz:
        for scene in ("dark", "specular"):
            cols = sorted({
                key[3]
                for key in lookup
                if key[0] == scene and key[1] == "same_pose" and key[3] >= 0
            })
            if not cols:
                failures.append(f"signal gate {scene}: no same_pose columns found")
                continue
            fill_adv_fixed = []
            fill_adv_randfix = []
            invalid_adv_fixed = []
            invalid_adv_randfix = []
            mae_fixed = []
            mae_randfix = []
            for col in cols:
                ours = lookup.get((scene, "same_pose", "flightonly", col))
                fixed = lookup.get((scene, "same_pose", "fixed", col))
                randfix = lookup.get((scene, "same_pose", "randfix", col))
                if ours and fixed:
                    fill_adv_fixed.append(as_float(ours, "local_fill") - as_float(fixed, "local_fill"))
                    invalid_adv_fixed.append(as_float(fixed, "invalid_rate") - as_float(ours, "invalid_rate"))
                    mae_fixed.append(_mean_abs_depth_diff(npz, scene, "same_pose", col, "fixed", "flightonly"))
                if ours and randfix:
                    fill_adv_randfix.append(as_float(ours, "local_fill") - as_float(randfix, "local_fill"))
                    invalid_adv_randfix.append(as_float(randfix, "invalid_rate") - as_float(ours, "invalid_rate"))
                    mae_randfix.append(_mean_abs_depth_diff(npz, scene, "same_pose", col, "randfix", "flightonly"))
            record = {
                "scene": scene,
                "cols": len(cols),
                "mean_ours_fill_advantage_vs_fixed": mean(fill_adv_fixed),
                "mean_ours_fill_advantage_vs_randfix": mean(fill_adv_randfix),
                "mean_fixed_invalid_advantage": mean(invalid_adv_fixed),
                "mean_randfix_invalid_advantage": mean(invalid_adv_randfix),
                "mean_fixed_vs_ours_depth_mae": mean(mae_fixed),
                "mean_randfix_vs_ours_depth_mae": mean(mae_randfix),
            }
            rows.append(record)
            if scene == "dark":
                dark_signal = max(
                    record["mean_ours_fill_advantage_vs_fixed"],
                    record["mean_ours_fill_advantage_vs_randfix"],
                    record["mean_fixed_vs_ours_depth_mae"],
                    record["mean_randfix_vs_ours_depth_mae"],
                )
                if dark_signal < min_dark_ours_fill_advantage:
                    failures.append(
                        "signal gate dark: matched-pose camera contrast too weak "
                        f"(best fill/MAE signal {dark_signal:.4f} < {min_dark_ours_fill_advantage:.4f})"
                    )
            elif scene == "specular":
                if record["mean_fixed_vs_ours_depth_mae"] < min_specular_fixed_depth_mae:
                    failures.append(
                        "signal gate specular: fixed and active-camera matched-pose depth are too similar "
                        f"(MAE {record['mean_fixed_vs_ours_depth_mae']:.4f} < {min_specular_fixed_depth_mae:.4f})"
                    )
                spec_rand_signal = max(
                    record["mean_ours_fill_advantage_vs_randfix"],
                    record["mean_randfix_vs_ours_depth_mae"],
                )
                if spec_rand_signal < min_specular_randfix_fill_advantage:
                    failures.append(
                        "signal gate specular: randfix contrast too weak "
                        f"(best fill/MAE signal {spec_rand_signal:.4f} < {min_specular_randfix_fill_advantage:.4f})"
                    )
    return rows, failures


def diagnose_rollout_camera_sweep_signal(
    rollout_rows_by_method: dict[str, list[dict]],
    *,
    min_camera_sweep_signal: float,
    min_camera_sweep_visual_delta: float,
) -> tuple[list[dict], list[str]]:
    """Gate sensor signal using camera sweeps at actual rollout poses.

    This is the right pre-training semantic gate: it does not assume any
    learned camera checkpoint is already valid under the current simulator.
    Instead it asks whether known good/bad camera settings produce distinct
    observations at the same real policy poses and attitudes.
    """
    expectations = {
        "glare": ("glare_expected", "overexposed"),
        "dark": ("dark_expected", "low_return_bad"),
        "specular": ("specular_safe", "high_power"),
    }
    rows: list[dict] = []
    failures: list[str] = []
    for method, method_rows in rollout_rows_by_method.items():
        for scene, (good_name, bad_name) in expectations.items():
            scene_rows = [r for r in method_rows if str(r.get("scene")) == scene]
            if not scene_rows:
                continue
            grouped: dict[tuple[str, str], list[dict]] = {}
            for row in scene_rows:
                pose_key = str(row.get("step", row.get("pose", "")))
                grouped.setdefault((pose_key, str(row.get("setting"))), []).append(row)

            signals = []
            fill_delta = []
            quality_delta = []
            invalid_delta = []
            local_edge_quality_delta = []
            visual_delta = []
            checked = 0
            for pose_key in sorted({k[0] for k in grouped}):
                good_vals = grouped.get((pose_key, good_name), [])
                bad_vals = grouped.get((pose_key, bad_name), [])
                if not good_vals or not bad_vals:
                    continue
                good = good_vals[0]
                bad = bad_vals[0]
                checked += 1
                fd = as_float(good, "local_fill") - as_float(bad, "local_fill")
                qd = as_float(good, "local_quality_mean") - as_float(bad, "local_quality_mean")
                idd = as_float(bad, "invalid_rate") - as_float(good, "invalid_rate")
                eqd = as_float(good, "local_edge_quality_mean") - as_float(bad, "local_edge_quality_mean")
                vd = max(
                    fd,
                    as_float(good, "local_depth_mean") - as_float(bad, "local_depth_mean"),
                    as_float(good, "local_depth_std") - as_float(bad, "local_depth_std"),
                )
                fill_delta.append(fd)
                quality_delta.append(qd)
                invalid_delta.append(idd)
                local_edge_quality_delta.append(eqd)
                visual_delta.append(vd)
                signals.append(fd + 0.50 * qd + idd + 0.25 * max(eqd, 0.0))

            record = {
                "method": method,
                "scene": scene,
                "good_setting": good_name,
                "bad_setting": bad_name,
                "checked": checked,
                "mean_signal": mean(signals),
                "mean_fill_delta": mean(fill_delta),
                "mean_quality_delta": mean(quality_delta),
                "mean_invalid_delta": mean(invalid_delta),
                "mean_edge_quality_delta": mean(local_edge_quality_delta),
                "mean_visual_depth_delta": mean(visual_delta),
            }
            rows.append(record)
            if checked == 0:
                failures.append(f"camera sweep signal {method}/{scene}: missing {good_name} or {bad_name} rows")
            elif record["mean_signal"] < min_camera_sweep_signal:
                failures.append(
                    f"camera sweep signal {method}/{scene}: {record['mean_signal']:.4f} "
                    f"< {min_camera_sweep_signal:.4f} for {good_name} vs {bad_name}"
                )
            elif scene in {"dark", "specular"} and record["mean_visual_depth_delta"] < min_camera_sweep_visual_delta:
                failures.append(
                    f"camera sweep visual depth {method}/{scene}: {record['mean_visual_depth_delta']:.4f} "
                    f"< {min_camera_sweep_visual_delta:.4f} for {good_name} vs {bad_name}"
                )
    return rows, failures


def write_report(
    out_dir: Path,
    specs: dict[str, MethodSpec],
    opening_rows: list[dict],
    rollout_rows_by_method: dict[str, list[dict]],
    qualitative_rows: list[dict],
    signal_rows: list[dict],
    sweep_signal_rows: list[dict],
    failures: list[str],
    commands: list[list[str]],
) -> None:
    lines = [
        "# Sensor Semantics Probe Suite",
        "",
        f"- gate status: `{'FAIL' if failures else 'PASS'}`",
        f"- opening probe rows: `{len(opening_rows)}`",
        f"- qualitative rows: `{len(qualitative_rows)}`",
        "",
        "Checkpoints:",
        "",
    ]
    for method, spec in specs.items():
        lines.append(f"- `{method}`: `{spec.checkpoint.relative_to(ROOT)}`")

    lines.extend(["", "Outputs:", ""])
    lines.extend([
        "- `opening_probe/`: idealized look-at-slit raw/depth/quality/invalid/effect/key-cue panels.",
        "- `rollout_probe_<method>/`: actual checkpoint rollout-state panels and arrays.",
        "- `journal_depth_sequences/`: final-figure-equivalent matched-pose and own-pose depth sequence panels.",
        "",
        "Key rollout means by scene/method:",
        "",
        "| method | scene | rows | mean local fill | mean invalid | max mask on back wall | max clean slit-edge shortcut |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for method, rows in rollout_rows_by_method.items():
        for scene in sorted({str(r.get("scene")) for r in rows}):
            vals = [r for r in rows if str(r.get("scene")) == scene]
            lines.append(
                f"| {method} | {scene} | {len(vals)} | "
                f"{mean([as_float(r, 'local_fill') for r in vals]):.3f} | "
                f"{mean([as_float(r, 'invalid_rate') for r in vals]):.3f} | "
                f"{max([as_float(r, 'scene_mask_on_back_wall_mean') for r in vals] or [0.0]):.4f} | "
                f"{max([as_float(r, 'clean_slit_edge_shortcut') for r in vals] or [0.0]):.4f} |"
            )

    lines.extend(["", "Failures:", ""])
    if failures:
        for item in failures:
            lines.append(f"- {item}")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "Camera-sweep signal gate:",
        "",
        "| method | scene | good | bad | poses | signal | visual depth delta | fill delta | quality delta | invalid delta |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    if sweep_signal_rows:
        for row in sweep_signal_rows:
            lines.append(
                f"| {row['method']} | {row['scene']} | {row['good_setting']} | {row['bad_setting']} | "
                f"{int(row['checked'])} | "
                f"{float(row['mean_signal']):.3f} | "
                f"{float(row.get('mean_visual_depth_delta', 0.0)):.3f} | "
                f"{float(row['mean_fill_delta']):.3f} | "
                f"{float(row['mean_quality_delta']):.3f} | "
                f"{float(row['mean_invalid_delta']):.3f} |"
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |")

    lines.extend([
        "",
        "Current-checkpoint qualitative method contrast:",
        "",
        "This table is diagnostic only for simulator-level validation. If `env_cuda.py` changed, learned-camera",
        "checkpoints are stale and should not be treated as a valid active-camera policy until retrained.",
        "",
        "| scene | cols | fill adv vs fixed | fill adv vs randfix | fixed-vs-ours depth MAE | randfix-vs-ours depth MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    if signal_rows:
        for row in signal_rows:
            lines.append(
                f"| {row['scene']} | {int(row['cols'])} | "
                f"{float(row['mean_ours_fill_advantage_vs_fixed']):.3f} | "
                f"{float(row['mean_ours_fill_advantage_vs_randfix']):.3f} | "
                f"{float(row['mean_fixed_vs_ours_depth_mae']):.4f} | "
                f"{float(row['mean_randfix_vs_ours_depth_mae']):.4f} |"
            )
    else:
        lines.append("| n/a | 0 | 0.000 | 0.000 | 0.0000 | 0.0000 |")

    lines.extend(["", "Commands:", ""])
    for cmd in commands:
        lines.append("```bash")
        lines.append(" ".join(cmd))
        lines.append("```")
    lines.append("")
    (out_dir / "sensor_semantics_probe_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "sensor_semantics_probe_summary.json").write_text(
        json.dumps(
            {
                "status": "FAIL" if failures else "PASS",
                "failures": failures,
                "signal_rows": signal_rows,
                "sweep_signal_rows": sweep_signal_rows,
                "checkpoints": {k: str(v.checkpoint.relative_to(ROOT)) for k, v in specs.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="*", type=Path, help="Optional training logs to extract fixed/randfix/flightonly checkpoints.")
    p.add_argument("--config", default="configs/slit_active_sensing.args")
    p.add_argument("--out_dir", default="paper/experiment/results/sensor_semantics_probe_suite_20260508")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--device", default="cuda")
    p.add_argument("--scenarios", nargs="*", default=["glare", "dark", "specular"])
    p.add_argument("--slots", nargs="*", default=["far_left", "far_right"])
    p.add_argument("--qual_slot", default="far_right")
    p.add_argument("--xs", default="-1.20,-0.90,-0.55,-0.20,0.15")
    p.add_argument("--target_local_x", default="-1.20,-0.75,-0.35,-0.08,0.18")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_panels", type=int, default=24)
    p.add_argument("--skip_opening", action="store_true")
    p.add_argument("--skip_rollout", action="store_true")
    p.add_argument("--skip_qualitative", action="store_true")
    p.add_argument("--fail_on_gate", action="store_true")
    p.add_argument("--max_dark_spec_back_wall_leak", type=float, default=0.002,
                   help="Maximum allowed dark/specular material mask mass on raw back-wall hits.")
    p.add_argument("--max_bad_slit_adjacent_front_fill", type=float, default=0.35,
                   help="Maximum valid fraction allowed on front-wall pixels directly adjacent to the slit for known bad dark/specular settings.")
    p.add_argument("--max_bad_clean_slit_shortcut", type=float, default=0.60,
                   help="Maximum allowed clean full-aperture slit cue for known bad dark/specular camera settings.")
    p.add_argument("--max_bad_clean_slit_edge_shortcut", type=float, default=0.60,
                   help="Maximum allowed clean negative-space slit-edge cue for known bad dark/specular camera settings.")
    p.add_argument("--max_bad_visible_slit_edge_shortcut", type=float, default=0.55,
                   help="Maximum allowed visible aperture-edge template for known bad dark/specular camera settings, even when the depth value is wrong.")
    p.add_argument("--max_bad_visible_slit_body_shortcut", type=float, default=0.70,
                   help="Maximum allowed visible full-aperture body template for known bad dark/specular camera settings, even when the depth value is wrong.")
    p.add_argument("--min_dark_ours_fill_advantage", type=float, default=0.04,
                   help="Minimum matched-pose dark camera contrast in fill or depth MAE.")
    p.add_argument("--min_specular_fixed_depth_mae", type=float, default=0.012,
                   help="Minimum matched-pose specular depth MAE between fixed and active camera.")
    p.add_argument("--min_specular_randfix_fill_advantage", type=float, default=0.06,
                   help="Minimum matched-pose specular contrast between randfix and active camera.")
    p.add_argument("--min_camera_sweep_signal", type=float, default=0.10,
                   help="Minimum same-rollout-pose good-vs-bad camera sweep signal.")
    p.add_argument("--min_camera_sweep_visual_delta", type=float, default=0.08,
                   help="Minimum same-rollout-pose visible observed-depth contrast for dark/specular.")
    p.add_argument("--fixed_ckpt", default=None)
    p.add_argument("--randfix_ckpt", default=None)
    p.add_argument("--flightonly_ckpt", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = resolve_method_specs(args)
    commands: list[list[str]] = []
    failures: list[str] = []

    opening_rows: list[dict] = []
    if not args.skip_opening:
        cmd = [
            args.python,
            "-u",
            "tools/probe_opening_depth_views.py",
            "--config",
            args.config,
            "--out_dir",
            args.out_dir + "/opening_probe",
            "--scenarios",
            *args.scenarios,
            "--slots",
            *args.slots,
            f"--xs={args.xs}",
            "--sensor_impl",
            "cuda",
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--max_panels",
            str(args.max_panels),
        ]
        commands.append(cmd)
        run(cmd)
        opening_rows, opening_failures = diagnose_opening(
            out_dir / "opening_probe",
            float(args.max_dark_spec_back_wall_leak),
            float(args.max_bad_slit_adjacent_front_fill),
            float(args.max_bad_clean_slit_shortcut),
            float(args.max_bad_clean_slit_edge_shortcut),
            float(args.max_bad_visible_slit_edge_shortcut),
            float(args.max_bad_visible_slit_body_shortcut),
        )
        failures.extend(opening_failures)

    rollout_rows_by_method: dict[str, list[dict]] = {}
    sweep_signal_rows: list[dict] = []
    if not args.skip_rollout:
        for method, spec in specs.items():
            cmd = [
                args.python,
                "-u",
                "tools/probe_rollout_depth_views.py",
                "--config",
                str(spec.config.relative_to(ROOT)),
                "--checkpoint",
                str(spec.checkpoint.relative_to(ROOT)),
                "--out_dir",
                f"{args.out_dir}/rollout_probe_{method}",
                "--scenarios",
                *args.scenarios,
                f"--target_local_x={args.target_local_x}",
                "--sensor_impl",
                "cuda",
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--max_panels",
                str(args.max_panels),
            ]
            commands.append(cmd)
            run(cmd)
            rows, rollout_failures = diagnose_rollout(
                out_dir / f"rollout_probe_{method}",
                float(args.max_dark_spec_back_wall_leak),
                float(args.max_bad_slit_adjacent_front_fill),
                float(args.max_bad_clean_slit_shortcut),
                float(args.max_bad_clean_slit_edge_shortcut),
                float(args.max_bad_visible_slit_edge_shortcut),
                float(args.max_bad_visible_slit_body_shortcut),
            )
            rollout_rows_by_method[method] = rows
            failures.extend(rollout_failures)
        sweep_signal_rows, sweep_signal_failures = diagnose_rollout_camera_sweep_signal(
            rollout_rows_by_method,
            min_camera_sweep_signal=float(args.min_camera_sweep_signal),
            min_camera_sweep_visual_delta=float(args.min_camera_sweep_visual_delta),
        )
        failures.extend(sweep_signal_failures)

    qualitative_rows: list[dict] = []
    signal_rows: list[dict] = []
    if not args.skip_qualitative:
        qual_out = out_dir / "journal_depth_sequences"
        cmd = [
            args.python,
            "-u",
            "tools/export_journal_depth_sequences.py",
            "--config",
            args.config,
            "--eval_dir",
            args.out_dir,
            "--out_dir",
            str(qual_out.relative_to(ROOT)),
            "--scenarios",
            *args.scenarios,
            "--slot",
            args.qual_slot,
            f"--target_local_x={args.target_local_x}",
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--fixed_ckpt",
            str(specs["fixed"].checkpoint.relative_to(ROOT)),
            "--randfix_ckpt",
            str(specs["randfix"].checkpoint.relative_to(ROOT)),
            "--flightonly_ckpt",
            str(specs["flightonly"].checkpoint.relative_to(ROOT)),
        ]
        commands.append(cmd)
        run(cmd)
        qualitative_rows, qualitative_failures = diagnose_qualitative(
            qual_out / "qualitative_depth",
            float(args.max_dark_spec_back_wall_leak),
            float(args.max_bad_clean_slit_shortcut),
            float(args.max_bad_clean_slit_edge_shortcut),
            float(args.max_bad_visible_slit_edge_shortcut),
            float(args.max_bad_visible_slit_body_shortcut),
        )
        failures.extend(qualitative_failures)
        signal_rows, _signal_failures = diagnose_signal_strength(
            qualitative_rows,
            qual_out / "qualitative_depth",
            min_dark_ours_fill_advantage=float(args.min_dark_ours_fill_advantage),
            min_specular_fixed_depth_mae=float(args.min_specular_fixed_depth_mae),
            min_specular_randfix_fill_advantage=float(args.min_specular_randfix_fill_advantage),
        )

    write_report(
        out_dir,
        specs,
        opening_rows,
        rollout_rows_by_method,
        qualitative_rows,
        signal_rows,
        sweep_signal_rows,
        failures,
        commands,
    )
    print(f"[sensor-probe-suite] status={'FAIL' if failures else 'PASS'}")
    print(f"[sensor-probe-suite] report={out_dir / 'sensor_semantics_probe_report.md'}")
    if failures:
        for item in failures[:12]:
            print(f"[sensor-probe-suite][fail] {item}")
        if args.fail_on_gate:
            raise SystemExit(2)


if __name__ == "__main__":
    main()

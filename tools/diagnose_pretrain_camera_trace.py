#!/usr/bin/env python3
"""Diagnose whether camera pretraining or flight-only rollout loses scene separation."""

from __future__ import annotations

import argparse
import csv
import shlex
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import build_parser, parse_diff_sensor_impl, parse_scenarios, validate_args  # noqa: E402
from model import Model  # noqa: E402
from tools.pretrain_camera_head import _camera_forward_sequence  # noqa: E402


def _read_args_file(path: Path) -> list[str]:
    tokens: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            tokens.extend(shlex.split(line))
    return tokens


def _load_project_args(path: Path):
    parser = build_parser()
    args = parser.parse_args(_read_args_file(path))
    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.scenarios = parse_scenarios(args.scenarios)
    args.wandb_disabled = True
    args.vis_enable = False
    validate_args(args)
    return args


def _make_model(args, device: torch.device) -> Model:
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


def _phase(local_x: float) -> str:
    if local_x < -0.25:
        return "before"
    if local_x <= 0.25:
        return "near"
    return "after"


def _mean(vals: list[float]) -> float:
    return sum(vals) / max(len(vals), 1)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _online_phase_summary(eval_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    trace_rows: list[dict] = []
    for path in sorted((eval_dir / "raw").glob("*_trace.csv")):
        trace_rows.extend(_read_csv(path))
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in trace_rows:
        lx = row.get("local_x") or row.get("x") or "0"
        key = (row.get("method", ""), row.get("scene_name", ""), _phase(float(lx)))
        grouped.setdefault(key, []).append(row)

    rows: list[dict] = []
    lookup: dict[str, dict] = {}
    for (method, scene, phase), vals in sorted(grouped.items()):
        out = {
            "source": "online_eval",
            "method": method,
            "scene": scene,
            "phase": phase,
            "n": len(vals),
            "power": _mean([float(v["power"]) for v in vals]),
            "exposure": _mean([float(v["exposure"]) for v in vals]),
            "gain": _mean([float(v["gain"]) for v in vals]),
        }
        rows.append(out)
        lookup[f"{method}:{scene}:{phase}"] = out
    return rows, lookup


def _offline_teacher_summary(
    *,
    config: Path,
    dataset: Path,
    checkpoint: Path,
    method_label: str,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict], dict[str, dict]]:
    args = _load_project_args(config)
    model = _make_model(args, device)
    model.load_state_dict(torch.load(str(checkpoint), map_location=device), strict=True)
    model.eval()

    data = torch.load(str(dataset), map_location="cpu")
    required = ["depth_obs", "state", "camera_state", "camera_motion_state", "teacher_camera", "scene_id", "local_x", "meta"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"dataset missing keys: {missing}")

    preds = []
    with torch.no_grad():
        for start in range(0, int(data["state"].shape[0]), int(batch_size)):
            end = start + int(batch_size)
            pred = _camera_forward_sequence(
                model,
                data["depth_obs"][start:end].to(device),
                data["state"][start:end].to(device),
                data["camera_state"][start:end].to(device),
                data["camera_motion_state"][start:end].to(device),
            )
            preds.append(pred.detach().cpu())
    pred_all = torch.cat(preds, dim=0)
    teacher = data["teacher_camera"]
    scene_id = data["scene_id"]
    local_x = data["local_x"]
    scenes = list(data["meta"].get("scenarios", []))

    rows: list[dict] = []
    lookup: dict[str, dict] = {}
    for sid, scene in enumerate(scenes):
        for phase in ("before", "near", "after"):
            mask = (scene_id == int(sid)) & torch.tensor(
                [[_phase(float(x)) == phase for x in seq] for seq in local_x],
                dtype=torch.bool,
            )
            if not bool(mask.any()):
                continue
            pred_m = pred_all[mask].mean(dim=0)
            teacher_m = teacher[mask].mean(dim=0)
            mae_m = (pred_all[mask] - teacher[mask]).abs().mean(dim=0)
            out = {
                "source": "offline_teacher_dataset",
                "method": method_label,
                "scene": scene,
                "phase": phase,
                "n": int(mask.sum().item()),
                "power": float(pred_m[0]),
                "exposure": float(pred_m[1]),
                "gain": float(pred_m[2]),
                "teacher_power": float(teacher_m[0]),
                "teacher_exposure": float(teacher_m[1]),
                "teacher_gain": float(teacher_m[2]),
                "mae_power": float(mae_m[0]),
                "mae_exposure": float(mae_m[1]),
                "mae_gain": float(mae_m[2]),
            }
            rows.append(out)
            lookup[f"{method_label}:{scene}:{phase}"] = out
    return rows, lookup


def _scene_l1(lookup: dict[str, dict], method: str, phase: str, a: str = "glare", b: str = "dark") -> tuple[float, list[float]]:
    ra = lookup[f"{method}:{a}:{phase}"]
    rb = lookup[f"{method}:{b}:{phase}"]
    diffs = [abs(float(ra[key]) - float(rb[key])) for key in ("power", "exposure", "gain")]
    return _mean(diffs), diffs


def _fmt_peg(row: dict) -> str:
    return f"{float(row['power']):.3f}/{float(row['exposure']):.3f}/{float(row['gain']):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--dataset", default="paper/experiment/results/closed_loop_teacher_camera_policy_v3d_full/camera_teacher_dataset.pt")
    parser.add_argument("--config", default="configs/slit_active_sensing.args")
    parser.add_argument("--pretrained_ckpt", required=True)
    parser.add_argument("--offline_method_label", default="pretrained")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=12)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out = Path(args.out) if args.out else eval_dir / "diagnosis.md"
    device = torch.device(args.device)

    online_rows, online_lookup = _online_phase_summary(eval_dir)
    offline_rows, offline_lookup = _offline_teacher_summary(
        config=Path(args.config),
        dataset=Path(args.dataset),
        checkpoint=Path(args.pretrained_ckpt),
        method_label=str(args.offline_method_label),
        device=device,
        batch_size=int(args.batch_size),
    )
    _write_csv(eval_dir / "pretrain_online_offline_phase_summary.csv", offline_rows + online_rows)

    offline_method = str(args.offline_method_label)
    offline_l1, offline_diff = _scene_l1(offline_lookup, offline_method, "near")
    online_methods = sorted({
        key.split(":", 1)[0]
        for key in online_lookup
        if key.endswith(":near")
        and f"{key.split(':', 1)[0]}:glare:near" in online_lookup
        and f"{key.split(':', 1)[0]}:dark:near" in online_lookup
    })
    online_l1 = {
        method: _scene_l1(online_lookup, method, "near")
        for method in online_methods
    }

    lines = [
        "# Pretrained Camera Trace Diagnosis",
        "",
        "## Result",
        "",
        (
            "The offline teacher-dataset check asks whether the checkpoint can reproduce "
            "`dark`/`glare` labels on its supervised data.  The online eval rows ask whether "
            "that separation survives closed-loop rollout."
        ),
        "",
        "| source | method | glare near p/e/g | dark near p/e/g | glare-dark L1 | per-param diff |",
        "|---|---|---:|---:|---:|---:|",
        (
            f"| offline teacher dataset | {offline_method} | {_fmt_peg(offline_lookup[f'{offline_method}:glare:near'])} | "
            f"{_fmt_peg(offline_lookup[f'{offline_method}:dark:near'])} | {offline_l1:.3f} | "
            f"{offline_diff[0]:.3f}/{offline_diff[1]:.3f}/{offline_diff[2]:.3f} |"
        ),
    ]
    for method in online_methods:
        l1, diff = online_l1[method]
        lines.append(
            f"| online eval | {method} | {_fmt_peg(online_lookup[f'{method}:glare:near'])} | "
            f"{_fmt_peg(online_lookup[f'{method}:dark:near'])} | {l1:.3f} | "
            f"{diff[0]:.3f}/{diff[1]:.3f}/{diff[2]:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        (
            "- The offline row checks whether supervised camera learning can represent the "
            "teacher labels on the relabeled dataset."
        ),
        (
            f"- Offline glare-dark L1 for `{offline_method}` is `{offline_l1:.3f}`.  "
            "Values around `0.18-0.22` mean the dataset and fitted camera head have a clear "
            "dark/glare distinction."
        ),
    ])
    for method in online_methods:
        l1, _ = online_l1[method]
        verdict = "clear" if l1 >= 0.12 else "weak"
        lines.append(f"- Online glare-dark L1 for `{method}` is `{l1:.3f}`: {verdict} separation.")
    lines.extend([
        "",
        "## Next Checks",
        "",
        (
            "1. If the DAgger checkpoint still has weak online separation, save a small batch of "
            "online `(depth_obs, state, camera_state, camera_motion_state)` tensors and run the "
            "teacher optimizer on those exact states.  That isolates whether the online dark state "
            "is truly ambiguous or merely underrepresented."
        ),
        (
            "2. If the DAgger checkpoint restores online separation, use it as the resume checkpoint "
            "for the next flight-only run."
        ),
        (
            "3. The immediate target is online glare-dark near L1 above about `0.12`, with dark "
            "exposure/gain clearly higher than glare."
        ),
        "",
        f"Detailed phase rows: `{(eval_dir / 'pretrain_online_offline_phase_summary.csv').as_posix()}`.",
    ])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[diagnose] wrote: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze whether a camera-teacher dataset contains useful active-sensing labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _fmt(vals) -> str:
    return "/".join(f"{float(v):.3f}" for v in vals)


def _mean_std(x: torch.Tensor) -> tuple[list[float], list[float]]:
    if x.numel() == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return [float(v) for v in x.mean(dim=0)], [float(v) for v in x.std(dim=0)]


def _safe_mean(x: torch.Tensor) -> float:
    return float(x.float().mean()) if x.numel() else 0.0


def _phase_loss_contrib(loss_terms: dict, mask: torch.Tensor) -> dict[str, float]:
    out = {}
    contrib_keys = [
        "contrib_smooth",
        "contrib_fill",
        "contrib_power",
        "contrib_blur",
        "contrib_noise",
        "contrib_nominal",
    ]
    contrib = {}
    for key in contrib_keys:
        value = loss_terms.get(key)
        if value is None:
            continue
        contrib[key] = _safe_mean(value.float().reshape(-1)[mask])
    total = sum(abs(v) for v in contrib.values())
    total = total if total > 1e-12 else 1e-12
    for key, value in contrib.items():
        short = key.removeprefix("contrib_")
        out[f"{short}_contrib"] = value
        out[f"{short}_share"] = abs(value) / total
    return out


def _phase_masks(local_x: torch.Tensor, near_width: float, after_width: float):
    return {
        "before": local_x < -float(near_width),
        "near": local_x.abs() <= float(near_width),
        "after": local_x > float(after_width),
    }


def _score_response(phase_rows: dict[str, dict], nominal: torch.Tensor) -> dict:
    before = torch.tensor(phase_rows.get("before", {}).get("mean_p_e_g", [0.0, 0.0, 0.0]))
    near = torch.tensor(phase_rows.get("near", {}).get("mean_p_e_g", [0.0, 0.0, 0.0]))
    after = torch.tensor(phase_rows.get("after", {}).get("mean_p_e_g", [0.0, 0.0, 0.0]))
    near_delta = (near - before).abs()
    recovery_delta = (after - nominal).abs()
    return {
        "near_response_l1": float(near_delta.mean()),
        "recovery_error_l1": float(recovery_delta.mean()),
        "near_response_p_e_g": [float(v) for v in near_delta],
        "recovery_error_p_e_g": [float(v) for v in recovery_delta],
    }


def analyze_dataset(
    dataset: str | Path,
    out: str | Path | None = None,
    *,
    near_width: float = 0.25,
    after_width: float = 0.35,
    min_response: float = 0.05,
    max_recovery_error: float = 0.08,
    min_fill: float | None = None,
    nominal_p_e_g: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> dict:
    path = Path(dataset)
    data = torch.load(path, map_location="cpu")
    teacher = data["teacher_camera"].float()
    if teacher.ndim != 3 or teacher.shape[-1] != 3:
        raise ValueError(f"teacher_camera must be [N,T,3], got {tuple(teacher.shape)}")
    scene_id = data.get("scene_id")
    local_x = data.get("local_x")
    teacher_fill = data.get("teacher_fill")
    teacher_loss = data.get("teacher_loss")
    loss_terms_raw = data.get("teacher_loss_terms", {})
    meta = data.get("meta", {})
    scenarios = list(meta.get("scenarios", []))
    if scene_id is None or local_x is None:
        raise KeyError("dataset needs scene_id and local_x for quality analysis")
    scene_id = scene_id.long()
    local_x = local_x.float()
    if scene_id.ndim == 1:
        scene_id = scene_id[:, None].expand_as(local_x)

    target_fill = (
        float(min_fill)
        if min_fill is not None
        else float(meta.get("diff_depth_min_fill_rate", 0.0) or 0.0)
    )
    nominal = torch.tensor(nominal_p_e_g, dtype=torch.float32)
    summary = {
        "dataset": str(path),
        "sequences": int(teacher.shape[0]),
        "timesteps": int(teacher.shape[1]),
        "samples": int(teacher.shape[0] * teacher.shape[1]),
        "overall_mean_p_e_g": [float(v) for v in teacher.mean(dim=(0, 1))],
        "overall_std_p_e_g": [float(v) for v in teacher.std(dim=(0, 1))],
        "target_fill": target_fill,
        "nominal_p_e_g": [float(v) for v in nominal],
        "by_scene": {},
        "verdict": [],
    }

    flat_teacher = teacher.reshape(-1, 3)
    flat_scene = scene_id.reshape(-1)
    flat_x = local_x.reshape(-1)
    flat_fill = teacher_fill.float().reshape(-1) if teacher_fill is not None else None
    flat_loss = teacher_loss.float().reshape(-1) if teacher_loss is not None else None
    flat_loss_terms = {
        key: value.float().reshape(-1)
        for key, value in loss_terms_raw.items()
        if isinstance(value, torch.Tensor)
    }

    scene_indices = sorted(int(v) for v in flat_scene.unique().tolist())
    for sid in scene_indices:
        name = scenarios[sid] if 0 <= sid < len(scenarios) else f"scene_{sid}"
        scene_mask = flat_scene == sid
        phase_rows = {}
        for phase, phase_mask_2d in _phase_masks(flat_x, near_width, after_width).items():
            mask = scene_mask & phase_mask_2d
            cam = flat_teacher[mask]
            mean, std = _mean_std(cam)
            row = {
                "samples": int(mask.sum()),
                "mean_p_e_g": mean,
                "std_p_e_g": std,
            }
            if flat_fill is not None:
                row["teacher_fill_mean"] = _safe_mean(flat_fill[mask])
                row["teacher_fill_below_target_frac"] = _safe_mean((flat_fill[mask] < target_fill).float())
            if flat_loss is not None:
                row["teacher_loss_mean"] = _safe_mean(flat_loss[mask])
            if flat_loss_terms:
                row["loss_contrib"] = _phase_loss_contrib(flat_loss_terms, mask)
            phase_rows[phase] = row
        score = _score_response(phase_rows, nominal)
        summary["by_scene"][name] = {"phases": phase_rows, "score": score}

        if score["near_response_l1"] < float(min_response):
            summary["verdict"].append(
                f"{name}: weak near-slit response "
                f"({score['near_response_l1']:.3f} < {float(min_response):.3f})"
            )
        if score["recovery_error_l1"] > float(max_recovery_error):
            summary["verdict"].append(
                f"{name}: poor recovery after slit "
                f"({score['recovery_error_l1']:.3f} > {float(max_recovery_error):.3f})"
            )
        near_row = phase_rows.get("near", {})
        if flat_fill is not None and target_fill > 0.0:
            below = float(near_row.get("teacher_fill_below_target_frac", 0.0))
            if below > 0.25:
                summary["verdict"].append(
                    f"{name}: teacher near-slit fill below target too often ({below:.2%})"
                )

    lines = [
        "# Camera Teacher Dataset Quality",
        "",
        f"- dataset: `{path}`",
        f"- sequences: `{summary['sequences']}`, timesteps: `{summary['timesteps']}`, samples: `{summary['samples']}`",
        f"- overall mean p/e/g: `{_fmt(summary['overall_mean_p_e_g'])}`",
        f"- overall std p/e/g: `{_fmt(summary['overall_std_p_e_g'])}`",
        f"- nominal p/e/g: `{_fmt(summary['nominal_p_e_g'])}`",
        f"- target fill: `{target_fill:.3f}`",
        "",
        "| scene | phase | samples | mean p/e/g | std p/e/g | fill | below target | loss |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scene, scene_row in summary["by_scene"].items():
        for phase, row in scene_row["phases"].items():
            lines.append(
                f"| {scene} | {phase} | {row['samples']} | "
                f"{_fmt(row['mean_p_e_g'])} | {_fmt(row['std_p_e_g'])} | "
                f"{row.get('teacher_fill_mean', 0.0):.3f} | "
                f"{row.get('teacher_fill_below_target_frac', 0.0):.3f} | "
                f"{row.get('teacher_loss_mean', 0.0):.4f} |"
            )
    if any(
        row.get("loss_contrib")
        for scene_row in summary["by_scene"].values()
        for row in scene_row["phases"].values()
    ):
        lines.extend([
            "",
            "## Weighted Loss Contribution Shares",
            "",
            "| scene | phase | smooth | fill | power | blur | noise | nominal |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for scene, scene_row in summary["by_scene"].items():
            for phase, row in scene_row["phases"].items():
                c = row.get("loss_contrib", {})
                lines.append(
                    f"| {scene} | {phase} | "
                    f"{c.get('smooth_share', 0.0):.3f} | "
                    f"{c.get('fill_share', 0.0):.3f} | "
                    f"{c.get('power_share', 0.0):.3f} | "
                    f"{c.get('blur_share', 0.0):.3f} | "
                    f"{c.get('noise_share', 0.0):.3f} | "
                    f"{c.get('nominal_share', 0.0):.3f} |"
                )
    lines.extend(["", "| scene | near response L1 | recovery error L1 | near response p/e/g | recovery error p/e/g |",
                  "|---|---:|---:|---:|---:|"])
    for scene, scene_row in summary["by_scene"].items():
        score = scene_row["score"]
        lines.append(
            f"| {scene} | {score['near_response_l1']:.3f} | {score['recovery_error_l1']:.3f} | "
            f"{_fmt(score['near_response_p_e_g'])} | {_fmt(score['recovery_error_p_e_g'])} |"
        )
    lines.append("")
    if summary["verdict"]:
        lines.append("## Warnings")
        lines.append("")
        for item in summary["verdict"]:
            lines.append(f"- {item}")
    else:
        lines.append("## Warnings")
        lines.append("")
        lines.append("- none")
    report = "\n".join(lines) + "\n"

    out = Path(out) if out else path.with_name("teacher_dataset_quality.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(report)
    print(f"[quality] report: {out}")
    print(f"[quality] json  : {out.with_suffix('.json')}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--near_width", type=float, default=0.25)
    parser.add_argument("--after_width", type=float, default=0.35)
    parser.add_argument("--min_response", type=float, default=0.05)
    parser.add_argument("--max_recovery_error", type=float, default=0.08)
    parser.add_argument("--min_fill", type=float, default=None)
    args = parser.parse_args()
    analyze_dataset(
        args.dataset,
        args.out,
        near_width=float(args.near_width),
        after_width=float(args.after_width),
        min_response=float(args.min_response),
        max_recovery_error=float(args.max_recovery_error),
        min_fill=args.min_fill,
    )


if __name__ == "__main__":
    main()

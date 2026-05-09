#!/usr/bin/env python3
"""Audit sensor probe outputs for reviewer-visible semantic failures.

This script complements the binary probe gate.  The gate catches a few known
failures, while this audit looks for two-sided problems across every exported
probe row:

- material leakage into the back wall;
- too-clean slit/back-wall templates under bad camera settings;
- over-degraded back-wall cues when the camera mostly sees the ordinary second
  wall through the slit;
- weak good-vs-bad camera separation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BAD_SETTINGS = {
    "dark": {"fixed_config", "fixed_mid", "baseline", "glare_expected", "specular_safe", "high_power", "low_return_bad"},
    # Baseline is not a deliberately bad specular setting: it uses moderate
    # active illumination and may legitimately preserve the ordinary second wall.
    "specular": {"fixed_config", "fixed_mid", "overexposed", "high_power", "dark_expected"},
}
GOOD_SETTINGS = {
    "dark": {"dark_expected", "overexposed"},
    "specular": {"specular_safe", "low_return_bad"},
}


def f(row, key, default=0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None or value == "":
            return default
        value = float(value)
        if value != value:
            return default
        return value
    except Exception:
        return default


def row_id(row) -> str:
    parts = [str(row.get("scene", "?"))]
    if "method" in row and str(row.get("method", "")) not in {"", "nan"}:
        parts.append(str(row.get("method")))
    if "slot" in row:
        parts.append(str(row.get("slot")))
    if "pose" in row:
        parts.append(str(row.get("pose")))
    if "step" in row:
        parts.append(f"step={row.get('step')}")
    parts.append(str(row.get("setting", "?")))
    return " / ".join(parts)


def audit_rows(df: pd.DataFrame, label: str) -> list[dict]:
    issues: list[dict] = []
    for _, row in df.iterrows():
        scene = str(row.get("scene", ""))
        setting = str(row.get("setting", ""))
        if scene not in {"dark", "specular"}:
            continue
        leak = f(row, "scene_mask_on_back_wall_mean")
        far_area = f(row, "far_slit_area")
        far_fill = f(row, "far_slit_fill")
        clean = f(row, "clean_slit_shortcut")
        edge_clean = f(row, "clean_slit_edge_shortcut")
        visible_edge = f(row, "visible_slit_edge_shortcut")
        visible_body = f(row, "visible_slit_body_shortcut")
        front_area = f(row, "local_front_area")
        front_fill = f(row, "local_front_fill")
        invalid = f(row, "invalid_rate")

        if leak > 0.002:
            issues.append({
                "severity": "fatal",
                "label": label,
                "id": row_id(row),
                "issue": f"dark/specular material leaks onto back-wall hits ({leak:.4f})",
            })

        mostly_back_wall = far_area >= 0.55 and front_area <= 0.30
        bad = setting in BAD_SETTINGS.get(scene, set())
        good = setting in GOOD_SETTINGS.get(scene, set())

        if bad and mostly_back_wall and far_fill < 0.22 and front_area <= 0.12:
            issues.append({
                "severity": "major",
                "label": label,
                "id": row_id(row),
                "issue": (
                    "over-degraded ordinary back-wall cue: "
                    f"far_area={far_area:.3f}, front_area={front_area:.3f}, "
                    f"far_fill={far_fill:.3f}, invalid={invalid:.3f}"
                ),
            })

        if bad and front_area >= 0.12 and clean > 0.65:
            issues.append({
                "severity": "major",
                "label": label,
                "id": row_id(row),
                "issue": (
                    "too-clean full slit/back-wall template under bad setting: "
                    f"clean={clean:.3f}, far_fill={far_fill:.3f}, front_fill={front_fill:.3f}"
                ),
            })

        if bad and front_area >= 0.20 and edge_clean > 0.45:
            issues.append({
                "severity": "major",
                "label": label,
                "id": row_id(row),
                "issue": (
                    "too-clean aperture edge under bad setting: "
                    f"edge_clean={edge_clean:.3f}, front_fill={front_fill:.3f}"
                ),
            })

        if bad and front_area >= 0.20 and visible_edge > 0.55:
            issues.append({
                "severity": "major",
                "label": label,
                "id": row_id(row),
                "issue": (
                    "too-visible aperture-edge template under bad setting: "
                    f"visible_edge={visible_edge:.3f}, front_fill={front_fill:.3f}"
                ),
            })

        if bad and front_area >= 0.35 and visible_body > 0.70:
            issues.append({
                "severity": "major",
                "label": label,
                "id": row_id(row),
                "issue": (
                    "too-visible full aperture body template under bad setting: "
                    f"visible_body={visible_body:.3f}, front_fill={front_fill:.3f}"
                ),
            })

        if good and far_area >= 0.25 and far_fill < 0.70:
            issues.append({
                "severity": "major",
                "label": label,
                "id": row_id(row),
                "issue": (
                    "expected recovery setting still loses most slit/back-wall cue: "
                    f"far_area={far_area:.3f}, far_fill={far_fill:.3f}, invalid={invalid:.3f}"
                ),
            })
    return issues


def load_csvs(base: Path) -> list[tuple[str, pd.DataFrame]]:
    out: list[tuple[str, pd.DataFrame]] = []
    candidates = [
        ("opening", base / "opening_probe" / "opening_depth_probe_detail.csv"),
        ("rollout_fixed", base / "rollout_probe_fixed" / "rollout_depth_probe_detail.csv"),
        ("rollout_randfix", base / "rollout_probe_randfix" / "rollout_depth_probe_detail.csv"),
        ("rollout_flightonly", base / "rollout_probe_flightonly" / "rollout_depth_probe_detail.csv"),
        ("journal_sequences", base / "journal_depth_sequences" / "qualitative_depth" / "depth_sequence_rows.csv"),
    ]
    for label, path in candidates:
        if path.exists():
            out.append((label, pd.read_csv(path)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--probe_dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    base = Path(args.probe_dir)
    all_issues: list[dict] = []
    summaries: list[str] = []
    for label, df in load_csvs(base):
        issues = audit_rows(df, label)
        all_issues.extend(issues)
        dark_spec = df[df["scene"].isin(["dark", "specular"])] if "scene" in df.columns else df
        summaries.append(
            f"- `{label}`: rows={len(df)}, dark/spec rows={len(dark_spec)}, issues={len(issues)}"
        )

    fatal = [x for x in all_issues if x["severity"] == "fatal"]
    major = [x for x in all_issues if x["severity"] == "major"]
    lines = [
        "# Sensor Probe Visual-Semantics Audit",
        "",
        f"- probe_dir: `{base}`",
        f"- fatal issues: `{len(fatal)}`",
        f"- major issues: `{len(major)}`",
        "",
        "## Coverage",
        "",
        *summaries,
        "",
        "## Issues",
        "",
    ]
    if not all_issues:
        lines.append("- none")
    else:
        for issue in all_issues[:120]:
            lines.append(
                f"- **{issue['severity']}** `{issue['label']}` `{issue['id']}`: {issue['issue']}"
            )
        if len(all_issues) > 120:
            lines.append(f"- ... {len(all_issues) - 120} more")
    text = "\n".join(lines) + "\n"
    out = Path(args.out) if args.out else base / "visual_semantics_audit.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"[audit] wrote: {out}")


if __name__ == "__main__":
    main()

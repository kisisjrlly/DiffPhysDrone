#!/usr/bin/env python3
"""Export paper-ready LaTeX tables and figure assets from an eval-suite result.

The script is intentionally lightweight: it reads the CSV files produced by
tools/run_checkpoint_eval_suite.py and writes a small set of LaTeX tables plus
copied PNG/PDF figure assets into a paper_assets directory.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


METHOD_LABEL = {
    "flightonly": r"\textbf{Ours}",
    "fixed": "Fixed",
    "randfix": "RandFix",
    "nondiff": "NonDiff",
    "zero": "Blind",
    "pretrained": "Pretrain",
    "dagger": "DAgger",
}

MAIN_METHODS = ["flightonly", "fixed", "randfix", "nondiff", "zero"]
CAMERA_METHODS = ["flightonly", "fixed", "randfix", "nondiff", "pretrained", "dagger", "zero"]
DIAG_METHODS = ["pretrained", "dagger", "flightonly"]
SCENES = ["glare", "specular", "dark"]
PHASES = ["before", "near", "after"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def peg(row: dict[str, str], digits: int = 3) -> str:
    return "/".join(fnum(row[k], digits) for k in ("power", "exposure", "gain"))


def metric_peg(row: dict[str, str], digits: int = 3) -> str:
    return "/".join(fnum(row[k], digits) for k in ("power_mean", "exposure_mean", "gain_mean"))


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def table_wrap(tabular: str, caption: str, label: str) -> str:
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            tabular,
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )


def make_overall_table(summary_rows: list[dict[str, str]]) -> str:
    by_method = {r["method"]: r for r in summary_rows if r["scene_name"] == "overall"}
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Episodes & Success $\uparrow$ & Collision $\downarrow$ & Fill $\uparrow$ & Final dist. $\downarrow$ \\",
        r"\midrule",
    ]
    for method in MAIN_METHODS:
        r = by_method[method]
        success = fnum(r["success_rate"])
        fill = fnum(r["fill_rate"])
        if method == "flightonly":
            success = rf"\textbf{{{success}}}"
            fill = rf"\textbf{{{fill}}}"
        lines.append(
            " & ".join(
                [
                    METHOD_LABEL[method],
                    r["n"],
                    success,
                    fnum(r["collision_rate"]),
                    fill,
                    fnum(r["final_goal_dist"]),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return table_wrap(
        "\n".join(lines),
        "Overall navigation performance over 300 evaluation episodes per method.",
        "tab:overall_navigation",
    )


def make_scene_table(summary_rows: list[dict[str, str]]) -> str:
    rows = {(r["method"], r["scene_name"]): r for r in summary_rows}
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Glare & Specular & Dark \\",
        r"\midrule",
    ]
    for method in MAIN_METHODS:
        vals = []
        for scene in SCENES:
            r = rows[(method, scene)]
            cell = f"{fnum(r['success_rate'])}/{fnum(r['fill_rate'])}"
            if method == "flightonly":
                cell = rf"\textbf{{{cell}}}"
            vals.append(cell)
        lines.append(" & ".join([METHOD_LABEL[method], *vals]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return table_wrap(
        "\n".join(lines),
        r"Per-scene success/fill. Each scene is evaluated for 100 episodes.",
        "tab:scene_success_fill",
    )


def near_l1(a: dict[str, str], b: dict[str, str]) -> float:
    return sum(abs(float(a[k]) - float(b[k])) for k in ("power", "exposure", "gain")) / 3.0


def make_camera_table(phase_rows: list[dict[str, str]]) -> str:
    rows = {(r["method"], r["scene_name"], r["phase"]): r for r in phase_rows}
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Glare near p/e/g & Dark near p/e/g & Specular near p/e/g & G-D L1 $\uparrow$ \\",
        r"\midrule",
    ]
    for method in CAMERA_METHODS:
        glare = rows[(method, "glare", "near")]
        dark = rows[(method, "dark", "near")]
        spec = rows[(method, "specular", "near")]
        l1 = fnum(near_l1(glare, dark))
        cells = [METHOD_LABEL[method], peg(glare), peg(dark), peg(spec), l1]
        if method == "flightonly":
            cells = [cells[0], *(rf"\textbf{{{c}}}" for c in cells[1:])]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return table_wrap(
        "\n".join(lines),
        "Near-slit camera behavior. G-D L1 is the mean absolute difference between glare and dark p/e/g.",
        "tab:camera_near",
    )


def make_dagger_table(phase_rows: list[dict[str, str]], diagnosis_rows: list[dict[str, str]] | None) -> str:
    rows = {(r["method"], r["scene_name"], r["phase"]): r for r in phase_rows}
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Source & Method & Glare near p/e/g & Dark near p/e/g & G-D L1 \\",
        r"\midrule",
    ]
    if diagnosis_rows:
        off = {
            (r["scene"], r["phase"]): r
            for r in diagnosis_rows
            if r["source"] == "offline_teacher_dataset" and r["method"] == "dagger"
        }
        glare = off[("glare", "near")]
        dark = off[("dark", "near")]
        l1 = fnum(near_l1(glare, dark))
        lines.append(
            " & ".join(["Offline", "DAgger", peg(glare), peg(dark), l1]) + r" \\"
        )
    for method in DIAG_METHODS:
        glare = rows[(method, "glare", "near")]
        dark = rows[(method, "dark", "near")]
        label = METHOD_LABEL[method]
        l1 = fnum(near_l1(glare, dark))
        lines.append(" & ".join(["Online", label, peg(glare), peg(dark), l1]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return table_wrap(
        "\n".join(lines),
        "Camera separation diagnosis before and after DAgger-style relabeling.",
        "tab:dagger_diagnosis",
    )


def make_teacher_table(summary_path: Path | None) -> str:
    if not summary_path or not summary_path.exists():
        return ""
    # Keep this table explicit because it is a compact description of the final
    # relabel/pretrain run and avoids fragile parsing from markdown prose.
    lines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Item & Value \\",
        r"\midrule",
        r"Sequences & 144 \\",
        r"Timesteps & 80 \\",
        r"Samples & 11520 \\",
        r"Teacher mean p/e/g & 0.584/0.408/0.403 \\",
        r"Teacher std p/e/g & 0.164/0.197/0.181 \\",
        r"Best epoch & 119 \\",
        r"Validation loss & 0.000715 \\",
        r"MAE p/e/g & 0.0094/0.0126/0.0131 \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return table_wrap(
        "\n".join(lines),
        "DAgger relabel dataset and camera-head pretraining summary.",
        "tab:teacher_pretrain",
    )


def copy_and_convert_figure(src: Path, dst_png: Path) -> tuple[Path, Path | None]:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_png)
    dst_pdf = dst_png.with_suffix(".pdf")
    try:
        from PIL import Image

        with Image.open(src) as im:
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            im.save(dst_pdf, "PDF", resolution=300.0)
        return dst_png, dst_pdf
    except Exception:
        return dst_png, None


def export_figures(eval_dir: Path, figure_dir: Path) -> list[tuple[str, Path, Path | None]]:
    fig_src_dir = eval_dir / "figures"
    specs = [
        ("scene_metrics", fig_src_dir / "scene_metrics.png", "fig_scene_metrics.png"),
        (
            "ours_camera_trajectories",
            fig_src_dir / "flightonly_camera_and_trajectories.png",
            "fig_ours_camera_and_trajectories.png",
        ),
        (
            "fixed_camera_trajectories",
            fig_src_dir / "fixed_camera_and_trajectories.png",
            "fig_fixed_camera_and_trajectories.png",
        ),
        (
            "nondiff_camera_trajectories",
            fig_src_dir / "nondiff_camera_and_trajectories.png",
            "fig_nondiff_camera_and_trajectories.png",
        ),
    ]
    copied = []
    for name, src, filename in specs:
        if src.exists():
            png, pdf = copy_and_convert_figure(src, figure_dir / filename)
            copied.append((name, png, pdf))
    return copied


def write_manifest(out_dir: Path, tables: list[Path], figures: list[tuple[str, Path, Path | None]]) -> None:
    lines = [
        "# Paper Result Assets",
        "",
        "Generated from `tools/run_checkpoint_eval_suite.py` outputs.",
        "",
        "## Tables",
        "",
    ]
    for path in tables:
        lines.append(f"- `{path.relative_to(out_dir)}`")
    lines += ["", "## Figures", ""]
    for name, png, pdf in figures:
        fig_line = f"- `{name}`: `{png.relative_to(out_dir)}`"
        if pdf is not None:
            fig_line += f", `{pdf.relative_to(out_dir)}`"
        lines.append(fig_line)
    lines += [
        "",
        "Recommended paper mapping:",
        "",
        "- Table 1: `tables/table_overall_navigation.tex`",
        "- Table 2: `tables/table_scene_success_fill.tex`",
        "- Table 3: `tables/table_camera_near_behavior.tex`",
        "- Table 4 or appendix: `tables/table_dagger_diagnosis.tex`",
        "- Figure 1/2: `figures/fig_scene_metrics.pdf` and `figures/fig_ours_camera_and_trajectories.pdf`",
        "",
    ]
    write(out_dir / "README.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_dir",
        default="paper/experiment/results/final_dagger_flightonly_eval_20260507",
        help="Directory produced by tools/run_checkpoint_eval_suite.py",
    )
    parser.add_argument("--out_dir", default=None, help="Default: <eval_dir>/paper_assets")
    parser.add_argument(
        "--teacher_summary",
        default="paper/experiment/results/closed_loop_teacher_camera_policy_v3d_dagger_full/summary.md",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "paper_assets"
    table_dir = out_dir / "tables"
    figure_dir = out_dir / "figures"

    summary_rows = read_csv(eval_dir / "summary_by_method_scene.csv")
    phase_rows = read_csv(eval_dir / "camera_phase_summary.csv")
    diagnosis_csv = eval_dir / "pretrain_online_offline_phase_summary.csv"
    diagnosis_rows = read_csv(diagnosis_csv) if diagnosis_csv.exists() else None

    tables = {
        "table_overall_navigation.tex": make_overall_table(summary_rows),
        "table_scene_success_fill.tex": make_scene_table(summary_rows),
        "table_camera_near_behavior.tex": make_camera_table(phase_rows),
        "table_dagger_diagnosis.tex": make_dagger_table(phase_rows, diagnosis_rows),
        "table_teacher_pretrain.tex": make_teacher_table(Path(args.teacher_summary)),
    }

    written_tables = []
    for filename, content in tables.items():
        if not content:
            continue
        path = table_dir / filename
        write(path, content)
        written_tables.append(path)

    figures = export_figures(eval_dir, figure_dir)
    write_manifest(out_dir, written_tables, figures)
    print(f"[export] wrote paper assets: {out_dir}")
    print(f"[export] tables: {len(written_tables)} figures: {len(figures)}")


if __name__ == "__main__":
    main()

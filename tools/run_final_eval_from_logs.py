#!/usr/bin/env python3
"""Run the final active-sensing evaluation suite from training logs.

The training logs contain the authoritative checkpoint paths for each mode. This
wrapper extracts those paths, runs the current checkpoint eval suite, then
regenerates journal assets and qualitative depth-sequence figures.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODE_ORDER = ["flightonly", "fixed", "randfix", "nondiff", "zero"]
MAIN_METHODS = ["flightonly", "fixed", "randfix", "nondiff", "zero"]
QUAL_METHODS = ["flightonly", "fixed", "randfix"]


def infer_mode(path: Path, text: str) -> str | None:
    name = path.name.lower()
    for mode in ["flightonly", "randfix", "nondiff", "zero", "fix"]:
        if f"_auto_{mode}" in name or f"-slit_active_sensing_auto_{mode}-" in name:
            return "fixed" if mode == "fix" else mode
    m = re.search(r"\[train-suite\]\s+mode=(\w+)", text)
    if m:
        mode = m.group(1).strip().lower()
        return "fixed" if mode == "fix" else mode
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
    missing = [m for m in MAIN_METHODS if m not in ckpts]
    if missing:
        found = ", ".join(f"{k}={v}" for k, v in sorted(ckpts.items()))
        raise SystemExit(f"missing checkpoints for modes {missing}; found: {found}")
    for mode, ckpt in ckpts.items():
        if not ckpt.is_file():
            raise SystemExit(f"{mode} checkpoint not found: {ckpt}")
    return ckpts


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("[final-eval] $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[final-eval] copied {src} -> {dst}")


def maybe_copy_diagnostics(args: argparse.Namespace, out_dir: Path) -> None:
    diag_dir = Path(args.diag_eval_dir)
    if not diag_dir.exists():
        print(f"[final-eval][warn] diagnostic eval dir not found, skipping: {diag_dir}")
        return
    for name in [
        "pretrained_episodes.csv",
        "pretrained_trace.csv",
        "dagger_episodes.csv",
        "dagger_trace.csv",
    ]:
        copy_if_exists(diag_dir / "raw" / name, out_dir / "raw" / name)
    offline_dir = Path(args.diag_summary_dir)
    copy_if_exists(offline_dir / "pretrain_online_offline_phase_summary.csv", out_dir / "pretrain_online_offline_phase_summary.csv")
    copy_if_exists(offline_dir / "diagnosis.md", out_dir / "diagnosis.md")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", type=Path)
    p.add_argument("--config", default="configs/slit_active_sensing.args")
    p.add_argument("--out_dir", default="paper/experiment/results/final_semantics_v3_eval_20260508")
    p.add_argument("--episodes_per_scene", type=int, default=100)
    p.add_argument("--scenarios", nargs="*", default=["glare", "specular", "dark"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip_eval", action="store_true", help="Do not rerun closed-loop eval; only regenerate assets.")
    p.add_argument("--skip_assets", action="store_true")
    p.add_argument("--skip_depth_sequences", action="store_true")
    p.add_argument("--diag_eval_dir", default="paper/experiment/results/pretrain_dagger_eval_semantics_v3_finaldiag_20260508")
    p.add_argument("--diag_summary_dir", default="paper/experiment/results/pretrain_dagger_eval_semantics_v3")
    p.add_argument("--qual_slot", default="far_right")
    p.add_argument("--qual_target_local_x", default="-1.20,-0.75,-0.35,-0.08,0.18")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logs = [p if p.is_absolute() else ROOT / p for p in args.logs]
    for log in logs:
        if not log.is_file():
            raise SystemExit(f"log not found: {log}")

    ckpts = extract_checkpoints(logs)
    for mode in MODE_ORDER:
        print(f"[final-eval] {mode}: {ckpts[mode].relative_to(ROOT)}")

    out_dir = ROOT / args.out_dir
    if not args.skip_eval:
        cmd = [
            args.python,
            "-u",
            "tools/run_checkpoint_eval_suite.py",
            "--config",
            args.config,
            "--out_dir",
            args.out_dir,
            "--episodes_per_scene",
            str(args.episodes_per_scene),
            "--scenarios",
            *args.scenarios,
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--methods",
            *MAIN_METHODS,
        ]
        for mode in MAIN_METHODS:
            cmd += [f"--{mode}_ckpt", str(ckpts[mode].relative_to(ROOT))]
        run(cmd)

    maybe_copy_diagnostics(args, out_dir)

    if not args.skip_assets:
        run([
            args.python,
            "tools/make_journal_assets.py",
            "--eval_dir",
            args.out_dir,
            "--out_dir",
            f"{args.out_dir}/journal_assets",
        ])

    if not args.skip_depth_sequences:
        cmd = [
            args.python,
            "-u",
            "tools/export_journal_depth_sequences.py",
            "--config",
            args.config,
            "--eval_dir",
            args.out_dir,
            "--out_dir",
            f"{args.out_dir}/journal_assets",
            "--scenarios",
            "glare",
            "dark",
            "specular",
            "--slot",
            args.qual_slot,
            f"--target_local_x={args.qual_target_local_x}",
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        for mode in QUAL_METHODS:
            cmd += [f"--{mode}_ckpt", str(ckpts[mode].relative_to(ROOT))]
        run(cmd)

    print(f"[final-eval] done: {out_dir}")


if __name__ == "__main__":
    main()

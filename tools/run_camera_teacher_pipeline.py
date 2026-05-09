#!/usr/bin/env python3
"""Orchestrate frozen-policy camera-teacher collection, pretraining, and eval.

This script is intentionally thin: the heavy work stays in
generate_camera_teacher_dataset.py, pretrain_camera_head.py, and eval.py.  The
value here is reproducibility and bookkeeping for the three-stage workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLIGHT_CKPT = ""


def _read_args_file(path: Path) -> list[str]:
    tokens: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            tokens.extend(shlex.split(line))
    return tokens


def _find_python() -> str:
    if os.environ.get("PYTHON_BIN"):
        return os.environ["PYTHON_BIN"]
    conda_python = Path.home() / "miniconda3/envs/mappo-mpc/bin/python"
    if conda_python.exists():
        return str(conda_python)
    return sys.executable


def _run_command(cmd: list[str], log_path: Path, *, dry_run: bool = False) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"[pipeline] $ {printable}")
    if dry_run:
        log_path.write_text(printable + "\n", encoding="utf-8")
        return ""

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as f:
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            lines.append(line)
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return "".join(lines)


def _as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / max(len(vals), 1)


def _summarize_episode_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keys = [
        "success_rate",
        "collision_rate",
        "goal_reach_rate",
        "final_goal_dist",
        "avg_speed",
        "fill_rate",
        "power_mean",
        "exposure_mean",
        "gain_mean",
    ]
    overall = {key: _mean(_as_float(row, key) for row in rows) for key in keys}
    by_scene: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("scene_name", "unknown")].append(row)
    for scene, scene_rows in grouped.items():
        by_scene[scene] = {key: _mean(_as_float(row, key) for row in scene_rows) for key in keys}
    return {"episodes": len(rows), "overall": overall, "by_scene": by_scene}


def _summarize_teacher_dataset(path: Path) -> dict:
    if not path.exists():
        return {}
    data = torch.load(path, map_location="cpu")
    teacher = data["teacher_camera"].float()
    scene_id = data.get("scene_id")
    local_x = data.get("local_x")
    meta = data.get("meta", {})
    scenes = list(meta.get("scenarios", []))
    summary = {
        "path": str(path),
        "sequences": int(teacher.shape[0]),
        "timesteps": int(teacher.shape[1]) if teacher.ndim >= 3 else 0,
        "samples": int(teacher.shape[0] * teacher.shape[1]) if teacher.ndim >= 3 else int(teacher.numel() // 3),
        "teacher_mean_p_e_g": [float(x) for x in teacher.mean(dim=(0, 1))],
        "teacher_std_p_e_g": [float(x) for x in teacher.std(dim=(0, 1))],
        "meta": meta,
    }
    if scene_id is not None and scenes:
        sid = scene_id[:, 0].long() if scene_id.ndim >= 2 else scene_id.long()
        by_scene = {}
        for idx, name in enumerate(scenes):
            mask = sid == idx
            if bool(mask.any()):
                cam = teacher[mask]
                by_scene[name] = {
                    "sequences": int(mask.sum()),
                    "mean_p_e_g": [float(x) for x in cam.mean(dim=(0, 1))],
                    "std_p_e_g": [float(x) for x in cam.std(dim=(0, 1))],
                }
        summary["by_scene"] = by_scene
    if local_x is not None:
        bins = {
            "before_wall_x_lt_-0.25": local_x < -0.25,
            "near_wall_abs_x_le_0.25": local_x.abs() <= 0.25,
            "after_wall_x_gt_0.25": local_x > 0.25,
        }
        by_x = {}
        for name, mask in bins.items():
            if bool(mask.any()):
                cam = teacher[mask]
                by_x[name] = {
                    "samples": int(mask.sum()),
                    "mean_p_e_g": [float(x) for x in cam.mean(dim=0)],
                    "std_p_e_g": [float(x) for x in cam.std(dim=0)],
                }
        summary["by_local_x_bin"] = by_x
    return summary


def _summarize_pretrain_log(path: Path) -> dict:
    if not path.exists():
        return {}
    epochs = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[camera-pretrain] epoch=" not in line:
            continue
        row: dict[str, object] = {"raw": line}
        for token in line.split():
            if token.startswith("epoch="):
                try:
                    row["epoch"] = int(token.split("=", 1)[1])
                except ValueError:
                    pass
            elif token.startswith("train="):
                row["train_loss"] = float(token.split("=", 1)[1])
            elif token.startswith("val="):
                row["val_loss"] = float(token.split("=", 1)[1])
            elif token.startswith("mae_p/e/g="):
                vals = token.split("=", 1)[1].split("/")
                if len(vals) == 3:
                    row["mae_p_e_g"] = [float(x) for x in vals]
        epochs.append(row)
    out = {"epochs": epochs}
    vals = [row for row in epochs if "val_loss" in row]
    if vals:
        best = min(vals, key=lambda row: float(row["val_loss"]))
        out["best"] = best
        out["final"] = vals[-1]
    return out


def _write_markdown(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Camera Teacher Pipeline Summary",
        "",
        f"- dataset: `{summary.get('dataset_path', '')}`",
        f"- teacher quality: `{summary.get('teacher_quality_report', '')}`",
        f"- pretrained checkpoint: `{summary.get('pretrain_checkpoint', '')}`",
        f"- best checkpoint: `{summary.get('best_checkpoint', '')}`",
        "",
    ]
    ds = summary.get("teacher_dataset", {})
    if ds:
        mean = ds.get("teacher_mean_p_e_g", [0, 0, 0])
        std = ds.get("teacher_std_p_e_g", [0, 0, 0])
        lines.extend([
            "## Teacher Dataset",
            "",
            f"- sequences: `{ds.get('sequences')}`, timesteps: `{ds.get('timesteps')}`, samples: `{ds.get('samples')}`",
            f"- teacher mean p/e/g: `{mean[0]:.3f}/{mean[1]:.3f}/{mean[2]:.3f}`",
            f"- teacher std p/e/g: `{std[0]:.3f}/{std[1]:.3f}/{std[2]:.3f}`",
            "",
        ])
        if ds.get("by_scene"):
            lines.append("| scene | seq | mean p/e/g | std p/e/g |")
            lines.append("|---|---:|---:|---:|")
            for scene, row in ds["by_scene"].items():
                m = row["mean_p_e_g"]
                s = row["std_p_e_g"]
                lines.append(
                    f"| {scene} | {row['sequences']} | "
                    f"{m[0]:.3f}/{m[1]:.3f}/{m[2]:.3f} | "
                    f"{s[0]:.3f}/{s[1]:.3f}/{s[2]:.3f} |"
                )
            lines.append("")
        if ds.get("by_local_x_bin"):
            lines.append("| local x bin | samples | mean p/e/g |")
            lines.append("|---|---:|---:|")
            for name, row in ds["by_local_x_bin"].items():
                m = row["mean_p_e_g"]
                lines.append(f"| {name} | {row['samples']} | {m[0]:.3f}/{m[1]:.3f}/{m[2]:.3f} |")
            lines.append("")

    pt = summary.get("pretrain", {})
    if pt:
        lines.extend(["## Camera Pretrain", ""])
        best = pt.get("best", {})
        final = pt.get("final", {})
        if best:
            mae = best.get("mae_p_e_g", [0, 0, 0])
            lines.append(
                f"- best epoch: `{best.get('epoch')}`, val: `{best.get('val_loss', 0.0):.6f}`, "
                f"MAE p/e/g: `{mae[0]:.4f}/{mae[1]:.4f}/{mae[2]:.4f}`"
            )
        if final:
            mae = final.get("mae_p_e_g", [0, 0, 0])
            lines.append(
                f"- final epoch: `{final.get('epoch')}`, val: `{final.get('val_loss', 0.0):.6f}`, "
                f"MAE p/e/g: `{mae[0]:.4f}/{mae[1]:.4f}/{mae[2]:.4f}`"
            )
        lines.append("")

    eval_summary = summary.get("eval", {})
    if eval_summary:
        lines.extend(["## Evaluation", ""])
        lines.append("| mode | episodes | success | collision | reach | final dist | fill | p/e/g |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for mode, row in eval_summary.items():
            overall = row.get("overall", {})
            lines.append(
                f"| {mode} | {row.get('episodes', 0)} | "
                f"{overall.get('success_rate', 0.0):.3f} | "
                f"{overall.get('collision_rate', 0.0):.3f} | "
                f"{overall.get('goal_reach_rate', 0.0):.3f} | "
                f"{overall.get('final_goal_dist', 0.0):.3f} | "
                f"{overall.get('fill_rate', 0.0):.3f} | "
                f"{overall.get('power_mean', 0.0):.3f}/"
                f"{overall.get('exposure_mean', 0.0):.3f}/"
                f"{overall.get('gain_mean', 0.0):.3f} |"
            )
        lines.append("")
        for mode, row in eval_summary.items():
            if not row.get("by_scene"):
                continue
            lines.append(f"### {mode} By Scene")
            lines.append("")
            lines.append("| scene | success | collision | final dist | fill | p/e/g |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for scene, sc in row["by_scene"].items():
                lines.append(
                    f"| {scene} | {sc.get('success_rate', 0.0):.3f} | "
                    f"{sc.get('collision_rate', 0.0):.3f} | "
                    f"{sc.get('final_goal_dist', 0.0):.3f} | "
                    f"{sc.get('fill_rate', 0.0):.3f} | "
                    f"{sc.get('power_mean', 0.0):.3f}/"
                    f"{sc.get('exposure_mean', 0.0):.3f}/"
                    f"{sc.get('gain_mean', 0.0):.3f} |"
                )
            lines.append("")
    lines.extend([
        "## Reading The Result",
        "",
        "- Pretrain is usually good enough when validation MAE is below roughly `0.05-0.08` per camera parameter and the eval trace keeps the expected time-varying shape.",
        "- The important comparison is not camera mean alone.  Check the trace CSV: parameters should change near the wall slit and relax after the hard region when the teacher also does so.",
        "- If learned camera does not beat fixed/random, first inspect the teacher dataset summary.  If teacher p/e/g has tiny std or little scene/local-x variation, improve teacher objective/data before changing the network.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "collect", "pretrain", "eval", "summary"], default="all")
    parser.add_argument("--config", default="configs/slit_active_sensing.args")
    parser.add_argument("--flight_checkpoint", default=DEFAULT_FLIGHT_CKPT)
    parser.add_argument("--work_dir", default="paper/experiment/results/camera_teacher_pipeline_20260503")
    parser.add_argument("--pretrain_dir", default="checkpoint/camera_pretrain_20260503")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--pretrain_out", default=None)
    parser.add_argument("--best_out", default=None)
    parser.add_argument("--python", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scenarios", nargs="*", default=["glare", "specular", "dark"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--teacher_source",
        choices=["closed_loop_diffopt", "rollout_local", "trajectory_diffopt"],
        default="closed_loop_diffopt",
        help=(
            "closed_loop_diffopt/rollout_local: fly the frozen checkpoint and "
            "optimize a camera target at each visited state.  trajectory_diffopt: "
            "use scripted wall-crossing trajectories."
        ),
    )

    parser.add_argument("--reuse_dataset", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--reuse_pretrain", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--force_collect", action="store_true")
    parser.add_argument("--force_pretrain", action="store_true")

    parser.add_argument("--rollouts_per_scene", type=int, default=12)
    parser.add_argument("--collect_batch_size", type=int, default=12)
    parser.add_argument("--timesteps", type=int, default=80)
    parser.add_argument("--teacher_steps", type=int, default=50)
    parser.add_argument("--teacher_lr", type=float, default=0.10)
    parser.add_argument("--teacher_every", type=int, default=1)
    parser.add_argument("--teacher_camera_ema", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--teacher_ema_alpha", type=float, default=0.7)
    parser.add_argument(
        "--rollout_camera_mode",
        choices=["fixed", "fixed_random_static", "learned"],
        default="fixed_random_static",
        help=(
            "Camera mode for collecting closed-loop states.  For a randfix-trained "
            "flight checkpoint, fixed_random_static usually matches the policy's "
            "training distribution best.  Use learned with --no-teacher_camera_ema "
            "for DAgger-style relabeling on a pretrained camera policy distribution."
        ),
    )
    parser.add_argument("--coef_nominal_when_healthy", type=float, default=0.5)
    parser.add_argument("--nominal_fill_margin", type=float, default=0.12)
    parser.add_argument("--slots", nargs="*", default=["left", "right"])
    parser.add_argument("--trajectory_teacher_per_scene_slot", type=int, default=None)
    parser.add_argument("--trajectory_xs", default="-1.20,-0.90,-0.60,-0.35,-0.18,-0.05,0.10,0.35,0.70,1.05,1.35")
    parser.add_argument("--trajectory_x_jitter", type=float, default=0.035)
    parser.add_argument("--trajectory_path_y_mode", default="slot", choices=["center", "blend", "slot"])
    parser.add_argument("--trajectory_target_mode", default="opening_then_goal", choices=["opening_then_goal", "opening", "goal"])
    parser.add_argument("--diffopt_random_restarts", type=int, default=4)
    parser.add_argument("--diffopt_randfix_k", type=int, default=24)
    parser.add_argument("--speed_mps", type=float, default=1.0)

    parser.add_argument("--pretrain_epochs", type=int, default=40)
    parser.add_argument("--pretrain_batch_size", type=int, default=8)
    parser.add_argument("--pretrain_lr", type=float, default=2e-4)
    parser.add_argument("--pretrain_weight_decay", type=float, default=1e-4)
    parser.add_argument("--temporal_smooth", type=float, default=0.02)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--resume_pretrain", default=None)
    parser.add_argument(
        "--train_shared_visual_encoder",
        action="store_true",
        help=(
            "During camera-head pretraining, also tune the shared depth encoder. "
            "Default trains only camera-specific visual modules and camera head."
        ),
    )

    parser.add_argument("--eval_modes", nargs="*", default=["fixed", "randfix", "learned"])
    parser.add_argument("--eval_episodes", type=int, default=30)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_trace", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_vis", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--vis_episode_idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_known_args()


def main() -> None:
    args, project_overrides = _parse_args()
    work_dir = Path(args.work_dir)
    pretrain_dir = Path(args.pretrain_dir)
    dataset = Path(args.dataset) if args.dataset else work_dir / "camera_teacher_dataset.pt"
    pretrain_out = Path(args.pretrain_out) if args.pretrain_out else pretrain_dir / "camera_head_pretrained.pth"
    best_out = Path(args.best_out) if args.best_out else pretrain_dir / "camera_head_pretrained_best.pth"
    logs_dir = work_dir / "logs"
    eval_dir = work_dir / "eval"
    summary_json = work_dir / "summary.json"
    summary_md = work_dir / "summary.md"
    python = args.python or _find_python()

    config = Path(args.config)
    flight_checkpoint = Path(args.flight_checkpoint)
    if not config.exists():
        raise FileNotFoundError(config)
    if not str(args.flight_checkpoint).strip():
        raise ValueError("--flight_checkpoint is required; train a flight policy first")
    if not flight_checkpoint.exists():
        raise FileNotFoundError(flight_checkpoint)

    collect_needed = args.stage in {"all", "collect"} and (
        args.force_collect or not (args.reuse_dataset and dataset.exists())
    )
    pretrain_needed = args.stage in {"all", "pretrain"} and (
        args.force_pretrain or not (args.reuse_pretrain and pretrain_out.exists())
    )
    eval_needed = args.stage in {"all", "eval"}

    if collect_needed:
        if args.teacher_source == "trajectory_diffopt":
            traj_per = (
                int(args.trajectory_teacher_per_scene_slot)
                if args.trajectory_teacher_per_scene_slot is not None
                else max(1, int(args.rollouts_per_scene))
            )
            cmd = [
                python, "-u", str(ROOT / "tools/generate_trajectory_diffopt_camera_dataset.py"),
                "--config", str(config),
                "--out", str(dataset),
                "--device", str(args.device),
                "--scenarios", *args.scenarios,
                "--slots", *args.slots,
                "--trajectories_per_scene_slot", str(traj_per),
                f"--xs={args.trajectory_xs}",
                "--x_jitter", str(args.trajectory_x_jitter),
                "--path_y_mode", str(args.trajectory_path_y_mode),
                "--target_mode", str(args.trajectory_target_mode),
                "--teacher_steps", str(args.teacher_steps),
                "--teacher_lr", str(args.teacher_lr),
                "--diffopt_random_restarts", str(args.diffopt_random_restarts),
                "--randfix_k", str(args.diffopt_randfix_k),
                "--teacher_ema_alpha", str(args.teacher_ema_alpha),
                "--speed_mps", str(args.speed_mps),
                "--seed", str(args.seed),
            ]
        else:
            cmd = [
                python, "-u", str(ROOT / "tools/generate_camera_teacher_dataset.py"),
                "--config", str(config),
                "--checkpoint", str(flight_checkpoint),
                "--out", str(dataset),
                "--device", str(args.device),
                "--scenarios", *args.scenarios,
                "--rollouts_per_scene", str(args.rollouts_per_scene),
                "--batch_size", str(args.collect_batch_size),
                "--timesteps", str(args.timesteps),
                "--teacher_steps", str(args.teacher_steps),
                "--teacher_lr", str(args.teacher_lr),
                "--teacher_every", str(args.teacher_every),
                "--teacher_ema_alpha", str(args.teacher_ema_alpha),
                "--rollout_camera_mode", str(args.rollout_camera_mode),
                "--coef_nominal_when_healthy", str(args.coef_nominal_when_healthy),
                "--nominal_fill_margin", str(args.nominal_fill_margin),
                "--quality_out", str(work_dir / "teacher_dataset_quality.md"),
                "--seed", str(args.seed),
            ]
            if not args.teacher_camera_ema:
                cmd.append("--no-teacher_camera_ema")
        cmd.extend(project_overrides)
        _run_command(cmd, logs_dir / "collect.log", dry_run=args.dry_run)
    elif args.stage in {"all", "collect"}:
        print(f"[pipeline] reusing dataset: {dataset}")

    if args.stage in {"all", "pretrain"} and not dataset.exists() and not args.dry_run:
        raise FileNotFoundError(f"dataset not found: {dataset}")

    if pretrain_needed:
        cmd = [
            python, "-u", str(ROOT / "tools/pretrain_camera_head.py"),
            "--config", str(config),
            "--checkpoint", str(flight_checkpoint),
            "--dataset", str(dataset),
            "--out", str(pretrain_out),
            "--best_out", str(best_out),
            "--device", str(args.device),
            "--epochs", str(args.pretrain_epochs),
            "--batch_size", str(args.pretrain_batch_size),
            "--lr", str(args.pretrain_lr),
            "--weight_decay", str(args.pretrain_weight_decay),
            "--temporal_smooth", str(args.temporal_smooth),
            "--val_fraction", str(args.val_fraction),
            "--seed", str(args.seed),
        ]
        if args.resume_pretrain:
            cmd.extend(["--resume", str(args.resume_pretrain)])
        if args.train_shared_visual_encoder:
            cmd.append("--train_shared_visual_encoder")
        cmd.extend(project_overrides)
        _run_command(cmd, logs_dir / "pretrain.log", dry_run=args.dry_run)
    elif args.stage in {"all", "pretrain"}:
        print(f"[pipeline] reusing pretrained camera checkpoint: {pretrain_out}")

    learned_ckpt = best_out if best_out.exists() else pretrain_out
    if eval_needed:
        if not learned_ckpt.exists() and not args.dry_run:
            raise FileNotFoundError(f"learned camera checkpoint not found: {learned_ckpt}")
        cfg_tokens = [tok for tok in _read_args_file(config) if tok != "--vis_enable"]
        mode_map = {
            "fixed": ("fixed", flight_checkpoint, "full"),
            "fix": ("fixed", flight_checkpoint, "full"),
            "randfix": ("fixed_random_static", flight_checkpoint, "full"),
            "fixed_random": ("fixed_random_static", flight_checkpoint, "full"),
            "learned": ("learned", learned_ckpt, "full"),
            "learned_detached": ("learned", learned_ckpt, "detached"),
        }
        for mode in args.eval_modes:
            key = mode.lower()
            if key not in mode_map:
                raise ValueError(f"unsupported eval mode {mode!r}; choose {sorted(mode_map)}")
            camera_mode, ckpt, sensor_grad_mode = mode_map[key]
            episode_csv = eval_dir / f"{key}_episodes.csv"
            trace_csv = eval_dir / f"{key}_trace.csv"
            cmd = [
                python, "-u", str(ROOT / "eval.py"),
                *cfg_tokens,
                *project_overrides,
                "--resume", str(ckpt),
                "--wandb_disabled",
                "--eval_episodes", str(args.eval_episodes),
                "--vis_episode_idx", str(args.vis_episode_idx),
                "--batch_size", str(args.eval_batch_size),
                "--scenarios", *args.scenarios,
                "--camera_control_mode", camera_mode,
                "--sensor_grad_mode", sensor_grad_mode,
                "--policy_depth_mode", "depth",
                "--seed", str(args.seed),
                "--eval_episode_csv", str(episode_csv),
            ]
            if args.eval_trace:
                cmd.extend(["--eval_trace_csv", str(trace_csv)])
            if args.eval_vis:
                cmd.append("--vis_enable")
            _run_command(cmd, logs_dir / f"eval_{key}.log", dry_run=args.dry_run)

    summary = {
        "config": str(config),
        "flight_checkpoint": str(flight_checkpoint),
        "teacher_source": str(args.teacher_source),
        "dataset_path": str(dataset),
        "teacher_quality_report": str(work_dir / "teacher_dataset_quality.md"),
        "pretrain_checkpoint": str(pretrain_out),
        "best_checkpoint": str(best_out),
        "teacher_dataset": _summarize_teacher_dataset(dataset) if dataset.exists() else {},
        "pretrain": _summarize_pretrain_log(logs_dir / "pretrain.log"),
        "eval": {},
    }
    for mode in args.eval_modes:
        key = mode.lower()
        episode_csv = eval_dir / f"{key}_episodes.csv"
        if episode_csv.exists():
            summary["eval"][key] = _summarize_episode_csv(episode_csv)
    work_dir.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(summary, summary_md)
    print(f"[pipeline] summary json: {summary_json}")
    print(f"[pipeline] summary md  : {summary_md}")


if __name__ == "__main__":
    main()

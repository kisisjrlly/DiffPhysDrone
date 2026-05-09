#!/usr/bin/env python3
"""Evaluate current active-sensing checkpoints and summarize paper-ready metrics."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    build_parser,
    parse_diff_sensor_impl,
    parse_scenarios,
    set_global_seed,
    validate_args,
)
from eval import _write_csv_rows, run_one_episode  # noqa: E402
from model import Model  # noqa: E402
from train_utils import build_env  # noqa: E402


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    checkpoint: Path
    camera_control_mode: str
    sensor_grad_mode: str
    policy_depth_mode: str
    train_flight_only: bool = False


def _read_args_file(path: Path) -> list[str]:
    import shlex

    tokens: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            tokens.extend(shlex.split(line))
    return tokens


def _load_base_args(config: Path):
    parser = build_parser()
    args = parser.parse_args(_read_args_file(config))
    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.scenarios = parse_scenarios(args.scenarios)
    args.wandb_disabled = True
    args.vis_enable = False
    validate_args(args)
    return args


def _make_model(args, device):
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


def _load_model(args, ckpt: Path, device):
    model = _make_model(args, device)
    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row and row[key] not in ("", None)]
    return sum(vals) / max(len(vals), 1)


def _group(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict]]:
    out: dict[tuple[str, ...], list[dict]] = {}
    for row in rows:
        group_key = tuple(str(row.get(k, "")) for k in keys)
        out.setdefault(group_key, []).append(row)
    return out


def _summarize_episodes(rows: list[dict]) -> list[dict]:
    metrics = [
        "success_rate",
        "collision_rate",
        "goal_reach_rate",
        "final_goal_dist",
        "avg_speed",
        "fill_rate",
        "power_mean",
        "exposure_mean",
        "gain_mean",
        "steps",
    ]
    out = []
    for key, vals in sorted(_group(rows, ("method", "scene_name")).items()):
        method, scene = key
        row = {"method": method, "scene_name": scene, "n": len(vals)}
        for metric in metrics:
            row[metric] = _mean(vals, metric)
        out.append(row)
    for key, vals in sorted(_group(rows, ("method",)).items()):
        (method,) = key
        row = {"method": method, "scene_name": "overall", "n": len(vals)}
        for metric in metrics:
            row[metric] = _mean(vals, metric)
        out.append(row)
    return out


def _phase(local_x: float) -> str:
    if local_x < -0.25:
        return "before"
    if local_x <= 0.25:
        return "near"
    return "after"


def _summarize_trace(trace_rows: list[dict]) -> list[dict]:
    metrics = ["power", "exposure", "gain", "scene_effect_mean", "min_margin", "clearance", "goal_dist"]
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in trace_rows:
        lx_key = "local_x" if row.get("local_x", "") not in ("", None) else "x"
        phase = _phase(float(row.get(lx_key, 0.0)))
        key = (str(row.get("method", "")), str(row.get("scene_name", "")), phase)
        grouped.setdefault(key, []).append(row)
    out = []
    for (method, scene, phase), vals in sorted(grouped.items()):
        item = {"method": method, "scene_name": scene, "phase": phase, "n": len(vals)}
        for metric in metrics:
            item[metric] = _mean(vals, metric)
        out.append(item)
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
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


def _camera_scene_separation(summary_trace: list[dict], method: str) -> dict:
    near = {
        row["scene_name"]: row
        for row in summary_trace
        if row["method"] == method and row["phase"] == "near"
    }
    out = {}
    for a, b in [("glare", "dark"), ("glare", "specular"), ("dark", "specular")]:
        if a not in near or b not in near:
            continue
        da = near[a]
        db = near[b]
        diffs = [abs(float(da[k]) - float(db[k])) for k in ("power", "exposure", "gain")]
        out[f"{a}_vs_{b}_near_l1"] = sum(diffs) / 3.0
        out[f"{a}_vs_{b}_near_p_e_g"] = diffs
    return out


def _make_plots(out_dir: Path, episode_rows: list[dict], trace_rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is best effort
        print(f"[suite][warn] plotting disabled: {exc}")
        return

    plot_dir = out_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted({row["method"] for row in episode_rows})
    scenes = sorted({row["scene_name"] for row in episode_rows})
    summary = _summarize_episodes(episode_rows)
    by_ms = {(row["method"], row["scene_name"]): row for row in summary}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ["success_rate", "fill_rate", "collision_rate"],
        ["Success", "Depth Fill", "Collision"],
    ):
        x = range(len(scenes))
        width = 0.8 / max(len(methods), 1)
        for i, method in enumerate(methods):
            vals = [float(by_ms.get((method, scene), {}).get(metric, 0.0)) for scene in scenes]
            ax.bar([xx + i * width for xx in x], vals, width=width, label=method)
        ax.set_title(title)
        ax.set_xticks([xx + width * (len(methods) - 1) / 2 for xx in x], scenes)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.savefig(plot_dir / "scene_metrics.png", dpi=180)
    plt.close(fig)

    for method in methods:
        fig, axes = plt.subplots(len(scenes), 2, figsize=(11, 3.2 * len(scenes)), constrained_layout=True)
        if len(scenes) == 1:
            axes = [axes]
        for r, scene in enumerate(scenes):
            rows = [row for row in trace_rows if row["method"] == method and row["scene_name"] == scene]
            ax = axes[r][0]
            for name, color in [("power", "tab:red"), ("exposure", "tab:green"), ("gain", "tab:blue")]:
                bins: dict[int, list[float]] = {}
                for row in rows:
                    step = int(float(row["step"]))
                    bins.setdefault(step, []).append(float(row[name]))
                xs = sorted(bins)
                ys = [sum(bins[x]) / len(bins[x]) for x in xs]
                ax.plot(xs, ys, label=name, color=color)
            ax.set_title(f"{method} camera - {scene}")
            ax.set_xlabel("step")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)

            ax = axes[r][1]
            for ep in sorted({int(float(row["episode_idx"])) for row in rows})[:30]:
                ers = [row for row in rows if int(float(row["episode_idx"])) == ep]
                if not ers:
                    continue
                lx_key = "local_x" if ers[0].get("local_x", "") not in ("", None) else "x"
                ax.plot([float(x[lx_key]) for x in ers], [float(x.get("local_y", x["y"])) for x in ers], alpha=0.35, linewidth=0.8)
            ax.axvline(0.0, color="k", linewidth=0.8, alpha=0.5)
            ax.set_title(f"{method} local trajectories - {scene}")
            ax.set_xlabel("local x")
            ax.set_ylabel("local y")
            ax.grid(alpha=0.25)
        fig.savefig(plot_dir / f"{method}_camera_and_trajectories.png", dpi=180)
        plt.close(fig)


def _write_report(out_dir: Path, episode_rows: list[dict], summary_rows: list[dict], trace_summary: list[dict]) -> None:
    lines = ["# Checkpoint Eval Suite", ""]
    lines.extend([
        "## Overall",
        "",
        "| method | n | success | collision | fill | final dist | p/e/g |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in [r for r in summary_rows if r["scene_name"] == "overall"]:
        lines.append(
            f"| {row['method']} | {row['n']} | {row['success_rate']:.3f} | "
            f"{row['collision_rate']:.3f} | {row['fill_rate']:.3f} | "
            f"{row['final_goal_dist']:.3f} | "
            f"{row['power_mean']:.3f}/{row['exposure_mean']:.3f}/{row['gain_mean']:.3f} |"
        )
    lines.extend(["", "## By Scene", "", "| method | scene | n | success | collision | fill | final dist | p/e/g |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for row in [r for r in summary_rows if r["scene_name"] != "overall"]:
        lines.append(
            f"| {row['method']} | {row['scene_name']} | {row['n']} | "
            f"{row['success_rate']:.3f} | {row['collision_rate']:.3f} | "
            f"{row['fill_rate']:.3f} | {row['final_goal_dist']:.3f} | "
            f"{row['power_mean']:.3f}/{row['exposure_mean']:.3f}/{row['gain_mean']:.3f} |"
        )
    lines.extend(["", "## Camera Phase Means", "",
                  "| method | scene | phase | n | p/e/g | scene effect | clearance |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for row in trace_summary:
        lines.append(
            f"| {row['method']} | {row['scene_name']} | {row['phase']} | {row['n']} | "
            f"{row['power']:.3f}/{row['exposure']:.3f}/{row['gain']:.3f} | "
            f"{row['scene_effect_mean']:.3f} | {row['clearance']:.3f} |"
        )
    lines.extend(["", "## Diagnostics", ""])
    for method in sorted({row["method"] for row in episode_rows}):
        sep = _camera_scene_separation(trace_summary, method)
        if sep:
            gd = sep.get("glare_vs_dark_near_l1", 0.0)
            msg = "OK" if gd >= 0.08 else "weak separation"
            vals = sep.get("glare_vs_dark_near_p_e_g", [0.0, 0.0, 0.0])
            lines.append(
                f"- `{method}` glare-vs-dark near camera L1: `{gd:.3f}` "
                f"({vals[0]:.3f}/{vals[1]:.3f}/{vals[2]:.3f}) -> {msg}."
            )
    lines.append("")
    lines.append("Figures are in `figures/`; raw episode and trace CSVs are in `raw/`.")
    (out_dir / "combined_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_method_specs(args) -> list[MethodSpec]:
    selected = list(args.methods or ["flightonly", "fixed", "randfix", "nondiff", "zero"])
    ckpt_by_method = {
        "pretrained": args.pretrained_ckpt,
        "dagger": args.dagger_ckpt,
        "flightonly": args.flightonly_ckpt,
        "fixed": args.fixed_ckpt,
        "randfix": args.randfix_ckpt,
        "nondiff": args.nondiff_ckpt,
        "zero": args.zero_ckpt,
    }
    spec_templates = {
        "pretrained": ("Pretrained camera head", "learned", "detached", "depth", False),
        "dagger": ("DAgger-relabel pretrained camera", "learned", "detached", "depth", False),
        "flightonly": ("Flight-only learned camera", "learned", "detached", "depth", True),
        "fixed": ("Fixed camera", "fixed", "detached", "depth", False),
        "randfix": ("Random static camera", "fixed_random_static", "detached", "depth", False),
        "nondiff": ("Non-diff learned camera", "learned", "detached", "depth", False),
        "zero": ("Blind zero-depth", "fixed", "detached", "zero", False),
    }
    specs = []
    for key in selected:
        if key not in spec_templates:
            raise ValueError(f"unsupported method {key!r}; choose from {sorted(spec_templates)}")
        ckpt = ckpt_by_method.get(key)
        if not ckpt:
            raise ValueError(f"--{key}_ckpt is required when method {key!r} is selected")
        label, camera_mode, sensor_grad, depth_mode, train_flight_only = spec_templates[key]
        specs.append(MethodSpec(key, label, Path(ckpt), camera_mode, sensor_grad, depth_mode, train_flight_only))
    for spec in specs:
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(f"{spec.key} checkpoint not found: {spec.checkpoint}")
    return specs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/slit_active_sensing.args")
    p.add_argument("--out_dir", default="paper/experiment/results/checkpoint_eval_suite_20260507")
    p.add_argument("--episodes_per_scene", type=int, default=100)
    p.add_argument("--scenarios", nargs="*", default=["glare", "specular", "dark"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--trace", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--pretrained_ckpt", default=None)
    p.add_argument("--dagger_ckpt", default=None)
    p.add_argument("--flightonly_ckpt", default=None)
    p.add_argument("--fixed_ckpt", default=None)
    p.add_argument("--randfix_ckpt", default=None)
    p.add_argument("--nondiff_ckpt", default=None)
    p.add_argument("--zero_ckpt", default=None)
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    out_dir = Path(cli.out_dir)
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    base_args = _load_base_args(Path(cli.config))
    set_global_seed(int(cli.seed), getattr(base_args, "deterministic", False))
    device = torch.device(cli.device)
    dummy_vis = type("DummyVis", (), {"enabled": False})()
    specs = _parse_method_specs(cli)

    all_episode_rows: list[dict] = []
    all_trace_rows: list[dict] = []
    for spec in specs:
        args = copy.deepcopy(base_args)
        args.batch_size = 1
        args.eval_episodes = int(cli.episodes_per_scene)
        args.vis_enable = False
        args.camera_control_mode = spec.camera_control_mode
        args.sensor_grad_mode = spec.sensor_grad_mode
        args.policy_depth_mode = spec.policy_depth_mode
        args.train_flight_only = spec.train_flight_only
        args.scenarios = list(cli.scenarios)
        args.resume = str(spec.checkpoint)
        model = _load_model(args, spec.checkpoint, device)
        env = build_env(args.batch_size, args, device, eval_mode=True)
        method_episode_rows: list[dict] = []
        method_trace_rows: list[dict] = []
        print(f"[suite] evaluating {spec.key}: {spec.checkpoint}")
        with torch.no_grad():
            for scene in cli.scenarios:
                for ep in range(int(cli.episodes_per_scene)):
                    # Same scene/episode seed across methods for paired comparisons.
                    set_global_seed(int(cli.seed) + 1000 * list(cli.scenarios).index(scene) + ep, getattr(args, "deterministic", False))
                    row, trace = run_one_episode(ep, scene, args, model, env, dummy_vis, device, collect_trace=bool(cli.trace))
                    row.update({
                        "episode_idx": ep,
                        "method": spec.key,
                        "method_label": spec.label,
                        "checkpoint": str(spec.checkpoint),
                    })
                    method_episode_rows.append(row)
                    all_episode_rows.append(row)
                    for tr in trace:
                        tr.update({"method": spec.key, "method_label": spec.label, "checkpoint": str(spec.checkpoint)})
                    method_trace_rows.extend(trace)
                    all_trace_rows.extend(trace)
        _write_csv(raw_dir / f"{spec.key}_episodes.csv", method_episode_rows)
        _write_csv(raw_dir / f"{spec.key}_trace.csv", method_trace_rows)

    summary_rows = _summarize_episodes(all_episode_rows)
    trace_summary = _summarize_trace(all_trace_rows)
    _write_csv(out_dir / "episode_metrics.csv", all_episode_rows)
    _write_csv(out_dir / "summary_by_method_scene.csv", summary_rows)
    _write_csv(out_dir / "camera_phase_summary.csv", trace_summary)
    (out_dir / "suite_config.json").write_text(
        json.dumps({
            "config": str(cli.config),
            "episodes_per_scene": int(cli.episodes_per_scene),
            "scenarios": list(cli.scenarios),
            "seed": int(cli.seed),
            "methods": [spec.__dict__ | {"checkpoint": str(spec.checkpoint)} for spec in specs],
        }, indent=2),
        encoding="utf-8",
    )
    _make_plots(out_dir, all_episode_rows, all_trace_rows)
    _write_report(out_dir, all_episode_rows, summary_rows, trace_summary)
    print(f"[suite] wrote: {out_dir}")


if __name__ == "__main__":
    main()

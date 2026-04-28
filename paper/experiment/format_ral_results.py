#!/usr/bin/env python3
"""
Format RAL experiment results into readable markdown and LaTeX tables.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]
METHOD_ORDER = ["ours", "ours_zero", "blind", "fixed", "fixed_random", "nondiff"]
SENSOR_SCENE_ORDER = ["glare", "specular", "dark"]
SLOT_ORDER = ["far_left", "left", "right", "far_right"]
ENTRY_PRE_STEPS = 5
ENTRY_POST_STEPS = 5


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v, default=0.0):
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _format_num(v: float, digits: int = 3) -> str:
    if math.isnan(v):
        return "-"
    return f"{v:.{digits}f}"


def _bold_if_best(value: float, best: float, lower_is_better: bool = False, digits: int = 3) -> str:
    if lower_is_better:
        is_best = abs(value - best) <= 1e-9 or value <= best + 1e-9
    else:
        is_best = abs(value - best) <= 1e-9 or value >= best - 1e-9
    s = _format_num(value, digits)
    return f"\\textbf{{{s}}}" if is_best else s


def _method_rank(method_key: str) -> int:
    try:
        return METHOD_ORDER.index(str(method_key))
    except ValueError:
        return len(METHOD_ORDER)


def _condition_rank(row: dict) -> tuple[int, int, str]:
    scene = str(row.get("scene_name", "") or str(row.get("condition", "")).split("_", 1)[0])
    slot = str(row.get("opening_slot", ""))
    try:
        scene_idx = SENSOR_SCENE_ORDER.index(scene)
    except ValueError:
        scene_idx = len(SENSOR_SCENE_ORDER)
    try:
        slot_idx = SLOT_ORDER.index(slot)
    except ValueError:
        slot_idx = len(SLOT_ORDER)
    return scene_idx, slot_idx, str(row.get("condition", ""))


def _condition_label(row: dict) -> str:
    scene = str(row.get("scene_name", "") or str(row.get("condition", "")).split("_", 1)[0])
    slot = str(row.get("opening_slot", ""))
    return f"{scene}/{slot}" if slot else scene


def _ordered_conditions(rows: List[dict]) -> list[str]:
    first_by_label: dict[str, dict] = {}
    for row in rows:
        first_by_label.setdefault(_condition_label(row), row)
    return sorted(first_by_label, key=lambda key: _condition_rank(first_by_label[key]))


def _latest_results_dir(results_root: Path) -> Path:
    dirs = [p for p in results_root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"no results directories under {results_root}")
    return sorted(dirs)[-1]


def _compute_t_entry(trace_rows: List[dict]) -> int | None:
    if not trace_rows:
        return None
    for row in trace_rows:
        if _to_float(row.get("scene_effect_mean", 0.0)) > 0.02:
            return int(row["step_idx"])
    zone_enter_x = _to_float(trace_rows[0].get("zone_enter_x", 0.0))
    for row in trace_rows:
        if _to_float(row.get("x", -1e9)) > zone_enter_x:
            return int(row["step_idx"])
    return None


def _window_mean(trace_rows: List[dict], key: str, step_lo: int, step_hi: int) -> float | None:
    vals: list[float] = []
    for row in trace_rows:
        step_idx = int(row["step_idx"])
        if step_idx < step_lo or step_idx > step_hi:
            continue
        raw = row.get(key, "")
        if raw in ("", None):
            continue
        val = _to_float(raw, default=math.nan)
        if math.isnan(val):
            continue
        vals.append(val)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _compute_post_entry_metrics_from_trace(trace_rows: List[dict]) -> dict:
    metrics = {
        "post_entry_available": 0.0,
        "t_entry_step": -1.0,
        "post_entry_local_glare_quality": 0.0,
        "post_entry_local_glare_invalid_rate": 0.0,
        "post_entry_fill_rate": 0.0,
        "post_entry_scene_effect_mean": 0.0,
        "post_entry_power_mean": 0.0,
        "post_entry_exposure_mean": 0.0,
        "post_entry_gain_mean": 0.0,
        "post_entry_power_delta": 0.0,
        "post_entry_exposure_delta": 0.0,
        "post_entry_gain_delta": 0.0,
    }
    if not trace_rows:
        return metrics

    rows = sorted(trace_rows, key=lambda x: int(x["step_idx"]))
    t_entry = _compute_t_entry(rows)
    if t_entry is None:
        return metrics

    pre_lo = max(int(rows[0]["step_idx"]), int(t_entry) - ENTRY_PRE_STEPS)
    pre_hi = int(t_entry) - 1
    post_lo = int(t_entry)
    post_hi = int(t_entry) + ENTRY_POST_STEPS

    pre_power = _window_mean(rows, "power", pre_lo, pre_hi)
    pre_exposure = _window_mean(rows, "exposure", pre_lo, pre_hi)
    pre_gain = _window_mean(rows, "gain", pre_lo, pre_hi)
    post_power = _window_mean(rows, "power", post_lo, post_hi)
    post_exposure = _window_mean(rows, "exposure", post_lo, post_hi)
    post_gain = _window_mean(rows, "gain", post_lo, post_hi)

    metrics["post_entry_available"] = 1.0
    metrics["t_entry_step"] = float(t_entry)
    metrics["post_entry_local_glare_quality"] = _window_mean(rows, "glare_quality_mean", post_lo, post_hi) or 0.0
    metrics["post_entry_local_glare_invalid_rate"] = _window_mean(rows, "glare_invalid_rate", post_lo, post_hi) or 0.0
    metrics["post_entry_fill_rate"] = _window_mean(rows, "fill_rate", post_lo, post_hi) or 0.0
    metrics["post_entry_scene_effect_mean"] = _window_mean(rows, "scene_effect_mean", post_lo, post_hi) or 0.0
    metrics["post_entry_power_mean"] = post_power or 0.0
    metrics["post_entry_exposure_mean"] = post_exposure or 0.0
    metrics["post_entry_gain_mean"] = post_gain or 0.0
    if pre_power is not None and post_power is not None:
        metrics["post_entry_power_delta"] = float(post_power - pre_power)
    if pre_exposure is not None and post_exposure is not None:
        metrics["post_entry_exposure_delta"] = float(post_exposure - pre_exposure)
    if pre_gain is not None and post_gain is not None:
        metrics["post_entry_gain_delta"] = float(post_gain - pre_gain)
    return metrics


def _augment_summary_with_post_entry(summary_rows: List[dict], trace_rows: List[dict]) -> List[dict]:
    grouped: Dict[tuple[str, str, str], List[dict]] = {}
    for row in trace_rows:
        key = (
            str(row.get("method_key", "")),
            str(row.get("condition", "")),
            str(row.get("episode_idx", "")),
        )
        grouped.setdefault(key, []).append(row)

    aggregate: Dict[tuple[str, str], Dict[str, List[float]]] = {}
    for (method_key, condition, _episode_idx), episode_rows in grouped.items():
        episode_metrics = _compute_post_entry_metrics_from_trace(episode_rows)
        agg_key = (method_key, condition)
        bucket = aggregate.setdefault(agg_key, {k: [] for k in episode_metrics.keys()})
        for key, value in episode_metrics.items():
            bucket[key].append(float(value))

    out: List[dict] = []
    for row in summary_rows:
        new_row = dict(row)
        agg_key = (str(row.get("method_key", "")), str(row.get("condition", "")))
        if agg_key in aggregate:
            for key, values in aggregate[agg_key].items():
                new_row[key] = float(sum(values) / len(values)) if values else 0.0
        else:
            defaults = _compute_post_entry_metrics_from_trace([])
            for key, value in defaults.items():
                new_row.setdefault(key, value)
        out.append(new_row)
    return out


def _build_sensor_scene_table(rows: List[dict]) -> str:
    if not rows:
        return "\n".join([
            "\\begin{tabular}{llccccc}",
            "\\toprule",
            "Method & Condition & Success $\\uparrow$ & Time $\\downarrow$ & LocalQ $\\uparrow$ & LocalInv $\\downarrow$ & Power \\\\",
            "\\midrule",
            "No data & - & - & - & - & - & - \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ])

    lines = []
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\toprule")
    lines.append("Method & Condition & Success $\\uparrow$ & Time $\\downarrow$ & LocalQ $\\uparrow$ & LocalInv $\\downarrow$ & Power \\\\")
    lines.append("\\midrule")
    condition_keys = _ordered_conditions(rows)
    for condition in condition_keys:
        condition_rows = [r for r in rows if _condition_label(r) == condition]
        if not condition_rows:
            continue
        condition_rows = sorted(condition_rows, key=lambda r: _method_rank(r.get("method_key", "")))
        best_success = max(_to_float(r["success_rate"]) for r in condition_rows)
        best_time = min(_to_float(r["time_to_goal"]) for r in condition_rows)
        best_q = max(_to_float(r["local_glare_quality"]) for r in condition_rows)
        best_inv = min(_to_float(r["local_glare_invalid_rate"]) for r in condition_rows)
        for idx, row in enumerate(condition_rows):
            method = row["method_label"]
            condition_cell = condition if idx == 0 else ""
            vals = [
                _bold_if_best(_to_float(row["success_rate"]), best_success, lower_is_better=False),
                _bold_if_best(_to_float(row["time_to_goal"]), best_time, lower_is_better=True),
                _bold_if_best(_to_float(row["local_glare_quality"]), best_q, lower_is_better=False),
                _bold_if_best(_to_float(row["local_glare_invalid_rate"]), best_inv, lower_is_better=True),
                _format_num(_to_float(row["power_mean"])),
            ]
            lines.append(f"{method} & {condition_cell} & " + " & ".join(vals) + " \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _build_camera_table(rows: List[dict]) -> str:
    if not rows:
        return "\n".join([
            "\\begin{tabular}{llccc}",
            "\\toprule",
            "Method & Condition & Power & Exposure & Gain \\\\",
            "\\midrule",
            "No data & - & - & - & - \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ])

    lines = []
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("Method & Condition & Power & Exposure & Gain \\\\")
    lines.append("\\midrule")
    condition_keys = _ordered_conditions(rows)
    for condition in condition_keys:
        condition_rows = [r for r in rows if _condition_label(r) == condition]
        for idx, row in enumerate(sorted(condition_rows, key=lambda r: _method_rank(r.get("method_key", "")))):
            condition_cell = condition if idx == 0 else ""
            lines.append(
                f"{row['method_label']} & {condition_cell} & "
                f"{_format_num(_to_float(row['power_mean']))} & "
                f"{_format_num(_to_float(row['exposure_mean']))} & "
                f"{_format_num(_to_float(row['gain_mean']))} \\\\"
            )
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _build_post_entry_table(rows: List[dict]) -> str:
    if not rows:
        return "\n".join([
            "\\begin{tabular}{llccccc}",
            "\\toprule",
            "Method & Condition & PostQ $\\uparrow$ & PostFill $\\uparrow$ & PostInv $\\downarrow$ & $\\Delta$Power & $\\Delta$Exposure \\\\",
            "\\midrule",
            "No data & - & - & - & - & - & - \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ])

    lines = []
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\toprule")
    lines.append("Method & Condition & PostQ $\\uparrow$ & PostFill $\\uparrow$ & PostInv $\\downarrow$ & $\\Delta$Power & $\\Delta$Exposure \\\\")
    lines.append("\\midrule")
    condition_keys = _ordered_conditions(rows)
    for condition in condition_keys:
        condition_rows = [r for r in rows if _condition_label(r) == condition]
        if not condition_rows:
            continue
        condition_rows = sorted(condition_rows, key=lambda r: _method_rank(r.get("method_key", "")))
        best_post_q = max(_to_float(r.get("post_entry_local_glare_quality", 0.0)) for r in condition_rows)
        best_post_fill = max(_to_float(r.get("post_entry_fill_rate", 0.0)) for r in condition_rows)
        best_post_inv = min(_to_float(r.get("post_entry_local_glare_invalid_rate", 0.0)) for r in condition_rows)
        for idx, row in enumerate(condition_rows):
            condition_cell = condition if idx == 0 else ""
            vals = [
                _bold_if_best(_to_float(row.get("post_entry_local_glare_quality", 0.0)), best_post_q, lower_is_better=False),
                _bold_if_best(_to_float(row.get("post_entry_fill_rate", 0.0)), best_post_fill, lower_is_better=False),
                _bold_if_best(_to_float(row.get("post_entry_local_glare_invalid_rate", 0.0)), best_post_inv, lower_is_better=True),
                _format_num(_to_float(row.get("post_entry_power_delta", 0.0))),
                _format_num(_to_float(row.get("post_entry_exposure_delta", 0.0))),
            ]
            lines.append(f"{row['method_label']} & {condition_cell} & " + " & ".join(vals) + " \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _build_markdown_summary(summary_rows: List[dict], results_dir: Path) -> str:
    sensor_scene_names = {"glare", "specular", "dark"}
    glare_rows = [
        r for r in summary_rows
        if str(r.get("scene_name", "")) in sensor_scene_names
        or str(r.get("condition", "")).split("_", 1)[0] in sensor_scene_names
    ]

    lines: List[str] = []
    lines.append("# RAL 实验结果报告")
    lines.append("")
    lines.append(f"结果目录：`{results_dir}`")
    lines.append("")
    lines.append("## 文件说明")
    lines.append("")
    lines.append("- `summary_metrics.csv`：每个方法、每个场景条件的一行汇总结果。")
    lines.append("- `episode_metrics.csv`：每个 episode 一行，可看波动和失败模式。")
    lines.append("- `trace_metrics.csv`：每个 timestep 一行，可画事件对齐曲线。")
    lines.append("- `success_by_scene.png`：按 glare/specular/dark 场景统计的成功率曲线。")
    lines.append("- `post_entry_by_scene.png`：进入局部传感器退化区域后的关键窗口指标曲线。")
    lines.append("- `event_aligned_<scene>_<slot>.png`：指定场景/开口下的参数时序图。")
    lines.append("- `trajectory_<scene>.png`：指定场景下的顶视轨迹图。")
    lines.append("")

    lines.append("## 当前结果一眼结论")
    lines.append("")
    if glare_rows:
        all_success = [_to_float(r["success_rate"]) for r in glare_rows]
        if min(all_success) >= 0.999:
            lines.append(
                "- 三种局部传感器场景当前全部 `100% success`，说明这个任务目前已经饱和，"
                "主结果表上很难再用 `success rate` 拉开差距。"
            )
            lines.append(
                "- 在这种情况下，更该关注 `post_entry` 指标、`local_glare_quality`、"
                "`power/exposure/gain` 以及事件对齐曲线，而不是只看成功率。"
            )
        best_q = max(glare_rows, key=lambda r: _to_float(r["local_glare_quality"]))
        lines.append(
            f"- 当前所有 glare 条件里，`local_glare_quality` 最高的单项结果来自 "
            f"`{best_q['method_label']}` @ `{best_q['condition']}`，数值约为 `{_format_num(_to_float(best_q['local_glare_quality']))}`。"
        )
        best_post_q = max(glare_rows, key=lambda r: _to_float(r.get("post_entry_local_glare_quality", 0.0)))
        lines.append(
            f"- 进入逆光区后的关键窗口里，`post_entry_local_glare_quality` 最高的单项结果来自 "
            f"`{best_post_q['method_label']}` @ `{best_post_q['condition']}`，数值约为 `{_format_num(_to_float(best_post_q.get('post_entry_local_glare_quality', 0.0)))}`。"
        )
    lines.append("")

    lines.append("## 场景/开口汇总")
    lines.append("")
    lines.append("| Method | Condition | Success | Time | LocalQ | LocalInv | Power | Exposure | Gain |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(glare_rows, key=lambda r: (_condition_rank(r), _method_rank(r.get("method_key", "")))):
        lines.append(
            f"| {row['method_label']} | {_condition_label(row)} | "
            f"{_format_num(_to_float(row['success_rate']))} | "
            f"{_format_num(_to_float(row['time_to_goal']))} | "
            f"{_format_num(_to_float(row['local_glare_quality']))} | "
            f"{_format_num(_to_float(row['local_glare_invalid_rate']))} | "
            f"{_format_num(_to_float(row['power_mean']))} | "
            f"{_format_num(_to_float(row['exposure_mean']))} | "
            f"{_format_num(_to_float(row['gain_mean']))} |"
        )
    lines.append("")

    lines.append("## Post-Entry 汇总")
    lines.append("")
    lines.append("| Method | Condition | PostQ | PostFill | PostInv | dPower | dExposure |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in sorted(glare_rows, key=lambda r: (_condition_rank(r), _method_rank(r.get("method_key", "")))):
        lines.append(
            f"| {row['method_label']} | {_condition_label(row)} | "
            f"{_format_num(_to_float(row.get('post_entry_local_glare_quality', 0.0)))} | "
            f"{_format_num(_to_float(row.get('post_entry_fill_rate', 0.0)))} | "
            f"{_format_num(_to_float(row.get('post_entry_local_glare_invalid_rate', 0.0)))} | "
            f"{_format_num(_to_float(row.get('post_entry_power_delta', 0.0)))} | "
            f"{_format_num(_to_float(row.get('post_entry_exposure_delta', 0.0)))} |"
        )
    lines.append("")

    lines.append("## 怎么分析这些指标")
    lines.append("")
    lines.append("- `success_rate / collision_rate`：决定任务是否完成，但一旦全部饱和就不再有区分度。")
    lines.append("- `post_entry_local_glare_quality`：最关键的主指标，专门看进入逆光区后那几步还能不能保住关键几何。")
    lines.append("- `post_entry_fill_rate / post_entry_local_glare_invalid_rate`：判断逆光区是否已经从“还能飞”退化到“局部盲飞”。")
    lines.append("- `post_entry_power_delta / post_entry_exposure_delta`：判断策略在事件发生后有没有做出你预期的相机反应。")
    lines.append("- `local_glare_quality`：整段 episode 的平均局部质量，适合看总体趋势，不适合单独当 hardest-case 指标。")
    lines.append("- `power_mean / exposure_mean / gain_mean`：用于解释策略整段任务期间的总体感知风格。")
    lines.append("- `time_to_goal`：如果成功率都一样，它能反映谁更保守、谁更果断。")
    lines.append("")

    lines.append("## 自动诊断建议")
    lines.append("")
    if glare_rows and min(_to_float(r["success_rate"]) for r in glare_rows) >= 0.999:
        lines.append("- 当前成功率完全饱和：建议继续提高局部传感退化难度，或者主要报告 post-entry 局部感知指标。")
    if glare_rows:
        ours_rows = [r for r in glare_rows if r["method_key"] == "ours"]
        fixed_rows = {r["condition"]: r for r in glare_rows if r["method_key"] == "fixed"}
        blind_rows = {r["condition"]: r for r in glare_rows if r["method_key"] == "blind"}
        paired_fixed = [(r, fixed_rows[r["condition"]]) for r in ours_rows if r["condition"] in fixed_rows]
        if paired_fixed:
            ours_worst, fixed_match = min(
                paired_fixed,
                key=lambda pair: _to_float(pair[0].get("post_entry_local_glare_quality", 0.0)) - _to_float(pair[1].get("post_entry_local_glare_quality", 0.0)),
            )
            q_gap = _to_float(ours_worst.get("post_entry_local_glare_quality", 0.0)) - _to_float(fixed_match.get("post_entry_local_glare_quality", 0.0))
            f_gap = _to_float(ours_worst.get("post_entry_fill_rate", 0.0)) - _to_float(fixed_match.get("post_entry_fill_rate", 0.0))
            dp = _to_float(ours_worst.get("post_entry_power_delta", 0.0))
            lines.append(
                f"- 在 `{ours_worst['condition']}` 下，`Ours` 相对 `Fixed Camera` 的 `PostQ` 提升约 `{_format_num(q_gap)}`，"
                f"`PostFill` 提升约 `{_format_num(f_gap)}`。这比只看整段平均值更能体现 hardest-case 感知恢复。"
            )
            if dp <= 0.0:
                lines.append("- 但这个条件下 `Ours` 的 `post_entry_power_delta` 仍未转正，说明当前仿真里 `power` 还不是必要动作。")
        paired_blind = [(r, blind_rows[r["condition"]]) for r in ours_rows if r["condition"] in blind_rows]
        if paired_blind:
            ours_worst, blind_match = min(
                paired_blind,
                key=lambda pair: _to_float(pair[0].get("success_rate", 0.0)) - _to_float(pair[1].get("success_rate", 0.0)),
            )
            success_gap = _to_float(ours_worst.get("success_rate", 0.0)) - _to_float(blind_match.get("success_rate", 0.0))
            post_q_gap = _to_float(ours_worst.get("post_entry_local_glare_quality", 0.0)) - _to_float(blind_match.get("post_entry_local_glare_quality", 0.0))
            if success_gap <= 0.05:
                lines.append("- `Blind / No Depth` 与 `Ours` 的最小成功率差距仍然很小，这通常意味着场景仍然可以被记忆轨迹或开环策略部分解决。")
            else:
                lines.append(
                    f"- `Blind / No Depth` 在 `{ours_worst['condition']}` 下相对 `Ours` 的成功率下降约 `{_format_num(success_gap)}`，"
                    f"`PostQ` 下降约 `{_format_num(post_q_gap)}`，这更能说明场景已经真正变成感知关键任务。"
                )
    lines.append("")

    lines.append("## 可直接引用的图片")
    lines.append("")
    for name in [
        "success_by_scene.png",
        "post_entry_by_scene.png",
        "success_by_slot.png",
        "trajectory_glare.png",
    ]:
        p = results_dir / name
        if p.is_file():
            lines.append(f"![{name}]({name})")
            lines.append("")
    return "\n".join(lines)


def format_results_dir(results_dir: Path):
    results_dir = results_dir.resolve()
    summary_csv = results_dir / "summary_metrics.csv"
    if not summary_csv.is_file():
        raise FileNotFoundError(f"missing summary csv: {summary_csv}")

    summary_rows = _read_csv(summary_csv)
    trace_csv = results_dir / "trace_metrics.csv"
    if trace_csv.is_file():
        summary_rows = _augment_summary_with_post_entry(summary_rows, _read_csv(trace_csv))
    sensor_scene_names = {"glare", "specular", "dark"}
    glare_rows = [
        r for r in summary_rows
        if str(r.get("scene_name", "")) in sensor_scene_names
        or str(r.get("condition", "")).split("_", 1)[0] in sensor_scene_names
    ]

    report_md = _build_markdown_summary(summary_rows, results_dir)
    (results_dir / "report.md").write_text(report_md, encoding="utf-8")

    latex_sensor_scene = _build_sensor_scene_table(glare_rows)
    (results_dir / "table_sensor_scene.tex").write_text(latex_sensor_scene, encoding="utf-8")

    latex_cam = _build_camera_table(glare_rows)
    (results_dir / "table_camera.tex").write_text(latex_cam, encoding="utf-8")

    latex_post = _build_post_entry_table(glare_rows)
    (results_dir / "table_post_entry.tex").write_text(latex_post, encoding="utf-8")

    return {
        "report_md": results_dir / "report.md",
        "table_sensor_scene_tex": results_dir / "table_sensor_scene.tex",
        "table_camera_tex": results_dir / "table_camera.tex",
        "table_post_entry_tex": results_dir / "table_post_entry.tex",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = ROOT / "paper" / "experiment" / "results"
    results_dir = Path(args.results_dir).resolve() if args.results_dir else _latest_results_dir(results_root)
    outputs = format_results_dir(results_dir)
    print("[format] done.")
    for name, path in outputs.items():
        print(f"[format] {name}: {path}")


if __name__ == "__main__":
    main()

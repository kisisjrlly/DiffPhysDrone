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
METHOD_ORDER = ["ours", "blind", "fixed", "nondiff"]
GLARE_LEVEL_ORDER = ["l0", "l1", "l2", "l3"]
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


def _group_by_condition(rows: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for row in rows:
        out.setdefault(str(row["condition"]), []).append(row)
    return out


def _method_rank(method_key: str) -> int:
    try:
        return METHOD_ORDER.index(str(method_key))
    except ValueError:
        return len(METHOD_ORDER)


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


def _build_base_table(rows: List[dict]) -> str:
    if not rows:
        return "\n".join([
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Method & Success $\\uparrow$ & Collision $\\downarrow$ & Time $\\downarrow$ & AvgSpeed $\\uparrow$ & Fill $\\uparrow$ \\\\",
            "\\midrule",
            "No data & - & - & - & - & - \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ])

    cols = [
        ("success_rate", False, "Success $\\uparrow$"),
        ("collision_rate", True, "Collision $\\downarrow$"),
        ("time_to_goal", True, "Time $\\downarrow$"),
        ("avg_speed", False, "AvgSpeed $\\uparrow$"),
        ("fill_rate", False, "Fill $\\uparrow$"),
    ]
    best = {}
    for key, lower, _ in cols:
        vals = [_to_float(r[key]) for r in rows]
        best[key] = min(vals) if lower else max(vals)

    lines = []
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("Method & Success $\\uparrow$ & Collision $\\downarrow$ & Time $\\downarrow$ & AvgSpeed $\\uparrow$ & Fill $\\uparrow$ \\\\")
    lines.append("\\midrule")
    for row in sorted(rows, key=lambda r: _method_rank(r.get("method_key", ""))):
        vals = []
        for key, lower, _ in cols:
            vals.append(_bold_if_best(_to_float(row[key]), best[key], lower_is_better=lower))
        lines.append(f"{row['method_label']} & " + " & ".join(vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _build_glare_table(rows: List[dict]) -> str:
    if not rows:
        return "\n".join([
            "\\begin{tabular}{llccccc}",
            "\\toprule",
            "Method & Level & Success $\\uparrow$ & Time $\\downarrow$ & LocalQ $\\uparrow$ & LocalInv $\\downarrow$ & Power $\\uparrow$ \\\\",
            "\\midrule",
            "No data & - & - & - & - & - & - \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ])

    lines = []
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\toprule")
    lines.append("Method & Level & Success $\\uparrow$ & Time $\\downarrow$ & LocalQ $\\uparrow$ & LocalInv $\\downarrow$ & Power $\\uparrow$ \\\\")
    lines.append("\\midrule")
    for level in GLARE_LEVEL_ORDER:
        level_rows = [r for r in rows if str(r.get("glare_level", "")) == level]
        if not level_rows:
            continue
        level_rows = sorted(level_rows, key=lambda r: _method_rank(r.get("method_key", "")))
        best_success = max(_to_float(r["success_rate"]) for r in level_rows)
        best_time = min(_to_float(r["time_to_goal"]) for r in level_rows)
        best_q = max(_to_float(r["local_glare_quality"]) for r in level_rows)
        best_inv = min(_to_float(r["local_glare_invalid_rate"]) for r in level_rows)
        best_power = max(_to_float(r["power_mean"]) for r in level_rows)
        for idx, row in enumerate(level_rows):
            method = row["method_label"]
            level_cell = level.upper() if idx == 0 else ""
            vals = [
                _bold_if_best(_to_float(row["success_rate"]), best_success, lower_is_better=False),
                _bold_if_best(_to_float(row["time_to_goal"]), best_time, lower_is_better=True),
                _bold_if_best(_to_float(row["local_glare_quality"]), best_q, lower_is_better=False),
                _bold_if_best(_to_float(row["local_glare_invalid_rate"]), best_inv, lower_is_better=True),
                _bold_if_best(_to_float(row["power_mean"]), best_power, lower_is_better=False),
            ]
            lines.append(f"{method} & {level_cell} & " + " & ".join(vals) + " \\\\")
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
            "Method & Level & Power & Exposure & Gain \\\\",
            "\\midrule",
            "No data & - & - & - & - \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ])

    lines = []
    lines.append("\\begin{tabular}{llccc}")
    lines.append("\\toprule")
    lines.append("Method & Level & Power & Exposure & Gain \\\\")
    lines.append("\\midrule")
    for level in GLARE_LEVEL_ORDER:
        level_rows = [r for r in rows if str(r.get("glare_level", "")) == level]
        for idx, row in enumerate(sorted(level_rows, key=lambda r: _method_rank(r.get("method_key", "")))):
            level_cell = level.upper() if idx == 0 else ""
            lines.append(
                f"{row['method_label']} & {level_cell} & "
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
            "Method & Level & PostQ $\\uparrow$ & PostFill $\\uparrow$ & PostInv $\\downarrow$ & $\\Delta$Power $\\uparrow$ & $\\Delta$Exposure $\\downarrow$ \\\\",
            "\\midrule",
            "No data & - & - & - & - & - & - \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ])

    lines = []
    lines.append("\\begin{tabular}{llccccc}")
    lines.append("\\toprule")
    lines.append("Method & Level & PostQ $\\uparrow$ & PostFill $\\uparrow$ & PostInv $\\downarrow$ & $\\Delta$Power $\\uparrow$ & $\\Delta$Exposure $\\downarrow$ \\\\")
    lines.append("\\midrule")
    for level in GLARE_LEVEL_ORDER:
        level_rows = [r for r in rows if str(r.get("glare_level", "")) == level]
        if not level_rows:
            continue
        level_rows = sorted(level_rows, key=lambda r: _method_rank(r.get("method_key", "")))
        best_post_q = max(_to_float(r.get("post_entry_local_glare_quality", 0.0)) for r in level_rows)
        best_post_fill = max(_to_float(r.get("post_entry_fill_rate", 0.0)) for r in level_rows)
        best_post_inv = min(_to_float(r.get("post_entry_local_glare_invalid_rate", 0.0)) for r in level_rows)
        best_dp = max(_to_float(r.get("post_entry_power_delta", 0.0)) for r in level_rows)
        best_de = min(_to_float(r.get("post_entry_exposure_delta", 0.0)) for r in level_rows)
        for idx, row in enumerate(level_rows):
            level_cell = level.upper() if idx == 0 else ""
            vals = [
                _bold_if_best(_to_float(row.get("post_entry_local_glare_quality", 0.0)), best_post_q, lower_is_better=False),
                _bold_if_best(_to_float(row.get("post_entry_fill_rate", 0.0)), best_post_fill, lower_is_better=False),
                _bold_if_best(_to_float(row.get("post_entry_local_glare_invalid_rate", 0.0)), best_post_inv, lower_is_better=True),
                _bold_if_best(_to_float(row.get("post_entry_power_delta", 0.0)), best_dp, lower_is_better=False),
                _bold_if_best(_to_float(row.get("post_entry_exposure_delta", 0.0)), best_de, lower_is_better=True),
            ]
            lines.append(f"{row['method_label']} & {level_cell} & " + " & ".join(vals) + " \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _build_markdown_summary(summary_rows: List[dict], results_dir: Path) -> str:
    cond_map = _group_by_condition(summary_rows)
    base_rows = cond_map.get("base", [])
    glare_rows = [r for r in summary_rows if str(r.get("scene_name", "")).startswith("sun_glare_")]

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
    lines.append("- `success_vs_glare.png`：随 glare 强度变化的成功率曲线。")
    lines.append("- `post_entry_vs_glare.png`：进入逆光区后的关键窗口指标曲线。")
    lines.append("- `event_aligned_l3.png`：L3 条件下的参数时序图。")
    lines.append("- `trajectory_l3.png`：L3 条件下的顶视轨迹图。")
    lines.append("")

    lines.append("## 当前结果一眼结论")
    lines.append("")
    if base_rows:
        best_base = max(base_rows, key=lambda r: _to_float(r["success_rate"]))
        lines.append(
            f"- `Base` 场景下当前最好的方法是 `{best_base['method_label']}`，"
            f"成功率约为 `{_format_num(_to_float(best_base['success_rate']))}`。"
        )
        lines.append(
            "- 但 `Base` 场景三种方法的成功率都偏低，说明当前基础导航本身还不够稳定，"
            "这会削弱论文里关于 `sun_glare` 的结论可信度。"
        )
    if glare_rows:
        all_success = [_to_float(r["success_rate"]) for r in glare_rows]
        if min(all_success) >= 0.999:
            lines.append(
                "- `Sun Glare` 四档强度下三种方法当前全部 `100% success`，说明这个逆光任务目前已经饱和，"
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

    if base_rows:
        lines.append("## Base 场景汇总")
        lines.append("")
        lines.append("| Method | Success | Collision | Time | AvgSpeed | Fill |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in base_rows:
            lines.append(
                f"| {row['method_label']} | "
                f"{_format_num(_to_float(row['success_rate']))} | "
                f"{_format_num(_to_float(row['collision_rate']))} | "
                f"{_format_num(_to_float(row['time_to_goal']))} | "
                f"{_format_num(_to_float(row['avg_speed']))} | "
                f"{_format_num(_to_float(row['fill_rate']))} |"
            )
        lines.append("")

    lines.append("## Sun Glare 汇总")
    lines.append("")
    lines.append("| Method | Level | Success | Time | LocalQ | LocalInv | Power | Exposure | Gain |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for level in GLARE_LEVEL_ORDER:
        level_rows = [r for r in glare_rows if str(r.get("glare_level", "")) == level]
        for row in sorted(level_rows, key=lambda r: _method_rank(r.get("method_key", ""))):
            lines.append(
                f"| {row['method_label']} | {level.upper()} | "
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
    lines.append("| Method | Level | PostQ | PostFill | PostInv | dPower | dExposure |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for level in GLARE_LEVEL_ORDER:
        level_rows = [r for r in glare_rows if str(r.get("glare_level", "")) == level]
        for row in sorted(level_rows, key=lambda r: _method_rank(r.get("method_key", ""))):
            lines.append(
                f"| {row['method_label']} | {level.upper()} | "
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
    if base_rows and max(_to_float(r["success_rate"]) for r in base_rows) < 0.6:
        lines.append("- `Base` 成功率太低：建议先提高基础避障稳定性，否则 reviewer 会质疑 glare 结论是否建立在不稳定导航之上。")
    if glare_rows and min(_to_float(r["success_rate"]) for r in glare_rows) >= 0.999:
        lines.append("- `Sun Glare` 成功率完全饱和：建议把 `L3` 再做难一点，或者增加更严格的局部感知指标。")
    if glare_rows:
        ours_l3 = next((r for r in glare_rows if r["method_key"] == "ours" and r["glare_level"] == "l3"), None)
        fixed_l3 = next((r for r in glare_rows if r["method_key"] == "fixed" and r["glare_level"] == "l3"), None)
        blind_l3 = next((r for r in glare_rows if r["method_key"] == "blind" and r["glare_level"] == "l3"), None)
        if ours_l3 and fixed_l3:
            q_gap = _to_float(ours_l3.get("post_entry_local_glare_quality", 0.0)) - _to_float(fixed_l3.get("post_entry_local_glare_quality", 0.0))
            f_gap = _to_float(ours_l3.get("post_entry_fill_rate", 0.0)) - _to_float(fixed_l3.get("post_entry_fill_rate", 0.0))
            dp = _to_float(ours_l3.get("post_entry_power_delta", 0.0))
            lines.append(
                f"- 在 `L3` 下，`Ours` 相对 `Fixed Camera` 的 `PostQ` 提升约 `{_format_num(q_gap)}`，"
                f"`PostFill` 提升约 `{_format_num(f_gap)}`。这比只看整段平均值更能体现 hardest-case 感知恢复。"
            )
            if dp <= 0.0:
                lines.append("- 但 `Ours` 在 `L3` 下的 `post_entry_power_delta` 仍未转正，说明当前仿真里 `power` 还不是必要动作，论文故事仍然主要靠 `exposure` 在支撑。")
        if ours_l3 and blind_l3:
            success_gap = _to_float(ours_l3.get("success_rate", 0.0)) - _to_float(blind_l3.get("success_rate", 0.0))
            post_q_gap = _to_float(ours_l3.get("post_entry_local_glare_quality", 0.0)) - _to_float(blind_l3.get("post_entry_local_glare_quality", 0.0))
            if success_gap <= 0.05:
                lines.append("- `Blind / No Depth` 在 `L3` 下与 `Ours` 的成功率差距仍然很小，这通常意味着场景仍然可以被记忆轨迹或开环策略部分解决。")
            else:
                lines.append(
                    f"- `Blind / No Depth` 在 `L3` 下相对 `Ours` 的成功率下降约 `{_format_num(success_gap)}`，"
                    f"`PostQ` 下降约 `{_format_num(post_q_gap)}`，这更能说明场景已经真正变成感知关键任务。"
                )
    lines.append("")

    lines.append("## 可直接引用的图片")
    lines.append("")
    for name in [
        "success_vs_glare.png",
        "post_entry_vs_glare.png",
        "event_aligned_l3.png",
        "trajectory_l3.png",
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
    base_rows = [r for r in summary_rows if r["condition"] == "base"]
    glare_rows = [r for r in summary_rows if str(r.get("scene_name", "")).startswith("sun_glare_")]

    report_md = _build_markdown_summary(summary_rows, results_dir)
    (results_dir / "report.md").write_text(report_md, encoding="utf-8")

    latex_main = _build_base_table(base_rows)
    (results_dir / "table_base.tex").write_text(latex_main, encoding="utf-8")

    latex_glare = _build_glare_table(glare_rows)
    (results_dir / "table_glare.tex").write_text(latex_glare, encoding="utf-8")

    latex_cam = _build_camera_table(glare_rows)
    (results_dir / "table_camera.tex").write_text(latex_cam, encoding="utf-8")

    latex_post = _build_post_entry_table(glare_rows)
    (results_dir / "table_post_entry.tex").write_text(latex_post, encoding="utf-8")

    return {
        "report_md": results_dir / "report.md",
        "table_base_tex": results_dir / "table_base.tex",
        "table_glare_tex": results_dir / "table_glare.tex",
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

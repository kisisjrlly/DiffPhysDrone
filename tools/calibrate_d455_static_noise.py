#!/usr/bin/env python3
"""Calibrate D455 static-wall depth noise for diff_depth camera semantics.

Recommended setup:
- Put D455 on a tripod or rigid stand.
- Face a flat matte wall at about 1.0m to 2.0m.
- Keep the camera and wall completely still during capture.
- Avoid sunlight, glass, glossy paint, and moving people in the view.

The script scans exposure/gain while keeping laser power fixed, estimates
temporal depth noise inside the center ROI, and recommends:
- cam_shot_noise_base
- cam_iso_gain_scale
- cam_iso_gain_gamma
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from datetime import datetime
from typing import Any

import numpy as np

from d455_calib_utils import (
    ensure_dir,
    exposure_proxy_from_us,
    gain_proxy_from_value,
    power_proxy_from_laser,
    write_json,
)


DEPTH_MODE_CANDIDATES = [
    (848, 480, 30),
    (640, 480, 30),
    (640, 480, 15),
    (424, 240, 30),
]


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_float_list(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    out = []
    for token in str(raw).replace(';', ',').split(','):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return out


def _clip_unique(values: list[float], lo: float, hi: float, step: float = 1.0) -> list[float]:
    out: list[float] = []
    eps = max(1e-6, abs(float(step)) * 0.25)
    for value in values:
        v = min(max(float(value), float(lo)), float(hi))
        if any(abs(v - old) <= eps for old in out):
            continue
        out.append(v)
    return out


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return v


def _center_crop(stack: np.ndarray, roi_fraction: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    frac = min(max(float(roi_fraction), 0.05), 1.0)
    _, h, w = stack.shape
    crop_h = max(1, int(round(h * frac)))
    crop_w = max(1, int(round(w * frac)))
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    y1 = min(h, y0 + crop_h)
    x1 = min(w, x0 + crop_w)
    return stack[:, y0:y1, x0:x1], (x0, y0, x1, y1)


def _nan_percentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.percentile(values, q))


def _nan_mean(values: np.ndarray, default: float = 0.0) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.mean(values))


def _nan_median(values: np.ndarray, default: float = 0.0) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(default)
    return float(np.median(values))


def start_pipeline():
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    profile = None
    chosen_mode = None
    last_error = None
    for width, height, fps in DEPTH_MODE_CANDIDATES:
        try:
            config = rs.config()
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            profile = pipeline.start(config)
            chosen_mode = (width, height, fps)
            break
        except RuntimeError as exc:
            last_error = exc
    if profile is None:
        raise RuntimeError(f'无法启动 D455 深度流，最后错误: {last_error}')
    return pipeline, profile, chosen_mode


def fetch_depth_m(pipeline) -> np.ndarray | None:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None
    depth_scale = depth_frame.get_units()
    return np.asanyarray(depth_frame.get_data()).astype(np.float32) * float(depth_scale)


def collect_stack(pipeline, frames: int, sleep_s: float = 0.0) -> np.ndarray:
    captured = []
    while len(captured) < int(frames):
        depth_m = fetch_depth_m(pipeline)
        if depth_m is None:
            continue
        captured.append(depth_m)
        if sleep_s > 0:
            time.sleep(float(sleep_s))
    return np.stack(captured, axis=0)


def compute_static_noise_metrics(
    stack_m: np.ndarray,
    depth_min_valid: float,
    depth_max_range: float,
    roi_fraction: float,
    valid_pixel_ratio: float,
) -> dict[str, float | tuple[int, int, int, int]]:
    roi, crop_xyxy = _center_crop(stack_m, roi_fraction)
    valid = np.isfinite(roi) & (roi >= float(depth_min_valid)) & (roi <= float(depth_max_range))
    frame_fill = valid.mean(axis=(1, 2))
    valid_per_pixel = valid.mean(axis=0)
    stable_mask = valid_per_pixel >= float(valid_pixel_ratio)
    if not np.any(stable_mask):
        stable_mask = valid_per_pixel > 0

    arr = roi.astype(np.float64)
    arr[~valid] = np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        pixel_mean = np.nanmean(arr, axis=0)
        pixel_std = np.nanstd(arr, axis=0)
        pixel_median = np.nanmedian(arr, axis=0)
        pixel_mad = np.nanmedian(np.abs(arr - pixel_median[None, :, :]), axis=0) * 1.4826

    use_std = pixel_std[stable_mask]
    use_mad = pixel_mad[stable_mask]
    use_mean = pixel_mean[stable_mask]

    x0, y0, x1, y1 = crop_xyxy
    return {
        'roi_x0': float(x0),
        'roi_y0': float(y0),
        'roi_x1': float(x1),
        'roi_y1': float(y1),
        'roi_fill_rate_mean': float(np.mean(frame_fill)) if frame_fill.size else 0.0,
        'roi_fill_rate_min': float(np.min(frame_fill)) if frame_fill.size else 0.0,
        'stable_pixel_ratio': float(np.mean(stable_mask)),
        'depth_mean_m': _nan_mean(use_mean),
        'temporal_std_mean_m': _nan_mean(use_std),
        'temporal_std_median_m': _nan_median(use_std),
        'temporal_std_p75_m': _nan_percentile(use_std, 75),
        'temporal_std_p90_m': _nan_percentile(use_std, 90),
        'temporal_mad_median_m': _nan_median(use_mad),
    }


def build_settings(args, exp_range, gain_range, laser_range) -> tuple[list[dict[str, float]], float]:
    exposure_values = _parse_float_list(args.exposure_us_values)
    if exposure_values is None:
        exposure_values = [3000.0, 10000.0, 30000.0]
    gain_values = _parse_float_list(args.gain_values)
    if gain_values is None:
        gain_values = [
            float(gain_range.min),
            32.0,
            64.0,
            128.0,
            float(gain_range.max),
        ]

    exposure_values = _clip_unique(exposure_values, exp_range.min, exp_range.max, exp_range.step)
    gain_values = _clip_unique(gain_values, gain_range.min, gain_range.max, gain_range.step)

    if args.laser_power is None:
        laser_power = float(laser_range.default)
    else:
        laser_power = float(args.laser_power)
    laser_power = min(max(laser_power, float(laser_range.min)), float(laser_range.max))

    settings = []
    idx = 0
    for exposure_us in exposure_values:
        for gain_value in gain_values:
            settings.append({
                'setting_id': float(idx),
                'exposure_us': float(exposure_us),
                'gain_value': float(gain_value),
                'laser_power': float(laser_power),
            })
            idx += 1
    return settings, laser_power


def _fit_gain_noise_model(rows: list[dict[str, float]], min_fill_rate: float) -> dict[str, float]:
    usable = [
        row for row in rows
        if row.get('roi_fill_rate_mean', 0.0) >= float(min_fill_rate)
        and row.get('temporal_std_median_m', 0.0) > 0.0
    ]
    if len(usable) < 3:
        return {
            'fit_ok': 0.0,
            'gain_noise_gamma': 1.2,
            'gain_noise_amp': 10.0,
            'gain_noise_ratio_max': 1.0,
            'fit_rmse': 0.0,
            'num_fit_points': float(len(usable)),
        }

    by_exp: dict[float, list[dict[str, float]]] = {}
    for row in usable:
        by_exp.setdefault(row['exposure_us'], []).append(row)

    xs = []
    ys = []
    for _, bucket in by_exp.items():
        bucket = sorted(bucket, key=lambda r: r['gain01'])
        base = next((r for r in bucket if r['temporal_std_median_m'] > 0.0), None)
        if base is None:
            continue
        base_noise = max(float(base['temporal_std_median_m']), 1e-9)
        for row in bucket:
            xs.append(float(row['gain01']))
            ys.append(float(row['temporal_std_median_m']) / base_noise)

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x = x[valid]
    y = y[valid]
    if x.size < 3:
        return {
            'fit_ok': 0.0,
            'gain_noise_gamma': 1.2,
            'gain_noise_amp': 10.0,
            'gain_noise_ratio_max': 1.0,
            'fit_rmse': 0.0,
            'num_fit_points': float(x.size),
        }

    best = None
    for gamma in np.linspace(0.5, 2.4, 77):
        basis = np.power(np.clip(x, 0.0, 1.0), gamma)
        denom = float(np.sum(basis * basis))
        if denom <= 1e-12:
            amp = 0.0
        else:
            amp = float(np.sum(basis * (y - 1.0)) / denom)
        amp = max(0.0, amp)
        pred = 1.0 + amp * basis
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        if best is None or rmse < best['fit_rmse']:
            best = {
                'gain_noise_gamma': float(gamma),
                'gain_noise_amp': float(amp),
                'gain_noise_ratio_max': float(1.0 + amp),
                'fit_rmse': rmse,
            }

    assert best is not None
    best['fit_ok'] = 1.0
    best['num_fit_points'] = float(x.size)
    return best


def _recommend_from_rows(
    rows: list[dict[str, float]],
    reference_exposure_us: float,
    sim_reference_noise_m: float,
    min_fit_fill_rate: float,
) -> dict[str, Any]:
    usable = [
        row for row in rows
        if row.get('roi_fill_rate_mean', 0.0) >= float(min_fit_fill_rate)
        and row.get('temporal_std_median_m', 0.0) > 0.0
    ]
    if not usable:
        return {
            'fit_ok': 0.0,
            'reason': '没有足够有效的静态墙面数据；请检查距离、曝光、激光功率和 ROI。',
        }

    # Reference: lowest gain, exposure closest to requested reference.
    usable_sorted = sorted(
        usable,
        key=lambda row: (
            abs(float(row['exposure_us']) - float(reference_exposure_us)),
            float(row['gain01']),
            float(row['temporal_std_median_m']),
        ),
    )
    ref = usable_sorted[0]
    ref_noise = max(float(ref['temporal_std_median_m']), 1e-9)
    shot_unclamped = 0.03 * ref_noise / max(float(sim_reference_noise_m), 1e-9)
    shot_recommended = min(max(shot_unclamped, 0.003), 0.12)

    gain_fit = _fit_gain_noise_model(usable, min_fill_rate=min_fit_fill_rate)
    gamma = float(gain_fit.get('gain_noise_gamma', 1.2))
    # The env uses gain_scale in several places, so keep the direct noise fit
    # available but avoid extreme zero-effect suggestions.
    amp = float(gain_fit.get('gain_noise_amp', 10.0))
    gain_scale_noise_fit = min(max(amp, 0.1), 14.0)

    return {
        'fit_ok': 1.0,
        'reference_setting': ref,
        'reference_noise_m': ref_noise,
        'sim_reference_noise_m': float(sim_reference_noise_m),
        'cam_shot_noise_base_unclamped': float(shot_unclamped),
        'cam_shot_noise_base': float(shot_recommended),
        'gain_noise_fit': gain_fit,
        'cam_iso_gain_base': 1.0,
        'cam_iso_gain_scale_noise_fit': float(gain_scale_noise_fit),
        'cam_iso_gain_gamma_noise_fit': gamma,
        'recommended_args': {
            'cam_iso_gain_base': 1.0,
            'cam_iso_gain_scale': float(gain_scale_noise_fit),
            'cam_iso_gain_gamma': gamma,
            'cam_shot_noise_base': float(shot_recommended),
        },
        'notes': [
            'cam_shot_noise_base 是按参考低增益静态墙面 temporal_std 反推的全局噪声尺度。',
            'cam_iso_gain_scale_noise_fit 只拟合 gain 对 temporal depth noise 的影响，不保证同时匹配暗光 fill 提升。',
            '如果该 scale 远低于当前训练配置，建议先小步试验，不要一次性完全替换。',
        ],
    }


def _args_snippet(recommended_args: dict[str, float]) -> str:
    keys = [
        'cam_iso_gain_base',
        'cam_iso_gain_scale',
        'cam_iso_gain_gamma',
        'cam_shot_noise_base',
    ]
    return '\n'.join(f'--{key} {recommended_args[key]:.6g}' for key in keys)


def read_summary_csv(csv_path: str) -> list[dict[str, float]]:
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({k: _safe_float(v) for k, v in row.items()})
    return rows


def write_summary_csv(csv_path: str, rows: list[dict[str, float]]):
    ensure_dir(os.path.dirname(csv_path) or '.')
    fieldnames = [
        'setting_id',
        'exposure_us',
        'gain_value',
        'laser_power',
        'exposure01',
        'gain01',
        'power01',
        'roi_x0',
        'roi_y0',
        'roi_x1',
        'roi_y1',
        'roi_fill_rate_mean',
        'roi_fill_rate_min',
        'stable_pixel_ratio',
        'depth_mean_m',
        'temporal_std_mean_m',
        'temporal_std_median_m',
        'temporal_std_p75_m',
        'temporal_std_p90_m',
        'temporal_mad_median_m',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, 0.0) for k in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect/analyze D455 static-wall temporal depth noise for cam_shot_noise_base calibration.',
    )
    parser.add_argument('--output-dir', default=None,
                        help='输出目录；默认写到仓库根目录 artifacts/d455_static_noise/<timestamp>')
    parser.add_argument('--analyze-csv', default=None,
                        help='只分析已有 summary.csv，不连接硬件')
    parser.add_argument('--frames-per-setting', type=int, default=90)
    parser.add_argument('--warmup-frames', type=int, default=30)
    parser.add_argument('--settle-frames', type=int, default=12)
    parser.add_argument('--frame-sleep-s', type=float, default=0.0)
    parser.add_argument('--depth-min-valid', type=float, default=0.3)
    parser.add_argument('--depth-max-range', type=float, default=6.0)
    parser.add_argument('--roi-fraction', type=float, default=0.45,
                        help='中心 ROI 尺寸占整幅图的比例；静态墙面建议 0.35~0.6')
    parser.add_argument('--valid-pixel-ratio', type=float, default=0.85,
                        help='一个 ROI 像素至少多少帧有效，才参与 temporal noise 统计')
    parser.add_argument('--exposure-us-values', default=None,
                        help='逗号分隔曝光列表，例如 3000,10000,30000')
    parser.add_argument('--gain-values', default=None,
                        help='逗号分隔 gain 列表，例如 16,32,64,128,248')
    parser.add_argument('--laser-power', type=float, default=None,
                        help='固定激光功率；默认使用 D455 当前模式的 laser_power default')
    parser.add_argument('--reference-exposure-us', type=float, default=3000.0,
                        help='用于反推 cam_shot_noise_base 的参考曝光')
    parser.add_argument('--sim-reference-noise-m', type=float, default=0.018,
                        help='当前仿真默认 shot noise 参考值；用于把真实 temporal_std 映射回 cam_shot_noise_base')
    parser.add_argument('--min-fit-fill-rate', type=float, default=0.90,
                        help='参与拟合的设置至少需要达到的 ROI fill rate')
    parser.add_argument('--notes', default='')
    return parser.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir is None:
        output_dir = os.path.join(_repo_root(), 'artifacts', 'd455_static_noise', ts)
    else:
        output_dir = args.output_dir
    output_dir = ensure_dir(output_dir)
    summary_csv = os.path.join(output_dir, 'summary.csv')
    fit_json = os.path.join(output_dir, 'fit_result.json')
    meta_json = os.path.join(output_dir, 'meta.json')

    if args.analyze_csv:
        rows = read_summary_csv(args.analyze_csv)
        fit = _recommend_from_rows(
            rows,
            reference_exposure_us=args.reference_exposure_us,
            sim_reference_noise_m=args.sim_reference_noise_m,
            min_fit_fill_rate=args.min_fit_fill_rate,
        )
        result = {
            'mode': 'analyze_csv',
            'source_csv': args.analyze_csv,
            'fit': fit,
            'rows': rows,
        }
        write_json(fit_json, result)
        print(f'[analyze] rows={len(rows)}')
        if fit.get('fit_ok', 0.0) > 0:
            print('[analyze] recommended args:')
            print(_args_snippet(fit['recommended_args']))
        else:
            print(f"[analyze] failed: {fit.get('reason', 'unknown')}")
        print(f'[analyze] wrote {fit_json}')
        return

    import pyrealsense2 as rs

    pipeline = None
    try:
        pipeline, profile, mode = start_pipeline()
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            time.sleep(0.1)

        exp_range = depth_sensor.get_option_range(rs.option.exposure)
        gain_range = depth_sensor.get_option_range(rs.option.gain)
        laser_range = depth_sensor.get_option_range(rs.option.laser_power)
        settings, laser_power = build_settings(args, exp_range, gain_range, laser_range)

        meta = {
            'output_dir': output_dir,
            'depth_mode': {
                'width': int(mode[0]),
                'height': int(mode[1]),
                'fps': int(mode[2]),
            },
            'ranges': {
                'exposure': {
                    'min': float(exp_range.min),
                    'max': float(exp_range.max),
                    'step': float(exp_range.step),
                    'default': float(exp_range.default),
                },
                'gain': {
                    'min': float(gain_range.min),
                    'max': float(gain_range.max),
                    'step': float(gain_range.step),
                    'default': float(gain_range.default),
                },
                'laser_power': {
                    'min': float(laser_range.min),
                    'max': float(laser_range.max),
                    'step': float(laser_range.step),
                    'default': float(laser_range.default),
                },
            },
            'frames_per_setting': int(args.frames_per_setting),
            'warmup_frames': int(args.warmup_frames),
            'settle_frames': int(args.settle_frames),
            'depth_min_valid': float(args.depth_min_valid),
            'depth_max_range': float(args.depth_max_range),
            'roi_fraction': float(args.roi_fraction),
            'valid_pixel_ratio': float(args.valid_pixel_ratio),
            'laser_power': float(laser_power),
            'notes': args.notes,
            'settings': settings,
        }
        write_json(meta_json, meta)

        print(f'[capture] output_dir={output_dir}')
        print(f'[capture] depth_mode={mode[0]}x{mode[1]}@{mode[2]}')
        print(f'[capture] settings={len(settings)} laser_power={laser_power:.0f}')
        print('[capture] 请保持 D455 和墙面完全静止，避免人经过或光照变化。')

        for _ in range(int(args.warmup_frames)):
            fetch_depth_m(pipeline)

        rows: list[dict[str, float]] = []
        for setting in settings:
            depth_sensor.set_option(rs.option.exposure, float(setting['exposure_us']))
            depth_sensor.set_option(rs.option.gain, float(setting['gain_value']))
            depth_sensor.set_option(rs.option.laser_power, float(setting['laser_power']))
            for _ in range(int(args.settle_frames)):
                fetch_depth_m(pipeline)

            stack = collect_stack(
                pipeline,
                frames=args.frames_per_setting,
                sleep_s=args.frame_sleep_s,
            )
            metrics = compute_static_noise_metrics(
                stack,
                depth_min_valid=args.depth_min_valid,
                depth_max_range=args.depth_max_range,
                roi_fraction=args.roi_fraction,
                valid_pixel_ratio=args.valid_pixel_ratio,
            )
            row = {
                'setting_id': float(setting['setting_id']),
                'exposure_us': float(setting['exposure_us']),
                'gain_value': float(setting['gain_value']),
                'laser_power': float(setting['laser_power']),
                'exposure01': exposure_proxy_from_us(setting['exposure_us'], exp_range.min, exp_range.max),
                'gain01': gain_proxy_from_value(setting['gain_value'], gain_range.min, gain_range.max),
                'power01': power_proxy_from_laser(setting['laser_power'], laser_range.max),
            }
            row.update({k: float(v) for k, v in metrics.items()})
            rows.append(row)
            print(
                f"[capture] setting={int(setting['setting_id']):02d} "
                f"exp={setting['exposure_us']:.0f} gain={setting['gain_value']:.0f} "
                f"fill={row['roi_fill_rate_mean']:.3f} "
                f"noise_med={row['temporal_std_median_m'] * 1000.0:.2f}mm "
                f"noise_p90={row['temporal_std_p90_m'] * 1000.0:.2f}mm"
            )

        write_summary_csv(summary_csv, rows)
        fit = _recommend_from_rows(
            rows,
            reference_exposure_us=args.reference_exposure_us,
            sim_reference_noise_m=args.sim_reference_noise_m,
            min_fit_fill_rate=args.min_fit_fill_rate,
        )
        result = {
            'mode': 'capture',
            'meta_path': meta_json,
            'summary_csv': summary_csv,
            'fit': fit,
            'rows': rows,
        }
        write_json(fit_json, result)

        print()
        print(f'[capture] wrote summary: {summary_csv}')
        print(f'[capture] wrote fit:     {fit_json}')
        if fit.get('fit_ok', 0.0) > 0:
            print()
            print('Recommended args snippet')
            print(_args_snippet(fit['recommended_args']))
            print()
            gf = fit['gain_noise_fit']
            print(
                f"Gain/noise fit: ratio_max={gf['gain_noise_ratio_max']:.3f}, "
                f"gamma={gf['gain_noise_gamma']:.3f}, "
                f"rmse={gf['fit_rmse']:.4f}, "
                f"points={int(gf['num_fit_points'])}"
            )
            print(
                f"Reference noise: {fit['reference_noise_m'] * 1000.0:.2f}mm -> "
                f"cam_shot_noise_base={fit['cam_shot_noise_base']:.5f}"
            )
        else:
            print(f"[capture] fit failed: {fit.get('reason', 'unknown')}")

    finally:
        if pipeline is not None:
            pipeline.stop()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Recommend diff_depth camera semantics aligned to a local D455 unit.

This script does two things:
1. Query D455 hardware ranges directly, or read them from a calibration `meta.json`.
2. Convert those ranges into the project's semantic config values:
   - cam_exposure_t_min
   - cam_exposure_t_span
   - cam_exposure_eff_min
   - cam_exposure_eff_max
   - cam_iso_gain_base
   - cam_iso_gain_scale
   - cam_iso_gain_gamma
   - cam_shot_noise_base

Important:
- Exposure mapping is the part that can be aligned most directly to D455.
- Gain / shot-noise mapping is still a semantic starting point, not a literal
  RealSense register-to-physics identity.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any


DEPTH_MODE_CANDIDATES = [
    (848, 480, 30),
    (640, 480, 30),
    (640, 480, 15),
    (424, 240, 30),
]


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _load_json(path: str) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _query_d455_ranges() -> dict[str, Any]:
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    profile = None
    last_error = None
    chosen_mode = None
    try:
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

        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)

        exp_range = depth_sensor.get_option_range(rs.option.exposure)
        gain_range = depth_sensor.get_option_range(rs.option.gain)
        laser_range = depth_sensor.get_option_range(rs.option.laser_power)
        return {
            'source': 'hardware',
            'depth_mode': {
                'width': int(chosen_mode[0]) if chosen_mode else None,
                'height': int(chosen_mode[1]) if chosen_mode else None,
                'fps': int(chosen_mode[2]) if chosen_mode else None,
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
        }
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


def _load_ranges_from_meta(meta_path: str) -> dict[str, Any]:
    meta = _load_json(meta_path)
    if 'ranges' not in meta:
        raise ValueError(f'meta.json 缺少 ranges 字段: {meta_path}')
    return {
        'source': 'meta',
        'depth_mode': meta.get('depth_mode', {}),
        'ranges': meta['ranges'],
    }


def _default_working_exposure_us(full_min: float, full_max: float) -> tuple[float, float]:
    """
    Conservative flight-oriented default exposure window.

    The current project defaults correspond to about 2.5ms -> 30ms:
      t = 0.25 -> 3.0
      delay = t * 0.01 sec
    """
    lo = max(float(full_min), 2500.0)
    hi = min(float(full_max), 30000.0)
    if hi <= lo + 1e-6:
        lo = float(full_min)
        hi = float(full_max)
    return float(lo), float(hi)


def _default_working_gain(full_min: float, full_max: float) -> tuple[float, float]:
    lo = float(full_min)
    hi = float(full_max)
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    return lo, hi


def _recommend_gain_scale(gain_min: float, gain_max: float) -> float:
    raw_ratio = float(gain_max) / max(float(gain_min), 1e-6)
    # Heuristic:
    # The sim uses gain_scale in both passive signal and noise paths, so using
    # the raw register ratio directly is usually too aggressive. Compress it.
    return _clamp(0.7 * (raw_ratio - 1.0), 6.0, 14.0)


def _recommend_semantics(
    exp_min_us: float,
    exp_max_us: float,
    gain_min: float,
    gain_max: float,
    laser_default: float,
    laser_max: float,
    exposure_divisor_us: float,
    gain_gamma: float,
    shot_noise_base: float,
) -> dict[str, float]:
    exp_min_us = float(exp_min_us)
    exp_max_us = float(exp_max_us)
    gain_min = float(gain_min)
    gain_max = float(gain_max)
    exposure_divisor_us = max(float(exposure_divisor_us), 1e-6)

    t_min = exp_min_us / exposure_divisor_us
    t_max = exp_max_us / exposure_divisor_us
    t_span = max(0.0, t_max - t_min)
    power_nominal = _clamp(float(laser_default) / max(float(laser_max), 1e-6), 0.0, 1.0)

    return {
        'cam_power_nominal': float(power_nominal),
        'cam_power_penalty_threshold': float(power_nominal),
        'cam_exposure_t_min': float(t_min),
        'cam_exposure_t_span': float(t_span),
        'cam_exposure_eff_min': float(t_min),
        'cam_exposure_eff_max': float(t_max),
        'cam_iso_gain_base': 1.0,
        'cam_iso_gain_scale': float(_recommend_gain_scale(gain_min, gain_max)),
        'cam_iso_gain_gamma': float(gain_gamma),
        'cam_shot_noise_base': float(shot_noise_base),
    }


def _format_args_block(values: dict[str, float]) -> str:
    lines = []
    for key in (
        'cam_power_nominal',
        'cam_power_penalty_threshold',
        'cam_exposure_t_min',
        'cam_exposure_t_span',
        'cam_exposure_eff_min',
        'cam_exposure_eff_max',
        'cam_iso_gain_base',
        'cam_iso_gain_scale',
        'cam_iso_gain_gamma',
        'cam_shot_noise_base',
    ):
        lines.append(f'--{key} {values[key]:.6g}')
    return '\n'.join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description='Recommend project camera-semantics params for a D455.')
    parser.add_argument(
        '--meta-json',
        type=str,
        default=None,
        help='可选：使用 collect_d455_calibration.py 生成目录中的 meta.json 离线推导，而不是直接访问硬件',
    )
    parser.add_argument(
        '--working-exposure-min-us',
        type=float,
        default=None,
        help='可选：手动指定想让 exposure01=0 对应的 D455 曝光下界（微秒）',
    )
    parser.add_argument(
        '--working-exposure-max-us',
        type=float,
        default=None,
        help='可选：手动指定想让 exposure01=1 对应的 D455 曝光上界（微秒）',
    )
    parser.add_argument(
        '--working-gain-min',
        type=float,
        default=None,
        help='可选：手动指定 gain01=0 对应的 D455 gain 下界',
    )
    parser.add_argument(
        '--working-gain-max',
        type=float,
        default=None,
        help='可选：手动指定 gain01=1 对应的 D455 gain 上界',
    )
    parser.add_argument(
        '--exposure-divisor-us',
        type=float,
        default=10000.0,
        help='把 D455 exposure(us) 映射到项目 time-scale 的除数；默认 10000，即 3000us -> 0.3',
    )
    parser.add_argument(
        '--gain-gamma',
        type=float,
        default=1.2,
        help='输出建议中的 cam_iso_gain_gamma；默认沿用当前项目经验值',
    )
    parser.add_argument(
        '--shot-noise-base',
        type=float,
        default=0.03,
        help='输出建议中的 cam_shot_noise_base；仅给初值，后续需用静态墙面数据再细调',
    )
    parser.add_argument(
        '--json-out',
        type=str,
        default=None,
        help='可选：把建议结果写到 JSON',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.meta_json:
        source_info = _load_ranges_from_meta(args.meta_json)
    else:
        source_info = _query_d455_ranges()

    exp_full = source_info['ranges']['exposure']
    gain_full = source_info['ranges']['gain']
    laser_full = source_info['ranges']['laser_power']

    if args.working_exposure_min_us is None or args.working_exposure_max_us is None:
        default_exp_lo, default_exp_hi = _default_working_exposure_us(exp_full['min'], exp_full['max'])
        exp_lo = default_exp_lo if args.working_exposure_min_us is None else args.working_exposure_min_us
        exp_hi = default_exp_hi if args.working_exposure_max_us is None else args.working_exposure_max_us
    else:
        exp_lo = args.working_exposure_min_us
        exp_hi = args.working_exposure_max_us

    gain_lo_default, gain_hi_default = _default_working_gain(gain_full['min'], gain_full['max'])
    gain_lo = gain_hi = 0.0
    gain_lo = gain_lo_default if args.working_gain_min is None else args.working_gain_min
    gain_hi = gain_hi_default if args.working_gain_max is None else args.working_gain_max

    exp_lo = _clamp(exp_lo, exp_full['min'], exp_full['max'])
    exp_hi = _clamp(exp_hi, exp_lo + 1e-6, exp_full['max'])
    gain_lo = _clamp(gain_lo, gain_full['min'], gain_full['max'])
    gain_hi = _clamp(gain_hi, gain_lo + 1e-6, gain_full['max'])

    recommended = _recommend_semantics(
        exp_min_us=exp_lo,
        exp_max_us=exp_hi,
        gain_min=gain_lo,
        gain_max=gain_hi,
        laser_default=laser_full['default'],
        laser_max=laser_full['max'],
        exposure_divisor_us=args.exposure_divisor_us,
        gain_gamma=args.gain_gamma,
        shot_noise_base=args.shot_noise_base,
    )

    result = {
        'source': source_info['source'],
        'depth_mode': source_info.get('depth_mode', {}),
        'ranges': source_info['ranges'],
        'working_window': {
            'exposure_us_min': float(exp_lo),
            'exposure_us_max': float(exp_hi),
            'gain_min': float(gain_lo),
            'gain_max': float(gain_hi),
            'laser_power_min': float(laser_full['min']),
            'laser_power_max': float(laser_full['max']),
            'exposure_divisor_us': float(args.exposure_divisor_us),
        },
        'recommended': recommended,
        'notes': [
            'cam_power_nominal / cam_power_penalty_threshold 已按 D455 默认 laser_power/max 归一化。',
            'cam_exposure_* 是与 D455 最容易直接对齐的一组参数。',
            'cam_iso_gain_* 是语义增益，不是 D455 原始 gain 寄存器本身；这里给的是可运行初值。',
            'cam_shot_noise_base 仅给初值，最好再用静态平面墙面数据按 depth_std 细调。',
            '当前项目里 power01 已经按 laser_power / laser_power_max 归一化，因此 power 相关推荐值最值得信任。',
        ],
    }

    if args.json_out:
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print('D455 range probe')
    print(f"source: {result['source']}")
    if result.get('depth_mode'):
        dm = result['depth_mode']
        if dm.get('width') is not None:
            print(f"depth_mode: {dm['width']}x{dm['height']}@{dm['fps']}")
    print()
    print('Hardware ranges')
    print(
        f"  exposure(us): {exp_full['min']:.0f} .. {exp_full['max']:.0f} "
        f"(default={exp_full['default']:.0f}, step={exp_full['step']:.0f})"
    )
    print(
        f"  gain:         {gain_full['min']:.0f} .. {gain_full['max']:.0f} "
        f"(default={gain_full['default']:.0f}, step={gain_full['step']:.0f})"
    )
    print(
        f"  laser_power:  {laser_full['min']:.0f} .. {laser_full['max']:.0f} "
        f"(default={laser_full['default']:.0f}, step={laser_full['step']:.0f})"
    )
    print()
    print('Chosen working window')
    print(f'  exposure(us): {exp_lo:.0f} .. {exp_hi:.0f}')
    print(f'  gain:         {gain_lo:.0f} .. {gain_hi:.0f}')
    print(f"  exposure_divisor_us: {args.exposure_divisor_us:.0f}")
    print()
    print('Recommended args snippet')
    print(_format_args_block(recommended))
    print()
    print('Interpretation')
    print(
        f"  power nominal -> {recommended['cam_power_nominal']:.4f} "
        f"(about {laser_full['default']:.0f}/{laser_full['max']:.0f})"
    )
    print(
        f"  exposure01=0 -> {recommended['cam_exposure_t_min']:.4f} "
        f"(about {exp_lo:.0f} us)"
    )
    print(
        f"  exposure01=1 -> "
        f"{recommended['cam_exposure_t_min'] + recommended['cam_exposure_t_span']:.4f} "
        f"(about {exp_hi:.0f} us)"
    )
    print(
        f"  gain01=1 -> gain_scale about "
        f"{recommended['cam_iso_gain_base'] + recommended['cam_iso_gain_scale']:.4f} "
        f"(semantic, not raw D455 gain)"
    )
    print()
    print('Notes')
    for note in result['notes']:
        print(f'  - {note}')
    if args.json_out:
        print()
        print(f'Wrote JSON to: {args.json_out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Collect D455 depth calibration data for diff_depth scene fitting."""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

from d455_calib_utils import (
    SUPPORTED_CALIB_SCENES,
    compute_depth_stats,
    ensure_dir,
    exposure_proxy_from_us,
    gain_proxy_from_value,
    power_proxy_from_laser,
    row_writer_from_path,
    write_json,
)


DEFAULT_DEPTH_MODES = [
    (848, 480, 30),
    (640, 480, 30),
    (640, 480, 15),
    (424, 240, 30),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Collect D455 calibration sweeps for diff_depth scene fitting.')
    parser.add_argument('--scene', required=True, choices=SUPPORTED_CALIB_SCENES)
    parser.add_argument('--condition-id', required=True,
                        help='场景内条件标签，例如 glare_frontlit / glass_panel / black_foam / dark_slit_fast')
    parser.add_argument('--output-dir', default='artifacts/d455_calibration')
    parser.add_argument('--frames-per-setting', type=int, default=20)
    parser.add_argument('--warmup-frames', type=int, default=10)
    parser.add_argument('--settle-frames', type=int, default=6,
                        help='每次写寄存器后丢弃的稳定帧数，用于估计参数生效延迟')
    parser.add_argument('--depth-min-valid', type=float, default=0.3)
    parser.add_argument('--depth-max-range', type=float, default=6.0)
    parser.add_argument('--vis-max-depth-mm', type=float, default=6000.0)
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def build_setting_grid(exp_range, gain_range, laser_range):
    exp_mid = float(np.clip(3000.0, exp_range.min, exp_range.max))
    gain_mid = float(np.clip(16.0, gain_range.min, gain_range.max))
    laser_mid = float(np.clip(150.0, laser_range.min, laser_range.max))

    def picks(rng, values):
        out = []
        for v in values:
            out.append(float(np.clip(v, rng.min, rng.max)))
        dedup = []
        for v in out:
            if not dedup or abs(dedup[-1] - v) > max(1e-6, float(rng.step) * 0.5):
                dedup.append(v)
        return dedup

    exp_values = picks(exp_range, [exp_range.min, exp_mid, 0.75 * exp_range.max, exp_range.max])
    gain_values = picks(gain_range, [gain_range.min, gain_mid, 0.7 * gain_range.max, gain_range.max])
    laser_values = picks(laser_range, [laser_range.min, 0.33 * laser_range.max, 0.66 * laser_range.max, laser_range.max])

    settings = []
    idx = 0
    for exposure_us in exp_values:
        settings.append({
            'setting_id': idx,
            'exposure_us': float(exposure_us),
            'gain_value': float(gain_mid),
            'laser_power': float(laser_mid),
            'scan_axis': 'exposure',
        })
        idx += 1
    for gain_value in gain_values:
        settings.append({
            'setting_id': idx,
            'exposure_us': float(exp_mid),
            'gain_value': float(gain_value),
            'laser_power': float(laser_mid),
            'scan_axis': 'gain',
        })
        idx += 1
    for laser_value in laser_values:
        settings.append({
            'setting_id': idx,
            'exposure_us': float(exp_mid),
            'gain_value': float(gain_mid),
            'laser_power': float(laser_value),
            'scan_axis': 'laser',
        })
        idx += 1
    return settings


def start_pipeline():
    pipeline = rs.pipeline()
    profile = None
    last_error = None
    chosen_mode = None
    for width, height, fps in DEFAULT_DEPTH_MODES:
        try:
            config = rs.config()
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            profile = pipeline.start(config)
            chosen_mode = (width, height, fps)
            break
        except RuntimeError as e:
            last_error = e
    if profile is None:
        raise RuntimeError(f'无法启动 D455 深度流，最后错误: {last_error}')
    return pipeline, profile, chosen_mode


def fetch_depth_frame(pipeline):
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        return None, None
    depth_scale = depth_frame.get_units()
    depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * float(depth_scale)
    return depth_frame, depth_m


def save_preview_image(depth_m, output_path: str, vis_max_depth_mm: float):
    depth_mm = np.clip(depth_m * 1000.0, 0.0, float(vis_max_depth_mm))
    depth_8u = cv2.convertScaleAbs(depth_mm, alpha=255.0 / max(float(vis_max_depth_mm), 1e-6))
    depth_vis = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, depth_vis)


def main():
    args = parse_args()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.scene}_{args.condition_id}_{ts}"
    output_dir = ensure_dir(os.path.join(args.output_dir, run_name))
    preview_dir = ensure_dir(os.path.join(output_dir, 'preview'))
    csv_path = os.path.join(output_dir, 'capture.csv')
    meta_path = os.path.join(output_dir, 'meta.json')

    pipeline = None
    profile = None
    mode = None

    try:
        if args.dry_run:
            print('[dry-run] skip hardware start')
            return

        pipeline, profile, mode = start_pipeline()
        depth_sensor = profile.get_device().first_depth_sensor()

        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            time.sleep(0.1)

        exp_range = depth_sensor.get_option_range(rs.option.exposure)
        gain_range = depth_sensor.get_option_range(rs.option.gain)
        laser_range = depth_sensor.get_option_range(rs.option.laser_power)
        settings = build_setting_grid(exp_range, gain_range, laser_range)

        fieldnames = [
            'timestamp', 'scene', 'condition_id', 'setting_id', 'scan_axis',
            'frame_index', 'frame_delay_estimate', 'width', 'height', 'fps',
            'exposure_us', 'gain_value', 'laser_power',
            'power01', 'exposure01', 'gain01',
            'fill_rate', 'invalid_ratio',
            'depth_mean_m', 'depth_std_m', 'depth_min_m', 'depth_max_m',
            'depth_p10_m', 'depth_p50_m', 'depth_p90_m',
            'depth_variance_m2', 'edge_std_m',
            'brightness_proxy',
            'notes', 'tag',
        ]
        csv_fh, writer = row_writer_from_path(csv_path, fieldnames)

        meta = {
            'scene': args.scene,
            'condition_id': args.condition_id,
            'output_dir': output_dir,
            'depth_mode': {
                'width': int(mode[0]) if mode else None,
                'height': int(mode[1]) if mode else None,
                'fps': int(mode[2]) if mode else None,
            },
            'ranges': {
                'exposure': {'min': exp_range.min, 'max': exp_range.max, 'step': exp_range.step, 'default': exp_range.default},
                'gain': {'min': gain_range.min, 'max': gain_range.max, 'step': gain_range.step, 'default': gain_range.default},
                'laser_power': {'min': laser_range.min, 'max': laser_range.max, 'step': laser_range.step, 'default': laser_range.default},
            },
            'frames_per_setting': int(args.frames_per_setting),
            'warmup_frames': int(args.warmup_frames),
            'settle_frames': int(args.settle_frames),
            'depth_min_valid': float(args.depth_min_valid),
            'depth_max_range': float(args.depth_max_range),
            'notes': args.notes,
            'tag': args.tag,
            'settings': settings,
        }
        write_json(meta_path, meta)

        print(f'[capture] output_dir={output_dir}')
        print(f'[capture] depth_mode={mode[0]}x{mode[1]}@{mode[2]}' if mode else '[capture] depth_mode=unknown')

        for _ in range(int(args.warmup_frames)):
            fetch_depth_frame(pipeline)

        for setting in settings:
            depth_sensor.set_option(rs.option.exposure, float(setting['exposure_us']))
            depth_sensor.set_option(rs.option.gain, float(setting['gain_value']))
            depth_sensor.set_option(rs.option.laser_power, float(setting['laser_power']))

            settle_measure = []
            for settle_idx in range(int(args.settle_frames)):
                _, depth_m = fetch_depth_frame(pipeline)
                if depth_m is None:
                    continue
                stats = compute_depth_stats(
                    depth_m,
                    min_valid_m=args.depth_min_valid,
                    max_valid_m=args.depth_max_range,
                )
                settle_measure.append(stats['fill_rate'])

            frame_delay_estimate = 0.0
            if len(settle_measure) >= 2:
                ref = settle_measure[-1]
                for idx, val in enumerate(settle_measure):
                    if abs(val - ref) <= 0.02:
                        frame_delay_estimate = float(idx + 1)
                        break
                if frame_delay_estimate <= 0.0:
                    frame_delay_estimate = float(len(settle_measure))

            for frame_index in range(int(args.frames_per_setting)):
                _, depth_m = fetch_depth_frame(pipeline)
                if depth_m is None:
                    continue
                stats = compute_depth_stats(
                    depth_m,
                    min_valid_m=args.depth_min_valid,
                    max_valid_m=args.depth_max_range,
                )
                preview_path = os.path.join(
                    preview_dir,
                    f"s{int(setting['setting_id']):03d}_f{frame_index:03d}.jpg",
                )
                if frame_index == 0:
                    save_preview_image(depth_m, preview_path, args.vis_max_depth_mm)

                row = {
                    'timestamp': time.time(),
                    'scene': args.scene,
                    'condition_id': args.condition_id,
                    'setting_id': int(setting['setting_id']),
                    'scan_axis': setting['scan_axis'],
                    'frame_index': frame_index,
                    'frame_delay_estimate': frame_delay_estimate,
                    'width': int(mode[0]) if mode else 0,
                    'height': int(mode[1]) if mode else 0,
                    'fps': int(mode[2]) if mode else 0,
                    'exposure_us': float(setting['exposure_us']),
                    'gain_value': float(setting['gain_value']),
                    'laser_power': float(setting['laser_power']),
                    'power01': power_proxy_from_laser(setting['laser_power'], laser_range.max),
                    'exposure01': exposure_proxy_from_us(setting['exposure_us'], exp_range.min, exp_range.max),
                    'gain01': gain_proxy_from_value(setting['gain_value'], gain_range.min, gain_range.max),
                    'brightness_proxy': float(np.mean(np.clip(depth_m, 0.0, args.depth_max_range) > 0.0)),
                    'notes': args.notes,
                    'tag': args.tag,
                }
                row.update(stats)
                writer.writerow(row)
            csv_fh.flush()
            print(
                f"[capture] scene={args.scene} condition={args.condition_id} "
                f"setting={setting['setting_id']:02d} axis={setting['scan_axis']} "
                f"exp={setting['exposure_us']:.0f} gain={setting['gain_value']:.0f} laser={setting['laser_power']:.0f}"
            )

        csv_fh.close()
        print(f'[capture] done. csv={csv_path}')

    finally:
        if pipeline is not None:
            pipeline.stop()


if __name__ == '__main__':
    main()

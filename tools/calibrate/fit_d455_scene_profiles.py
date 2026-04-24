#!/usr/bin/env python3
"""Fit diff_depth scene parameters from collected D455 calibration CSV files."""
from __future__ import annotations

import argparse
import os

from tools.calibrate.d455_calib_utils import (
    SUPPORTED_CALIB_SCENES,
    fit_scene_profile,
    load_csv_rows,
    read_json,
    to_float_rows,
    write_json,
)


NUMERIC_KEYS = [
    'setting_id', 'frame_index', 'frame_delay_estimate',
    'width', 'height', 'fps',
    'exposure_us', 'gain_value', 'laser_power',
    'power01', 'exposure01', 'gain01',
    'fill_rate', 'invalid_ratio',
    'depth_mean_m', 'depth_std_m', 'depth_min_m', 'depth_max_m',
    'depth_p10_m', 'depth_p50_m', 'depth_p90_m',
    'depth_variance_m2', 'edge_std_m',
    'brightness_proxy',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Fit diff_depth scene profiles from D455 captures.')
    parser.add_argument('--input-dir', default='artifacts/d455_calibration',
                        help='collect_d455_calibration.py 输出的根目录')
    parser.add_argument('--output-json', default='artifacts/d455_calibration/scene_fit_profiles.json')
    parser.add_argument('--scene', action='append', choices=SUPPORTED_CALIB_SCENES,
                        help='只拟合指定场景；可多次传入')
    return parser.parse_args()


def discover_capture_dirs(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        if os.path.isfile(os.path.join(full, 'capture.csv')) and os.path.isfile(os.path.join(full, 'meta.json')):
            out.append(full)
    return out


def main():
    args = parse_args()
    capture_dirs = discover_capture_dirs(args.input_dir)
    if not capture_dirs:
        raise FileNotFoundError(f'未找到采集目录: {args.input_dir}')

    all_rows = []
    capture_meta = []
    for cap_dir in capture_dirs:
        meta = read_json(os.path.join(cap_dir, 'meta.json'))
        rows = to_float_rows(load_csv_rows(os.path.join(cap_dir, 'capture.csv')), NUMERIC_KEYS)
        capture_meta.append(meta)
        all_rows.extend(rows)

    target_scenes = list(args.scene) if args.scene else list(SUPPORTED_CALIB_SCENES)
    scene_results = {}
    for scene in target_scenes:
        res = fit_scene_profile(scene, all_rows)
        scene_results[scene] = res.to_dict()

    out = {
        'captures': [
            {
                'scene': m.get('scene'),
                'condition_id': m.get('condition_id'),
                'output_dir': m.get('output_dir'),
                'depth_mode': m.get('depth_mode'),
                'frames_per_setting': m.get('frames_per_setting'),
            }
            for m in capture_meta
        ],
        'scene_profiles': {
            scene: payload['sim_profile']
            for scene, payload in scene_results.items()
        },
        'fit_details': scene_results,
        'usage': {
            'note': 'scene_profiles 字段可直接作为 env_cuda.py 当前场景 effects 参数调参参考。',
            'scenes': target_scenes,
        },
    }
    write_json(args.output_json, out)
    print(f'[fit] wrote {args.output_json}')
    for scene in target_scenes:
        profile = out['scene_profiles'].get(scene, {})
        print(f'[fit] {scene}: {profile}')


if __name__ == '__main__':
    main()

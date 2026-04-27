#!/usr/bin/env python3
"""Quick interactive D455 depth sanity check.

正式标定采集请使用:
  python3 tools/collect_d455_calibration.py --scene glare --condition-id glare_front
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs


SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'depth.jpg')
DEPTH_MODE_CANDIDATES = [
    (848, 480, 30),
    (640, 480, 30),
    (640, 480, 15),
    (424, 240, 30),
]


def main():
    pipeline = rs.pipeline()
    profile = None
    chosen_mode = None

    try:
        for width, height, fps in DEPTH_MODE_CANDIDATES:
            try:
                config = rs.config()
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                profile = pipeline.start(config)
                chosen_mode = (width, height, fps)
                break
            except RuntimeError:
                continue

        if profile is None:
            raise RuntimeError('无法启动 D455 深度流，请检查 USB3 连接、供电和占用进程。')

        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)

        exp_range = depth_sensor.get_option_range(rs.option.exposure)
        gain_range = depth_sensor.get_option_range(rs.option.gain)
        laser_range = depth_sensor.get_option_range(rs.option.laser_power)

        exposure_us = float(np.clip(3000.0, exp_range.min, exp_range.max))
        gain_value = float(np.clip(16.0, gain_range.min, gain_range.max))
        laser_power = float(np.clip(150.0, laser_range.min, laser_range.max))
        depth_sensor.set_option(rs.option.exposure, exposure_us)
        depth_sensor.set_option(rs.option.gain, gain_value)
        depth_sensor.set_option(rs.option.laser_power, laser_power)

        print(f'[quick-check] mode={chosen_mode[0]}x{chosen_mode[1]}@{chosen_mode[2]}' if chosen_mode else '[quick-check] mode=unknown')
        print(f'[quick-check] exposure={exposure_us:.0f} gain={gain_value:.0f} laser={laser_power:.0f}')
        print(f'[quick-check] preview={SAVE_PATH}')

        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            depth_scale = depth_frame.get_units()
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * float(depth_scale)
            valid = np.isfinite(depth_m) & (depth_m >= 0.3) & (depth_m <= 6.0)
            fill_rate = float(valid.mean())
            depth_mm = np.clip(depth_m * 1000.0, 0.0, 6000.0)
            depth_8u = cv2.convertScaleAbs(depth_mm, alpha=255.0 / 6000.0)
            depth_vis = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
            cv2.imwrite(SAVE_PATH, depth_vis)
            sys.stdout.write(f'\rfill_rate={fill_rate:.3f} saved={SAVE_PATH}     ')
            sys.stdout.flush()
            time.sleep(0.05)

    except KeyboardInterrupt:
        print('\n[quick-check] stopped by user')
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass


if __name__ == '__main__':
    main()

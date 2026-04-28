#!/usr/bin/env python3
"""Run a trained DiffPhysDrone policy on real hardware.

This script bridges four pieces into one realtime loop:
1. Load a trained checkpoint and the same project args used in sim.
2. Read realtime depth from an Intel RealSense D455.
3. Map policy camera outputs (power/exposure/gain in [0,1]) back to D455 registers.
4. Convert policy motion outputs into PX4 offboard setpoints via pymavlink.

Design goal:
- Reuse the project's existing policy semantics as much as possible.
- Keep the script self-contained so it can run directly on Jetson/Orin after
  the required runtime dependencies are installed.

Required extra runtime dependencies on the robot:
- pyrealsense2
- pymavlink

Typical example:
  python3 tools/run_real_policy.py \
      --checkpoint checkpoint/2026-04-23-12-12-57/checkpoint0014.pth \
      --args-file configs/paper_final_full.args \
      --px4-connection udp:127.0.0.1:14540 \
      --arm \
      --auto-takeoff \
      --takeoff-height-m 1.5 \
      --goal-forward-m 5.0 \
      --goal-left-m 0.0 \
      --goal-up-m 0.0 \
      --finish-action hold

Notes:
- The script is intentionally conservative about safety:
  - PX4 is only armed when `--arm` is passed.
  - The mission starts after an explicit offboard warmup and optional takeoff.
  - On error, it tries to switch to a safe hold/land behavior.
- The real vehicle must already have a valid local-position estimate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from camera_semantics import from_args as camera_semantics_from_args
from config import (
    build_parser,
    parse_diff_sensor_impl,
    parse_scenarios,
    set_global_seed,
    validate_args,
    print_runtime_mode,
)
from model import Model
from rollout_ops import (
    build_local_frame,
    build_state_vector,
    compute_depth_fill_rate,
    compute_target_velocity,
    decode_action_direct,
    diff_depth_exposure_to_time,
    init_camera_params,
    select_policy_depth_obs,
    update_camera_params,
)


DEFAULT_D455_MODES = [
    (640, 480, 30),
    (848, 480, 30),
    (640, 480, 15),
    (424, 240, 30),
]

PX4_CUSTOM_MAIN_MODE_MANUAL = 1
PX4_CUSTOM_MAIN_MODE_ALTCTL = 2
PX4_CUSTOM_MAIN_MODE_POSCTL = 3
PX4_CUSTOM_MAIN_MODE_AUTO = 4
PX4_CUSTOM_MAIN_MODE_ACRO = 5
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6


def _bool_parser_flag(default: bool = False) -> dict[str, Any]:
    return {
        'default': default,
        'action': argparse.BooleanOptionalAction,
    }


def parse_cli():
    parser = argparse.ArgumentParser(
        description='Run a trained DiffPhysDrone checkpoint on real D455 + PX4 hardware.',
    )
    parser.add_argument('--args-file', default='configs/paper_final_full.args',
                        help='训练/评测时使用的项目参数文件')
    parser.add_argument('--checkpoint', required=True,
                        help='训练好的策略 checkpoint 路径 (.pth)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='PyTorch 推理设备')

    parser.add_argument('--px4-connection', default='udp:127.0.0.1:14540',
                        help='pymavlink 连接串，例如 udp:127.0.0.1:14540 或 serial:/dev/ttyUSB0:921600')
    parser.add_argument('--telemetry-timeout-s', type=float, default=1.0,
                        help='等待 PX4 遥测的超时时间')
    parser.add_argument('--control-rate-hz', type=float, default=None,
                        help='真实控制回路频率；默认继承项目 args 中的 base_control_freq')
    parser.add_argument('--px4-control-mode', default='vel_accel', choices=['accel', 'vel_accel'],
                        help='PX4 offboard 输出模式：纯加速度，或速度+加速度前馈')
    parser.add_argument('--offboard-warmup-s', type=float, default=1.5,
                        help='切到 OFFBOARD 之前的预热 setpoint 时长')
    parser.add_argument('--arm', **_bool_parser_flag(default=False),
                        help='是否在脚本中直接执行 ARM')
    parser.add_argument('--auto-takeoff', **_bool_parser_flag(default=False),
                        help='是否在进入策略控制前先自动起飞到指定高度')
    parser.add_argument('--takeoff-height-m', type=float, default=1.5,
                        help='相对于当前起始点的起飞高度')
    parser.add_argument('--takeoff-tolerance-m', type=float, default=0.12,
                        help='自动起飞认为到位的高度误差')
    parser.add_argument('--takeoff-timeout-s', type=float, default=20.0,
                        help='自动起飞超时')
    parser.add_argument('--finish-action', default='hold', choices=['hold', 'land', 'disarm'],
                        help='任务结束或异常时的收尾动作')

    parser.add_argument('--goal-forward-m', type=float, default=5.0,
                        help='任务目标在 mission frame 中相对起点的前向距离')
    parser.add_argument('--goal-left-m', type=float, default=0.0,
                        help='任务目标在 mission frame 中相对起点的左向距离')
    parser.add_argument('--goal-up-m', type=float, default=0.0,
                        help='任务目标在 mission frame 中相对起点的上向距离')
    parser.add_argument('--goal-tolerance-m', type=float, default=0.35,
                        help='认为到达目标的距离阈值')
    parser.add_argument('--mission-timeout-s', type=float, default=None,
                        help='任务最大时长；默认按 timesteps/base_control_freq 推导')

    parser.add_argument('--policy-max-speed-mps', type=float, default=None,
                        help='真实部署时给策略状态构造使用的 max_speed；默认按场景推断')
    parser.add_argument('--policy-margin-m', type=float, default=None,
                        help='真实部署时给策略状态构造使用的 margin；默认按场景推断')

    parser.add_argument('--d455-width', type=int, default=640)
    parser.add_argument('--d455-height', type=int, default=480)
    parser.add_argument('--d455-fps', type=int, default=30)
    parser.add_argument('--d455-serial', default=None,
                        help='可选：指定 D455 序列号')
    parser.add_argument('--d455-enable-emitter', **_bool_parser_flag(default=True),
                        help='是否保持 D455 emitter 使能')
    parser.add_argument('--d455-exposure-divisor-us', type=float, default=10000.0,
                        help='把项目曝光语义时间映射回 D455 微秒的除数；与 recommend_d455_semantics.py 保持一致')
    parser.add_argument('--d455-working-exposure-min-us', type=float, default=None,
                        help='可选：手动覆盖 D455 曝光下界')
    parser.add_argument('--d455-working-exposure-max-us', type=float, default=None,
                        help='可选：手动覆盖 D455 曝光上界')
    parser.add_argument('--d455-working-gain-min', type=float, default=None,
                        help='可选：手动覆盖 D455 gain 下界')
    parser.add_argument('--d455-working-gain-max', type=float, default=None,
                        help='可选：手动覆盖 D455 gain 上界')
    parser.add_argument('--d455-working-laser-min', type=float, default=None,
                        help='可选：手动覆盖 D455 laser_power 下界')
    parser.add_argument('--d455-working-laser-max', type=float, default=None,
                        help='可选：手动覆盖 D455 laser_power 上界')
    parser.add_argument('--camera-warmup-frames', type=int, default=20,
                        help='相机启动后丢弃的预热帧数')
    parser.add_argument('--camera-frame-timeout-ms', type=int, default=1000,
                        help='等待 D455 帧的超时时间')
    parser.add_argument('--resize-depth-to-policy', **_bool_parser_flag(default=True),
                        help='是否先把 D455 深度缩放到训练时的 depth_width/depth_height')

    parser.add_argument('--log-dir', default=os.path.join('artifacts', 'real_policy_runs'),
                        help='真实部署日志目录')

    cli_args, project_overrides = parser.parse_known_args()
    cli_args.project_overrides = project_overrides
    return cli_args


def _strip_comments_and_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        tokens.extend(shlex.split(line))
    return tokens


def load_project_args(args_file: str, checkpoint: str, override_tokens: list[str]):
    args_path = args_file
    if not os.path.isabs(args_path):
        args_path = os.path.join(REPO_ROOT, args_path)
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f'project args file not found: {args_path}')

    with open(args_path, 'r', encoding='utf-8') as f:
        cfg_tokens = _strip_comments_and_tokens(f.read())
    cfg_tokens.extend(['--resume', checkpoint])
    cfg_tokens.extend(list(override_tokens))

    parser = build_parser()
    args = parser.parse_args(cfg_tokens)
    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.scenarios = parse_scenarios(args.scenarios)
    set_global_seed(args.seed, args.deterministic)
    validate_args(args)
    return args, args_path, cfg_tokens


def infer_policy_speed_and_margin(project_args):
    max_speed = 1.0
    margin = 0.05
    return max_speed, margin


def infer_mission_timeout(project_args) -> float:
    base_hz = max(float(getattr(project_args, 'base_control_freq', 15.0)), 1e-3)
    steps = max(int(getattr(project_args, 'timesteps', 120)), 1)
    return max(30.0, float(steps) / base_hz + 10.0)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('requested --device cuda but CUDA is not available')
        return torch.device('cuda')
    if device_arg == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model_from_args(project_args, checkpoint: str, device: torch.device):
    obs_dim = 7 if project_args.no_odom else 10
    model = Model(
        obs_dim,
        6,
        include_camera_state_in_obs=project_args.include_camera_state_in_obs,
        use_policy_intent=project_args.policy_output_intent,
        intent_dim=9,
        depth_nn_width=project_args.depth_nn_width,
        depth_nn_height=project_args.depth_nn_height,
        depth_use_pipeline=project_args.depth_use_pipeline,
        depth_min_valid=project_args.depth_min_valid,
        depth_max_range=project_args.depth_max_range,
    ).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print('[real][warn] missing checkpoint keys:', missing)
    if unexpected:
        print('[real][warn] unexpected checkpoint keys:', unexpected)
    model.eval()
    return model


def quat_or_rpy_rotation_frd_to_zup(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return body FLU -> world Z-up rotation used by the policy.

    Input roll/pitch/yaw are MAVLink/PX4 ATTITUDE angles in the common FRD/NED convention.
    The returned matrix has columns [forward, left, up] in world Z-up coordinates.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # Body FRD -> world NED.
    r_b2n = np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp,     sr * cp,                cr * cp],
    ], dtype=np.float32)

    t_world = np.diag([1.0, 1.0, -1.0]).astype(np.float32)  # NED -> Z-up
    t_body = np.diag([1.0, -1.0, -1.0]).astype(np.float32)  # FRD -> FLU
    return t_world @ r_b2n @ t_body


def rotate_mission_delta_to_world(delta_forward_left_up: np.ndarray, yaw0_rad: float) -> np.ndarray:
    cy = math.cos(yaw0_rad)
    sy = math.sin(yaw0_rad)
    rot = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return rot @ delta_forward_left_up.astype(np.float32)


def choose_yaw_cmd(yaw_current: float, v_pred_world: np.ndarray, target_vec_world: np.ndarray) -> float:
    if np.linalg.norm(v_pred_world[:2]) > 0.10:
        return float(math.atan2(v_pred_world[1], v_pred_world[0]))
    if np.linalg.norm(target_vec_world[:2]) > 0.10:
        return float(math.atan2(target_vec_world[1], target_vec_world[0]))
    return float(yaw_current)


def resize_depth_for_policy(depth_m: np.ndarray, width: int, height: int) -> np.ndarray:
    if depth_m.shape[1] == width and depth_m.shape[0] == height:
        return depth_m.astype(np.float32, copy=False)
    return cv2.resize(depth_m.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)


def normalized_iso_gain_curve(cam_sem, gain01: float) -> float:
    gain01 = float(min(max(gain01, 0.0), 1.0))
    gain_lo = float(cam_sem.iso_to_gain(0.0))
    gain_hi = float(cam_sem.iso_to_gain(1.0))
    denom = gain_hi - gain_lo
    if abs(denom) <= 1e-12:
        return gain01
    gain_sem = float(cam_sem.iso_to_gain(gain01))
    return float(min(max((gain_sem - gain_lo) / denom, 0.0), 1.0))


@dataclass
class D455SettingInfo:
    power01: float
    exposure01: float
    gain01: float
    laser_power: float
    exposure_us: float
    gain_value: float


class D455Runtime:
    def __init__(self, project_args, cli_args):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                '缺少 pyrealsense2。请在 Orin/Jetson 上先安装 librealsense Python 绑定。'
            ) from exc

        self.rs = rs
        self.project_args = project_args
        self.cli_args = cli_args
        self.cam_sem = camera_semantics_from_args(project_args)
        self.pipeline = None
        self.profile = None
        self.depth_sensor = None
        self.mode = None
        self.last_settings: D455SettingInfo | None = None
        self.option_ranges: dict[str, Any] = {}

    def start(self):
        rs = self.rs
        pipeline = rs.pipeline()
        last_error = None
        modes = [(self.cli_args.d455_width, self.cli_args.d455_height, self.cli_args.d455_fps)]
        for cand in DEFAULT_D455_MODES:
            if cand not in modes:
                modes.append(cand)

        profile = None
        for width, height, fps in modes:
            try:
                config = rs.config()
                if self.cli_args.d455_serial:
                    config.enable_device(self.cli_args.d455_serial)
                config.enable_stream(rs.stream.depth, int(width), int(height), rs.format.z16, int(fps))
                profile = pipeline.start(config)
                self.mode = (int(width), int(height), int(fps))
                break
            except RuntimeError as exc:
                last_error = exc
        if profile is None:
            raise RuntimeError(f'无法启动 D455 深度流，最后错误: {last_error}')

        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.enable_auto_exposure):
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            time.sleep(0.05)
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(
                rs.option.emitter_enabled,
                1.0 if self.cli_args.d455_enable_emitter else 0.0,
            )

        self.pipeline = pipeline
        self.profile = profile
        self.depth_sensor = depth_sensor
        self.option_ranges = {
            'exposure': depth_sensor.get_option_range(rs.option.exposure),
            'gain': depth_sensor.get_option_range(rs.option.gain),
            'laser_power': depth_sensor.get_option_range(rs.option.laser_power),
        }

        init_power01 = float(getattr(self.project_args, 'cam_power_baseline', 0.55))
        init_exposure01 = 0.5
        init_gain01 = 0.5
        self.apply_normalized(init_power01, init_exposure01, init_gain01, force=True)

        for _ in range(max(int(self.cli_args.camera_warmup_frames), 0)):
            self.read_depth()

    def stop(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None

    def _quantize(self, raw: float, key: str) -> float:
        rng = self.option_ranges[key]
        lo = float(rng.min)
        hi = float(rng.max)
        step = float(rng.step) if float(rng.step) > 0.0 else 1.0
        value = min(max(float(raw), lo), hi)
        q = round((value - lo) / step) * step + lo
        return float(min(max(q, lo), hi))

    def _working_range(self, key: str) -> tuple[float, float]:
        rng = self.option_ranges[key]
        lo = float(rng.min)
        hi = float(rng.max)
        if key == 'exposure':
            auto_lo = float(diff_depth_exposure_to_time(torch.tensor(0.0), camera_semantics=self.cam_sem)) * self.cli_args.d455_exposure_divisor_us
            auto_hi = float(diff_depth_exposure_to_time(torch.tensor(1.0), camera_semantics=self.cam_sem)) * self.cli_args.d455_exposure_divisor_us
            lo = max(lo, float(self.cli_args.d455_working_exposure_min_us) if self.cli_args.d455_working_exposure_min_us is not None else auto_lo)
            hi = min(hi, float(self.cli_args.d455_working_exposure_max_us) if self.cli_args.d455_working_exposure_max_us is not None else auto_hi)
        elif key == 'gain':
            if self.cli_args.d455_working_gain_min is not None:
                lo = max(lo, float(self.cli_args.d455_working_gain_min))
            if self.cli_args.d455_working_gain_max is not None:
                hi = min(hi, float(self.cli_args.d455_working_gain_max))
        elif key == 'laser_power':
            if self.cli_args.d455_working_laser_min is not None:
                lo = max(lo, float(self.cli_args.d455_working_laser_min))
            if self.cli_args.d455_working_laser_max is not None:
                hi = min(hi, float(self.cli_args.d455_working_laser_max))
        if hi <= lo:
            hi = lo
        return float(lo), float(hi)

    def apply_normalized(self, power01: float, exposure01: float, gain01: float, force: bool = False) -> D455SettingInfo:
        rs = self.rs

        power01 = float(min(max(power01, 0.0), 1.0))
        exposure01 = float(min(max(exposure01, 0.0), 1.0))
        gain01 = float(min(max(gain01, 0.0), 1.0))

        laser_lo, laser_hi = self._working_range('laser_power')
        exp_lo, exp_hi = self._working_range('exposure')
        gain_lo, gain_hi = self._working_range('gain')

        laser_power = laser_lo + (laser_hi - laser_lo) * power01
        exposure_us = float(diff_depth_exposure_to_time(
            torch.tensor(exposure01),
            camera_semantics=self.cam_sem,
        )) * float(self.cli_args.d455_exposure_divisor_us)
        exposure_us = min(max(exposure_us, exp_lo), exp_hi)

        gain_curve = normalized_iso_gain_curve(self.cam_sem, gain01)
        gain_value = gain_lo + (gain_hi - gain_lo) * gain_curve

        laser_power = self._quantize(laser_power, 'laser_power')
        exposure_us = self._quantize(exposure_us, 'exposure')
        gain_value = self._quantize(gain_value, 'gain')

        if force or self.last_settings is None or abs(laser_power - self.last_settings.laser_power) > 1e-6:
            self.depth_sensor.set_option(rs.option.laser_power, laser_power)
        if force or self.last_settings is None or abs(exposure_us - self.last_settings.exposure_us) > 1e-6:
            self.depth_sensor.set_option(rs.option.exposure, exposure_us)
        if force or self.last_settings is None or abs(gain_value - self.last_settings.gain_value) > 1e-6:
            self.depth_sensor.set_option(rs.option.gain, gain_value)

        info = D455SettingInfo(
            power01=power01,
            exposure01=exposure01,
            gain01=gain01,
            laser_power=float(laser_power),
            exposure_us=float(exposure_us),
            gain_value=float(gain_value),
        )
        self.last_settings = info
        return info

    def read_depth(self) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError('D455 pipeline is not started')
        frames = self.pipeline.wait_for_frames(timeout_ms=int(self.cli_args.camera_frame_timeout_ms))
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            raise RuntimeError('D455 depth_frame is empty')
        depth_scale = float(depth_frame.get_units())
        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
        return depth_m

    def runtime_meta(self) -> dict[str, Any]:
        return {
            'mode': self.mode,
            'ranges': {
                k: {
                    'min': float(v.min),
                    'max': float(v.max),
                    'step': float(v.step),
                    'default': float(v.default),
                }
                for k, v in self.option_ranges.items()
            },
            'last_settings': asdict(self.last_settings) if self.last_settings is not None else None,
        }


@dataclass
class Px4TelemetryState:
    pos_ned: np.ndarray
    vel_ned: np.ndarray
    roll: float
    pitch: float
    yaw_ned: float
    t_pos: float
    t_att: float

    @property
    def pos_zup(self) -> np.ndarray:
        return np.array([self.pos_ned[0], self.pos_ned[1], -self.pos_ned[2]], dtype=np.float32)

    @property
    def vel_zup(self) -> np.ndarray:
        return np.array([self.vel_ned[0], self.vel_ned[1], -self.vel_ned[2]], dtype=np.float32)

    @property
    def R_policy(self) -> np.ndarray:
        return quat_or_rpy_rotation_frd_to_zup(self.roll, self.pitch, self.yaw_ned)

    @property
    def yaw_policy(self) -> float:
        fwd = self.R_policy[:, 0]
        return float(math.atan2(float(fwd[1]), float(fwd[0])))


class Px4Bridge:
    def __init__(self, connection: str):
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise RuntimeError(
                '缺少 pymavlink。请先在 Orin/Jetson 上安装 `pip install pymavlink`。'
            ) from exc

        self.mavutil = mavutil
        self.connection_str = connection
        self.master = None
        self._last_pos = None
        self._last_vel = None
        self._last_att = None
        self._t_pos = 0.0
        self._t_att = 0.0

    def connect(self, timeout_s: float = 10.0):
        self.master = self.mavutil.mavlink_connection(
            self.connection_str,
            autoreconnect=True,
            source_system=255,
            source_component=self.mavutil.mavlink.MAV_COMP_ID_ONBOARD_COMPUTER,
        )
        hb = self.master.wait_heartbeat(timeout=timeout_s)
        if hb is None:
            raise RuntimeError('等待 PX4 heartbeat 超时')
        print(f'[real] PX4 heartbeat: sys={self.master.target_system} comp={self.master.target_component}')
        self.request_default_streams()

    def request_default_streams(self):
        if self.master is None:
            return
        mv = self.mavutil.mavlink
        try:
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mv.MAV_DATA_STREAM_POSITION,
                30,
                1,
            )
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mv.MAV_DATA_STREAM_EXTRA1,
                30,
                1,
            )
        except Exception:
            pass
        for msg_id, hz in (
            (mv.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 30.0),
            (mv.MAVLINK_MSG_ID_ATTITUDE, 50.0),
        ):
            try:
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mv.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,
                    float(msg_id),
                    float(1e6 / hz),
                    0,
                    0,
                    0,
                    0,
                    0,
                )
            except Exception:
                pass

    def poll(self, blocking: bool = False, timeout: float = 0.0):
        if self.master is None:
            raise RuntimeError('PX4 connection is not established')
        msg = self.master.recv_match(blocking=blocking, timeout=timeout)
        while msg is not None:
            msg_type = msg.get_type()
            now = time.time()
            if msg_type == 'LOCAL_POSITION_NED':
                self._last_pos = np.array([float(msg.x), float(msg.y), float(msg.z)], dtype=np.float32)
                self._last_vel = np.array([float(msg.vx), float(msg.vy), float(msg.vz)], dtype=np.float32)
                self._t_pos = now
            elif msg_type == 'ATTITUDE':
                self._last_att = (float(msg.roll), float(msg.pitch), float(msg.yaw))
                self._t_att = now
            msg = self.master.recv_match(blocking=False)

    def wait_for_state(self, timeout_s: float) -> Px4TelemetryState:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self.poll(blocking=True, timeout=min(0.2, max(0.0, deadline - time.time())))
            try:
                return self.current_state(max_age_s=timeout_s)
            except RuntimeError:
                pass
        raise RuntimeError('等待 PX4 LOCAL_POSITION_NED / ATTITUDE 超时')

    def current_state(self, max_age_s: float = 0.5) -> Px4TelemetryState:
        self.poll(blocking=False, timeout=0.0)
        now = time.time()
        if self._last_pos is None or self._last_vel is None or self._last_att is None:
            raise RuntimeError('PX4 状态尚未准备好')
        if (now - self._t_pos) > max_age_s:
            raise RuntimeError('PX4 LOCAL_POSITION_NED 已过期')
        if (now - self._t_att) > max_age_s:
            raise RuntimeError('PX4 ATTITUDE 已过期')
        roll, pitch, yaw = self._last_att
        return Px4TelemetryState(
            pos_ned=self._last_pos.copy(),
            vel_ned=self._last_vel.copy(),
            roll=roll,
            pitch=pitch,
            yaw_ned=yaw,
            t_pos=self._t_pos,
            t_att=self._t_att,
        )

    def _send_set_mode_main(self, main_mode: int):
        mv = self.mavutil.mavlink
        try:
            if hasattr(self.master, 'set_mode_px4'):
                mode_name = {
                    PX4_CUSTOM_MAIN_MODE_POSCTL: 'POSCTL',
                    PX4_CUSTOM_MAIN_MODE_OFFBOARD: 'OFFBOARD',
                }.get(main_mode)
                if mode_name is not None:
                    self.master.set_mode_px4(mode_name)
                    return
        except Exception:
            pass

        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mv.MAV_CMD_DO_SET_MODE,
            0,
            float(mv.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
            float(main_mode),
            0,
            0,
            0,
            0,
            0,
        )

    def set_offboard_mode(self):
        self._send_set_mode_main(PX4_CUSTOM_MAIN_MODE_OFFBOARD)

    def arm(self):
        mv = self.mavutil.mavlink
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mv.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def disarm(self):
        mv = self.mavutil.mavlink
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mv.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0.0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def command_land(self):
        mv = self.mavutil.mavlink
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mv.MAV_CMD_NAV_LAND,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

    def send_position_yaw_ned(self, pos_ned: np.ndarray, yaw_ned: float):
        mv = self.mavutil.mavlink
        mask = (
            mv.POSITION_TARGET_TYPEMASK_VX_IGNORE |
            mv.POSITION_TARGET_TYPEMASK_VY_IGNORE |
            mv.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
            mv.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mv.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mv.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mv.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mv.MAV_FRAME_LOCAL_NED,
            mask,
            float(pos_ned[0]),
            float(pos_ned[1]),
            float(pos_ned[2]),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            float(yaw_ned),
            0.0,
        )

    def send_accel_yaw_ned(self, accel_ned: np.ndarray, yaw_ned: float,
                           vel_ned: np.ndarray | None = None,
                           use_velocity_feedforward: bool = False):
        mv = self.mavutil.mavlink
        mask = (
            mv.POSITION_TARGET_TYPEMASK_X_IGNORE |
            mv.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            mv.POSITION_TARGET_TYPEMASK_Z_IGNORE |
            mv.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
        )
        if not use_velocity_feedforward:
            mask |= (
                mv.POSITION_TARGET_TYPEMASK_VX_IGNORE |
                mv.POSITION_TARGET_TYPEMASK_VY_IGNORE |
                mv.POSITION_TARGET_TYPEMASK_VZ_IGNORE
            )
            vx = vy = vz = 0.0
        else:
            if vel_ned is None:
                raise ValueError('vel_ned is required when use_velocity_feedforward=True')
            vx, vy, vz = float(vel_ned[0]), float(vel_ned[1]), float(vel_ned[2])
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mv.MAV_FRAME_LOCAL_NED,
            mask,
            0.0,
            0.0,
            0.0,
            vx,
            vy,
            vz,
            float(accel_ned[0]),
            float(accel_ned[1]),
            float(accel_ned[2]),
            float(yaw_ned),
            0.0,
        )


class CsvLogger:
    def __init__(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, 'trace.csv')
        self._fh = open(self.path, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._fh, fieldnames=[
            't_wall',
            'phase',
            'step',
            'pos_x',
            'pos_y',
            'pos_z',
            'vel_x',
            'vel_y',
            'vel_z',
            'goal_dist',
            'yaw_policy',
            'yaw_cmd',
            'power01',
            'exposure01',
            'gain01',
            'laser_power',
            'exposure_us',
            'gain_value',
            'fill_rate',
            'accel_cmd_x',
            'accel_cmd_y',
            'accel_cmd_z',
            'vel_cmd_x',
            'vel_cmd_y',
            'vel_cmd_z',
        ])
        self._writer.writeheader()

    def write(self, row: dict[str, Any]):
        self._writer.writerow(row)
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


def mission_hold(bridge: Px4Bridge, state: Px4TelemetryState, duration_s: float, rate_hz: float):
    deadline = time.time() + max(float(duration_s), 0.0)
    dt = 1.0 / max(float(rate_hz), 1e-3)
    pos_ned = state.pos_ned.copy()
    yaw_ned = state.yaw_ned
    while time.time() < deadline:
        bridge.send_position_yaw_ned(pos_ned, yaw_ned)
        time.sleep(dt)


def auto_takeoff(bridge: Px4Bridge, takeoff_height_m: float, tol_m: float, timeout_s: float, rate_hz: float) -> Px4TelemetryState:
    start = bridge.current_state(max_age_s=1.0)
    start_zup = float(start.pos_zup[2])
    target_zup = start_zup + float(takeoff_height_m)
    target_ned = start.pos_ned.copy()
    target_ned[2] = -target_zup
    yaw_ned = start.yaw_ned

    deadline = time.time() + max(float(timeout_s), 0.0)
    dt = 1.0 / max(float(rate_hz), 1e-3)
    while time.time() < deadline:
        state = bridge.current_state(max_age_s=1.0)
        bridge.send_position_yaw_ned(target_ned, yaw_ned)
        if abs(float(state.pos_zup[2]) - target_zup) <= float(tol_m):
            return state
        time.sleep(dt)
    raise RuntimeError('自动起飞超时')


def safe_finish(bridge: Px4Bridge, finish_action: str):
    try:
        state = bridge.current_state(max_age_s=2.0)
    except Exception:
        state = None

    if finish_action == 'land':
        try:
            bridge.command_land()
            print('[real] commanded PX4 LAND')
            return
        except Exception as exc:
            print(f'[real][warn] LAND command failed: {exc}')

    if finish_action == 'disarm':
        try:
            bridge.disarm()
            print('[real] sent DISARM command')
            return
        except Exception as exc:
            print(f'[real][warn] DISARM failed: {exc}')

    if state is not None:
        try:
            mission_hold(bridge, state, duration_s=1.5, rate_hz=10.0)
            print('[real] sent short HOLD setpoint sequence')
        except Exception as exc:
            print(f'[real][warn] HOLD sequence failed: {exc}')


def main():
    cli_args = parse_cli()

    checkpoint_path = cli_args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(REPO_ROOT, checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')

    project_args, args_path, project_tokens = load_project_args(
        cli_args.args_file,
        checkpoint_path,
        cli_args.project_overrides,
    )

    if bool(project_args.use_dmpc) or bool(project_args.policy_output_intent):
        raise RuntimeError(
            '当前真机脚本只支持 direct-action checkpoint；'
            '你当前的 args 打开了 use_dmpc / policy_output_intent。'
        )

    control_rate_hz = float(cli_args.control_rate_hz or project_args.base_control_freq)
    policy_max_speed_default, policy_margin_default = infer_policy_speed_and_margin(project_args)
    policy_max_speed_mps = float(cli_args.policy_max_speed_mps if cli_args.policy_max_speed_mps is not None else policy_max_speed_default)
    policy_margin_m = float(cli_args.policy_margin_m if cli_args.policy_margin_m is not None else policy_margin_default)
    mission_timeout_s = float(cli_args.mission_timeout_s if cli_args.mission_timeout_s is not None else infer_mission_timeout(project_args))

    device = choose_device(cli_args.device)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print('\n' + '=' * 30 + ' Real Deployment Configuration ' + '=' * 30)
    print(f'args_file                    : {args_path}')
    print(f'checkpoint                   : {checkpoint_path}')
    print(f'device                       : {device}')
    print(f'px4_connection               : {cli_args.px4_connection}')
    print(f'control_rate_hz              : {control_rate_hz}')
    print(f'policy_max_speed_mps         : {policy_max_speed_mps}')
    print(f'policy_margin_m              : {policy_margin_m}')
    print(f'mission_timeout_s            : {mission_timeout_s}')
    print_runtime_mode(project_args)
    print('=' * 90 + '\n')

    model = build_model_from_args(project_args, checkpoint_path, device)
    d455 = D455Runtime(project_args, cli_args)
    bridge = Px4Bridge(cli_args.px4_connection)

    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(REPO_ROOT, cli_args.log_dir, run_ts)
    os.makedirs(run_dir, exist_ok=True)
    logger = CsvLogger(run_dir)

    with open(os.path.join(run_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'run_ts': run_ts,
            'checkpoint': checkpoint_path,
            'args_file': args_path,
            'project_tokens': project_tokens,
            'cli_args': vars(cli_args),
            'policy_max_speed_mps': policy_max_speed_mps,
            'policy_margin_m': policy_margin_m,
        }, f, ensure_ascii=False, indent=2)

    cam_power_baseline = float(project_args.cam_power_baseline)
    fixed_camera_power = float(project_args.fixed_camera_power)
    if fixed_camera_power < 0.0:
        fixed_camera_power = cam_power_baseline
    cam_env = SimpleNamespace(
        cam_power_baseline=cam_power_baseline,
        camera_control_mode=str(project_args.camera_control_mode).lower(),
        fixed_camera_power=fixed_camera_power,
        fixed_camera_exposure=float(project_args.fixed_camera_exposure),
        fixed_camera_gain=float(project_args.fixed_camera_gain),
        fixed_random_power_range=(
            float(project_args.fixed_random_power_min),
            float(project_args.fixed_random_power_max),
        ),
        fixed_random_exposure_range=(
            float(project_args.fixed_random_exposure_min),
            float(project_args.fixed_random_exposure_max),
        ),
        fixed_random_gain_range=(
            float(project_args.fixed_random_gain_min),
            float(project_args.fixed_random_gain_max),
        ),
    )

    try:
        d455.start()
        bridge.connect(timeout_s=max(float(cli_args.telemetry_timeout_s), 3.0))
        state0 = bridge.wait_for_state(timeout_s=max(float(cli_args.telemetry_timeout_s), 3.0))

        with open(os.path.join(run_dir, 'meta.json'), 'r', encoding='utf-8') as f:
            meta = json.load(f)
        meta['d455'] = d455.runtime_meta()
        with open(os.path.join(run_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f'[real] initial state zup={state0.pos_zup} yaw={state0.yaw_policy:.3f}')

        if cli_args.arm:
            print(f'[real] offboard warmup for {cli_args.offboard_warmup_s:.2f}s')
            mission_hold(bridge, state0, duration_s=cli_args.offboard_warmup_s, rate_hz=max(control_rate_hz, 10.0))
            bridge.set_offboard_mode()
            time.sleep(0.2)
            bridge.arm()
            print('[real] arm command sent')
            time.sleep(0.8)
        else:
            print('[real] running without ARM; commands will still be computed and streamed, but the vehicle should remain disarmed')

        state_start = state0
        if cli_args.arm and cli_args.auto_takeoff:
            print(f'[real] auto takeoff to +{cli_args.takeoff_height_m:.2f}m')
            state_start = auto_takeoff(
                bridge,
                takeoff_height_m=cli_args.takeoff_height_m,
                tol_m=cli_args.takeoff_tolerance_m,
                timeout_s=cli_args.takeoff_timeout_s,
                rate_hz=max(control_rate_hz, 10.0),
            )
            print(f'[real] takeoff reached z={state_start.pos_zup[2]:.3f}m')

        # Mission frame is anchored after optional takeoff.
        mission_origin_pos = state_start.pos_zup.copy()
        mission_origin_yaw = state_start.yaw_policy
        goal_local = np.array([
            float(cli_args.goal_forward_m),
            float(cli_args.goal_left_m),
            float(cli_args.goal_up_m),
        ], dtype=np.float32)
        goal_world = mission_origin_pos + rotate_mission_delta_to_world(goal_local, mission_origin_yaw)
        print(f'[real] mission origin={mission_origin_pos} yaw0={mission_origin_yaw:.3f}')
        print(f'[real] goal_world={goal_world}')

        h = None
        power, exposure, gain = init_camera_params(cam_env, 1, device)
        d455.apply_normalized(float(power.item()), float(exposure.item()), float(gain.item()), force=True)

        g_std = torch.tensor([0.0, 0.0, -9.80665], device=device)
        thr_est_error = torch.ones((1,), device=device)
        dt = 1.0 / max(control_rate_hz, 1e-3)

        step = 0
        mission_begin = time.time()
        last_state = state_start

        while True:
            loop_t0 = time.time()
            if (loop_t0 - mission_begin) > mission_timeout_s:
                print('[real][warn] mission timeout reached')
                break

            depth_m = d455.read_depth()
            if cli_args.resize_depth_to_policy:
                depth_policy_np = resize_depth_for_policy(depth_m, int(project_args.depth_width), int(project_args.depth_height))
            else:
                depth_policy_np = depth_m.astype(np.float32, copy=False)
            fill_rate = float(compute_depth_fill_rate(
                torch.from_numpy(depth_policy_np[None]),
                min_valid_depth=project_args.depth_min_valid,
            ).item())

            last_state = bridge.current_state(max_age_s=max(0.5, cli_args.telemetry_timeout_s))
            pos_world = last_state.pos_zup
            vel_world = last_state.vel_zup
            R_world = last_state.R_policy
            yaw_policy = last_state.yaw_policy

            goal_vec_world = goal_world - pos_world
            goal_dist = float(np.linalg.norm(goal_vec_world))
            if goal_dist <= float(cli_args.goal_tolerance_m):
                print(f'[real] goal reached at step={step}, dist={goal_dist:.3f}m')
                break

            ctrl_env = SimpleNamespace(
                v=torch.from_numpy(vel_world[None]).to(device=device, dtype=torch.float32),
                R=torch.from_numpy(R_world[None]).to(device=device, dtype=torch.float32),
                margin=torch.full((1,), policy_margin_m, device=device),
                max_speed=torch.full((1, 1), policy_max_speed_mps, device=device),
                g_std=g_std,
                thr_est_error=thr_est_error,
            )

            target_v_raw = torch.from_numpy(goal_vec_world[None]).to(device=device, dtype=torch.float32)
            R_local = build_local_frame(ctrl_env)
            target_v = compute_target_velocity(target_v_raw, ctrl_env)
            state_vec, _ = build_state_vector(
                ctrl_env,
                target_v,
                R_local,
                power,
                exposure,
                gain,
                project_args.no_odom,
                project_args.include_camera_state_in_obs,
            )

            depth_tensor = torch.from_numpy(depth_policy_np[None]).to(device=device, dtype=torch.float32)
            policy_depth_obs = select_policy_depth_obs(depth_tensor, str(project_args.policy_depth_mode))

            with torch.no_grad():
                act_raw, cam_params, h = model(
                    state_vec,
                    h,
                    depth_obs=policy_depth_obs,
                    add_noise=False,
                )
                act_raw = act_raw.float()
                cam_params = cam_params.float()
                power, exposure, gain, _ = update_camera_params(cam_params, power, exposure, gain, cam_env)
                accel_cmd_t, v_pred_t = decode_action_direct(act_raw, R_local, ctrl_env, 1, project_args.max_acc_cmd)

            camera_info = d455.apply_normalized(
                float(power.item()),
                float(exposure.item()),
                float(gain.item()),
            )

            accel_cmd_world = accel_cmd_t[0].detach().cpu().numpy().astype(np.float32)
            v_pred_world = v_pred_t[0].detach().cpu().numpy().astype(np.float32)
            yaw_cmd_policy = choose_yaw_cmd(yaw_policy, v_pred_world, goal_vec_world)
            yaw_cmd_ned = -yaw_cmd_policy

            accel_cmd_ned = np.array([
                accel_cmd_world[0],
                accel_cmd_world[1],
                -accel_cmd_world[2],
            ], dtype=np.float32)

            vel_cmd_world = np.clip(
                vel_world + accel_cmd_world * dt,
                -policy_max_speed_mps,
                policy_max_speed_mps,
            )
            speed_norm = max(float(np.linalg.norm(vel_cmd_world)), 1e-6)
            if speed_norm > policy_max_speed_mps:
                vel_cmd_world = vel_cmd_world / speed_norm * policy_max_speed_mps
            vel_cmd_ned = np.array([
                vel_cmd_world[0],
                vel_cmd_world[1],
                -vel_cmd_world[2],
            ], dtype=np.float32)

            bridge.send_accel_yaw_ned(
                accel_cmd_ned,
                yaw_cmd_ned,
                vel_ned=vel_cmd_ned,
                use_velocity_feedforward=(cli_args.px4_control_mode == 'vel_accel'),
            )

            logger.write({
                't_wall': loop_t0,
                'phase': 'mission',
                'step': step,
                'pos_x': float(pos_world[0]),
                'pos_y': float(pos_world[1]),
                'pos_z': float(pos_world[2]),
                'vel_x': float(vel_world[0]),
                'vel_y': float(vel_world[1]),
                'vel_z': float(vel_world[2]),
                'goal_dist': goal_dist,
                'yaw_policy': float(yaw_policy),
                'yaw_cmd': float(yaw_cmd_policy),
                'power01': float(camera_info.power01),
                'exposure01': float(camera_info.exposure01),
                'gain01': float(camera_info.gain01),
                'laser_power': float(camera_info.laser_power),
                'exposure_us': float(camera_info.exposure_us),
                'gain_value': float(camera_info.gain_value),
                'fill_rate': fill_rate,
                'accel_cmd_x': float(accel_cmd_world[0]),
                'accel_cmd_y': float(accel_cmd_world[1]),
                'accel_cmd_z': float(accel_cmd_world[2]),
                'vel_cmd_x': float(vel_cmd_world[0]),
                'vel_cmd_y': float(vel_cmd_world[1]),
                'vel_cmd_z': float(vel_cmd_world[2]),
            })

            if step % 10 == 0:
                print(
                    f"[real] step={step:04d} goal_dist={goal_dist:.3f} "
                    f"fill={fill_rate:.3f} power={camera_info.power01:.3f} "
                    f"exp={camera_info.exposure01:.3f} gain={camera_info.gain01:.3f}"
                )

            step += 1
            elapsed = time.time() - loop_t0
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print('\n[real] interrupted by user')
    except Exception as exc:
        print(f'[real][error] {exc}')
        raise
    finally:
        try:
            safe_finish(bridge, cli_args.finish_action)
        except Exception as exc:
            print(f'[real][warn] safe_finish failed: {exc}')
        try:
            d455.stop()
        except Exception:
            pass
        logger.close()
        print(f'[real] logs saved to {run_dir}')


if __name__ == '__main__':
    main()

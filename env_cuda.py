import math
import json
import os
import random
from collections import OrderedDict
import torch
import torch.nn.functional as F
import quadsim_cuda

from utils import g_decay
from autograd_ops import (
    run,
    diff_depth,
)
from camera_semantics import CameraSemantics


class Env:
    """
    diff_depth-only 的 Sun Glare 固定小地图环境。

    地图语义如下：
    - 地图中心: (0, 0, 0)
    - 地图范围: x/y 均为 [-5, 5]
    - 起点: (-2.8, start_y, 1.5)
    - 终点: (3.0, 0, 1.5)
    - 几何主骨架是 `_build_sun_glare_voxel_layout()` 的 occluder/gate/gap

    当前版本的设计目标是：
    - 公开场景只有 `glare` / `specular` / `dark`；
    - 三个场景共享同一张门洞地图，只改变开口附近的局部传感器退化模式；
    - 避免多套几何地图混在一起干扰主动感知调参验证。
    """
    @staticmethod
    def _ordered_range(lo, hi):
        lo_f = float(lo)
        hi_f = float(hi)
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f
        return lo_f, hi_f

    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 eval_mode=False,
                 cam_angle=10, ellipsoid_a=0.0, ellipsoid_c=0.0,
                 camera_preset='high',
                 cam_enable_specular=True,
                 cam_enable_motion_blur=True,
                 cam_noise_scale=1.0,
                 cam_blur_scale=1.0,
                 cam_fog_scale=1.0,
                 cam_lighting_scale=1.0,
                 cam_model_randomize=True,
                 cam_model_randomize_scale=0.08,
                 cam_power_baseline=0.55,
                 camera_control_mode='learned',
                 sensor_grad_mode='full',
                 fixed_camera_power=-1.0,
                 fixed_camera_exposure=0.5,
                 fixed_camera_gain=0.5,
                 fixed_random_power_min=0.55,
                 fixed_random_power_max=0.90,
                 fixed_random_exposure_min=0.16,
                 fixed_random_exposure_max=0.60,
                 fixed_random_gain_min=0.02,
                 fixed_random_gain_max=0.42,
                 cam_exposure_t_min=0.25,
                 cam_exposure_t_span=2.75,
                 cam_exposure_eff_min=0.15,
                 cam_exposure_eff_max=4.0,
                 cam_iso_gain_base=1.0,
                 cam_iso_gain_scale=10.0,
                 cam_iso_gain_gamma=1.2,
                 cam_shot_noise_base=0.03,
                 depth_min_valid=0.3,
                 depth_max_range=6.0,
                 scenarios=None,
                 sun_glare_levels=None,
                 sun_glare_eval_level=None,
                 sun_glare_eval_slot=None,
                 sun_glare_randomize=False,
                 sun_glare_ambient_min=0.06,
                 sun_glare_ambient_max=0.28,
                 sun_glare_dir_min=0.25,
                 sun_glare_dir_max=1.05,
                 sun_glare_airlight_min=0.06,
                 sun_glare_airlight_max=0.34,
                 sun_glare_fog_beta_min=0.006,
                 sun_glare_fog_beta_max=0.045,
                 sun_glare_mat_obstacle_min=0.42,
                 sun_glare_mat_obstacle_max=0.82,
                 sun_glare_mat_spec_min=0.02,
                 sun_glare_mat_spec_max=0.16,
                 sun_glare_sun_sigma_u_min=0.18,
                 sun_glare_sun_sigma_u_max=0.34,
                 sun_glare_sun_sigma_v_min=0.16,
                 sun_glare_sun_sigma_v_max=0.30,
                 sun_glare_sun_y_jitter=0.18,
                 sun_glare_sun_z_jitter=0.12,
                 sun_glare_occluder_x_jitter=0.10,
                 sun_glare_occluder_half_y_min=0.36,
                 sun_glare_occluder_half_y_max=0.50,
                 sun_glare_divider_x_jitter=0.08,
                 sun_glare_gate_x_jitter=0.08,
                 sun_glare_gap_half_w_min=0.16,
                 sun_glare_gap_half_w_max=0.24,
                 sun_glare_start_y_jitter=0.18,
                 scene_fit_profiles_path=None,
                 diff_sensor_impl=None) -> None:
        self.device = device
        self.batch_size = batch_size
        self.width = max(int(width), 1)
        self.height = max(int(height), 1)
        self.grad_decay = grad_decay
        self.depth_min_valid = max(float(depth_min_valid), 1e-3)
        self.depth_max_range = max(float(depth_max_range), self.depth_min_valid + 1e-3)
        self.fov_x_half_tan = float(fov_x_half_tan)
        self.cam_angle = float(cam_angle)
        self.eval_mode = bool(eval_mode)

        # 椭球体碰撞模型参数
        self.ellipsoid_a = ellipsoid_a
        self.ellipsoid_c = ellipsoid_c
        self.use_ellipsoid = ellipsoid_a > 0 and ellipsoid_c > 0

        self.g_std = torch.tensor([0.0, 0.0, -9.80665], device=device)
        self.v_wind_w = torch.tensor([1.0, 1.0, 0.2], device=device)
        self.sub_div = torch.linspace(0, 1. / 15, 10, device=device).reshape(-1, 1, 1)

        self.flow = torch.empty((batch_size, 0, self.height, self.width), device=device)

        # 固定地图定义
        self.map_half_extent_x = 5.0
        self.map_half_extent_y = 5.0
        self.scene_min = torch.tensor([-5.0, -5.0, 0.0], device=device)
        self.scene_max = torch.tensor([5.0, 5.0, 3.0], device=device)
        self.start_position = torch.tensor([-5.0, 0.0, 1.5], device=device)
        self.goal_position = torch.tensor([5.0, 0.0, 1.5], device=device)
        self.fixed_max_speed = 1
        self.fixed_drone_radius = 0.12
        self.fixed_margin = 0.05
        self.fixed_pitch_ctl_delay = 12.0
        self.fixed_yaw_ctl_delay = 6.0
        self.fixed_drag_linear = 0.35
        self.fixed_wind_scale = 0.03
        self.supported_scenarios = (
            'glare',
            'specular',
            'dark',
        )
        self.scene_name_to_id = {name: idx for idx, name in enumerate(self.supported_scenarios)}
        self.scenarios = self._normalize_scenarios(scenarios)
        self.current_scene_name = self.scenarios[0]
        self.current_scene_id = self.scene_name_to_id[self.current_scene_name]
        self.sun_glare_supported_levels = ('l0', 'l1', 'l2', 'l3')
        self.sun_glare_levels = self._normalize_sun_glare_levels(sun_glare_levels)
        self.sun_glare_eval_level = self._canonical_sun_glare_level(sun_glare_eval_level)
        self.sun_glare_eval_slot = self._canonical_sun_glare_slot(sun_glare_eval_slot)
        self.current_scene_variant = None
        self.current_scene_tag = self.current_scene_name
        self.current_sun_glare_level = None
        self.current_scene_effects = {}
        self.last_diff_depth_debug = None
        self.last_diff_depth_train_aux = None
        self.scene_fit_profiles_path = None
        self.scene_sensor_profile_overrides = {}
        self.scene_effect_overrides = {}

        _impl = {'diff_depth': 'python'}
        if diff_sensor_impl is not None:
            _impl.update({str(k): str(v).lower() for k, v in dict(diff_sensor_impl).items()})
        self.diff_sensor_impl = _impl

        self.camera_preset = str(camera_preset).lower()
        self.cam_enable_specular = bool(cam_enable_specular)
        self.cam_enable_motion_blur = bool(cam_enable_motion_blur)
        self.cam_noise_scale = float(cam_noise_scale)
        self.cam_blur_scale = float(cam_blur_scale)
        self.cam_fog_scale = float(cam_fog_scale)
        self.cam_lighting_scale = float(cam_lighting_scale)
        self.cam_model_randomize = bool(cam_model_randomize)
        self.cam_model_randomize_scale = float(cam_model_randomize_scale)
        self.cam_power_baseline = float(cam_power_baseline)
        self.camera_control_mode = str(camera_control_mode).lower()
        self.sensor_grad_mode = str(sensor_grad_mode).lower()
        self.fixed_camera_power = float(self.cam_power_baseline if float(fixed_camera_power) < 0.0 else fixed_camera_power)
        self.fixed_camera_exposure = float(fixed_camera_exposure)
        self.fixed_camera_gain = float(fixed_camera_gain)
        self.fixed_random_power_range = self._ordered_range(fixed_random_power_min, fixed_random_power_max)
        self.fixed_random_exposure_range = self._ordered_range(fixed_random_exposure_min, fixed_random_exposure_max)
        self.fixed_random_gain_range = self._ordered_range(fixed_random_gain_min, fixed_random_gain_max)
        self.sun_glare_randomize = bool(sun_glare_randomize)
        self.sun_glare_ambient_range = self._ordered_range(sun_glare_ambient_min, sun_glare_ambient_max)
        self.sun_glare_dir_range = self._ordered_range(sun_glare_dir_min, sun_glare_dir_max)
        self.sun_glare_airlight_range = self._ordered_range(sun_glare_airlight_min, sun_glare_airlight_max)
        self.sun_glare_fog_beta_range = self._ordered_range(sun_glare_fog_beta_min, sun_glare_fog_beta_max)
        self.sun_glare_mat_obstacle_range = self._ordered_range(sun_glare_mat_obstacle_min, sun_glare_mat_obstacle_max)
        self.sun_glare_mat_spec_range = self._ordered_range(sun_glare_mat_spec_min, sun_glare_mat_spec_max)
        self.sun_glare_sun_sigma_u_range = self._ordered_range(sun_glare_sun_sigma_u_min, sun_glare_sun_sigma_u_max)
        self.sun_glare_sun_sigma_v_range = self._ordered_range(sun_glare_sun_sigma_v_min, sun_glare_sun_sigma_v_max)
        self.sun_glare_sun_y_jitter = max(float(sun_glare_sun_y_jitter), 0.0)
        self.sun_glare_sun_z_jitter = max(float(sun_glare_sun_z_jitter), 0.0)
        self.sun_glare_occluder_x_jitter = max(float(sun_glare_occluder_x_jitter), 0.0)
        self.sun_glare_occluder_half_y_range = self._ordered_range(sun_glare_occluder_half_y_min, sun_glare_occluder_half_y_max)
        self.sun_glare_divider_x_jitter = max(float(sun_glare_divider_x_jitter), 0.0)
        self.sun_glare_gate_x_jitter = max(float(sun_glare_gate_x_jitter), 0.0)
        self.sun_glare_gap_half_w_range = self._ordered_range(sun_glare_gap_half_w_min, sun_glare_gap_half_w_max)
        self.sun_glare_start_y_jitter = max(float(sun_glare_start_y_jitter), 0.0)

        self.cam_ambient_min = 0.08
        self.cam_ambient_max = 0.35
        self.cam_dir_min = 0.4
        self.cam_dir_max = 1.6
        self.cam_fog_beta_min = 0.01
        self.cam_fog_beta_max = 0.12
        self.cam_airlight_min = 0.2
        self.cam_airlight_max = 0.8

        self.cam_read_noise = 0.0025

        self.cam_motion_blur_gain = 0.09

        self.cam_sem = CameraSemantics(
            exposure_t_min=float(cam_exposure_t_min),
            exposure_t_span=float(cam_exposure_t_span),
            exposure_eff_min=float(cam_exposure_eff_min),
            exposure_eff_max=float(cam_exposure_eff_max),
            iso_gain_base=float(cam_iso_gain_base),
            iso_gain_scale=float(cam_iso_gain_scale),
            iso_gain_gamma=float(cam_iso_gain_gamma),
            shot_noise_base=float(cam_shot_noise_base),
        )

        self._configure_camera_preset(self.camera_preset)

        self._cam_ambient = torch.full((batch_size,), 0.2, device=device)
        self._cam_dir_intensity = torch.full((batch_size,), 1.0, device=device)
        self._cam_fog_beta = torch.full((batch_size,), 0.02, device=device)
        self._cam_airlight = torch.full((batch_size,), 0.4, device=device)
        self._cam_mat_obstacle = torch.full((batch_size,), 0.6, device=device)
        self._cam_mat_spec = torch.full((batch_size,), 0.08, device=device)
        self._img_grid_u = torch.linspace(-1.0, 1.0, self.width, device=device)[None, None, :]
        self._img_grid_v = torch.linspace(-1.0, 1.0, self.height, device=device)[None, :, None]
        self._sensor_model_base_params = OrderedDict()
        self._sensor_model_params = OrderedDict()
        self._build_sensor_model_base_params()

        self._load_scene_fit_profiles(scene_fit_profiles_path)
        self.reset()

    def _configure_camera_preset(self, preset: str):
        """根据档位配置 diff_depth 传感器退化强度。"""
        p = preset.lower()
        if p == 'low':
            self.cam_ambient_min, self.cam_ambient_max = 0.12, 0.28
            self.cam_dir_min, self.cam_dir_max = 0.45, 1.0
            self.cam_fog_beta_min, self.cam_fog_beta_max = 0.005, 0.06
            self.cam_read_noise = 0.0018
            self.cam_motion_blur_gain = 0.05
        elif p == 'medium':
            self.cam_ambient_min, self.cam_ambient_max = 0.10, 0.32
            self.cam_dir_min, self.cam_dir_max = 0.4, 1.3
            self.cam_fog_beta_min, self.cam_fog_beta_max = 0.008, 0.09
            self.cam_read_noise = 0.0022
            self.cam_motion_blur_gain = 0.075
        elif p == 'ultra':
            self.cam_ambient_min, self.cam_ambient_max = 0.06, 0.42
            self.cam_dir_min, self.cam_dir_max = 0.5, 1.9
            self.cam_fog_beta_min, self.cam_fog_beta_max = 0.015, 0.15
            self.cam_read_noise = 0.0032
            self.cam_motion_blur_gain = 0.11
        else:  # high
            self.cam_ambient_min, self.cam_ambient_max = 0.08, 0.35
            self.cam_dir_min, self.cam_dir_max = 0.4, 1.6
            self.cam_fog_beta_min, self.cam_fog_beta_max = 0.01, 0.12
            self.cam_read_noise = 0.0025
            self.cam_motion_blur_gain = 0.09

        self.cam_dir_min *= self.cam_lighting_scale
        self.cam_dir_max *= self.cam_lighting_scale
        self.cam_ambient_min *= self.cam_lighting_scale
        self.cam_ambient_max *= self.cam_lighting_scale
        self.cam_fog_beta_min *= self.cam_fog_scale
        self.cam_fog_beta_max *= self.cam_fog_scale
        self.cam_read_noise *= self.cam_noise_scale
        self.cam_motion_blur_gain *= self.cam_blur_scale

    def _reset_camera_states(self):
        """重置 diff_depth 传感器随机环境参数。"""
        B = self.batch_size
        device = self.device

        self._cam_ambient = torch.empty((B,), device=device).uniform_(self.cam_ambient_min, self.cam_ambient_max)
        self._cam_dir_intensity = torch.empty((B,), device=device).uniform_(self.cam_dir_min, self.cam_dir_max)
        self._cam_fog_beta = torch.empty((B,), device=device).uniform_(self.cam_fog_beta_min, self.cam_fog_beta_max)
        self._cam_airlight = torch.empty((B,), device=device).uniform_(self.cam_airlight_min, self.cam_airlight_max)
        self._cam_mat_obstacle = torch.empty((B,), device=device).uniform_(0.45, 0.85)
        self._cam_mat_spec = torch.empty((B,), device=device).uniform_(0.02, 0.18)

    def _build_sensor_model_base_params(self):
        """收口 diff_depth 传感器模型魔数为少量分组参数。"""
        self._sensor_model_base_params = OrderedDict([
            ('edge_depth_bias', 0.15),
            ('edge_gain', 1.0),
            ('frontality_edge_slope', 1.2),
            ('active_signal_gain', 5.0),
            ('active_signal_depth_bias', 0.08),
            ('passive_edge_base', 0.15),
            ('passive_edge_gain', 0.85),
            ('passive_albedo_base', 0.35),
            ('passive_albedo_gain', 0.65),
            ('active_range_base', 0.9),
            ('active_range_min_frac', 0.15),
            ('active_range_gain_frac', 0.85),
            ('active_range_gain_boost', 0.35),
            ('passive_range_base', 0.7),
            ('passive_range_exposure_frac', 0.08),
            ('passive_range_ambient_frac', 0.18),
            ('active_gate_width', 0.22),
            ('passive_gate_width', 0.28),
            ('signal_passive_mix', 0.45),
            ('washout_bias', 0.12),
            ('snr_ambient_weight', 0.45),
            ('snr_gain_weight', 0.12),
            ('snr_spec_weight', 0.35),
            ('snr_motion_weight', 0.25),
            ('quality_snr_gain', 2.6),
            ('quality_passive_gain', 0.9),
            ('quality_washout_penalty', 1.4),
            ('quality_spec_penalty', 2.0),
            ('quality_motion_penalty', 1.6),
            ('quality_far_penalty', 2.2),
            ('motion_blend_gain', 0.55),
            ('motion_blend_max', 0.85),
            ('flying_base', 0.12),
            ('flying_quality_gain', 0.88),
            ('flying_motion_base', 0.35),
            ('flying_motion_gain', 0.65),
            ('noise_floor_gain_weight', 0.18),
            ('noise_signal_gain', 0.018),
            ('noise_signal_range_gain', 0.8),
            ('noise_signal_bias', 0.08),
            ('noise_motion_gain', 0.03),
            ('noise_motion_edge_base', 0.3),
            ('noise_motion_edge_gain', 0.7),
            ('noise_spec_gain', 0.05),
            ('noise_std_min', 0.002),
            ('noise_std_max', 0.75),
            ('valid_threshold_base', 0.45),
            ('valid_bias_max', 0.35),
            ('valid_sigmoid_width', 0.08),
        ])
        self._sensor_model_params = OrderedDict(self._sensor_model_base_params)

    def _sensor_model_randomizable_keys(self):
        """仅对结构性缩放项做轻微随机化，避免把稳定项扰乱过头。"""
        return {
            'edge_depth_bias',
            'frontality_edge_slope',
            'active_signal_gain',
            'active_signal_depth_bias',
            'active_range_gain_boost',
            'passive_range_exposure_frac',
            'passive_range_ambient_frac',
            'active_gate_width',
            'passive_gate_width',
            'signal_passive_mix',
            'snr_ambient_weight',
            'snr_gain_weight',
            'snr_spec_weight',
            'snr_motion_weight',
            'quality_snr_gain',
            'quality_passive_gain',
            'quality_washout_penalty',
            'quality_spec_penalty',
            'quality_motion_penalty',
            'quality_far_penalty',
            'motion_blend_gain',
            'flying_quality_gain',
            'noise_floor_gain_weight',
            'noise_signal_gain',
            'noise_signal_range_gain',
            'noise_motion_gain',
            'noise_spec_gain',
            'valid_threshold_base',
            'valid_sigmoid_width',
        }

    def _sample_sensor_model_params(self):
        scale = max(float(self.cam_model_randomize_scale), 0.0)
        randomized = OrderedDict()
        randomizable = self._sensor_model_randomizable_keys()
        for key, value in self._sensor_model_base_params.items():
            v = float(value)
            if self.cam_model_randomize and scale > 0.0 and key in randomizable:
                jitter = 1.0 + random.uniform(-scale, scale)
                if key in {'active_gate_width', 'passive_gate_width', 'valid_sigmoid_width'}:
                    jitter = 1.0 + random.uniform(-0.5 * scale, 0.5 * scale)
                v = max(v * jitter, 1e-6)
            randomized[key] = float(v)
        self._sensor_model_params = randomized

    def _sensor_param(self, key: str) -> float:
        return float(self._sensor_model_params.get(key, self._sensor_model_base_params[key]))

    def _normalize_scenarios(self, scenarios):
        if scenarios is None:
            return ['glare', 'specular', 'dark']
        out = []
        for raw in scenarios:
            if raw is None:
                continue
            for token in str(raw).split(','):
                name = token.strip().lower().replace('-', '_')
                if not name:
                    continue
                if name not in self.supported_scenarios:
                    raise ValueError(
                        f"不支持的场景 '{name}'，仅支持: {list(self.supported_scenarios)}"
                    )
                if name not in out:
                    out.append(name)
        return out or ['glare', 'specular', 'dark']

    def _canonical_scene_name(self, name):
        return str(name).strip().lower().replace('-', '_')

    def _canonical_sun_glare_level(self, level):
        if level is None:
            return None
        token = str(level).strip().lower()
        aliases = {
            '0': 'l0',
            'l0': 'l0',
            'weak': 'l0',
            'low': 'l0',
            '1': 'l1',
            'l1': 'l1',
            'mild': 'l1',
            'midlow': 'l1',
            '2': 'l2',
            'l2': 'l2',
            'mid': 'l2',
            'medium': 'l2',
            'default': 'l2',
            '3': 'l3',
            'l3': 'l3',
            'strong': 'l3',
            'high': 'l3',
        }
        out = aliases.get(token, token)
        if out not in self.sun_glare_supported_levels:
            raise ValueError(
                f"不支持的 sun_glare 档位 '{level}'，仅支持 {list(self.sun_glare_supported_levels)}"
            )
        return out

    def _normalize_sun_glare_levels(self, levels):
        if levels is None:
            return ['l0', 'l1', 'l2', 'l3']
        out = []
        for raw in levels:
            if raw is None:
                continue
            for token in str(raw).split(','):
                name = self._canonical_sun_glare_level(token)
                if name not in out:
                    out.append(name)
        return out or ['l0', 'l1', 'l2', 'l3']

    def _canonical_sun_glare_slot(self, slot):
        if slot is None:
            return None
        token = str(slot).strip().lower().replace('-', '_')
        aliases = {
            'fl': 'far_left',
            'farleft': 'far_left',
            'far_left': 'far_left',
            '-1.5': 'far_left',
            'l': 'left',
            'left': 'left',
            '-0.5': 'left',
            'r': 'right',
            'right': 'right',
            '0.5': 'right',
            'fr': 'far_right',
            'farright': 'far_right',
            'far_right': 'far_right',
            '1.5': 'far_right',
        }
        name = aliases.get(token, token)
        valid = {item['name'] for item in self._sun_glare_opening_candidates()}
        if name not in valid:
            raise ValueError(
                f"不支持的 sun_glare 开口 '{slot}'，仅支持 {sorted(valid)}"
            )
        return name

    def _canonical_sensor_scene(self, scene_name):
        if scene_name is None:
            return None
        token = str(scene_name).strip().lower().replace('-', '_')
        name = token
        if name not in self.supported_scenarios:
            raise ValueError(
                f"不支持的场景模式 '{scene_name}'，仅支持 {list(self.supported_scenarios)}"
            )
        return name

    def _choose_sun_glare_level(self, scene_variant=None):
        if scene_variant is not None:
            return self._canonical_sun_glare_level(scene_variant)
        if self.eval_mode and self.sun_glare_eval_level is not None:
            return self.sun_glare_eval_level
        return random.choice(self.sun_glare_levels)

    def _sun_glare_opening_candidates(self):
        """
        Four candidate opening lanes, ordered from negative-y to positive-y.

        This makes blind or fixed-template policies much harder to script: even
        after choosing a coarse left/right side, they still need to identify the
        correct lane among two candidates on that side.
        """
        return (
            {'name': 'far_left', 'side': 'left', 'y': -1.12, 'id': -1.5},
            {'name': 'left', 'side': 'left', 'y': -0.56, 'id': -0.5},
            {'name': 'right', 'side': 'right', 'y': 0.56, 'id': 0.5},
            {'name': 'far_right', 'side': 'right', 'y': 1.12, 'id': 1.5},
        )

    def _choose_sun_glare_open_slot(self):
        """随机选择四选一的 opening lane。"""
        if self.eval_mode and self.sun_glare_eval_slot is not None:
            for item in self._sun_glare_opening_candidates():
                if item['name'] == self.sun_glare_eval_slot:
                    return dict(item)
            raise RuntimeError(f"invalid sun_glare_eval_slot={self.sun_glare_eval_slot}")
        return dict(random.choice(self._sun_glare_opening_candidates()))

    def _choose_sun_glare_open_side(self):
        """
        Backward-compatible helper retained for old analysis scripts.

        Returns only the coarse left/right side, while the actual scene logic
        uses `_choose_sun_glare_open_slot()` for four-way opening randomization.
        """
        return self._choose_sun_glare_open_slot()['side']

    def _apply_sun_glare_level(self, effects, level):
        cfg = {
            'l0': {
                'severity_id': 0.0,
                'local_effect_mul': 0.58,
                'ambient_add_mul': 0.45,
                'active_drop_mul': 0.70,
                'active_recover_mul': 0.92,
                'glare_bias_mul': 0.70,
                'glare_exposure_gain_mul': 0.72,
                'glare_power_bias_mul': 1.00,
                'glare_power_gain_mul': 0.98,
                'power_rescue_bias_mul': 1.00,
                'power_rescue_exposure_gain_mul': 1.00,
                'power_quality_bonus_mul': 0.68,
                'quality_penalty_mul': 0.58,
                'valid_bias_scale_mul': 0.70,
                'sun_sigma_u_mul': 0.92,
                'sun_sigma_v_mul': 0.92,
            },
            'l1': {
                'severity_id': 1.0,
                'local_effect_mul': 0.78,
                'ambient_add_mul': 0.72,
                'active_drop_mul': 0.86,
                'active_recover_mul': 0.96,
                'glare_bias_mul': 0.86,
                'glare_exposure_gain_mul': 0.88,
                'glare_power_bias_mul': 1.00,
                'glare_power_gain_mul': 0.99,
                'power_rescue_bias_mul': 1.00,
                'power_rescue_exposure_gain_mul': 1.00,
                'power_quality_bonus_mul': 0.84,
                'quality_penalty_mul': 0.82,
                'valid_bias_scale_mul': 0.86,
                'sun_sigma_u_mul': 0.97,
                'sun_sigma_v_mul': 0.97,
            },
            'l2': {
                'severity_id': 2.0,
                'local_effect_mul': 1.00,
                'ambient_add_mul': 1.00,
                'active_drop_mul': 1.00,
                'active_recover_mul': 1.00,
                'glare_bias_mul': 1.00,
                'glare_exposure_gain_mul': 1.00,
                'glare_power_bias_mul': 1.00,
                'glare_power_gain_mul': 1.00,
                'power_rescue_bias_mul': 1.00,
                'power_rescue_exposure_gain_mul': 1.00,
                'power_quality_bonus_mul': 1.00,
                'quality_penalty_mul': 1.00,
                'valid_bias_scale_mul': 1.00,
                'sun_sigma_u_mul': 1.00,
                'sun_sigma_v_mul': 1.00,
            },
            'l3': {
                'severity_id': 3.0,
                'local_effect_mul': 1.22,
                'ambient_add_mul': 1.26,
                'active_drop_mul': 1.10,
                'active_recover_mul': 1.05,
                'glare_bias_mul': 1.14,
                'glare_exposure_gain_mul': 1.16,
                'glare_power_bias_mul': 1.00,
                'glare_power_gain_mul': 1.06,
                'power_rescue_bias_mul': 1.00,
                'power_rescue_exposure_gain_mul': 1.00,
                'power_quality_bonus_mul': 1.18,
                'quality_penalty_mul': 1.24,
                'valid_bias_scale_mul': 1.16,
                'sun_sigma_u_mul': 1.06,
                'sun_sigma_v_mul': 1.06,
            },
        }[level]

        out = dict(effects)
        scaled_keys = {
            'ambient_add': 'ambient_add_mul',
            'active_drop': 'active_drop_mul',
            'active_recover': 'active_recover_mul',
            'glare_bias': 'glare_bias_mul',
            'glare_exposure_gain': 'glare_exposure_gain_mul',
            'glare_power_bias': 'glare_power_bias_mul',
            'glare_power_gain': 'glare_power_gain_mul',
            'power_rescue_bias': 'power_rescue_bias_mul',
            'power_rescue_exposure_gain': 'power_rescue_exposure_gain_mul',
            'power_quality_bonus': 'power_quality_bonus_mul',
            'quality_penalty': 'quality_penalty_mul',
            'valid_bias_scale': 'valid_bias_scale_mul',
            'sun_sigma_u': 'sun_sigma_u_mul',
            'sun_sigma_v': 'sun_sigma_v_mul',
        }
        for key, mul_key in scaled_keys.items():
            if key in out:
                out[key] = float(out[key]) * float(cfg[mul_key])
        local_mul = float(cfg['local_effect_mul'])
        for key in (
            'spec_add',
            'power_washout_penalty',
            'power_washout_valid_bias',
            'dark_albedo_drop',
            'dark_active_drop',
            'dark_underexposure_penalty',
            'dark_passive_rescue',
        ):
            if key in out:
                out[key] = float(out[key]) * local_mul
        out['glare_level'] = level
        out['glare_level_id'] = float(cfg['severity_id'])
        return out

    def _sensor_scene_effects(self, scene_name):
        scene_name = self._canonical_sensor_scene(scene_name)
        if scene_name == 'specular':
            return {
                'sensor_regime_name': 'specular',
                'sensor_regime_id': 1.0,
                'spec_add': 0.82,
                'spec_mask_sun_mix': 0.0,
                'spec_mask_hazard_mix': 1.0,
                'power_washout_gamma': 1.65,
                'power_washout_penalty': 1.25,
                'power_washout_valid_bias': 0.22,
                'hazard_mask_mix': 1.0,
            }
        if scene_name == 'dark':
            return {
                'sensor_regime_name': 'dark',
                'sensor_regime_id': 2.0,
                'dark_albedo_drop': 0.72,
                'dark_mask_hazard_mix': 1.0,
                'dark_mask_sun_mix': 0.0,
                'dark_exposure_target': 0.62,
                'dark_gain_target': 0.25,
                'dark_gain_weight': 0.32,
                'dark_underexposure_penalty': 1.18,
                'dark_exposure_bonus': 0.34,
                'dark_gain_bonus': 0.10,
                'dark_active_drop': 0.86,
                'dark_passive_rescue': 0.95,
                'hazard_mask_mix': 1.0,
            }
        return {
            'sensor_regime_name': 'glare',
            'sensor_regime_id': 0.0,
            'ambient_add': 4.4,
            'active_drop': 0.76,
            'active_recover': 1.02,
            'glare_bias': 0.32,
            'glare_exposure_gain': 2.55,
            'glare_power_bias': 0.14,
            'glare_power_gain': 1.55,
            'power_rescue_bias': 0.14,
            'power_rescue_exposure_gain': 0.58,
            'power_quality_bonus': 0.88,
            'quality_penalty': 2.95,
            'valid_bias_scale': 0.18,
            'hazard_mask_mix': 0.35,
        }

    def _normalize_profile_dict(self, data):
        out = {}
        if not isinstance(data, dict):
            return out
        for raw_name, payload in data.items():
            name = self._canonical_scene_name(raw_name)
            if name not in self.supported_scenarios or not isinstance(payload, dict):
                continue
            out[name] = dict(payload)
        return out

    def _load_scene_fit_profiles(self, scene_fit_profiles_path):
        self.scene_fit_profiles_path = None
        self.scene_sensor_profile_overrides = {}
        self.scene_effect_overrides = {}
        if not scene_fit_profiles_path:
            return

        path = os.path.expanduser(str(scene_fit_profiles_path))
        if not os.path.isfile(path):
            print(f"[warn] scene_fit_profiles 文件不存在，忽略: {path}")
            return

        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        self.scene_fit_profiles_path = path
        self.scene_sensor_profile_overrides = self._normalize_profile_dict(
            payload.get('sensor_profiles', {})
        )

        effect_payload = payload.get('scene_effects', {})
        if not effect_payload and isinstance(payload.get('scene_profiles'), dict):
            effect_payload = payload.get('scene_profiles', {})
        self.scene_effect_overrides = self._normalize_profile_dict(effect_payload)

    def _scene_profile_range(self, scene_name, key, default_lo, default_hi):
        profile = self.scene_sensor_profile_overrides.get(scene_name, {})
        if key not in profile:
            return float(default_lo), float(default_hi)
        value = profile[key]
        if isinstance(value, (int, float)):
            val = float(value)
            return val, val
        if isinstance(value, (list, tuple)) and len(value) == 2:
            lo = float(value[0])
            hi = float(value[1])
            if hi < lo:
                lo, hi = hi, lo
            return lo, hi
        return float(default_lo), float(default_hi)

    def _sample_scene_profile(self, scene_name, key, default_lo, default_hi):
        lo, hi = self._scene_profile_range(scene_name, key, default_lo, default_hi)
        if abs(hi - lo) <= 1e-9:
            return torch.full((self.batch_size,), lo, device=self.device)
        return self._sample_scene_tensor(lo, hi)

    def _merge_scene_effects(self, scene_name, effects):
        merged = dict(effects)
        merged.update(self.scene_effect_overrides.get(scene_name, {}))
        return merged

    def _realign_sun_glare_effects(self, effects):
        """
        Keep geometry-dependent anchors consistent with the sampled opening side.

        `scene_fit_profiles.json` can override scene-effect values at runtime, but
        geometry-dependent anchors should follow the current opening side rather
        than a fixed centerline value from the profile file.
        """
        aligned = dict(effects)
        slot_y = aligned.get('decision_open_slot_y', None)
        if slot_y is None:
            open_side = str(aligned.get('decision_open_side', 'right')).strip().lower()
            gap_y_center = -0.56 if open_side == 'left' else 0.56
        else:
            gap_y_center = float(slot_y)

        def _replace_y(key, default_xyz, offset_key=None):
            raw = aligned.get(key, default_xyz)
            if isinstance(raw, torch.Tensor):
                raw = raw.detach().cpu().tolist()
            if not isinstance(raw, (list, tuple)) or len(raw) < 3:
                raw = default_xyz
            y_offset = float(aligned.get(offset_key, 0.0)) if offset_key else 0.0
            return [float(raw[0]), float(gap_y_center) + y_offset, float(raw[2])]

        if 'sun_anchor' in aligned:
            aligned['sun_anchor'] = _replace_y('sun_anchor', [3.00, gap_y_center, 1.65], 'sun_y_offset')
        aligned['hazard_center'] = _replace_y('hazard_center', [1.82, gap_y_center, 1.50])
        return aligned

    def _choose_scene_name(self, scene_name=None):
        if scene_name is not None:
            name = str(scene_name).strip().lower()
            if name not in self.supported_scenarios:
                raise ValueError(
                    f"reset(scene_name={scene_name!r}) 不支持；仅支持 {list(self.supported_scenarios)}"
                )
            return name
        return random.choice(self.scenarios)

    def _set_scene_name(self, scene_name, scene_variant=None):
        self.current_scene_name = str(scene_name)
        self.current_scene_id = self.scene_name_to_id[self.current_scene_name]
        self.current_scene_variant = scene_variant
        self.current_sun_glare_level = scene_variant
        self.current_scene_tag = f'{self.current_scene_name}_{scene_variant}' if scene_variant else self.current_scene_name

    def _sample_scene_tensor(self, lo, hi):
        return torch.empty((self.batch_size,), device=self.device).uniform_(float(lo), float(hi))

    def _require_finite_tensor(self, name, value, scene_name=None):
        if not torch.is_tensor(value):
            return value
        if torch.isfinite(value).all():
            return value

        bad = ~torch.isfinite(value)
        bad_count = int(bad.sum().detach().cpu().item())
        flat_bad = bad.reshape(-1).nonzero(as_tuple=False)
        first_bad = int(flat_bad[0, 0].detach().cpu().item()) if flat_bad.numel() > 0 else -1
        scene = scene_name or self.current_scene_name or 'unknown'
        raise RuntimeError(
            f"[env_cuda:{scene}] non-finite tensor detected in {name}: "
            f"shape={tuple(value.shape)} bad_count={bad_count} first_bad_flat_idx={first_bad}"
        )

    def _spatial_mean(self, x):
        if x is None:
            return None
        if not torch.is_tensor(x):
            return x
        if x.ndim < 3:
            return x
        return x.mean(dim=(-2, -1))

    def _store_last_diff_depth_debug(self, debug):
        if debug is None:
            self.last_diff_depth_debug = None
            return

        stored = {}
        for key, value in debug.items():
            if isinstance(value, dict):
                inner = {}
                for sub_key, sub_value in value.items():
                    if torch.is_tensor(sub_value):
                        inner[sub_key] = sub_value.detach()
                    else:
                        inner[sub_key] = sub_value
                stored[key] = inner
            elif torch.is_tensor(value):
                stored[key] = value.detach()
            else:
                stored[key] = value
        self.last_diff_depth_debug = stored

    def _store_last_diff_depth_train_aux(self, aux):
        self.last_diff_depth_train_aux = aux

    def get_last_diff_depth_train_aux(self):
        return self.last_diff_depth_train_aux or {}

    def export_last_diff_depth_debug(self, env_idx=0):
        debug = self.last_diff_depth_debug or {}
        out = {
            'scene_name': str(debug.get('scene_name', self.current_scene_name)),
            'images': {},
            'scalars': {},
        }

        for key in ('quality_map', 'invalid_mask', 'scene_effect_map', 'scene_mask'):
            value = debug.get(key, None)
            if torch.is_tensor(value) and value.ndim >= 3 and value.shape[0] > 0:
                idx = int(min(max(env_idx, 0), value.shape[0] - 1))
                out['images'][key] = value[idx].detach().cpu().numpy()

        for key, value in (debug.get('scalars', {}) or {}).items():
            if torch.is_tensor(value):
                if value.ndim == 0:
                    out['scalars'][key] = float(value.detach().cpu().item())
                elif value.shape[0] > 0:
                    idx = int(min(max(env_idx, 0), value.shape[0] - 1))
                    out['scalars'][key] = float(value[idx].detach().cpu().item())
            elif isinstance(value, (int, float)):
                out['scalars'][key] = float(value)
        return out

    def _apply_scene_sensor_profile(self, scene_name):
        if scene_name in self.supported_scenarios:
            if self.sun_glare_randomize:
                regime = str(scene_name)
                if regime == 'specular':
                    self._cam_ambient = self._sample_scene_tensor(0.16, 0.36)
                    self._cam_dir_intensity = self._sample_scene_tensor(0.55, 1.35)
                    self._cam_fog_beta = self._sample_scene_tensor(0.012, 0.060)
                    self._cam_airlight = self._sample_scene_tensor(0.12, 0.42)
                    self._cam_mat_obstacle = self._sample_scene_tensor(0.58, 0.90)
                    self._cam_mat_spec = self._sample_scene_tensor(0.32, 0.72)
                elif regime == 'dark':
                    self._cam_ambient = self._sample_scene_tensor(0.012, 0.070)
                    self._cam_dir_intensity = self._sample_scene_tensor(0.035, 0.18)
                    self._cam_fog_beta = self._sample_scene_tensor(0.002, 0.020)
                    self._cam_airlight = self._sample_scene_tensor(0.010, 0.070)
                    self._cam_mat_obstacle = self._sample_scene_tensor(0.22, 0.48)
                    self._cam_mat_spec = self._sample_scene_tensor(0.00, 0.035)
                else:
                    self._cam_ambient = self._sample_scene_tensor(*self.sun_glare_ambient_range)
                    self._cam_dir_intensity = self._sample_scene_tensor(*self.sun_glare_dir_range)
                    self._cam_fog_beta = self._sample_scene_tensor(*self.sun_glare_fog_beta_range)
                    self._cam_airlight = self._sample_scene_tensor(*self.sun_glare_airlight_range)
                    self._cam_mat_obstacle = self._sample_scene_tensor(*self.sun_glare_mat_obstacle_range)
                    self._cam_mat_spec = self._sample_scene_tensor(*self.sun_glare_mat_spec_range)
            else:
                if scene_name == 'specular':
                    self._cam_ambient = self._sample_scene_profile(scene_name, 'cam_ambient', 0.16, 0.28)
                    self._cam_dir_intensity = self._sample_scene_profile(scene_name, 'cam_dir_intensity', 0.55, 1.00)
                    self._cam_fog_beta = self._sample_scene_profile(scene_name, 'cam_fog_beta', 0.012, 0.040)
                    self._cam_airlight = self._sample_scene_profile(scene_name, 'cam_airlight', 0.12, 0.30)
                    self._cam_mat_obstacle = self._sample_scene_profile(scene_name, 'cam_mat_obstacle', 0.58, 0.82)
                    self._cam_mat_spec = self._sample_scene_profile(scene_name, 'cam_mat_spec', 0.34, 0.58)
                elif scene_name == 'dark':
                    self._cam_ambient = self._sample_scene_profile(scene_name, 'cam_ambient', 0.018, 0.055)
                    self._cam_dir_intensity = self._sample_scene_profile(scene_name, 'cam_dir_intensity', 0.040, 0.14)
                    self._cam_fog_beta = self._sample_scene_profile(scene_name, 'cam_fog_beta', 0.002, 0.014)
                    self._cam_airlight = self._sample_scene_profile(scene_name, 'cam_airlight', 0.010, 0.050)
                    self._cam_mat_obstacle = self._sample_scene_profile(scene_name, 'cam_mat_obstacle', 0.24, 0.42)
                    self._cam_mat_spec = self._sample_scene_profile(scene_name, 'cam_mat_spec', 0.00, 0.030)
                else:
                    self._cam_ambient = self._sample_scene_profile(scene_name, 'cam_ambient', 0.10, 0.18)
                    self._cam_dir_intensity = self._sample_scene_profile(scene_name, 'cam_dir_intensity', 0.35, 0.75)
                    self._cam_fog_beta = self._sample_scene_profile(scene_name, 'cam_fog_beta', 0.010, 0.030)
                    self._cam_airlight = self._sample_scene_profile(scene_name, 'cam_airlight', 0.12, 0.25)
                    self._cam_mat_obstacle = self._sample_scene_profile(scene_name, 'cam_mat_obstacle', 0.52, 0.78)
                    self._cam_mat_spec = self._sample_scene_profile(scene_name, 'cam_mat_spec', 0.04, 0.10)
        else:
            raise ValueError(f'未知场景: {scene_name}')

    def _build_voxels(self, rows):
        if not rows:
            return torch.empty((0, 6), device=self.device)
        return torch.tensor(rows, device=self.device, dtype=torch.float32)

    def _build_sun_glare_voxel_layout(self, gap_y_center, *, occluder_x=0.88,
                                      occluder_half_y=0.48, divider_x=1.58,
                                      gate_x=1.82, gap_half_w=0.18):
        """
        Sun Glare probe-then-commit 场景：
        中央遮挡板 + 三条 lane divider fins + 四选一单开口墙。

        设计目标：
        - 在 occluder 前仍然只看到一个居中的遮挡体，避免“提前背答案”；
        - 在 occluder 后留一小段 probe zone，使正确 lane 更早暴露 cue；
        - 通过三条很薄的 divider fins 把通道切成 4 条候选 lane；
        - 真正 opening 只在其中 1 条 lane 上开放，且每回合随机；
        - 正确策略应当先 probe 再 commit，而不是先固定走某个模板 lane。
        """
        voxel_half_w = 0.25
        voxel_half_h = 1.5

        guide = self._build_voxels([
            [-1.65, -1.48, 1.5, voxel_half_w, voxel_half_w, voxel_half_h],
            [-1.65,  1.48, 1.5, voxel_half_w, voxel_half_w, voxel_half_h],
        ])
        occluder = self._build_voxels([
            [float(occluder_x), 0.00, 1.5, 0.10, float(occluder_half_y), voxel_half_h],
        ])
        lane_dividers = self._build_voxels([
            # Keep the lanes well-separated near the gate, but leave enough
            # x-distance after the occluder so all four candidate lanes remain
            # physically reachable under the current collision model.
            [float(divider_x), -0.84, 1.5, 0.22, 0.05, voxel_half_h],
            [float(divider_x),  0.00, 1.5, 0.22, 0.05, voxel_half_h],
            [float(divider_x),  0.84, 1.5, 0.22, 0.05, voxel_half_h],
        ])
        big = 4.0
        wall_thickness = 0.15
        gap_z_center = 1.50
        gap_half_h = 1.05
        gate_wall = self._build_voxels([
            [float(gate_x), gap_y_center - float(gap_half_w) - big, gap_z_center, wall_thickness, big, big],
            [float(gate_x), gap_y_center + float(gap_half_w) + big, gap_z_center, wall_thickness, big, big],
            [float(gate_x), gap_y_center, gap_z_center + gap_half_h + big, wall_thickness, float(gap_half_w), big],
            [float(gate_x), gap_y_center, gap_z_center - gap_half_h - big, wall_thickness, float(gap_half_w), big],
        ])
        back_wall = self._build_voxels([
            [float(gate_x) + 1.83, 0.00, 1.5, 0.10, 1.30, voxel_half_h],
        ])
        return torch.cat([guide, occluder, lane_dividers, gate_wall, back_wall], dim=0)

    def _project_world_point(self, point, R_cam_world, dtype):
        point_t = torch.as_tensor(point, device=self.device, dtype=dtype)
        if point_t.ndim == 1:
            point_t = point_t.unsqueeze(0).expand(self.batch_size, -1)
        elif point_t.ndim == 2 and point_t.shape[0] == 1:
            point_t = point_t.expand(self.batch_size, -1)
        point_t = self._require_finite_tensor('scene_anchor_point', point_t)
        origin = self._require_finite_tensor('camera_position', self.p.to(dtype))
        rel = self._require_finite_tensor('scene_anchor_rel', point_t - origin)
        cam = self._require_finite_tensor(
            'scene_anchor_cam_coords',
            torch.einsum('bij,bj->bi', R_cam_world.transpose(1, 2), rel),
        )
        cam_x = cam[:, 0]
        fov_x_half = max(float(self._fov_x_half_tan), 1e-4)
        fov_y_half = max(fov_x_half * float(self.height) / max(float(self.width), 1.0), 1e-4)
        denom = cam_x.abs().clamp_min(1e-4)
        u = self._require_finite_tensor(
            'scene_anchor_proj_u',
            (cam[:, 1] / (denom * fov_x_half)).clamp(-4.0, 4.0),
        )
        v = self._require_finite_tensor(
            'scene_anchor_proj_v',
            (cam[:, 2] / (denom * fov_y_half)).clamp(-4.0, 4.0),
        )
        visible = (cam_x > 0.05).to(dtype)
        return u, v, cam_x, visible

    def _voxel_line_of_sight_mask(self, point, dtype, start_margin=1e-4, end_margin=0.02):
        """
        Return whether a free-space anchor point is visible from the camera origin
        under the current voxel geometry.

        Notes:
        - This is only used for Sun Glare scene-effect anchors such as
          `sun_anchor` and `hazard_center`.
        - The active benchmark is voxel-only, so checking `self.voxels` is
          sufficient.
        - `end_margin` is in normalized segment parameter t in [0,1]. It avoids
          classifying points very near the target anchor as "occluded by the
          target itself" due to finite wall thickness / numerical noise.
        """
        point_t = torch.as_tensor(point, device=self.device, dtype=dtype)
        if point_t.ndim == 1:
            point_t = point_t.unsqueeze(0).expand(self.batch_size, -1)
        elif point_t.ndim == 2 and point_t.shape[0] == 1:
            point_t = point_t.expand(self.batch_size, -1)
        point_t = self._require_finite_tensor('los_anchor_point', point_t)
        origin = self._require_finite_tensor('los_camera_origin', self.p.to(dtype))
        direction = self._require_finite_tensor('los_direction', point_t - origin)

        if self.voxels.numel() == 0:
            return torch.ones((self.batch_size,), device=self.device, dtype=dtype)

        # Segment parameterization: x(t) = origin + t * direction, t in [0, 1].
        o = origin[:, None, :]
        d = direction[:, None, :]
        vox = self.voxels.to(dtype)
        center = vox[:, :, :3]
        half = vox[:, :, 3:6].clamp_min(1e-6)

        eps = 1e-8
        parallel = d.abs() <= eps
        slab_min = center - half
        slab_max = center + half
        outside_parallel = parallel & ((o < slab_min) | (o > slab_max))
        miss_parallel = outside_parallel.any(dim=-1)

        safe_d = torch.where(parallel, torch.ones_like(d), d)
        t1 = (slab_min - o) / safe_d
        t2 = (slab_max - o) / safe_d

        neg_inf = torch.full_like(t1, -1e9)
        pos_inf = torch.full_like(t1, 1e9)
        t_near_axis = torch.where(parallel, neg_inf, torch.minimum(t1, t2))
        t_far_axis = torch.where(parallel, pos_inf, torch.maximum(t1, t2))

        t_enter = t_near_axis.amax(dim=-1)
        t_exit = t_far_axis.amin(dim=-1)
        t_enter = self._require_finite_tensor('los_t_enter', t_enter)
        t_exit = self._require_finite_tensor('los_t_exit', t_exit)

        intersects = (~miss_parallel) & (t_exit >= t_enter) & (t_exit >= float(start_margin))
        first_hit = torch.where(t_enter > float(start_margin), t_enter, t_exit)
        blocked = intersects & (first_hit < (1.0 - float(end_margin)))
        return (~blocked.any(dim=-1)).to(dtype)

    def _project_rect_half_extents(self, cam_x, half_y, half_z, dtype):
        fov_x_half = max(float(self._fov_x_half_tan), 1e-4)
        fov_y_half = max(fov_x_half * float(self.height) / max(float(self.width), 1.0), 1e-4)
        forward = cam_x.abs().clamp_min(0.25)
        half_u = (float(half_y) / (forward * fov_x_half)).clamp(0.04, 1.25)
        half_v = (float(half_z) / (forward * fov_y_half)).clamp(0.04, 1.25)
        return half_u.to(dtype=dtype), half_v.to(dtype=dtype)

    def _make_gaussian_mask(self, center_u, center_v, sigma_u, sigma_v, dtype):
        grid_u = self._img_grid_u.to(dtype=dtype)
        grid_v = self._img_grid_v.to(dtype=dtype)
        sigma_u = max(float(sigma_u), 1e-3)
        sigma_v = max(float(sigma_v), 1e-3)
        du = (grid_u - center_u[:, None, None]) / sigma_u
        dv = (grid_v - center_v[:, None, None]) / sigma_v
        return torch.exp(-0.5 * (du.square() + dv.square()))

    def _make_box_mask(self, center_u, center_v, half_u, half_v, softness, dtype):
        grid_u = self._img_grid_u.to(dtype=dtype)
        grid_v = self._img_grid_v.to(dtype=dtype)
        half_u = half_u[:, None, None]
        half_v = half_v[:, None, None]
        softness = max(float(softness), 1e-4)
        mask_u = torch.sigmoid((half_u - (grid_u - center_u[:, None, None]).abs()) / softness)
        mask_v = torch.sigmoid((half_v - (grid_v - center_v[:, None, None]).abs()) / softness)
        return mask_u * mask_v

    def _scene_sensor_adjustments(self, depth, power01, exposure_s, gain_scale, motion, R_cam_world):
        dtype = depth.dtype
        zeros = torch.zeros_like(depth)
        ones = torch.ones_like(depth)
        adj = {
            'ambient_mul': ones,
            'ambient_add': zeros,
            'albedo_mul': ones,
            'active_mul': ones,
            'passive_mul': ones,
            'spec_add': zeros,
            'motion_mul': ones,
            'quality_add': zeros,
            'far_override': zeros,
            'valid_bias': zeros,
            'debug_scene_mask': zeros,
            'debug_effect_map': zeros,
            'debug_scalars': {},
        }

        fx = self.current_scene_effects or {}
        scene_name = self.current_scene_name

        if scene_name in self.supported_scenarios:
            regime = str(fx.get('sensor_regime_name', scene_name))
            use_sun = regime == 'glare' and 'sun_anchor' in fx
            if use_sun:
                center_u, center_v, _, visible = self._project_world_point(
                    fx['sun_anchor'], R_cam_world, dtype)
                sun_los = self._voxel_line_of_sight_mask(fx['sun_anchor'], dtype)
                sun_mask = self._make_gaussian_mask(
                    center_u, center_v,
                    fx.get('sun_sigma_u', 0.30),
                    fx.get('sun_sigma_v', 0.24),
                    dtype,
                )
                strength = self._require_finite_tensor(
                    'sun_glare/strength',
                    sun_mask * visible[:, None, None] * sun_los[:, None, None],
                    scene_name,
                )
            else:
                strength = zeros
                sun_los = torch.ones((self.batch_size,), device=self.device, dtype=dtype)

            hazard_mask = zeros
            if 'hazard_center' in fx:
                hazard_u, hazard_v, hazard_x, hazard_visible = self._project_world_point(
                    fx['hazard_center'], R_cam_world, dtype)
                hazard_los = self._voxel_line_of_sight_mask(fx['hazard_center'], dtype)
                half_u, half_v = self._project_rect_half_extents(
                    hazard_x,
                    fx.get('hazard_half_y', 0.32),
                    fx.get('hazard_half_z', 1.35),
                    dtype,
                )
                hazard_mask = self._make_box_mask(
                    hazard_u,
                    hazard_v,
                    half_u,
                    half_v,
                    fx.get('hazard_softness', 0.055),
                    dtype,
                ) * hazard_visible[:, None, None] * hazard_los[:, None, None]
                hazard_mask = self._require_finite_tensor('sun_glare/hazard_mask', hazard_mask, scene_name)
            else:
                hazard_los = torch.ones((self.batch_size,), device=self.device, dtype=dtype)

            local_mask = (strength + hazard_mask * float(fx.get('hazard_mask_mix', 0.35))).clamp(0.0, 1.0)
            glare_penalty = torch.zeros_like(strength)
            extra_effect = torch.zeros_like(strength)

            if regime == 'glare':
                effect_strength = strength
                glare_penalty = self._require_finite_tensor(
                    'sun_glare/glare_penalty',
                    effect_strength * (
                        float(fx.get('glare_bias', 0.24)) +
                        float(fx.get('glare_exposure_gain', 1.70)) * exposure_s[:, None, None]
                    ) / (
                        float(fx.get('glare_power_bias', 0.18)) +
                        float(fx.get('glare_power_gain', 1.55)) * power01[:, None, None]
                    ),
                    scene_name,
                )
                power_rescue = effect_strength * power01[:, None, None] / (
                    float(fx.get('power_rescue_bias', 0.22)) +
                    float(fx.get('power_rescue_exposure_gain', 0.85)) * exposure_s[:, None, None]
                )
                adj['ambient_add'] = adj['ambient_add'] + strength * float(fx.get('ambient_add', 2.2))
                adj['active_mul'] = adj['active_mul'] * (
                    1.0
                    - float(fx.get('active_drop', 0.50)) * effect_strength
                    + float(fx.get('active_recover', 0.55)) * power01[:, None, None] * effect_strength
                ).clamp_min(0.05)
                adj['quality_add'] = adj['quality_add'] - float(fx.get('quality_penalty', 1.8)) * glare_penalty
                adj['quality_add'] = adj['quality_add'] + float(fx.get('power_quality_bonus', 0.38)) * power_rescue
                adj['valid_bias'] = adj['valid_bias'] + float(fx.get('valid_bias_scale', 0.08)) * effect_strength

            spec_add = float(fx.get('spec_add', 0.0))
            if spec_add > 0.0:
                spec_mask = (
                    strength * float(fx.get('spec_mask_sun_mix', 1.0)) +
                    hazard_mask * float(fx.get('spec_mask_hazard_mix', 0.75))
                ).clamp(0.0, 1.0)
                power_overdrive = spec_mask * power01[:, None, None].pow(
                    float(fx.get('power_washout_gamma', 1.6))
                )
                extra_effect = torch.maximum(extra_effect, power_overdrive.clamp(0.0, 1.0))
                adj['spec_add'] = adj['spec_add'] + spec_mask * spec_add
                adj['quality_add'] = adj['quality_add'] - (
                    float(fx.get('power_washout_penalty', 0.0)) * power_overdrive
                )
                adj['valid_bias'] = adj['valid_bias'] + (
                    float(fx.get('power_washout_valid_bias', 0.0)) * power_overdrive
                )

            dark_drop = float(fx.get('dark_albedo_drop', 0.0))
            if dark_drop > 0.0:
                dark_mask = (
                    hazard_mask * float(fx.get('dark_mask_hazard_mix', 0.85)) +
                    strength * float(fx.get('dark_mask_sun_mix', 0.25))
                ).clamp(0.0, 1.0)
                exposure01_est = (
                    (exposure_s - float(self.cam_sem.exposure_t_min)) /
                    max(float(self.cam_sem.exposure_t_span), 1e-6)
                ).clamp(0.0, 1.0)
                gain01_est = (
                    (gain_scale - float(self.cam_sem.iso_gain_base)) /
                    max(float(self.cam_sem.iso_gain_scale), 1e-6)
                ).clamp(0.0, 1.0)
                underexposed = (
                    torch.relu(float(fx.get('dark_exposure_target', 0.62)) - exposure01_est) +
                    float(fx.get('dark_gain_weight', 0.35)) *
                    torch.relu(float(fx.get('dark_gain_target', 0.28)) - gain01_est)
                )
                dark_bonus = (
                    float(fx.get('dark_exposure_bonus', 0.20)) * exposure01_est +
                    float(fx.get('dark_gain_bonus', 0.10)) * gain01_est
                )
                adj['albedo_mul'] = adj['albedo_mul'] * (1.0 - dark_drop * dark_mask).clamp_min(0.04)
                adj['active_mul'] = adj['active_mul'] * (
                    1.0 - float(fx.get('dark_active_drop', 0.0)) * dark_mask
                ).clamp_min(0.04)
                adj['passive_mul'] = adj['passive_mul'] * (
                    1.0 +
                    float(fx.get('dark_passive_rescue', 0.0)) *
                    dark_mask *
                    (exposure01_est + 0.45 * gain01_est)[:, None, None]
                )
                extra_effect = torch.maximum(
                    extra_effect,
                    (dark_mask * underexposed[:, None, None]).clamp(0.0, 1.0),
                )
                adj['quality_add'] = adj['quality_add'] - (
                    float(fx.get('dark_underexposure_penalty', 1.0)) *
                    dark_mask * underexposed[:, None, None]
                )
                adj['quality_add'] = adj['quality_add'] + dark_mask * dark_bonus[:, None, None]

            adj['debug_scene_mask'] = hazard_mask
            debug_effect = (glare_penalty + extra_effect + local_mask * 0.10).clamp(0.0, 1.0)
            adj['debug_effect_map'] = debug_effect
            adj['debug_scalars'] = {
                'scene_mask_mean': self._spatial_mean(hazard_mask),
                'scene_effect_mean': self._spatial_mean(debug_effect),
                'sensor_regime_id': torch.full(
                    (self.batch_size,),
                    float(fx.get('sensor_regime_id', 0.0)),
                    device=self.device,
                    dtype=dtype,
                ),
                'sun_mask_mean': self._spatial_mean(strength),
                'hazard_mask_mean': self._spatial_mean(hazard_mask),
                'sun_los_mean': sun_los.detach(),
                'hazard_los_mean': hazard_los.detach(),
                'decision_open_side_id': torch.full(
                    (self.batch_size,),
                    float(fx.get('decision_open_side_id', 0.0)),
                    device=self.device,
                    dtype=dtype,
                ),
                'decision_open_slot_id': torch.full(
                    (self.batch_size,),
                    float(fx.get('decision_open_side_id', 0.0)),
                    device=self.device,
                    dtype=dtype,
                ),
                'decision_open_slot_y': torch.full(
                    (self.batch_size,),
                    float(fx.get('decision_open_slot_y', 0.0)),
                    device=self.device,
                    dtype=dtype,
                ),
                'glare_level_id': torch.full(
                    (self.batch_size,),
                    float(fx.get('glare_level_id', 0.0)),
                    device=self.device,
                    dtype=dtype,
                ),
            }

        return adj

    def _build_scene_geometry(self, scene_name, scene_variant=None):
        if scene_name not in self.supported_scenarios:
            raise ValueError(f'未知场景: {scene_name}')

        selected_variant = self._choose_sun_glare_level(scene_variant)
        open_slot = self._choose_sun_glare_open_slot()
        sensor_regime = scene_name
        gap_y_center = float(open_slot['y'])
        start_y = 0.0
        occluder_x = 0.88
        occluder_half_y = 0.48
        divider_x = 1.58
        gate_x = 1.82
        gap_half_w = 0.18
        sun_sigma_u = 0.24
        sun_sigma_v = 0.22
        sun_y_offset = 0.0
        sun_z_offset = 0.0
        if self.sun_glare_randomize:
            start_y = random.uniform(-self.sun_glare_start_y_jitter, self.sun_glare_start_y_jitter)
            occluder_x = 0.88 + random.uniform(-self.sun_glare_occluder_x_jitter, self.sun_glare_occluder_x_jitter)
            occluder_half_y = random.uniform(*self.sun_glare_occluder_half_y_range)
            divider_x = 1.58 + random.uniform(-self.sun_glare_divider_x_jitter, self.sun_glare_divider_x_jitter)
            gate_x = 1.82 + random.uniform(-self.sun_glare_gate_x_jitter, self.sun_glare_gate_x_jitter)
            divider_x = max(divider_x, occluder_x + 0.45)
            gate_x = max(gate_x, divider_x + 0.20)
            gap_half_w = random.uniform(*self.sun_glare_gap_half_w_range)
            sun_sigma_u = random.uniform(*self.sun_glare_sun_sigma_u_range)
            sun_sigma_v = random.uniform(*self.sun_glare_sun_sigma_v_range)
            sun_y_offset = random.uniform(-self.sun_glare_sun_y_jitter, self.sun_glare_sun_y_jitter)
            sun_z_offset = random.uniform(-self.sun_glare_sun_z_jitter, self.sun_glare_sun_z_jitter)

        start = torch.tensor([-2.8, start_y, 1.5], device=self.device)
        goal = torch.tensor([3.00, 0.0, 1.5], device=self.device)
        max_speed = 1.15
        margin = self.fixed_margin
        voxels = self._build_sun_glare_voxel_layout(
            gap_y_center,
            occluder_x=occluder_x,
            occluder_half_y=occluder_half_y,
            divider_x=divider_x,
            gate_x=gate_x,
            gap_half_w=gap_half_w,
        )
        effects = {
            'hazard_center': [gate_x, gap_y_center, 1.5],
            'hazard_half_y': max(0.18, gap_half_w),
            'hazard_half_z': 1.20,
            'hazard_softness': 0.045,
            'decision_open_side': str(open_slot['side']),
            'decision_open_side_id': float(open_slot['id']),
            'decision_open_slot_name': str(open_slot['name']),
            'decision_open_slot_y': float(gap_y_center),
            'geometry_occluder_x': float(occluder_x),
            'geometry_occluder_half_y': float(occluder_half_y),
            'geometry_divider_x': float(divider_x),
            'geometry_gate_x': float(gate_x),
            'geometry_gap_half_w': float(gap_half_w),
            'geometry_start_y': float(start_y),
        }
        if sensor_regime == 'glare':
            effects.update({
                'ambient_add': 4.2,
                'active_drop': 0.72,
                'active_recover': 0.95,
                'glare_bias': 0.30,
                'glare_exposure_gain': 2.30,
                'glare_power_bias': 0.16,
                'glare_power_gain': 1.45,
                'power_rescue_bias': 0.16,
                'power_rescue_exposure_gain': 0.60,
                'power_quality_bonus': 0.78,
                'quality_penalty': 2.75,
                'valid_bias_scale': 0.16,
                'sun_anchor': [3.00, gap_y_center + sun_y_offset, 1.65 + sun_z_offset],
                'sun_y_offset': sun_y_offset,
                'sun_sigma_u': sun_sigma_u,
                'sun_sigma_v': sun_sigma_v,
            })
        effects.update(self._sensor_scene_effects(sensor_regime))
        effects = self._merge_scene_effects(scene_name, effects)
        if self.sun_glare_randomize:
            random_effects = {
                'hazard_center': [gate_x, gap_y_center, 1.5],
                'hazard_half_y': max(0.18, gap_half_w),
                'geometry_occluder_x': float(occluder_x),
                'geometry_occluder_half_y': float(occluder_half_y),
                'geometry_divider_x': float(divider_x),
                'geometry_gate_x': float(gate_x),
                'geometry_gap_half_w': float(gap_half_w),
                'geometry_start_y': float(start_y),
            }
            if sensor_regime == 'glare':
                random_effects.update({
                    'sun_anchor': [3.00, gap_y_center + sun_y_offset, 1.65 + sun_z_offset],
                    'sun_y_offset': sun_y_offset,
                    'sun_sigma_u': sun_sigma_u,
                    'sun_sigma_v': sun_sigma_v,
                })
            effects.update(random_effects)
        effects = self._apply_sun_glare_level(effects, selected_variant)
        effects = self._realign_sun_glare_effects(effects)
        return voxels, start, goal, max_speed, margin, effects, selected_variant

    def reset(self, scene_name=None, scene_variant=None):
        """重置为固定小地图任务，并按 `scenarios` 选择简化场景。"""
        B = self.batch_size
        device = self.device
        scene_name = self._choose_scene_name(scene_name)
        selected_variant = scene_variant
        selected_variant = self._choose_sun_glare_level(scene_variant)
        self._set_scene_name(scene_name, selected_variant)
        self.last_diff_depth_debug = None
        self.last_diff_depth_train_aux = None

        cam_angle = torch.full(
            (B,),
            float(self.cam_angle) * math.pi / 180.0,
            device=device,
        )
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
            zeros, ones, zeros,
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),
        ], -1).reshape(B, 3, 3)
        self._fov_x_half_tan = float(self.fov_x_half_tan)

        self.n_drones_per_group = 1
        self.drone_radius = self.fixed_drone_radius
        voxels, start, goal, max_speed, margin, effects, selected_variant = self._build_scene_geometry(
            scene_name, selected_variant)
        self._set_scene_name(scene_name, selected_variant)
        self.current_scene_effects = effects
        self.max_speed = torch.full((B, 1), max_speed, device=device)
        self.margin = torch.full((B,), margin, device=device)
        self.thr_est_error = torch.ones((B,), device=device)
        self.pitch_ctl_delay = torch.full((B, 1), self.fixed_pitch_ctl_delay, device=device)
        self.yaw_ctl_delay = torch.full((B, 1), self.fixed_yaw_ctl_delay, device=device)
        self.drag_2 = torch.zeros((B, 2), device=device)
        self.drag_2[:, 1] = self.fixed_drag_linear
        self.z_drag_coef = torch.ones((B, 1), device=device)

        self.balls = torch.empty((B, 0, 4), device=device)
        self.cyl = torch.empty((B, 0, 3), device=device)
        self.cyl_h = torch.empty((B, 0, 3), device=device)
        self.voxels = voxels.unsqueeze(0).repeat(B, 1, 1).clone()

        self.p = start.unsqueeze(0).repeat(B, 1).clone()
        self.p_target = goal.unsqueeze(0).repeat(B, 1).clone()

        self.v = torch.zeros((B, 3), device=device)
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w * self.fixed_wind_scale
        self.act = torch.zeros((B, 3), device=device)
        self.a = torch.zeros((B, 3), device=device)
        self.dg = torch.randn((B, 3), device=device) * 0.03

        R = torch.zeros((B, 3, 3), device=device)
        v_dir = F.normalize(self.p_target - self.p, 2, -1)
        self.R = quadsim_cuda.update_state_vec(
            R,
            self.act,
            v_dir,
            torch.zeros_like(self.yaw_ctl_delay),
            5,
        )
        self.R_old = self.R.clone()
        self.p_old = self.p.clone()

        self._reset_camera_states()
        self._apply_scene_sensor_profile(scene_name)
        self._sample_sensor_model_params()

    @staticmethod
    @torch.no_grad()
    def update_state_vec(R, a_thr, v_pred, alpha, yaw_inertia=5):
        """
        根据推力加速度和预测速度更新无人机的姿态矩阵 (旋转矩阵 R)。
        模拟无人机为了产生特定方向的加速度，必须倾斜机身的物理特性。
        """
        self_forward_vec = R[..., 0] # 当前机头朝向 (X轴)
        g_std = torch.tensor([0, 0, -9.80665], device=R.device)
        a_thr = a_thr - g_std # 抵消重力后的净推力加速度
        thrust = torch.norm(a_thr, 2, -1, True)
        self_up_vec = a_thr / thrust # 推力方向即为机身上方 (Z轴)
        
        # 计算新的机头朝向 (结合偏航惯性和预测速度方向)
        forward_vec = self_forward_vec * yaw_inertia + v_pred
        forward_vec = self_forward_vec * alpha + F.normalize(forward_vec, 2, -1) * (1 - alpha)
        # 确保机头方向与机身上方垂直 (正交化)
        forward_vec[:, 2] = (forward_vec[:, 0] * self_up_vec[:, 0] + forward_vec[:, 1] * self_up_vec[:, 1]) / -self_up_vec[2]
        self_forward_vec = F.normalize(forward_vec, 2, -1)
        
        # 通过叉乘计算机身左侧 (Y轴)
        self_left_vec = torch.cross(self_up_vec, self_forward_vec)
        
        # 组合成新的旋转矩阵 [X, Y, Z]
        return torch.stack([
            self_forward_vec,
            self_left_vec,
            self_up_vec,
        ], -1)

    def render(self, ctl_dt):
        """
        标准渲染函数 (不可微 FOV)。
        调用 CUDA 核心渲染当前视角的深度图。
        """
        canvas = torch.empty((self.batch_size, self.height, self.width), device=self.device)
        # R @ self.R_cam: 将相机的局部旋转叠加到无人机的机身旋转上
        quadsim_cuda.render(canvas, self.flow, self.balls, self.cyl, self.cyl_h,
                            self.voxels, self.R @ self.R_cam, self.R_old, self.p,
                            self.p_old, self.drone_radius, self.n_drones_per_group,
                            self._fov_x_half_tan)
        return canvas, None

    def render_diff_depth(self, power, exposure, gain, max_range=None):
        """
        可微主动深度相机渲染（Diff Depth Sensor）。
        说明：
        - 几何层使用 CUDA 深度渲染 `quadsim_cuda.render_depth`
        - 其后的 D455 风格传感器链使用 Torch 张量算子实现，保证不同实现后端下的物理语义一致
        - 历史上的 fused diff_depth CUDA 后处理路径保留在扩展中，但默认不再走该分支，
          以避免训练/评估在 power / exposure / gain 梯度上出现不一致
        """
        impl = self.diff_sensor_impl.get('diff_depth', 'python')
        if impl == 'python':
            return self._render_diff_depth_python(power, exposure, gain, max_range=max_range)
        if impl == 'cuda':
            return self._render_diff_depth_cuda(power, exposure, gain, max_range=max_range)
        else:
            raise ValueError(f"不支持的 diff_sensor_impl[diff_depth]={impl}，仅支持 python/cuda")

    def _diff_depth_exposure_scale(self, exposure):
        return self.cam_sem.exposure_to_time(exposure)

    def _apply_diff_depth_sensor_model(self, depth, power, exposure, gain, max_range=None):
        """
        D455 风格主动双目深度观测模型。
        目标不是逐寄存器复刻，而是把以下关键失效模式做对：
        - 激光功率 / 曝光 / 增益三者的非对称 trade-off
        - 远距离与弱纹理场景下的软退化
        - 高速运动时沿相机速度方向的拖影
        - 深度边缘 flying pixels、镜面高光失效与空洞输出
        - 环境红外（ambient IR）对主动散斑的淹没
        """
        max_range = float(self.depth_max_range if max_range is None else max_range)
        min_valid = float(self.depth_min_valid)
        depth = depth.clamp(max(min_valid * 0.1, 0.03), max(float(max_range) * 2.0, 12.0))

        power01 = power.clamp(0.0, 1.0)
        exposure_s = self._diff_depth_exposure_scale(exposure)
        gain01 = gain.clamp(0.0, 1.0)
        gain_scale = self.cam_sem.iso_to_gain(gain01).clamp_min(1.0)
        sp = self._sensor_param

        ps = power01[:, None, None]
        es = exposure_s[:, None, None]
        gs = gain_scale[:, None, None]

        depth4 = depth[:, None]
        depth_far = F.max_pool2d(depth4, 3, stride=1, padding=1)[:, 0]
        depth_near = -F.max_pool2d(-depth4, 3, stride=1, padding=1)[:, 0]

        edge = (
            sp('edge_gain') * (depth_far - depth_near) /
            (depth + sp('edge_depth_bias'))
        ).clamp(0.0, 1.5)
        frontality = torch.exp(-sp('frontality_edge_slope') * edge)
        fog_trans = torch.exp(-self._cam_fog_beta[:, None, None] * depth)

        ambient_ir = (
            0.12
            + 0.55 * self._cam_ambient[:, None, None]
            + 0.25 * self._cam_dir_intensity[:, None, None]
            + 0.18 * self._cam_airlight[:, None, None]
        ) * (1.0 + 1.5 * self._cam_fog_beta[:, None, None])
        albedo = (0.25 + 0.75 * self._cam_mat_obstacle[:, None, None]).clamp(0.1, 1.0)
        if self.cam_enable_specular:
            spec = self._cam_mat_spec[:, None, None].clamp(0.0, 1.0)
        else:
            spec = torch.zeros_like(albedo)

        if self.cam_enable_motion_blur:
            speed = self.v.norm(2, -1)
            motion = (speed[:, None, None] * es * self.cam_motion_blur_gain).clamp(0.0, 1.25)
        else:
            motion = torch.zeros_like(depth)

        R_cam_world = (self.R @ self.R_cam).contiguous()
        scene_adj = self._scene_sensor_adjustments(
            depth,
            power01,
            exposure_s,
            gain_scale,
            motion,
            R_cam_world,
        )
        ambient_ir = ambient_ir * scene_adj['ambient_mul'] + scene_adj['ambient_add']
        albedo = (albedo * scene_adj['albedo_mul']).clamp(0.01, 1.0)
        spec = (spec + scene_adj['spec_add']).clamp(0.0, 1.5)
        motion = motion * scene_adj['motion_mul']

        signal_active = (
            sp('active_signal_gain') * ps * es * albedo * frontality * fog_trans /
            (depth.square() + sp('active_signal_depth_bias'))
        )
        signal_active = signal_active.clamp_max(1e6)  # 防止 depth 极小时 Inf → NaN
        signal_active = signal_active * scene_adj['active_mul']
        signal_passive = (
            es * ambient_ir *
            (sp('passive_edge_base') + sp('passive_edge_gain') * edge) *
            (sp('passive_albedo_base') + sp('passive_albedo_gain') * albedo) *
            torch.sqrt(gs)
        )
        signal_passive = signal_passive * scene_adj['passive_mul']
        spec_bloom = spec * ps * (0.6 + 0.4 * ambient_ir) * (1.0 + edge)

        gain_boost = torch.log(gs).clamp_min(0.0)
        active_range = (
            sp('active_range_base')
            + float(max_range) * (
                sp('active_range_min_frac') +
                sp('active_range_gain_frac') * torch.sqrt((ps * es).clamp_min(1e-6))
            )
            + sp('active_range_gain_boost') * gain_boost
        )
        passive_range = (
            sp('passive_range_base') +
            float(max_range) * (
                sp('passive_range_exposure_frac') +
                sp('passive_range_ambient_frac') * es * ambient_ir
            )
        )
        active_gate = torch.sigmoid((active_range - depth) / sp('active_gate_width'))
        passive_gate = torch.sigmoid((passive_range - depth) / sp('passive_gate_width'))

        signal = signal_active * active_gate + sp('signal_passive_mix') * signal_passive * passive_gate
        washout = ambient_ir / (signal_active + sp('washout_bias'))
        snr = signal / (
            0.08
            + sp('snr_ambient_weight') * ambient_ir
            + sp('snr_gain_weight') * gs
            + sp('snr_spec_weight') * spec_bloom
            + sp('snr_motion_weight') * motion
        )
        far = torch.relu(depth / (active_range + 1e-3) - 0.9)
        quality = torch.sigmoid(
            sp('quality_snr_gain') * snr
            + sp('quality_passive_gain') * signal_passive
            - sp('quality_washout_penalty') * washout
            - sp('quality_spec_penalty') * spec_bloom
            - sp('quality_motion_penalty') * motion * edge
            - sp('quality_far_penalty') * far
        )
        quality = (quality + scene_adj['quality_add']).clamp(0.0, 1.0)

        v_cam = torch.einsum('bij,bj->bi', R_cam_world.transpose(1, 2), self.v)
        motion_h = v_cam[:, 1].abs()
        motion_v = v_cam[:, 2].abs()
        motion_sum = (motion_h + motion_v).clamp_min(1e-6)
        w_h = (motion_h / motion_sum)[:, None, None]
        w_v = (motion_v / motion_sum)[:, None, None]

        pad = F.pad(depth4, (1, 1, 1, 1), mode='replicate')
        left = pad[:, 0, 1:-1, :-2]
        right = pad[:, 0, 1:-1, 2:]
        up = pad[:, 0, :-2, 1:-1]
        down = pad[:, 0, 2:, 1:-1]
        blur_h = 0.25 * (left + 2.0 * depth + right)
        blur_v = 0.25 * (up + 2.0 * depth + down)
        directional_blur = blur_h * w_h + blur_v * w_v

        if self.cam_enable_motion_blur:
            motion_blend = (sp('motion_blend_gain') * motion).clamp(0.0, sp('motion_blend_max'))
            depth_blur = depth * (1.0 - motion_blend) + directional_blur * motion_blend
        else:
            depth_blur = depth

        flying = (
            sp('flying_base') +
            sp('flying_quality_gain') * (1.0 - quality)
        ) * edge * (
            sp('flying_motion_base') +
            sp('flying_motion_gain') * (motion + spec_bloom).clamp(0.0, 1.5)
        )
        flying = flying.clamp(0.0, 1.0)
        depth_corrupt = depth_blur + flying * (depth_far - depth_blur)

        range_ratio = (depth / max(float(max_range), 1e-6)).clamp(0.0, 1.5)
        shot_noise_scale = float(self.cam_sem.shot_noise_base / 0.03)
        noise_floor = self.cam_read_noise * (1.0 + sp('noise_floor_gain_weight') * gs)
        noise_signal = (
            sp('noise_signal_gain') * shot_noise_scale *
            (1.0 + sp('noise_signal_range_gain') * range_ratio.square()) /
            (signal + sp('noise_signal_bias'))
        )
        noise_motion = sp('noise_motion_gain') * motion * (
            sp('noise_motion_edge_base') + sp('noise_motion_edge_gain') * edge
        )
        noise_spec = sp('noise_spec_gain') * spec_bloom
        noise_std = (noise_floor + noise_signal + noise_motion + noise_spec).clamp(
            sp('noise_std_min'),
            sp('noise_std_max'),
        )

        noisy_depth = depth_corrupt + torch.randn_like(depth_corrupt) * noise_std
        far_override = scene_adj['far_override'].clamp(0.0, 1.0)
        noisy_depth = torch.lerp(
            noisy_depth,
            torch.full_like(noisy_depth, float(max_range)),
            far_override,
        )
        noisy_depth = noisy_depth.clamp(min_valid, float(max_range))
        valid_threshold = sp('valid_threshold_base') + scene_adj['valid_bias'].clamp(0.0, sp('valid_bias_max'))
        valid = torch.sigmoid((quality - valid_threshold) / sp('valid_sigmoid_width'))
        noisy_depth = noisy_depth * valid
        quality = quality * valid
        scene_mask = scene_adj.get('debug_scene_mask', torch.zeros_like(depth))
        scene_effect_map = scene_adj.get('debug_effect_map', torch.zeros_like(depth))
        invalid_mask = (1.0 - valid).clamp(0.0, 1.0)
        debug_scalars = dict(scene_adj.get('debug_scalars', {}))
        train_aux = {}
        if self.current_scene_name in self.supported_scenarios:
            glare_mass = scene_mask.sum(dim=(-2, -1)).clamp_min(1e-6)
            glare_quality = (quality * scene_mask).sum(dim=(-2, -1)) / glare_mass
            glare_invalid = (invalid_mask * scene_mask).sum(dim=(-2, -1)) / glare_mass
            debug_scalars['glare_quality_mean'] = glare_quality.detach()
            debug_scalars['glare_invalid_rate'] = glare_invalid.detach()
        debug_scalars.update({
            **{f'sensor/{k}': float(v) for k, v in self._sensor_model_params.items()},
            'quality_mean': self._spatial_mean(quality),
            'invalid_rate': self._spatial_mean(invalid_mask),
            'ambient_ir_mean': self._spatial_mean(ambient_ir),
            'signal_active_mean': self._spatial_mean(signal_active),
            'signal_passive_mean': self._spatial_mean(signal_passive),
            'spec_bloom_mean': self._spatial_mean(spec_bloom),
            'motion_blur_mean': self._spatial_mean(motion),
            'washout_mean': self._spatial_mean(washout),
            'far_override_mean': self._spatial_mean(far_override),
        })
        self._store_last_diff_depth_train_aux(train_aux)
        self._store_last_diff_depth_debug({
            'scene_name': self.current_scene_name,
            'quality_map': quality,
            'invalid_mask': invalid_mask,
            'scene_effect_map': scene_effect_map,
            'scene_mask': scene_mask,
            'scalars': debug_scalars,
        })
        return noisy_depth, quality

    def _render_diff_depth_python(self, power, exposure, gain, max_range=None):
        """
        可微主动深度相机渲染（Diff Depth Sensor）。
        输入:
            power: 激光发射功率 [0, 1] 标量 (内部可缩放至物理数值如 0~360)
            exposure: 曝光时间 [0, 1] 标量 (控制运动模糊，反比于速率限制)
            gain: 接收增益 [0, 1] 标量 (在暗处放大信号，但增加噪声)
        输出:
            depth_obs: 包含可微噪声的深度图 (B, H_depth, W_depth)
        """
        B = power.shape[0]
        device = power.device
        
        # 1. 基础几何渲染
        R_cam_world = (self.R @ self.R_cam).contiguous()
        pos = self.p.contiguous()
        depth = torch.empty((B, self.height, self.width), device=device, dtype=power.dtype)
        quadsim_cuda.render_depth(
            depth,
            self.balls,
            self.cyl,
            self.cyl_h,
            self.voxels,
            R_cam_world,
            pos,
            self.n_drones_per_group,
            float(self._fov_x_half_tan),
        )
        noisy_depth, quality = self._apply_diff_depth_sensor_model(
            depth,
            power,
            exposure,
            gain,
            max_range=max_range,
        )
        return noisy_depth, quality

    def _render_diff_depth_cuda(self, power, exposure, gain, max_range=None):
        """可微主动深度相机渲染（CUDA backend）。"""
        B = power.shape[0]
        device = power.device
        max_range = float(self.depth_max_range if max_range is None else max_range)

        R_cam_world = (self.R @ self.R_cam).contiguous()
        pos = self.p.contiguous()

        noisy_depth, _ = diff_depth(
            self._fov_x_half_tan,
            power,
            exposure,
            gain,
            self.v.contiguous(),
            R_cam_world,
            pos,
            self.balls,
            self.cyl,
            self.cyl_h,
            self.voxels,
            self.n_drones_per_group,
            self.height,
            self.width,
            float(max_range),
        )
        self.last_diff_depth_debug = None
        return noisy_depth, None  # CUDA path does not expose quality separately

    def find_vec_to_nearest_pt(self):
        """
        寻找无人机到最近障碍物表面的向量。
        用于计算避障惩罚 (Obstacle Avoidance Loss) 和碰撞检测。
        """
        # 预测未来一小段时间内的轨迹点 (用于连续碰撞检测)
        p = self.p + self.v * self.sub_div
        nearest_pt = torch.empty_like(p)
        if self.use_ellipsoid:
            # 使用椭球体模型进行更精确的碰撞检测 (考虑无人机的姿态)
            quadsim_cuda.find_nearest_pt_ellipsoid(
                nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p,
                self.R.contiguous(), self.drone_radius, self.n_drones_per_group,
                self.ellipsoid_a, self.ellipsoid_c)
        else:
            # 使用简单的球体模型进行碰撞检测
            quadsim_cuda.find_nearest_pt(
                nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p,
                self.drone_radius, self.n_drones_per_group)
        return nearest_pt - p

    def run(self, act_pred, ctl_dt=1/15, v_pred=None):
        """
        执行一个物理控制步 (带梯度回传)。
        Args:
            act_pred: 策略网络输出的预测动作 (目标推力加速度)
            ctl_dt: 控制步长 (默认 1/15 秒)
            v_pred: 预测速度方向 (用于姿态更新)
        """
        # 更新环境扰动 (风阻/重力噪声)，使用一阶马尔可夫过程保持时间连续性
        self.dg = self.dg * math.sqrt(1 - ctl_dt / 4) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        self.p_old = self.p
        
        # 调用自定义的 RunFunction (封装了 CUDA 前向和反向传播)
        self.act, self.p, self.v, self.a = run(
            self.R, self.dg, self.z_drag_coef, self.drag_2, self.pitch_ctl_delay,
            act_pred, self.act, self.p, self.v, self.v_wind, self.a,
            self.grad_decay, ctl_dt, 0.5)
            
        # 更新无人机姿态 (考虑偏航控制延迟)
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)

    def save_state(self):
        """
        保存当前环境的完整状态快照。
        用于教师-学生训练的内部优化循环 (Inner Loop)。
        在教师网络进行多次梯度下降寻找最优轨迹时，需要反复重置到同一个初始状态。
        """
        return {
            'p': self.p.clone(),
            'v': self.v.clone(),
            'a': self.a.clone(),
            'act': self.act.clone(),
            'R': self.R.clone(),
            'R_old': self.R_old.clone(),
            'p_old': self.p_old.clone() if isinstance(self.p_old, torch.Tensor) else self.p_old,
            'dg': self.dg.clone(),
            'v_wind': self.v_wind.clone(),
        }

    def restore_state(self, snapshot):
        """
        从快照中恢复环境状态。
        用于教师-学生训练的内部优化循环。
        """
        self.p = snapshot['p'].clone()
        self.v = snapshot['v'].clone()
        self.a = snapshot['a'].clone()
        self.act = snapshot['act'].clone()
        self.R = snapshot['R'].clone()
        self.R_old = snapshot['R_old'].clone()
        self.p_old = snapshot['p_old'].clone() if isinstance(snapshot['p_old'], torch.Tensor) else snapshot['p_old']
        self.dg = snapshot['dg'].clone()
        self.v_wind = snapshot['v_wind'].clone()

    def _run(self, act_pred, ctl_dt=1/15, v_pred=None):
        """
        纯 PyTorch 实现的物理步进函数 (不使用 CUDA 扩展)。
        主要用于调试、验证 CUDA 实现的正确性，或者在不支持 CUDA 的环境下运行。
        包含了与 CUDA 版本相同的物理逻辑：控制延迟、空气阻力、运动学积分等。
        """
        # 1. 模拟底层飞控的俯仰/滚转延迟 (一阶低通滤波)
        alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
        self.act = act_pred * (1 - alpha) + self.act * alpha
        
        # 2. 更新环境扰动
        self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
        
        # 3. 计算空气阻力
        z_drag = 0
        if self.z_drag_coef is not None:
            # Z轴方向的诱导阻力 (与旋翼转速和垂直速度有关)
            v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
            v_prep = self.v - v_up
            motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
            z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
            
        # 二次空气阻力 (与速度平方成正比)
        drag = self.drag_2 * self.v * self.v.norm(2, -1, True)
        
        # 4. 计算净加速度
        a_next = self.act + self.dg - z_drag - drag
        
        # 5. 运动学积分 (使用 g_decay 缓解长序列梯度爆炸)
        self.p_old = self.p
        self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
        self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
        self.a = a_next

        # 6. 更新姿态 (考虑偏航控制延迟)
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)

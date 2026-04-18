import math
import json
import os
import random
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
    diff_depth-only 的固定小地图场景环境。

    当前版本保留 `scenarios` 作为“可微感知能力测试”开关，但所有场景都共享：
    地图语义如下：
    - 地图中心: (0, 0, 0)
    - 地图范围: x/y 均为 [-5, 5]
    - 起点: (-5, 0, 1.5)
    - 终点: (5, 0, 1.5)
    - 几何主骨架固定且小型，不要求复杂避障

    当前版本的设计目标是：
    - 让飞行/避障任务足够简单
    - 让 `sun_glare` / `specular_trap` / `vantablack_gap` / `dark_morphing`
      这些论文场景在仿真中对应到局部光照、材质与几何事件
    - 避免回退到旧的大地图随机世界
    """
    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 cam_angle=10, ellipsoid_a=0.0, ellipsoid_c=0.0,
                 camera_preset='high',
                 cam_enable_specular=True,
                 cam_enable_motion_blur=True,
                 cam_noise_scale=1.0,
                 cam_blur_scale=1.0,
                 cam_fog_scale=1.0,
                 cam_lighting_scale=1.0,
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
        self.fixed_max_speed = 1.8
        self.fixed_drone_radius = 0.12
        self.fixed_margin = 0.05
        self.fixed_pitch_ctl_delay = 12.0
        self.fixed_yaw_ctl_delay = 6.0
        self.fixed_drag_linear = 0.35
        self.fixed_wind_scale = 0.03
        self.supported_scenarios = (
            'base',
            'sun_glare',
            'specular_trap',
            'vantablack_gap',
            'dark_morphing',
        )
        self.scene_name_to_id = {name: idx for idx, name in enumerate(self.supported_scenarios)}
        self.scenarios = self._normalize_scenarios(scenarios)
        self.current_scene_name = self.scenarios[0]
        self.current_scene_id = self.scene_name_to_id[self.current_scene_name]
        self.current_scene_has_opening = False
        self.current_scene_effects = {}
        self.scene_fit_profiles_path = None
        self.scene_sensor_profile_overrides = {}
        self.scene_effect_overrides = {}
        self.base_voxels_template = self._build_base_voxel_layout()

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

    def _normalize_scenarios(self, scenarios):
        if scenarios is None:
            return ['base']
        out = []
        aliases = {
            'random_base': 'base',
            'random': 'base',
            'random_scene': 'base',
            'black_gap': 'vantablack_gap',
            'dark_slit_lite': 'dark_morphing',
        }
        for raw in scenarios:
            if raw is None:
                continue
            for token in str(raw).split(','):
                name = aliases.get(token.strip().lower(), token.strip().lower())
                if not name:
                    continue
                if name not in self.supported_scenarios:
                    raise ValueError(
                        f"不支持的场景 '{name}'，仅支持: {list(self.supported_scenarios)}"
                    )
                if name not in out:
                    out.append(name)
        return out or ['base']

    def _canonical_scene_name(self, name):
        aliases = {
            'random_base': 'base',
            'random': 'base',
            'random_scene': 'base',
            'black_gap': 'vantablack_gap',
            'dark_slit_lite': 'dark_morphing',
        }
        return aliases.get(str(name).strip().lower(), str(name).strip().lower())

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

    def _choose_scene_name(self, scene_name=None):
        if scene_name is not None:
            name = str(scene_name).strip().lower()
            if name not in self.supported_scenarios:
                raise ValueError(
                    f"reset(scene_name={scene_name!r}) 不支持；仅支持 {list(self.supported_scenarios)}"
                )
            return name
        return random.choice(self.scenarios)

    def _set_scene_name(self, scene_name):
        self.current_scene_name = str(scene_name)
        self.current_scene_id = self.scene_name_to_id[self.current_scene_name]
        self.current_scene_has_opening = self.current_scene_name in {'vantablack_gap', 'dark_morphing'}

    def _sample_scene_tensor(self, lo, hi):
        return torch.empty((self.batch_size,), device=self.device).uniform_(float(lo), float(hi))

    def _apply_scene_sensor_profile(self, scene_name):
        if scene_name == 'sun_glare':
            self._cam_ambient = self._sample_scene_profile(scene_name, 'cam_ambient', 0.10, 0.18)
            self._cam_dir_intensity = self._sample_scene_profile(scene_name, 'cam_dir_intensity', 0.35, 0.75)
            self._cam_fog_beta = self._sample_scene_profile(scene_name, 'cam_fog_beta', 0.010, 0.030)
            self._cam_airlight = self._sample_scene_profile(scene_name, 'cam_airlight', 0.12, 0.25)
            self._cam_mat_obstacle = self._sample_scene_profile(scene_name, 'cam_mat_obstacle', 0.52, 0.78)
            self._cam_mat_spec = self._sample_scene_profile(scene_name, 'cam_mat_spec', 0.04, 0.10)
        elif scene_name == 'specular_trap':
            self._cam_ambient = self._sample_scene_profile(scene_name, 'cam_ambient', 0.08, 0.16)
            self._cam_dir_intensity = self._sample_scene_profile(scene_name, 'cam_dir_intensity', 0.18, 0.42)
            self._cam_fog_beta = self._sample_scene_profile(scene_name, 'cam_fog_beta', 0.006, 0.018)
            self._cam_airlight = self._sample_scene_profile(scene_name, 'cam_airlight', 0.05, 0.12)
            self._cam_mat_obstacle = self._sample_scene_profile(scene_name, 'cam_mat_obstacle', 0.45, 0.72)
            self._cam_mat_spec = self._sample_scene_profile(scene_name, 'cam_mat_spec', 0.18, 0.38)
        elif scene_name == 'vantablack_gap':
            self._cam_ambient = self._sample_scene_profile(scene_name, 'cam_ambient', 0.02, 0.06)
            self._cam_dir_intensity = self._sample_scene_profile(scene_name, 'cam_dir_intensity', 0.05, 0.16)
            self._cam_fog_beta = self._sample_scene_profile(scene_name, 'cam_fog_beta', 0.003, 0.015)
            self._cam_airlight = self._sample_scene_profile(scene_name, 'cam_airlight', 0.02, 0.08)
            self._cam_mat_obstacle = self._sample_scene_profile(scene_name, 'cam_mat_obstacle', 0.30, 0.48)
            self._cam_mat_spec = self._sample_scene_profile(scene_name, 'cam_mat_spec', 0.00, 0.02)
        elif scene_name == 'dark_morphing':
            self._cam_ambient = self._sample_scene_profile(scene_name, 'cam_ambient', 0.006, 0.020)
            self._cam_dir_intensity = self._sample_scene_profile(scene_name, 'cam_dir_intensity', 0.015, 0.070)
            self._cam_fog_beta = self._sample_scene_profile(scene_name, 'cam_fog_beta', 0.004, 0.020)
            self._cam_airlight = self._sample_scene_profile(scene_name, 'cam_airlight', 0.005, 0.020)
            self._cam_mat_obstacle = self._sample_scene_profile(scene_name, 'cam_mat_obstacle', 0.22, 0.38)
            self._cam_mat_spec = self._sample_scene_profile(scene_name, 'cam_mat_spec', 0.00, 0.01)
        elif scene_name == 'base':
            self._cam_ambient = self._sample_scene_profile(scene_name, 'cam_ambient', self.cam_ambient_min, self.cam_ambient_max)
            self._cam_dir_intensity = self._sample_scene_profile(scene_name, 'cam_dir_intensity', self.cam_dir_min, self.cam_dir_max)
            self._cam_fog_beta = self._sample_scene_profile(scene_name, 'cam_fog_beta', self.cam_fog_beta_min, self.cam_fog_beta_max)
            self._cam_airlight = self._sample_scene_profile(scene_name, 'cam_airlight', self.cam_airlight_min, self.cam_airlight_max)
            self._cam_mat_obstacle = self._sample_scene_profile(scene_name, 'cam_mat_obstacle', 0.45, 0.85)
            self._cam_mat_spec = self._sample_scene_profile(scene_name, 'cam_mat_spec', 0.02, 0.18)

    def _build_base_voxels_layout(self):
        return self._build_base_voxel_layout()

    def _build_base_voxel_layout(self):
        """
        基础小地图：6 个等尺寸、左右交替的高柱体。

        设计目标是保持“简洁固定柱体地图”的同时，让从 (-5, 0) 到 (5, 0)
        的直线路径被连续切断，迫使无人机在左右两侧交替绕行，形成明显
        的 S 型避障轨迹。
        """
        voxel_half_w = 0.25
        voxel_half_h = 1.5
        layout = torch.tensor([
            [-3.80,  0.10, 1.5, voxel_half_w, voxel_half_w, voxel_half_h],
            [-2.20, -0.80, 1.5, voxel_half_w, voxel_half_w, voxel_half_h],
            [-0.60,  0.50, 1.5, voxel_half_w, voxel_half_w, voxel_half_h],
            [ 1.00, -0.80, 1.5, voxel_half_w, voxel_half_w, voxel_half_h],
            [ 2.60,  0.50, 1.5, voxel_half_w, voxel_half_w, voxel_half_h],
            [ 4.20, -0.50, 1.5, voxel_half_w, voxel_half_w, voxel_half_h],
        ], device=self.device)
        return layout

    def _build_voxels(self, rows):
        if not rows:
            return torch.empty((0, 6), device=self.device)
        return torch.tensor(rows, device=self.device, dtype=torch.float32)

    def _build_opening_wall(self, wall_x, gap_y_center, gap_z_center, gap_half_w, gap_half_h):
        big = 4.0
        wall_thickness = 0.15
        device = self.device
        return torch.tensor([
            [wall_x, gap_y_center - gap_half_w - big, gap_z_center, wall_thickness, big, big],
            [wall_x, gap_y_center + gap_half_w + big, gap_z_center, wall_thickness, big, big],
            [wall_x, gap_y_center, gap_z_center + gap_half_h + big, wall_thickness, gap_half_w, big],
            [wall_x, gap_y_center, gap_z_center - gap_half_h - big, wall_thickness, gap_half_w, big],
        ], device=device)

    def _build_sun_glare_voxel_layout(self):
        wall_t = 0.18
        wall_h = 1.5
        rows = [
            [-1.45, -1.55, 1.5, 3.10, wall_t, wall_h],
            [-1.45,  1.55, 1.5, 3.10, wall_t, wall_h],
            [ 2.10, -1.55, 1.5, 1.15, wall_t, wall_h],
            [ 2.10,  1.55, 1.5, 1.15, wall_t, wall_h],
            [ 3.45, -1.08, 1.5, 0.12, 0.46, wall_h],
            [ 3.45,  1.08, 1.5, 0.12, 0.46, wall_h],
            [ 3.45,  0.00, 2.55, 0.12, 0.62, 0.35],
        ]
        return self._build_voxels(rows)

    def _build_specular_trap_layout(self):
        rows = [
            [-2.8,  0.95, 1.5, 0.25, 0.25, 1.5],
            [-1.4, -0.95, 1.5, 0.25, 0.25, 1.5],
            [ 0.0,  0.00, 1.45, 0.05, 0.95, 1.15],
            [ 1.8,  1.00, 1.5, 0.25, 0.25, 1.5],
            [ 3.2, -1.00, 1.5, 0.25, 0.25, 1.5],
        ]
        return self._build_voxels(rows)

    def _project_world_point(self, point, R_cam_world, dtype):
        point_t = torch.as_tensor(point, device=self.device, dtype=dtype)
        if point_t.ndim == 1:
            point_t = point_t.unsqueeze(0).expand(self.batch_size, -1)
        elif point_t.ndim == 2 and point_t.shape[0] == 1:
            point_t = point_t.expand(self.batch_size, -1)
        rel = point_t - self.p.to(dtype)
        cam = torch.einsum('bij,bj->bi', R_cam_world.transpose(1, 2), rel)
        cam_x = cam[:, 0]
        fov_x_half = max(float(self._fov_x_half_tan), 1e-4)
        fov_y_half = max(fov_x_half * float(self.height) / max(float(self.width), 1.0), 1e-4)
        denom = cam_x.abs().clamp_min(1e-4)
        u = (cam[:, 1] / (denom * fov_x_half)).clamp(-4.0, 4.0)
        v = (cam[:, 2] / (denom * fov_y_half)).clamp(-4.0, 4.0)
        visible = (cam_x > 0.05).to(dtype)
        return u, v, cam_x, visible

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
        }

        fx = self.current_scene_effects or {}
        scene_name = self.current_scene_name

        if scene_name == 'sun_glare':
            center_u, center_v, _, visible = self._project_world_point(
                fx['sun_anchor'], R_cam_world, dtype)
            sun_mask = self._make_gaussian_mask(
                center_u, center_v,
                fx.get('sun_sigma_u', 0.30),
                fx.get('sun_sigma_v', 0.24),
                dtype,
            )
            zone_gate = torch.sigmoid(
                (self.p[:, 0].to(dtype) - float(fx.get('zone_enter_x', 1.8))) /
                float(fx.get('zone_softness', 0.35))
            )[:, None, None]
            strength = sun_mask * zone_gate * visible[:, None, None]
            glare_penalty = strength * (0.30 + 2.10 * exposure_s[:, None, None]) / (
                0.35 + 0.95 * power01[:, None, None]
            )
            adj['ambient_add'] = adj['ambient_add'] + strength * float(fx.get('ambient_add', 2.2))
            adj['active_mul'] = adj['active_mul'] * (1.0 - float(fx.get('active_drop', 0.50)) * strength)
            adj['quality_add'] = adj['quality_add'] - float(fx.get('quality_penalty', 1.8)) * glare_penalty
            adj['valid_bias'] = adj['valid_bias'] + float(fx.get('valid_bias_scale', 0.08)) * strength

        elif scene_name == 'specular_trap':
            center_u, center_v, cam_x, visible = self._project_world_point(
                fx['panel_center'], R_cam_world, dtype)
            half_u, half_v = self._project_rect_half_extents(
                cam_x,
                fx.get('panel_half_y', 0.95),
                fx.get('panel_half_z', 1.15),
                dtype,
            )
            panel_mask = self._make_box_mask(center_u, center_v, half_u, half_v, 0.06, dtype)
            x_gate = torch.sigmoid(
                (float(fx.get('interaction_radius_x', 1.5)) - (self.p[:, 0].to(dtype) - float(fx['panel_center'][0])).abs()) /
                0.25
            )[:, None, None]
            panel_mask = panel_mask * x_gate * visible[:, None, None]
            laser_harm = power01[:, None, None].pow(1.35)
            passive_rescue = (
                (1.0 - power01[:, None, None]) *
                exposure_s[:, None, None] *
                torch.sqrt(gain_scale[:, None, None])
            ).clamp(0.0, 1.5)
            spec_boost = (
                float(fx.get('spec_boost_base', 0.45)) +
                float(fx.get('spec_boost_scale', 0.90)) * laser_harm
            )
            adj['spec_add'] = adj['spec_add'] + panel_mask * spec_boost
            adj['passive_mul'] = adj['passive_mul'] * (
                1.0 + float(fx.get('passive_rescue_scale', 0.35)) * panel_mask * passive_rescue
            )
            adj['quality_add'] = adj['quality_add'] - float(fx.get('quality_penalty', 1.85)) * panel_mask * laser_harm
            adj['far_override'] = torch.maximum(
                adj['far_override'],
                (
                    float(fx.get('far_override_scale', 0.92)) *
                    panel_mask *
                    laser_harm *
                    (1.0 - float(fx.get('far_override_rescue_discount', 0.45)) * passive_rescue)
                ).clamp(0.0, 1.0),
            )
            adj['valid_bias'] = adj['valid_bias'] + float(fx.get('valid_bias_scale', 0.25)) * panel_mask * laser_harm

        elif scene_name == 'vantablack_gap':
            center_u, center_v, cam_x, visible = self._project_world_point(
                fx['gap_center'], R_cam_world, dtype)
            gap_half_u, gap_half_v = self._project_rect_half_extents(
                cam_x,
                fx.get('gap_half_w', 0.55),
                fx.get('gap_half_h', 0.95),
                dtype,
            )
            outer = self._make_box_mask(center_u, center_v, gap_half_u * 1.55, gap_half_v * 1.20, 0.07, dtype)
            inner = self._make_box_mask(center_u, center_v, gap_half_u * 0.72, gap_half_v * 0.72, 0.07, dtype)
            frame_mask = (outer - inner).clamp(0.0, 1.0) * visible[:, None, None]
            adj['albedo_mul'] = adj['albedo_mul'] * (1.0 - float(fx.get('albedo_drop', 0.88)) * frame_mask)
            adj['ambient_mul'] = adj['ambient_mul'] * (1.0 - float(fx.get('ambient_drop', 0.45)) * frame_mask)
            adj['passive_mul'] = adj['passive_mul'] * (1.0 - float(fx.get('passive_drop', 0.72)) * frame_mask)
            adj['motion_mul'] = adj['motion_mul'] * (1.0 + float(fx.get('motion_boost', 1.10)) * frame_mask)
            adj['quality_add'] = adj['quality_add'] - float(fx.get('quality_penalty', 0.30)) * frame_mask * exposure_s[:, None, None]

        elif scene_name == 'dark_morphing':
            center_u, center_v, cam_x, visible = self._project_world_point(
                fx['slit_center'], R_cam_world, dtype)
            slit_half_u, slit_half_v = self._project_rect_half_extents(
                cam_x,
                fx.get('gap_half_w', 0.32),
                fx.get('gap_half_h', 0.88),
                dtype,
            )
            outer = self._make_box_mask(center_u, center_v, slit_half_u * 1.70, slit_half_v * 1.25, 0.06, dtype)
            inner = self._make_box_mask(center_u, center_v, slit_half_u * 0.78, slit_half_v * 0.76, 0.06, dtype)
            frame_mask = (outer - inner).clamp(0.0, 1.0) * visible[:, None, None]
            slit_mask = inner * visible[:, None, None]
            adj['ambient_mul'] = adj['ambient_mul'] * float(fx.get('ambient_global_mul', 0.40))
            adj['albedo_mul'] = adj['albedo_mul'] * (1.0 - float(fx.get('albedo_drop', 0.78)) * frame_mask)
            adj['passive_mul'] = adj['passive_mul'] * (1.0 - float(fx.get('passive_drop', 0.82)) * frame_mask)
            motion_mix = frame_mask + float(fx.get('slit_motion_mix', 0.30)) * slit_mask
            adj['motion_mul'] = adj['motion_mul'] * (1.0 + float(fx.get('motion_boost', 1.45)) * motion_mix)
            adj['quality_add'] = adj['quality_add'] - float(fx.get('quality_penalty', 0.42)) * frame_mask * exposure_s[:, None, None]
            adj['quality_add'] = adj['quality_add'] + float(fx.get('slit_power_bonus', 0.18)) * slit_mask * power01[:, None, None]

        return adj

    def _build_scene_geometry(self, scene_name):
        start = self.start_position
        goal = self.goal_position
        max_speed = self.fixed_max_speed
        margin = self.fixed_margin
        effects = {}

        if scene_name == 'base':
            voxels = self.base_voxels_template
        elif scene_name == 'sun_glare':
            voxels = self._build_sun_glare_voxel_layout()
            effects = {
                'sun_anchor': [7.2, 0.0, 1.8],
                'zone_enter_x': 1.8,
                'zone_softness': 0.35,
                'sun_sigma_u': 0.34,
                'sun_sigma_v': 0.26,
                'ambient_add': 2.4,
                'active_drop': 0.50,
                'quality_penalty': 1.9,
                'valid_bias_scale': 0.08,
            }
        elif scene_name == 'specular_trap':
            voxels = self._build_specular_trap_layout()
            effects = {
                'panel_center': [0.0, 0.0, 1.45],
                'panel_half_y': 0.95,
                'panel_half_z': 1.15,
                'interaction_radius_x': 1.6,
                'spec_boost_base': 0.45,
                'spec_boost_scale': 0.90,
                'passive_rescue_scale': 0.35,
                'quality_penalty': 1.85,
                'far_override_scale': 0.92,
                'far_override_rescue_discount': 0.45,
                'valid_bias_scale': 0.25,
            }
        elif scene_name == 'vantablack_gap':
            voxels = self._build_opening_wall(
                wall_x=0.0,
                gap_y_center=0.85,
                gap_z_center=1.5,
                gap_half_w=0.58,
                gap_half_h=0.95,
            )
            max_speed = 1.35
            effects = {
                'gap_center': [0.0, 0.85, 1.5],
                'gap_half_w': 0.58,
                'gap_half_h': 0.95,
                'albedo_drop': 0.88,
                'ambient_drop': 0.45,
                'passive_drop': 0.72,
                'motion_boost': 1.10,
                'quality_penalty': 0.30,
            }
        elif scene_name == 'dark_morphing':
            voxels = self._build_opening_wall(
                wall_x=0.0,
                gap_y_center=-0.80,
                gap_z_center=1.5,
                gap_half_w=0.32,
                gap_half_h=0.88,
            )
            max_speed = 0.95
            margin = 0.03
            effects = {
                'slit_center': [0.0, -0.80, 1.5],
                'gap_half_w': 0.32,
                'gap_half_h': 0.88,
                'ambient_global_mul': 0.40,
                'albedo_drop': 0.78,
                'passive_drop': 0.82,
                'motion_boost': 1.45,
                'quality_penalty': 0.42,
                'slit_motion_mix': 0.30,
                'slit_power_bonus': 0.18,
            }
        else:
            raise ValueError(f'未知场景: {scene_name}')

        effects = self._merge_scene_effects(scene_name, effects)
        return voxels, start, goal, max_speed, margin, effects

    def reset(self, scene_name=None):
        """重置为固定小地图任务，并按 `scenarios` 选择简化场景。"""
        B = self.batch_size
        device = self.device
        scene_name = self._choose_scene_name(scene_name)
        self._set_scene_name(scene_name)

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
        voxels, start, goal, max_speed, margin, effects = self._build_scene_geometry(scene_name)
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

        ps = power01[:, None, None]
        es = exposure_s[:, None, None]
        gs = gain_scale[:, None, None]

        depth4 = depth[:, None]
        depth_far = F.max_pool2d(depth4, 3, stride=1, padding=1)[:, 0]
        depth_near = -F.max_pool2d(-depth4, 3, stride=1, padding=1)[:, 0]

        edge = ((depth_far - depth_near) / (depth + 0.15)).clamp(0.0, 1.5)
        frontality = torch.exp(-1.2 * edge)
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

        signal_active = 5.0 * ps * es * albedo * frontality * fog_trans / (depth.square() + 0.08)
        signal_active = signal_active.clamp_max(1e6)  # 防止 depth 极小时 Inf → NaN
        signal_active = signal_active * scene_adj['active_mul']
        signal_passive = (
            es * ambient_ir * (0.15 + 0.85 * edge) * (0.35 + 0.65 * albedo) * torch.sqrt(gs)
        )
        signal_passive = signal_passive * scene_adj['passive_mul']
        spec_bloom = spec * ps * (0.6 + 0.4 * ambient_ir) * (1.0 + edge)

        gain_boost = torch.log(gs).clamp_min(0.0)
        active_range = (
            0.9
            + float(max_range) * (0.15 + 0.85 * torch.sqrt((ps * es).clamp_min(1e-6)))
            + 0.35 * gain_boost
        )
        passive_range = 0.7 + float(max_range) * (0.08 + 0.18 * es * ambient_ir)
        active_gate = torch.sigmoid((active_range - depth) / 0.22)
        passive_gate = torch.sigmoid((passive_range - depth) / 0.28)

        signal = signal_active * active_gate + 0.45 * signal_passive * passive_gate
        washout = ambient_ir / (signal_active + 0.12)
        snr = signal / (0.08 + 0.45 * ambient_ir + 0.12 * gs + 0.35 * spec_bloom + 0.25 * motion)
        far = torch.relu(depth / (active_range + 1e-3) - 0.9)
        quality = torch.sigmoid(
            2.6 * snr
            + 0.9 * signal_passive
            - 1.4 * washout
            - 2.0 * spec_bloom
            - 1.6 * motion * edge
            - 2.2 * far
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
            motion_blend = (0.55 * motion).clamp(0.0, 0.85)
            depth_blur = depth * (1.0 - motion_blend) + directional_blur * motion_blend
        else:
            depth_blur = depth

        flying = (0.12 + 0.88 * (1.0 - quality)) * edge * (0.35 + 0.65 * (motion + spec_bloom).clamp(0.0, 1.5))
        flying = flying.clamp(0.0, 1.0)
        depth_corrupt = depth_blur + flying * (depth_far - depth_blur)

        range_ratio = (depth / max(float(max_range), 1e-6)).clamp(0.0, 1.5)
        shot_noise_scale = float(self.cam_sem.shot_noise_base / 0.03)
        noise_floor = self.cam_read_noise * (1.0 + 0.18 * gs)
        noise_signal = 0.018 * shot_noise_scale * (1.0 + 0.8 * range_ratio.square()) / (signal + 0.08)
        noise_motion = 0.03 * motion * (0.3 + 0.7 * edge)
        noise_spec = 0.05 * spec_bloom
        noise_std = (noise_floor + noise_signal + noise_motion + noise_spec).clamp(0.002, 0.75)

        noisy_depth = depth_corrupt + torch.randn_like(depth_corrupt) * noise_std
        far_override = scene_adj['far_override'].clamp(0.0, 1.0)
        noisy_depth = torch.lerp(
            noisy_depth,
            torch.full_like(noisy_depth, float(max_range)),
            far_override,
        )
        noisy_depth = noisy_depth.clamp(min_valid, float(max_range))
        valid_threshold = 0.45 + scene_adj['valid_bias'].clamp(0.0, 0.35)
        valid = torch.sigmoid((quality - valid_threshold) / 0.08)
        noisy_depth = noisy_depth * valid
        quality = quality * valid
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

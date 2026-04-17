import math
import random
import time
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
    无人机物理仿真环境类。
    负责管理无人机状态、障碍物生成、碰撞检测、可微渲染以及物理步进。
    支持大规模并行仿真 (Batch processing)。

    当前分支为 diff_depth-only：
    - 仅维护可微深度相机主链
    - 非 diff_depth 的公共渲染接口会直接报错
    """
    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 single=False, gate=False, ground_voxels=False, scaffold=False, speed_mtp=1,
                 random_rotation=False, cam_angle=10,
                 wall_slit=False, ellipsoid_a=0.0, ellipsoid_c=0.0,
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
                 diff_sensor_impl=None) -> None:
        self.device = device
        self.batch_size = batch_size
        self.width = max(int(width), 1)      # diff_depth 渲染宽度
        self.height = max(int(height), 1)    # diff_depth 渲染高度
        self.grad_decay = grad_decay # 梯度衰减系数
        self.wall_slit = wall_slit   # 是否启用狭缝穿越环境
        self.depth_min_valid = max(float(depth_min_valid), 1e-3)
        self.depth_max_range = max(float(depth_max_range), self.depth_min_valid + 1e-3)
        self.supported_scenarios = ('random_base', 'sun_glare', 'black_gap', 'dark_slit_lite')
        self.scene_name_to_id = {
            'random_base': 0,
            'sun_glare': 1,
            'black_gap': 2,
            'dark_slit_lite': 3,
            'wall_slit': 4,
        }
        self.scenarios = self._normalize_scenarios(scenarios)
        self.current_scene_name = 'random_base'
        self.current_scene_id = self.scene_name_to_id[self.current_scene_name]
        self.current_scene_has_opening = False
        self._clear_opening_scene_metadata()
        
        # 椭球体碰撞模型参数 (用于更精确的无人机碰撞检测)
        self.ellipsoid_a = ellipsoid_a # 椭球体 XY 半轴 (螺旋桨平面半径)
        self.ellipsoid_c = ellipsoid_c # 椭球体 Z 半轴 (无人机半高)
        self.use_ellipsoid = ellipsoid_a > 0 and ellipsoid_c > 0
        
        # 障碍物生成的基准参数 (w: 范围/宽度, b: 偏移/基准值)
        self.ball_w = torch.tensor([8., 18, 6, 0.2], device=device) # 球体障碍物
        self.ball_b = torch.tensor([0., -9, -1, 0.4], device=device)
        self.voxel_w = torch.tensor([8., 18, 6, 0.1, 0.1, 0.1], device=device) # 立方体障碍物
        self.voxel_b = torch.tensor([0., -9, -1, 0.2, 0.2, 0.2], device=device)
        self.ground_voxel_w = torch.tensor([8., 18,  0, 2.9, 2.9, 1.9], device=device) # 地面障碍物
        self.ground_voxel_b = torch.tensor([0., -9, -1, 0.1, 0.1, 0.1], device=device)
        self.cyl_w = torch.tensor([8., 18, 0.35], device=device) # 垂直圆柱体
        self.cyl_b = torch.tensor([0., -9, 0.05], device=device)
        self.cyl_h_w = torch.tensor([8., 6, 0.1], device=device) # 水平圆柱体
        self.cyl_h_b = torch.tensor([0., 0, 0.05], device=device)
        self.gate_w = torch.tensor([2.,  2,  1.0, 0.5], device=device) # 穿越门
        self.gate_b = torch.tensor([3., -1,  0.0, 0.5], device=device)
        
        self.v_wind_w = torch.tensor([1,  1,  0.2], device=device) # 风扰动权重
        self.g_std = torch.tensor([0., 0, -9.80665], device=device) # 标准重力加速度
        self.roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5], device=device) # 屋顶障碍物附加值
        
        # 物理步进的子步划分 (用于更精确的碰撞检测)
        self.sub_div = torch.linspace(0, 1. / 15, 10, device=device).reshape(-1, 1, 1)
        
        # 无人机初始位置和目标位置的基准点 (支持多机编队)
        self.p_init = torch.as_tensor([
            [-1.5, -3.,  1], [ 9.5, -3.,  1], [-0.5,  1.,  1], [ 8.5,  1.,  1],
            [ 0.0,  3.,  1], [ 8.0,  3.,  1], [-1.0, -1.,  1], [ 9.0, -1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        self.p_end = torch.as_tensor([
            [8.,  3.,  1], [0.,  3.,  1], [8., -1.,  1], [0., -1.,  1],
            [8., -3.,  1], [0., -3.,  1], [8.,  1.,  1], [0.,  1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        
        # 光流张量 (当前未使用，预留接口)
        self.flow = torch.empty((batch_size, 0, self.height, self.width), device=device)
        
        # 环境配置标志
        self.single = single             # 是否单机模式
        self.gate = gate                 # 是否生成穿越门
        self.ground_voxels = ground_voxels # 是否生成地面复杂地形
        self.scaffold = scaffold         # 是否生成脚手架障碍物
        self.speed_mtp = speed_mtp       # 速度乘数
        self.random_rotation = random_rotation # 是否随机旋转整个场景
        self.cam_angle = cam_angle       # 相机俯仰角
        self.fov_x_half_tan = fov_x_half_tan # 基础视场角 (tan(FOV/2))
        _impl = {'diff_depth': 'python'}
        if diff_sensor_impl is not None:
            _impl.update({str(k): str(v).lower() for k, v in dict(diff_sensor_impl).items()})
        self.diff_sensor_impl = _impl
        
        if wall_slit:
            self.single = True  # 狭缝穿越任务强制使用单机模式

        # ==================== 高保真可微相机参数（7层管线） ====================
        self.camera_preset = str(camera_preset).lower()
        self.cam_enable_specular = bool(cam_enable_specular)
        self.cam_enable_motion_blur = bool(cam_enable_motion_blur)
        self.cam_noise_scale = float(cam_noise_scale)
        self.cam_blur_scale = float(cam_blur_scale)
        self.cam_fog_scale = float(cam_fog_scale)
        self.cam_lighting_scale = float(cam_lighting_scale)

        # 光照层
        self.cam_ambient_min = 0.08
        self.cam_ambient_max = 0.35
        self.cam_dir_min = 0.4
        self.cam_dir_max = 1.6
        self.cam_fog_beta_min = 0.01
        self.cam_fog_beta_max = 0.12
        self.cam_airlight_min = 0.2
        self.cam_airlight_max = 0.8

        # 传感器层
        self.cam_read_noise = 0.0025

        # 时序层（运动模糊）
        self.cam_motion_blur_gain = 0.09

        # 统一相机语义常数（曝光/ISO/噪声映射）
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

        # 相机状态容器（在 reset 中刷新为随机状态）
        self._cam_ambient = torch.full((batch_size,), 0.2, device=device)
        self._cam_dir_intensity = torch.full((batch_size,), 1.0, device=device)
        self._cam_fog_beta = torch.full((batch_size,), 0.02, device=device)
        self._cam_airlight = torch.full((batch_size,), 0.4, device=device)
        self._cam_mat_obstacle = torch.full((batch_size,), 0.6, device=device)
        self._cam_mat_spec = torch.full((batch_size,), 0.08, device=device)
            
        # 初始化环境状态
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

    def _normalize_scenarios(self, scenarios):
        if scenarios is None:
            return ['random_base']
        out = []
        for raw in scenarios:
            name = str(raw).strip().lower()
            if not name:
                continue
            if name not in self.supported_scenarios:
                raise ValueError(
                    f"不支持的场景 '{name}'，仅支持: {list(self.supported_scenarios)}"
                )
            if name not in out:
                out.append(name)
        return out or ['random_base']

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

    def _clear_opening_scene_metadata(self):
        self.wall_x = None
        self.slit_y_center = None
        self.slit_z_center = None
        self.slit_half_w = None
        self.slit_half_h = None
        self.wall_thickness = None

    def _choose_scene_name(self, scene_name=None):
        if self.wall_slit:
            return 'wall_slit'
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
        self.current_scene_has_opening = self.current_scene_name in {'black_gap', 'dark_slit_lite', 'wall_slit'}

    def _sample_scene_tensor(self, lo, hi):
        return torch.empty((self.batch_size,), device=self.device).uniform_(float(lo), float(hi))

    def _apply_scene_sensor_profile(self, scene_name):
        if scene_name == 'sun_glare':
            self._cam_ambient = self._sample_scene_tensor(0.30, 0.52)
            self._cam_dir_intensity = self._sample_scene_tensor(1.8, 3.0)
            self._cam_fog_beta = self._sample_scene_tensor(0.03, 0.12)
            self._cam_airlight = self._sample_scene_tensor(0.65, 1.00)
            self._cam_mat_obstacle = self._sample_scene_tensor(0.55, 0.95)
            self._cam_mat_spec = self._sample_scene_tensor(0.04, 0.18)
        elif scene_name == 'black_gap':
            self._cam_ambient = self._sample_scene_tensor(0.02, 0.09)
            self._cam_dir_intensity = self._sample_scene_tensor(0.08, 0.35)
            self._cam_fog_beta = self._sample_scene_tensor(0.005, 0.03)
            self._cam_airlight = self._sample_scene_tensor(0.04, 0.14)
            self._cam_mat_obstacle = self._sample_scene_tensor(0.03, 0.12)
            self._cam_mat_spec = self._sample_scene_tensor(0.00, 0.03)
        elif scene_name == 'dark_slit_lite':
            self._cam_ambient = self._sample_scene_tensor(0.03, 0.12)
            self._cam_dir_intensity = self._sample_scene_tensor(0.12, 0.50)
            self._cam_fog_beta = self._sample_scene_tensor(0.006, 0.04)
            self._cam_airlight = self._sample_scene_tensor(0.05, 0.18)
            self._cam_mat_obstacle = self._sample_scene_tensor(0.07, 0.20)
            self._cam_mat_spec = self._sample_scene_tensor(0.01, 0.05)

    def _reset_rect_opening_scene(self, B, device, scene_name,
                                  slit_half_w_range, slit_half_h_range,
                                  dist_from_wall_range, max_speed_range,
                                  lateral_noise, vertical_noise,
                                  margin, drone_radius, wind_scale):
        wall_x = random.uniform(2.2, 5.2)
        slit_y_center = random.uniform(-1.2, 1.2)
        slit_z_center = random.uniform(0.35, 1.45)
        slit_half_w = random.uniform(*slit_half_w_range)
        slit_half_h = random.uniform(*slit_half_h_range)
        wall_thickness = 0.16

        self.wall_x = wall_x
        self.slit_y_center = slit_y_center
        self.slit_z_center = slit_z_center
        self.slit_half_w = slit_half_w
        self.slit_half_h = slit_half_h
        self.wall_thickness = wall_thickness

        big = 10.0
        wall_voxels = torch.zeros((B, 4, 6), device=device)
        wall_voxels[:, 0, 0] = wall_x
        wall_voxels[:, 0, 1] = slit_y_center - slit_half_w - big
        wall_voxels[:, 0, 2] = slit_z_center
        wall_voxels[:, 0, 3] = wall_thickness
        wall_voxels[:, 0, 4] = big
        wall_voxels[:, 0, 5] = big

        wall_voxels[:, 1, 0] = wall_x
        wall_voxels[:, 1, 1] = slit_y_center + slit_half_w + big
        wall_voxels[:, 1, 2] = slit_z_center
        wall_voxels[:, 1, 3] = wall_thickness
        wall_voxels[:, 1, 4] = big
        wall_voxels[:, 1, 5] = big

        wall_voxels[:, 2, 0] = wall_x
        wall_voxels[:, 2, 1] = slit_y_center
        wall_voxels[:, 2, 2] = slit_z_center + slit_half_h + big
        wall_voxels[:, 2, 3] = wall_thickness
        wall_voxels[:, 2, 4] = slit_half_w
        wall_voxels[:, 2, 5] = big

        wall_voxels[:, 3, 0] = wall_x
        wall_voxels[:, 3, 1] = slit_y_center
        wall_voxels[:, 3, 2] = slit_z_center - slit_half_h - big
        wall_voxels[:, 3, 3] = wall_thickness
        wall_voxels[:, 3, 4] = slit_half_w
        wall_voxels[:, 3, 5] = big

        self.balls[:, :, 2] = -200
        self.cyl[:, :, 2] = 0.001
        self.cyl_h[:, :, 2] = 0.001
        self.voxels = wall_voxels

        dist_from_wall = random.uniform(*dist_from_wall_range)
        noise_y = torch.randn(B, device=device) * lateral_noise
        noise_z = torch.randn(B, device=device) * vertical_noise
        self.p = torch.stack([
            torch.full((B,), wall_x - dist_from_wall, device=device),
            torch.full((B,), slit_y_center, device=device) + noise_y,
            torch.full((B,), slit_z_center, device=device) + noise_z,
        ], -1)
        self.p_target = torch.stack([
            torch.full((B,), wall_x + dist_from_wall, device=device),
            torch.full((B,), slit_y_center, device=device) + noise_y * 0.4,
            torch.full((B,), slit_z_center, device=device) + noise_z * 0.4,
        ], -1)

        self.n_drones_per_group = 1
        self.drone_radius = drone_radius
        self.max_speed = torch.full(
            (B, 1), random.uniform(*max_speed_range), device=device
        ) * self.speed_mtp
        self.margin = torch.full((B,), margin, device=device)

        self.v = torch.randn((B, 3), device=device) * 0.08
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w * wind_scale
        self.act = torch.randn_like(self.v) * 0.04
        self.a = self.act
        self.dg = torch.randn((B, 3), device=device) * 0.08

        R = torch.zeros((B, 3, 3), device=device)
        v_dir = F.normalize(self.p_target - self.p, 2, -1)
        self.R = quadsim_cuda.update_state_vec(
            R, self.act, torch.randn((B, 3), device=device) * 0.1 + v_dir,
            torch.zeros_like(self.yaw_ctl_delay), 5)
        self.R_old = self.R.clone()
        self.p_old = self.p
        self._current_scale = 1.0
        self._current_y_stretch = 1.0

    def reset(self, scene_name=None):
        """
        重置环境状态。
        在每个 episode 开始时调用，随机生成障碍物、无人机初始状态、目标点等。
        """
        B = self.batch_size
        device = self.device
        scene_name = self._choose_scene_name(scene_name)
        self._set_scene_name(scene_name)
        self._clear_opening_scene_metadata()

        # 1. 初始化相机旋转矩阵 (R_cam)
        # 相机默认有一个向下的俯仰角 (cam_angle)，并加入少量随机噪声
        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
            zeros, ones, zeros,
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),
        ], -1).reshape(B, 3, 3)

        # 2. 随机生成环境障碍物
        # balls: 球体 (x, y, z, r)
        # voxels: 立方体 (x, y, z, rx, ry, rz)
        # cyl: 垂直圆柱体 (x, y, r)
        # cyl_h: 水平圆柱体 (x, z, r)
        self.balls = torch.rand((B, 30, 4), device=device) * self.ball_w + self.ball_b
        self.voxels = torch.rand((B, 30, 6), device=device) * self.voxel_w + self.voxel_b
        self.cyl = torch.rand((B, 30, 3), device=device) * self.cyl_w + self.cyl_b
        self.cyl_h = torch.rand((B, 2, 3), device=device) * self.cyl_h_w + self.cyl_h_b

        # 随机化基础 FOV
        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan
        
        # 确定每组无人机的数量 (编队飞行)
        self.n_drones_per_group = random.choice([4, 8])
        self.drone_radius = random.uniform(0.1, 0.15) # 无人机碰撞半径
        if self.single:
            self.n_drones_per_group = 1 # 单机模式

        # 随机生成最大飞行速度限制
        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        self.max_speed = (0.75 + 2.5 * rd) * self.speed_mtp
        scene_scale = (self.max_speed - 0.5).clamp_min(1) # 根据速度缩放场景大小
        scene_y_stretch = (self.max_speed + 4) / scene_scale   # Y 轴拉伸系数（用于障碍物与起终点保持一致）

        # 推力估计误差 (模拟真实世界中电机推力的不确定性)
        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        # 3. 场景变体：屋顶环境 (Roof)
        # 50% 的概率生成带有屋顶的受限空间，迫使无人机在低空飞行
        roof = torch.rand((B,)) < 0.5
        self.balls[~roof, :15, :2] = self.cyl[~roof, :15, :2]
        self.voxels[~roof, :15, :2] = self.cyl[~roof, 15:, :2]
        self.balls[~roof, :15] = self.balls[~roof, :15] + self.roof_add[:4]
        self.voxels[~roof, :15] = self.voxels[~roof, :15] + self.roof_add
        # 限制障碍物的 X 坐标范围，确保起点和终点有足够的空间
        self.balls[..., 0] = torch.minimum(torch.maximum(self.balls[..., 0], self.balls[..., 3] + 0.3 / scene_scale), 8 - 0.3 / scene_scale - self.balls[..., 3])
        self.voxels[..., 0] = torch.minimum(torch.maximum(self.voxels[..., 0], self.voxels[..., 3] + 0.3 / scene_scale), 8 - 0.3 / scene_scale - self.voxels[..., 3])
        self.cyl[..., 0] = torch.minimum(torch.maximum(self.cyl[..., 0], self.cyl[..., 2] + 0.3 / scene_scale), 8 - 0.3 / scene_scale - self.cyl[..., 2])
        self.cyl_h[..., 0] = torch.minimum(torch.maximum(self.cyl_h[..., 0], self.cyl_h[..., 2] + 0.3 / scene_scale), 8 - 0.3 / scene_scale - self.cyl_h[..., 2])
        # 设置屋顶的高度
        self.voxels[roof, 0, 2] = self.voxels[roof, 0, 2] * 0.5 + 201
        self.voxels[roof, 0, 3:] = 200

        # 4. 场景变体：复杂地面 (Ground Voxels)
        ground_balls_r_ground = torch.zeros((B, 2), device=device)
        if self.ground_voxels:
            # 生成起伏的地面球体和平台
            ground_balls_r = 8 + torch.rand((B, 2), device=device) * 6
            ground_balls_r_ground = 2 + torch.rand((B, 2), device=device) * 4
            ground_balls_h = ground_balls_r - (ground_balls_r.pow(2) - ground_balls_r_ground.pow(2)).sqrt()
            # |   ground_balls_h
            # ----- ground_balls_r_ground
            # |  /
            # | / ground_balls_r
            # |/
            self.balls[:, :2, 3] = ground_balls_r
            self.balls[:, :2, 2] = ground_balls_h - ground_balls_r - 1

            # planner shape in (0.1-2.0) times (0.1-2.0)
            ground_voxels = torch.rand((B, 10, 6), device=device) * self.ground_voxel_w + self.ground_voxel_b
            ground_voxels[:, :, 2] = ground_voxels[:, :, 5] - 1
            self.voxels = torch.cat([self.voxels, ground_voxels], 1)

        # 根据最大速度拉伸场景的 Y 轴 (速度越快，场景越长)
        self.voxels[:, :, 1] *= scene_y_stretch
        self.balls[:, :, 1] *= scene_y_stretch
        self.cyl[:, :, 1] *= scene_y_stretch

        # 5. 场景变体：穿越门 (Gates)
        if self.gate:
            # 随机生成门的位置和大小
            gate = torch.rand((B, 4), device=device) * self.gate_w + self.gate_b
            p = gate[None, :, :3]
            nearest_pt = torch.empty_like(p)
            # 检查门是否与其他障碍物重叠，如果重叠则将其移出场景 (x=-50)
            quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, 1)
            gate_x, gate_y, gate_z, gate_r = gate.unbind(-1)
            gate_x[(nearest_pt - p).norm(2, -1)[0] < 0.5] = -50
            ones = torch.ones_like(gate_x)
            # 用 4 个长条形 voxel 拼成一个方形的门框
            gate = torch.stack([
                torch.stack([gate_x, gate_y + gate_r + 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1), # 上边框
                torch.stack([gate_x, gate_y, gate_z + gate_r + 5, ones * 0.05, ones * 5, ones * 5], -1), # 右边框
                torch.stack([gate_x, gate_y - gate_r - 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1), # 下边框
                torch.stack([gate_x, gate_y, gate_z - gate_r - 5, ones * 0.05, ones * 5, ones * 5], -1), # 左边框
            ], 1)

            self.voxels = torch.cat([self.voxels, gate], 1)
            
        # 根据 scale 缩放所有障碍物的 X 坐标
        self.voxels[..., 0] *= scene_scale
        self.balls[..., 0] *= scene_scale
        self.cyl[..., 0] *= scene_scale
        self.cyl_h[..., 0] *= scene_scale
        if self.ground_voxels:
            self.balls[:, :2, 0] = torch.minimum(torch.maximum(self.balls[:, :2, 0], ground_balls_r_ground + 0.3), scene_scale * 8 - 0.3 - ground_balls_r_ground)

        # 6. 初始化无人机动力学参数
        # 俯仰/滚转控制延迟 (模拟底层飞控的响应时间)
        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device)
        # 偏航控制延迟
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)

        # 7. 初始化无人机位置 (p) 和目标位置 (p_target)
        pos_scale = torch.cat([
            scene_scale,
            scene_y_stretch,
            torch.rand_like(scene_scale) - 0.5], -1)
        self.p = self.p_init * pos_scale + torch.randn_like(pos_scale) * 0.1
        self.p_target = self.p_end * pos_scale + torch.randn_like(pos_scale) * 0.1

        # 8. 场景变体：随机旋转整个场景 (增加泛化能力)
        if self.random_rotation:
            yaw_bias = torch.rand(B//self.n_drones_per_group, device=device).repeat_interleave(self.n_drones_per_group, 0) * 1.5 - 0.75
            c = torch.cos(yaw_bias)
            s = torch.sin(yaw_bias)
            l = torch.ones_like(yaw_bias)
            o = torch.zeros_like(yaw_bias)
            R = torch.stack([c,-s, o, s, c, o, o, o, l], -1).reshape(B, 3, 3)
            # 旋转无人机位置、目标位置和所有障碍物
            self.p = torch.squeeze(R @ self.p[..., None], -1)
            self.p_target = torch.squeeze(R @ self.p_target[..., None], -1)
            self.voxels[..., :3] = (R @ self.voxels[..., :3].transpose(1, 2)).transpose(1, 2)
            self.balls[..., :3] = (R @ self.balls[..., :3].transpose(1, 2)).transpose(1, 2)
            self.cyl[..., :3] = (R @ self.cyl[..., :3].transpose(1, 2)).transpose(1, 2)

        # 9. 场景变体：脚手架 (Scaffold) - 密集的细小障碍物
        if self.scaffold and random.random() < 0.5:
            x = torch.arange(1, 6, dtype=torch.float, device=device)
            y = torch.arange(-3, 4, dtype=torch.float, device=device)
            z = torch.arange(1, 4, dtype=torch.float, device=device)
            _x, _y = torch.meshgrid(x, y)
            # 生成垂直脚手架杆
            scaf_v = torch.stack([_x, _y, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            x_bias = torch.rand_like(self.max_speed) * self.max_speed
            scaf_scale = 1 + torch.rand((B, 1, 1), device=device)
            scaf_v = scaf_v * scaf_scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed),
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            self.cyl = torch.cat([self.cyl, scaf_v], 1)
            # 生成水平脚手架杆
            _x, _z = torch.meshgrid(x, z)
            scaf_h = torch.stack([_x, _z, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            scaf_h = scaf_h * scaf_scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed) * 0.1,
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            self.cyl_h = torch.cat([self.cyl_h, scaf_h], 1)

        # 10. 初始化无人机运动状态
        self.v = torch.randn((B, 3), device=device) * 0.2 # 初始速度
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w # 初始风速
        self.act = torch.randn_like(self.v) * 0.1 # 初始动作 (推力加速度)
        self.a = self.act # 初始加速度
        self.dg = torch.randn((B, 3), device=device) * 0.2 # 初始重力/风阻扰动

        # 初始化姿态矩阵 R (机体坐标系到世界坐标系的旋转矩阵)
        R = torch.zeros((B, 3, 3), device=device)
        self.R = quadsim_cuda.update_state_vec(R, self.act, torch.randn((B, 3), device=device) * 0.2 + F.normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay), 5)
        self.R_old = self.R.clone()
        self.p_old = self.p
        
        # 碰撞安全边距 (margin)
        self.margin = torch.rand((B,), device=device) * 0.2 + 0.1

        # ==================== 论文场景：开口墙几何 ====================
        if scene_name == 'black_gap':
            self._reset_rect_opening_scene(
                B, device, scene_name,
                slit_half_w_range=(0.38, 0.70),
                slit_half_h_range=(0.34, 0.58),
                dist_from_wall_range=(1.8, 3.0),
                max_speed_range=(0.18, 0.32),
                lateral_noise=0.14,
                vertical_noise=0.10,
                margin=0.04,
                drone_radius=0.13,
                wind_scale=0.20,
            )
        elif scene_name == 'dark_slit_lite':
            self._reset_rect_opening_scene(
                B, device, scene_name,
                slit_half_w_range=(0.22, 0.32),
                slit_half_h_range=(0.34, 0.55),
                dist_from_wall_range=(1.8, 2.8),
                max_speed_range=(0.14, 0.24),
                lateral_noise=0.10,
                vertical_noise=0.08,
                margin=0.03,
                drone_radius=0.12,
                wind_scale=0.16,
            )
        elif self.wall_slit:
            self._reset_wall_slit(B, device)

        # 11. 初始化空气阻力系数
        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3 # 二次阻力系数
        self.drag_2[:, 0] = 0
        self.z_drag_coef = torch.ones((B, 1), device=device) # Z轴阻力系数

        # 12. 保存场景缩放参数（用于可视化AABB计算）
        # 这些值在reset时计算，用于log_environment中的动态AABB
        if not self.current_scene_has_opening:
            self._current_scale = scene_scale.reshape(-1)[0].item() if isinstance(scene_scale, torch.Tensor) else float(scene_scale)
            self._current_y_stretch = scene_y_stretch.reshape(-1)[0].item() if isinstance(scene_y_stretch, torch.Tensor) else float(scene_y_stretch)

        # 13. 初始化高保真可微相机状态
        self._reset_camera_states()
        self._apply_scene_sensor_profile(scene_name)

    def _reset_wall_slit(self, B, device):
        self._reset_rect_opening_scene(
            B, device, 'wall_slit',
            slit_half_w_range=(0.10, 0.18),
            slit_half_h_range=(0.35, 0.60),
            dist_from_wall_range=(1.5, 3.0),
            max_speed_range=(0.08, 0.20),
            lateral_noise=0.30,
            vertical_noise=0.20,
            margin=0.02,
            drone_radius=0.15,
            wind_scale=0.30,
        )

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

        signal_active = 5.0 * ps * es * albedo * frontality * fog_trans / (depth.square() + 0.08)
        signal_active = signal_active.clamp_max(1e6)  # 防止 depth 极小时 Inf → NaN
        signal_passive = (
            es * ambient_ir * (0.15 + 0.85 * edge) * (0.35 + 0.65 * albedo) * torch.sqrt(gs)
        )
        spec_bloom = spec * ps * (0.6 + 0.4 * ambient_ir) * (1.0 + edge)

        if self.cam_enable_motion_blur:
            speed = self.v.norm(2, -1)
            motion = (speed[:, None, None] * es * self.cam_motion_blur_gain).clamp(0.0, 1.25)
        else:
            motion = torch.zeros_like(depth)

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

        R_cam_world = (self.R @ self.R_cam).contiguous()
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
        noisy_depth = noisy_depth.clamp(min_valid, float(max_range))
        valid = torch.sigmoid((quality - 0.45) / 0.08)
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

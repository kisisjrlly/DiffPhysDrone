import math
import random
import time
import torch
import torch.nn.functional as F
import quadsim_cuda

# =============================================================================
# 1. 自定义 PyTorch 自动求导函数 (Autograd Functions)
# =============================================================================

class GDecay(torch.autograd.Function):
    """
    梯度衰减函数 (Gradient Decay)。
    在强化学习/轨迹优化中，长序列的反向传播容易导致梯度爆炸。
    这个函数在前向传播时保持值不变，但在反向传播时，将梯度乘以一个衰减系数 alpha。
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha # 保存衰减系数用于反向传播
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时，将传回来的梯度乘以 alpha
        return grad_output * ctx.alpha, None

# 实例化为可调用的函数
g_decay = GDecay.apply


class RunFunction(torch.autograd.Function):
    """
    无人机物理动力学的前向与反向传播封装。
    调用底层的 CUDA C++ 扩展 (quadsim_cuda) 来加速物理仿真，并支持可微物理 (Differentiable Physics)。
    """
    @staticmethod
    def forward(ctx, R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, grad_decay, ctl_dt, airmode):
        # 调用 CUDA 核心进行前向物理步进
        # R: 旋转矩阵, dg: 扰动重力/风阻, z_drag_coef: Z轴阻力系数, drag_2: 二次阻力系数
        # pitch_ctl_delay: 俯仰控制延迟, act_pred: 预测动作, act: 当前动作, p: 位置, v: 速度, a: 加速度
        act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, airmode)
        
        # 保存前向计算的中间变量，供反向传播计算梯度使用
        ctx.save_for_backward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next)
        ctx.grad_decay = grad_decay
        ctx.ctl_dt = ctl_dt
        return act_next, p_next, v_next, a_next

    @staticmethod
    def backward(ctx, d_act_next, d_p_next, d_v_next, d_a_next):
        # 提取保存的变量
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next = ctx.saved_tensors
        # 调用 CUDA 核心进行反向物理求导，计算出对各个输入的梯度
        d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, d_act_next, d_p_next, d_v_next, d_a_next,
            ctx.grad_decay, ctx.ctl_dt)
        # 返回梯度，对应 forward 的输入参数。不需要梯度的参数返回 None
        return None, None, None, None, None, d_act_pred, d_act, d_p, d_v, None, d_a, None, None, None

run = RunFunction.apply


class DiffRenderFunction(torch.autograd.Function):
    """
    可微渲染函数 (Differentiable Rendering)。
    支持对每个 batch 的 FOV (视场角) 张量进行求导。
    通过 CUDA 实现深度图的渲染，并允许梯度从深度图回传到相机的 FOV 参数。
    """
    @staticmethod
    def forward(ctx, fov_x_half_tan, R_cam, pos, balls, cyl, cyl_h, voxels,
                n_drones_per_group, height, width):
        B = pos.shape[0]
        # 确保内存连续，这是 CUDA 扩展的要求
        fov_x_half_tan = fov_x_half_tan.contiguous()
        R_cam = R_cam.contiguous()
        pos = pos.contiguous()
        # 初始化深度图画布
        canvas = torch.empty((B, height, width), device=pos.device)
        # 调用 CUDA 渲染核心，生成深度图
        quadsim_cuda.render_diff_fov(canvas, balls, cyl, cyl_h, voxels,
                                     R_cam, pos, n_drones_per_group, fov_x_half_tan)
        # 保存变量用于反向传播
        ctx.save_for_backward(fov_x_half_tan, canvas, R_cam, pos, balls, cyl, cyl_h, voxels)
        ctx.n_drones_per_group = n_drones_per_group
        return canvas

    @staticmethod
    def backward(ctx, grad_output):
        # 提取保存的变量
        fov, canvas, R_cam, pos, balls, cyl, cyl_h, voxels = ctx.saved_tensors
        grad_fov = torch.zeros_like(fov)
        # 调用 CUDA 核心计算深度图对 FOV 的梯度
        quadsim_cuda.render_backward_fov(grad_fov, grad_output.contiguous(), canvas,
                                         balls, cyl, cyl_h, voxels, R_cam, pos,
                                         ctx.n_drones_per_group, fov)
        # 仅返回对 fov_x_half_tan 的梯度，其他环境参数（如位置、障碍物）的梯度在此处不计算（或在其他地方计算）
        return grad_fov, None, None, None, None, None, None, None, None, None

diff_render = DiffRenderFunction.apply


class DiffRenderYuvYFunction(torch.autograd.Function):
    """
    Y 通道可微渲染函数。
    前向调用 CUDA 扩展的 render_diff_yuv_y_forward，
    反向调用 render_diff_yuv_y_backward，
    对 fov/exposure/iso 提供严格梯度。
    """
    @staticmethod
    def forward(ctx, fov_x_half_tan, exposure, iso,
                R_cam, pos, balls, cyl, cyl_h, voxels,
                n_drones_per_group, height, width):
        fov_x_half_tan = fov_x_half_tan.contiguous()
        exposure = exposure.contiguous()
        iso = iso.contiguous()
        R_cam = R_cam.contiguous()
        pos = pos.contiguous()
        y, depth_raw = quadsim_cuda.render_diff_yuv_y_forward(
            fov_x_half_tan, exposure, iso,
            R_cam, pos, balls, cyl, cyl_h, voxels,
            n_drones_per_group, height, width)
        ctx.save_for_backward(
            depth_raw, fov_x_half_tan, exposure, iso,
            R_cam, pos, balls, cyl, cyl_h, voxels)
        ctx.n_drones_per_group = n_drones_per_group
        return y

    @staticmethod
    def backward(ctx, grad_output):
        depth_raw, fov, exposure, iso, R_cam, pos, balls, cyl, cyl_h, voxels = ctx.saved_tensors
        grad_fov, grad_exposure, grad_iso = quadsim_cuda.render_diff_yuv_y_backward(
            grad_output.contiguous(),
            depth_raw,
            fov,
            exposure,
            iso,
            R_cam,
            pos,
            balls,
            cyl,
            cyl_h,
            voxels,
            ctx.n_drones_per_group)
        return grad_fov, grad_exposure, grad_iso, None, None, None, None, None, None, None, None, None


diff_render_yuv_y = DiffRenderYuvYFunction.apply


# =============================================================================
# 2. 论文 §2.3 提出的可微相机传感器效应 (Optical Perception Potentials)
# =============================================================================

def apply_camera_effects(depth, exposure, iso):
    """
    将可微的相机传感器效应应用到渲染出的纯净深度图上。
    模拟真实相机的曝光、ISO 噪点等物理效应。

    Args:
        depth: (B, H, W) 渲染器输出的原始深度图
        exposure: (B,) 曝光参数，范围 [0, 1] (通常由策略网络输出并经过 sigmoid)
        iso: (B,) ISO 参数，范围 [0, 1]
    Returns:
        (B, H, W) 带有传感器效应的深度图
    """
    # 将 [0,1] 的网络输出映射到物理范围
    exposure_phys = exposure * 10 + 0.5       # 曝光时间: [0.5, 10.5] 毫秒
    iso_phys = iso * 6400 + 100               # ISO 感光度: [100, 6500]

    # 1. 有效感知范围 (Effective sensing range): 传感器能探测到的最大深度
    #    曝光时间越长 / ISO 越高 -> 探测距离越远
    max_range = 2.0 + 1.5 * exposure_phys + 0.001 * iso_phys  # 大致范围 ~[3, 24] 米
    max_range = max_range[:, None, None]  # 扩展维度以匹配 (B, H, W)
    # 平滑截断 (Smooth clamp): 将超出最大探测距离的深度值平滑地映射到 max_range，保持可微性
    depth = max_range - F.softplus(max_range - depth, beta=2.0)

    # 2. 深度噪点 (Depth noise): 模拟散斑噪声
    #    ISO 越高噪点越大，曝光时间越长噪点越小
    noise_sigma = 0.03 * (1.0 + 2.0 * iso) / (exposure + 0.3)  # (B,)
    # 噪点随距离增加而放大 (远处的物体测距更不准)
    depth_dist_scale = depth.detach().clamp(0.3, 20) / 5.0  
    # 添加高斯噪声
    depth = depth + torch.randn_like(depth) * noise_sigma[:, None, None] * depth_dist_scale

    return depth


def _safe_normalize(x, dim=-1, eps=1e-6):
    return x / torch.clamp(torch.norm(x, 2, dim=dim, keepdim=True), min=eps)


def _make_separable_gaussian_kernel1d(sigma, device, dtype):
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(3.0 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / torch.clamp(k.sum(), min=1e-12)
    return k


def _separable_gaussian_blur(img, sigma):
    """
    对单通道图像执行可微高斯模糊。
    img: (B, H, W)
    """
    if sigma <= 1e-4:
        return img
    k = _make_separable_gaussian_kernel1d(sigma, img.device, img.dtype)
    r = (k.numel() - 1) // 2
    x = img[:, None]
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    x = F.pad(x, (r, r, 0, 0), mode='reflect')
    x = F.conv2d(x, kx)
    x = F.pad(x, (0, 0, r, r), mode='reflect')
    x = F.conv2d(x, ky)
    return x[:, 0]


class Env:
    """
    无人机物理仿真环境类。
    负责管理无人机状态、障碍物生成、碰撞检测、可微渲染以及物理步进。
    支持大规模并行仿真 (Batch processing)。
    """
    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 single=False, gate=False, ground_voxels=False, scaffold=False, speed_mtp=1,
                 random_rotation=False, cam_angle=10,
                 wall_slit=False, ellipsoid_a=0.0, ellipsoid_c=0.0,
                 tof_downsample=4, tof_width=None, tof_height=None,
                 camera_preset='high',
                 cam_enable_shadow=True,
                 cam_enable_specular=True,
                 cam_enable_distortion=True,
                 cam_enable_flare=True,
                 cam_enable_motion_blur=True,
                 cam_enable_rolling=True,
                 cam_noise_scale=1.0,
                 cam_blur_scale=1.0,
                 cam_fog_scale=1.0,
                 cam_lighting_scale=1.0,
                 cam_ae_target=0.42) -> None:
        self.device = device
        self.batch_size = batch_size
        self.width = width      # 主相机渲染宽度
        self.height = height    # 主相机渲染高度
        self.grad_decay = grad_decay # 梯度衰减系数
        self.wall_slit = wall_slit   # 是否启用狭缝穿越环境
        
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
        self.flow = torch.empty((batch_size, 0, height, width), device=device)
        
        # 环境配置标志
        self.single = single             # 是否单机模式
        self.gate = gate                 # 是否生成穿越门
        self.ground_voxels = ground_voxels # 是否生成地面复杂地形
        self.scaffold = scaffold         # 是否生成脚手架障碍物
        self.speed_mtp = speed_mtp       # 速度乘数
        self.random_rotation = random_rotation # 是否随机旋转整个场景
        self.cam_angle = cam_angle       # 相机俯仰角
        self.fov_x_half_tan = fov_x_half_tan # 基础视场角 (tan(FOV/2))
        self.tof_downsample = max(int(tof_downsample), 1)
        if tof_width is None:
            self.tof_width = max(int(self.width) // self.tof_downsample, 1)
        else:
            self.tof_width = max(int(tof_width), 1)
        if tof_height is None:
            self.tof_height = max(int(self.height) // self.tof_downsample, 1)
        else:
            self.tof_height = max(int(tof_height), 1)
        
        if wall_slit:
            self.single = True  # 狭缝穿越任务强制使用单机模式

        # ==================== 高保真可微相机参数（7层管线） ====================
        self.camera_preset = str(camera_preset).lower()
        self.cam_enable_shadow = bool(cam_enable_shadow)
        self.cam_enable_specular = bool(cam_enable_specular)
        self.cam_enable_distortion = bool(cam_enable_distortion)
        self.cam_enable_flare = bool(cam_enable_flare)
        self.cam_enable_motion_blur = bool(cam_enable_motion_blur)
        self.cam_enable_rolling = bool(cam_enable_rolling)
        self.cam_noise_scale = float(cam_noise_scale)
        self.cam_blur_scale = float(cam_blur_scale)
        self.cam_fog_scale = float(cam_fog_scale)
        self.cam_lighting_scale = float(cam_lighting_scale)

        # 几何/材质层
        self.cam_ground_z = -1.0
        self.cam_ground_soft_band = 0.08

        # 光照层
        self.cam_ambient_min = 0.08
        self.cam_ambient_max = 0.35
        self.cam_dir_min = 0.4
        self.cam_dir_max = 1.6
        self.cam_fog_beta_min = 0.01
        self.cam_fog_beta_max = 0.12
        self.cam_airlight_min = 0.2
        self.cam_airlight_max = 0.8

        # 镜头层
        self.cam_vignette_a = 0.28
        self.cam_vignette_b = 0.22
        self.cam_dist_k1_range = (-0.12, 0.08)
        self.cam_dist_k2_range = (-0.06, 0.04)
        self.cam_flare_strength_max = 0.16

        # 传感器层
        self.cam_base_gain = 0.14
        self.cam_read_noise = 0.0025
        self.cam_black_level = 0.01
        self.cam_prnu_std = 0.02
        self.cam_dsnu_std = 0.005

        # ISP 层
        self.cam_gamma_min = 1.9
        self.cam_gamma_max = 2.4
        self.cam_sharpen_amount = 0.35

        # 时序层（自动曝光 + 运动模糊）
        self.cam_ae_target = float(cam_ae_target)
        self.cam_ae_kp = 0.18
        self.cam_ae_ki = 0.015
        self.cam_ae_log_t_min = math.log(0.2)
        self.cam_ae_log_t_max = math.log(3.0)
        self.cam_motion_blur_gain = 0.09
        self.cam_rolling_blur_prob = 0.5

        self._configure_camera_preset(self.camera_preset)

        # 相机状态容器（在 reset 中刷新为随机状态）
        self._cam_light_dir = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(batch_size, 1)
        self._cam_ambient = torch.full((batch_size,), 0.2, device=device)
        self._cam_dir_intensity = torch.full((batch_size,), 1.0, device=device)
        self._cam_fog_beta = torch.full((batch_size,), 0.02, device=device)
        self._cam_airlight = torch.full((batch_size,), 0.4, device=device)
        self._cam_mat_ground = torch.full((batch_size,), 0.4, device=device)
        self._cam_mat_obstacle = torch.full((batch_size,), 0.6, device=device)
        self._cam_mat_spec = torch.full((batch_size,), 0.08, device=device)
        self._cam_dist_k1 = torch.zeros((batch_size,), device=device)
        self._cam_dist_k2 = torch.zeros((batch_size,), device=device)
        self._cam_flare_strength = torch.zeros((batch_size,), device=device)
        self._cam_gamma = torch.full((batch_size,), 2.2, device=device)
        self._cam_prnu = torch.zeros((batch_size, height, width), device=device)
        self._cam_dsnu = torch.zeros((batch_size, height, width), device=device)
        self._cam_prev_y = torch.zeros((batch_size, height, width), device=device)
        self._cam_ae_log_t = torch.zeros((batch_size,), device=device)
        self._cam_ae_integral = torch.zeros((batch_size,), device=device)
        self._cam_use_rolling = torch.zeros((batch_size,), device=device)
            
        # 初始化环境状态
        self.reset()

    def _configure_camera_preset(self, preset: str):
        """根据档位配置高保真可微相机强度。"""
        p = preset.lower()
        if p == 'low':
            self.cam_ambient_min, self.cam_ambient_max = 0.12, 0.28
            self.cam_dir_min, self.cam_dir_max = 0.45, 1.0
            self.cam_fog_beta_min, self.cam_fog_beta_max = 0.005, 0.06
            self.cam_vignette_a, self.cam_vignette_b = 0.18, 0.12
            self.cam_read_noise = 0.0018
            self.cam_prnu_std, self.cam_dsnu_std = 0.012, 0.003
            self.cam_sharpen_amount = 0.20
            self.cam_motion_blur_gain = 0.05
        elif p == 'medium':
            self.cam_ambient_min, self.cam_ambient_max = 0.10, 0.32
            self.cam_dir_min, self.cam_dir_max = 0.4, 1.3
            self.cam_fog_beta_min, self.cam_fog_beta_max = 0.008, 0.09
            self.cam_vignette_a, self.cam_vignette_b = 0.24, 0.16
            self.cam_read_noise = 0.0022
            self.cam_prnu_std, self.cam_dsnu_std = 0.016, 0.004
            self.cam_sharpen_amount = 0.28
            self.cam_motion_blur_gain = 0.075
        elif p == 'ultra':
            self.cam_ambient_min, self.cam_ambient_max = 0.06, 0.42
            self.cam_dir_min, self.cam_dir_max = 0.5, 1.9
            self.cam_fog_beta_min, self.cam_fog_beta_max = 0.015, 0.15
            self.cam_vignette_a, self.cam_vignette_b = 0.34, 0.26
            self.cam_read_noise = 0.0032
            self.cam_prnu_std, self.cam_dsnu_std = 0.028, 0.006
            self.cam_sharpen_amount = 0.42
            self.cam_motion_blur_gain = 0.11
        else:  # high
            self.cam_ambient_min, self.cam_ambient_max = 0.08, 0.35
            self.cam_dir_min, self.cam_dir_max = 0.4, 1.6
            self.cam_fog_beta_min, self.cam_fog_beta_max = 0.01, 0.12
            self.cam_vignette_a, self.cam_vignette_b = 0.28, 0.22
            self.cam_read_noise = 0.0025
            self.cam_prnu_std, self.cam_dsnu_std = 0.02, 0.005
            self.cam_sharpen_amount = 0.35
            self.cam_motion_blur_gain = 0.09

        # 全局缩放
        self.cam_dir_min *= self.cam_lighting_scale
        self.cam_dir_max *= self.cam_lighting_scale
        self.cam_ambient_min *= self.cam_lighting_scale
        self.cam_ambient_max *= self.cam_lighting_scale
        self.cam_fog_beta_min *= self.cam_fog_scale
        self.cam_fog_beta_max *= self.cam_fog_scale
        self.cam_read_noise *= self.cam_noise_scale
        self.cam_prnu_std *= self.cam_noise_scale
        self.cam_dsnu_std *= self.cam_noise_scale
        self.cam_motion_blur_gain *= self.cam_blur_scale
        self.cam_rolling_blur_prob = self.cam_rolling_blur_prob if self.cam_enable_rolling else 0.0

    def _reset_camera_states(self):
        """重置高保真可微相机的随机参数与时序状态。"""
        B = self.batch_size
        device = self.device

        # 光照参数
        light = torch.randn((B, 3), device=device)
        light[:, 2] = torch.abs(light[:, 2]) + 0.2  # 主光源更多来自“上方”
        self._cam_light_dir = _safe_normalize(light, -1)
        self._cam_ambient = torch.empty((B,), device=device).uniform_(self.cam_ambient_min, self.cam_ambient_max)
        self._cam_dir_intensity = torch.empty((B,), device=device).uniform_(self.cam_dir_min, self.cam_dir_max)
        self._cam_fog_beta = torch.empty((B,), device=device).uniform_(self.cam_fog_beta_min, self.cam_fog_beta_max)
        self._cam_airlight = torch.empty((B,), device=device).uniform_(self.cam_airlight_min, self.cam_airlight_max)

        # 材质先验
        self._cam_mat_ground = torch.empty((B,), device=device).uniform_(0.30, 0.55)
        self._cam_mat_obstacle = torch.empty((B,), device=device).uniform_(0.45, 0.85)
        self._cam_mat_spec = torch.empty((B,), device=device).uniform_(0.02, 0.18)

        # 镜头参数
        k1_lo, k1_hi = self.cam_dist_k1_range
        k2_lo, k2_hi = self.cam_dist_k2_range
        self._cam_dist_k1 = torch.empty((B,), device=device).uniform_(k1_lo, k1_hi)
        self._cam_dist_k2 = torch.empty((B,), device=device).uniform_(k2_lo, k2_hi)
        self._cam_flare_strength = torch.empty((B,), device=device).uniform_(0.0, self.cam_flare_strength_max)

        # ISP 参数
        self._cam_gamma = torch.empty((B,), device=device).uniform_(self.cam_gamma_min, self.cam_gamma_max)

        # 固定图样噪声
        self._cam_prnu = torch.randn((B, self.height, self.width), device=device) * self.cam_prnu_std
        self._cam_dsnu = torch.randn((B, self.height, self.width), device=device) * self.cam_dsnu_std

        # 时序状态
        self._cam_prev_y = torch.zeros((B, self.height, self.width), device=device)
        self._cam_ae_log_t = torch.zeros((B,), device=device)
        self._cam_ae_integral = torch.zeros((B,), device=device)
        self._cam_use_rolling = (torch.rand((B,), device=device) < self.cam_rolling_blur_prob).float()

    def _build_camera_rays(self, fov_tensor, R_cam_world):
        """
        构造每个像素在世界坐标系下的单位光线方向。
        Returns:
            dir_world: (B, H, W, 3)
            dir_cam:   (B, H, W, 3)
        """
        B = R_cam_world.shape[0]
        H, W = self.height, self.width
        device = R_cam_world.device
        dtype = R_cam_world.dtype

        u = torch.arange(H, device=device, dtype=dtype)
        v = torch.arange(W, device=device, dtype=dtype)
        uu, vv = torch.meshgrid(u, v, indexing='ij')
        uu = uu[None].expand(B, -1, -1)
        vv = vv[None].expand(B, -1, -1)

        fov = fov_tensor[:, None, None]
        fov_y = fov / W * H
        fu = (2.0 * (uu + 0.5) / H - 1.0) * fov_y
        fv = (2.0 * (vv + 0.5) / W - 1.0) * fov

        dir_cam = torch.stack([
            torch.ones_like(fu),
            -fv,
            -fu,
        ], -1)
        dir_cam = _safe_normalize(dir_cam, -1)

        dir_world = torch.einsum('bij,bhwj->bhwi', R_cam_world, dir_cam)
        dir_world = _safe_normalize(dir_world, -1)
        return dir_world, dir_cam

    def _estimate_normals_from_depth(self, depth):
        """
        从深度图估计近似法线（相机坐标系），再由外部变换到世界系。
        """
        x = depth[:, None]
        sobel_x = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], device=depth.device, dtype=depth.dtype) / 8.0
        sobel_y = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], device=depth.device, dtype=depth.dtype) / 8.0
        dx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), sobel_x.view(1, 1, 3, 3))[:, 0]
        dy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), sobel_y.view(1, 1, 3, 3))[:, 0]
        n = torch.stack([-dx, -dy, torch.ones_like(depth)], -1)
        return _safe_normalize(n, -1)

    def _material_prior(self, points_world, normals_world):
        """根据几何位置 + 法线估计材质反照率与镜面先验。"""
        z = points_world[..., 2]
        nz = torch.abs(normals_world[..., 2])

        # 地面软分类：接近 z=-1 且法线接近竖直
        near_ground = torch.exp(-((z - self.cam_ground_z) ** 2) / (2 * self.cam_ground_soft_band ** 2))
        flatness = torch.clamp((nz - 0.55) / 0.45, 0.0, 1.0)
        w_ground = torch.clamp(near_ground * flatness, 0.0, 1.0)

        w_obs = 1.0 - w_ground
        albedo = (
            w_ground * self._cam_mat_ground[:, None, None]
            + w_obs * self._cam_mat_obstacle[:, None, None]
        )
        spec = w_obs * self._cam_mat_spec[:, None, None]
        return albedo, spec

    def _screen_space_shadow(self, depth, light_dir_cam):
        """
        屏幕空间阴影近似：沿主光方向在图像平面采样更近深度作为遮挡证据。
        """
        if not self.cam_enable_shadow:
            return torch.ones_like(depth)
        B, H, W = depth.shape
        device = depth.device
        in_dtype = depth.dtype
        work_dtype = torch.float16 if depth.is_cuda else in_dtype

        depth_w = depth.to(work_dtype)
        light_w = light_dir_cam.to(work_dtype)

        base_y = torch.linspace(-1, 1, H, device=device, dtype=work_dtype)
        base_x = torch.linspace(-1, 1, W, device=device, dtype=work_dtype)
        gy, gx = torch.meshgrid(base_y, base_x, indexing='ij')
        gx = gx[None].expand(B, -1, -1)
        gy = gy[None].expand(B, -1, -1)

        # 光线在成像面上的方向（x: 水平, y: 垂直）
        lx = light_w[:, 1]
        ly = light_w[:, 2]
        lz = torch.clamp(light_w[:, 0].abs(), min=0.15)
        dir_u = -(ly / lz)[:, None, None]
        dir_v = -(lx / lz)[:, None, None]

        d = depth_w[:, None]
        occ = 0.0
        for t in (1.5, 3.0):
            sx = gx + dir_v * (2.0 * t / max(W, 1))
            sy = gy + dir_u * (2.0 * t / max(H, 1))
            grid = torch.stack([sx, sy], -1)
            d_shift = F.grid_sample(d, grid, mode='bilinear', padding_mode='border', align_corners=True)[:, 0]
            # 若偏移位置更“近”，说明可能遮挡主光
            occ = occ + torch.sigmoid((depth_w - d_shift - 0.03) / 0.02)
        occ = occ / 2.0
        shadow = torch.clamp(1.0 - 0.65 * occ, 0.2, 1.0)
        return shadow.to(in_dtype)

    def _apply_lens_model(self, y):
        """镜头层：暗角 + 畸变 + flare 近似。"""
        B, H, W = y.shape
        device = y.device
        dtype = y.dtype

        # 暗角（vignetting）
        yy = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xx = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing='ij')
        r2 = gx * gx + gy * gy
        vignette = torch.clamp(1.0 - self.cam_vignette_a * r2 - self.cam_vignette_b * (r2 ** 2), 0.25, 1.0)
        y = y * vignette[None]

        # 畸变（radial distortion）
        if self.cam_enable_distortion:
            gx = gx[None].expand(B, -1, -1)
            gy = gy[None].expand(B, -1, -1)
            r2b = gx * gx + gy * gy
            scale = 1.0 + self._cam_dist_k1[:, None, None] * r2b + self._cam_dist_k2[:, None, None] * (r2b ** 2)
            sx = torch.clamp(gx * scale, -1.2, 1.2)
            sy = torch.clamp(gy * scale, -1.2, 1.2)
            grid = torch.stack([sx, sy], -1)
            y = F.grid_sample(y[:, None], grid, mode='bilinear', padding_mode='border', align_corners=True)[:, 0]

        # flare（对高亮区域大核扩散）
        if self.cam_enable_flare:
            bright = torch.relu(y - 0.82)
            flare = _separable_gaussian_blur(bright, sigma=4.0)
            y = y + self._cam_flare_strength[:, None, None] * flare
        return y

    def _apply_sensor_model(self, irradiance, exposure, iso):
        """传感器层：曝光积分 + shot/read noise + PRNU/DSNU。"""
        exposure01 = torch.sigmoid(exposure)
        iso01 = torch.sigmoid(iso)

        # 命令曝光（ms 标度化到比例）+ AE 动态乘子
        t_cmd = 0.25 + 2.75 * exposure01
        t_ae = torch.exp(self._cam_ae_log_t)
        t_eff = torch.clamp(t_cmd * t_ae, 0.15, 4.0)

        # 基础电子计数
        electrons = irradiance * t_eff[:, None, None] * self.cam_base_gain

        # ISO 提升（模拟模拟/数字增益）
        iso_gain = 1.0 + 10.0 * (iso01 ** 1.2)
        electrons = electrons * iso_gain[:, None, None]

        # Shot noise（泊松高斯近似）
        shot_std = torch.sqrt(torch.clamp(electrons, min=1e-6)) * 0.03 * self.cam_noise_scale
        # Read noise
        read_std = self.cam_read_noise * (1.0 + 2.5 * iso01)

        noisy = electrons + torch.randn_like(electrons) * shot_std
        noisy = noisy + torch.randn_like(electrons) * read_std[:, None, None]

        # 固定图样噪声
        noisy = noisy * (1.0 + self._cam_prnu) + self._cam_dsnu
        return noisy, iso01

    def _apply_isp(self, raw, iso01):
        """ISP层：黑电平、增益、tone mapping、gamma、锐化/去噪近似。"""
        # 黑电平与归一化
        x = torch.relu(raw - self.cam_black_level)

        # tone mapping（Reinhard）
        x = x / (1.0 + x)

        # 降噪（高 ISO 时更强）
        denoise_strength = 0.08 + 0.28 * iso01
        smooth = _separable_gaussian_blur(x, sigma=1.0)
        x = x * (1.0 - denoise_strength[:, None, None]) + smooth * denoise_strength[:, None, None]

        # 锐化（unsharp）
        blur_small = _separable_gaussian_blur(x, sigma=0.8)
        x = x + self.cam_sharpen_amount * (x - blur_small)

        # gamma
        gamma = self._cam_gamma[:, None, None]
        x = torch.clamp(x, 0.0, 1.0) ** (1.0 / gamma)
        return torch.clamp(x, 0.0, 1.0)

    def _update_ae_state(self, y):
        """自动曝光 PI 状态机（时序层）。"""
        with torch.no_grad():
            mean_luma = y.detach().flatten(1).mean(-1)
            err = self.cam_ae_target - mean_luma
            self._cam_ae_integral = torch.clamp(self._cam_ae_integral + err, -4.0, 4.0)
            self._cam_ae_log_t = torch.clamp(
                self._cam_ae_log_t + self.cam_ae_kp * err + self.cam_ae_ki * self._cam_ae_integral,
                self.cam_ae_log_t_min,
                self.cam_ae_log_t_max,
            )

    def _apply_motion_blur(self, y):
        """时序层：全局/滚动快门风格运动模糊。"""
        if not self.cam_enable_motion_blur:
            self._cam_prev_y = y.detach()
            return y
        with torch.no_grad():
            speed = torch.norm(self.v, 2, -1)
            blur_alpha = torch.clamp(speed * self.cam_motion_blur_gain, 0.0, 0.72)

        prev = self._cam_prev_y
        if prev is None:
            self._cam_prev_y = y.detach()
            return y

        B, H, W = y.shape
        row = torch.linspace(0.0, 1.0, H, device=y.device, dtype=y.dtype)[None, :, None]
        row = row.expand(B, -1, W)

        # global shutter blur
        yg = y * (1.0 - blur_alpha[:, None, None]) + prev * blur_alpha[:, None, None]
        # rolling shutter blur（底部行受到更晚曝光影响）
        a_roll = blur_alpha[:, None, None] * row
        yr = y * (1.0 - a_roll) + prev * a_roll

        use_roll = self._cam_use_rolling[:, None, None] if self.cam_enable_rolling else torch.zeros_like(row)
        out = yg * (1.0 - use_roll) + yr * use_roll
        self._cam_prev_y = out.detach()
        return out

    def reset(self):
        """
        重置环境状态。
        在每个 episode 开始时调用，随机生成障碍物、无人机初始状态、目标点等。
        """
        B = self.batch_size
        device = self.device

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
        scale = (self.max_speed - 0.5).clamp_min(1) # 根据速度缩放场景大小
        y_stretch = (self.max_speed + 4) / scale   # Y 轴拉伸系数（用于障碍物与起终点保持一致）

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
        self.balls[..., 0] = torch.minimum(torch.maximum(self.balls[..., 0], self.balls[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.balls[..., 3])
        self.voxels[..., 0] = torch.minimum(torch.maximum(self.voxels[..., 0], self.voxels[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.voxels[..., 3])
        self.cyl[..., 0] = torch.minimum(torch.maximum(self.cyl[..., 0], self.cyl[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl[..., 2])
        self.cyl_h[..., 0] = torch.minimum(torch.maximum(self.cyl_h[..., 0], self.cyl_h[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl_h[..., 2])
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
        self.voxels[:, :, 1] *= y_stretch
        self.balls[:, :, 1] *= y_stretch
        self.cyl[:, :, 1] *= y_stretch

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
        self.voxels[..., 0] *= scale
        self.balls[..., 0] *= scale
        self.cyl[..., 0] *= scale
        self.cyl_h[..., 0] *= scale
        if self.ground_voxels:
            self.balls[:, :2, 0] = torch.minimum(torch.maximum(self.balls[:, :2, 0], ground_balls_r_ground + 0.3), scale * 8 - 0.3 - ground_balls_r_ground)

        # 6. 初始化无人机动力学参数
        # 俯仰/滚转控制延迟 (模拟底层飞控的响应时间)
        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device)
        # 偏航控制延迟
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)

        # 7. 初始化无人机位置 (p) 和目标位置 (p_target)
        scale = torch.cat([
            scale,
            y_stretch,
            torch.rand_like(scale) - 0.5], -1)
        self.p = self.p_init * scale + torch.randn_like(scale) * 0.1
        self.p_target = self.p_end * scale + torch.randn_like(scale) * 0.1

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
            scale = 1 + torch.rand((B, 1, 1), device=device)
            scaf_v = scaf_v * scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed),
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            self.cyl = torch.cat([self.cyl, scaf_v], 1)
            # 生成水平脚手架杆
            _x, _z = torch.meshgrid(x, z)
            scaf_h = torch.stack([_x, _z, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            scaf_h = scaf_h * scale + torch.stack([
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

        # ==================== 论文 §4.2 狭缝穿越环境 (Wall-Slit Environment) ====================
        if self.wall_slit:
            self._reset_wall_slit(B, device)

        # 11. 初始化空气阻力系数
        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3 # 二次阻力系数
        self.drag_2[:, 0] = 0
        self.z_drag_coef = torch.ones((B, 1), device=device) # Z轴阻力系数

        # 12. 初始化高保真可微相机状态
        self._reset_camera_states()

    def _reset_wall_slit(self, B, device):
        """
        重写障碍物和无人机位置，用于狭缝穿越 (Wall-Slit) 场景。
        创建一个带有狭窄垂直缝隙的墙壁 (沿 YZ 平面)。
        缝隙的高度大于宽度，因此无人机必须侧向翻滚 (Roll/Tilt) 才能穿过。
        无人机从墙的一侧起飞，目标点在墙的另一侧。
        """
        # 墙壁参数 (每次 reset 随机生成，但在同一个 batch 内共享)
        wall_x = 2.0 + random.random() * 4.0      # 墙壁的 X 坐标位置 [2, 6]
        slit_y_center = random.uniform(-1.0, 1.0)  # 缝隙的 Y 轴中心
        slit_z_center = random.uniform(0.0, 1.5)   # 缝隙的 Z 轴中心 (离地高度)
        slit_half_w = random.uniform(0.10, 0.18)    # 缝隙半宽 (非常窄, 总宽 ~0.20-0.36m)
        slit_half_h = random.uniform(0.35, 0.60)    # 缝隙半高 (较高, 总高 ~0.70-1.20m)
        wall_thickness = 0.15                        # 墙壁的半厚度 (X 轴方向)

        # 保存墙壁参数，用于评估和日志记录
        self.wall_x = wall_x
        self.slit_y_center = slit_y_center
        self.slit_z_center = slit_z_center
        self.slit_half_w = slit_half_w
        self.slit_half_h = slit_half_h

        # 使用 4 个 voxel 拼接成一面带有矩形开口的墙:
        #   左墙: 开口左侧的所有区域
        #   右墙: 开口右侧的所有区域
        #   顶墙: 开口上方的区域 (Y 跨度与开口相同)
        #   底墙: 开口下方的区域 (Y 跨度与开口相同)
        big = 10.0  # 足够大的半跨度，以覆盖整个场景

        wall_voxels = torch.zeros((B, 4, 6), device=device)
        # 左墙: center_y = slit_y - slit_half_w - big, ry = big
        wall_voxels[:, 0, 0] = wall_x
        wall_voxels[:, 0, 1] = slit_y_center - slit_half_w - big
        wall_voxels[:, 0, 2] = slit_z_center
        wall_voxels[:, 0, 3] = wall_thickness
        wall_voxels[:, 0, 4] = big
        wall_voxels[:, 0, 5] = big

        # 右墙: center_y = slit_y + slit_half_w + big, ry = big
        wall_voxels[:, 1, 0] = wall_x
        wall_voxels[:, 1, 1] = slit_y_center + slit_half_w + big
        wall_voxels[:, 1, 2] = slit_z_center
        wall_voxels[:, 1, 3] = wall_thickness
        wall_voxels[:, 1, 4] = big
        wall_voxels[:, 1, 5] = big

        # 顶墙: center_z = slit_z + slit_half_h + big, rz = big, ry = slit_half_w
        wall_voxels[:, 2, 0] = wall_x
        wall_voxels[:, 2, 1] = slit_y_center
        wall_voxels[:, 2, 2] = slit_z_center + slit_half_h + big
        wall_voxels[:, 2, 3] = wall_thickness
        wall_voxels[:, 2, 4] = slit_half_w
        wall_voxels[:, 2, 5] = big

        # 底墙: center_z = slit_z - slit_half_h - big, rz = big, ry = slit_half_w
        wall_voxels[:, 3, 0] = wall_x
        wall_voxels[:, 3, 1] = slit_y_center
        wall_voxels[:, 3, 2] = slit_z_center - slit_half_h - big
        wall_voxels[:, 3, 3] = wall_thickness
        wall_voxels[:, 3, 4] = slit_half_w
        wall_voxels[:, 3, 5] = big

        # 用墙壁 voxel 替换掉所有其他随机生成的障碍物
        # 将原有的障碍物移出场景
        self.balls[:, :, 2] = -200  # 移到地下深处
        self.cyl[:, :, 2] = 0.001   # 缩小到忽略不计
        self.cyl_h[:, :, 2] = 0.001
        self.voxels = wall_voxels

        # 无人机放置: 起点在墙前，终点在墙后
        dist_from_wall = 1.5 + random.random() * 1.5  # 距离墙 1.5-3.0m
        noise_y = torch.randn(B, device=device) * 0.3
        noise_z = torch.randn(B, device=device) * 0.2
        self.p = torch.stack([
            torch.full((B,), wall_x - dist_from_wall, device=device),
            torch.full((B,), slit_y_center, device=device) + noise_y,
            torch.full((B,), slit_z_center, device=device) + noise_z,
        ], -1)
        self.p_target = torch.stack([
            torch.full((B,), wall_x + dist_from_wall, device=device),
            torch.full((B,), slit_y_center, device=device) + noise_y * 0.5,
            torch.full((B,), slit_z_center, device=device) + noise_z * 0.5,
        ], -1)

        # 强制单机模式
        self.n_drones_per_group = 1
        self.drone_radius = 0.15

        # 降低最大速度，以便进行精确的机动
        self.max_speed = torch.full((B, 1), 0.5 + random.random() * 1.0, device=device) * self.speed_mtp

        # 减小安全边距 (因为缝隙很窄，允许无人机贴近障碍物)
        self.margin = torch.full((B,), 0.02, device=device)

        # 使用更新后的位置重新初始化速度、姿态等
        self.v = torch.randn((B, 3), device=device) * 0.1
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w * 0.3  # 减小风扰动
        self.act = torch.randn_like(self.v) * 0.05
        self.a = self.act
        self.dg = torch.randn((B, 3), device=device) * 0.1

        R = torch.zeros((B, 3, 3), device=device)
        self.R = quadsim_cuda.update_state_vec(
            R, self.act,
            torch.randn((B, 3), device=device) * 0.2 + F.normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay), 5)
        self.R_old = self.R.clone()
        self.p_old = self.p

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

    def render_diff(self, fov_tensor):
        """
        可微渲染函数 (Differentiable Rendering)。
        允许梯度从深度图回传到 fov_tensor (视场角参数)。
        用于训练主动感知 (Active Perception) 策略。
        """
        canvas = diff_render(fov_tensor, self.R @ self.R_cam, self.p,
                             self.balls, self.cyl, self.cyl_h, self.voxels,
                             self.n_drones_per_group, self.height, self.width)
        return canvas

    def render_main_luma(self, ctl_dt):
        """
        主相机亮度图 (Y) 渲染（非可微相机参数路径）。
        该路径要求 CUDA 扩展提供原生 YUV/Y 渲染实现，不使用任何代理转换。
        返回:
            y: (B, H, W)
        """
        y = torch.empty((self.batch_size, self.height, self.width), device=self.device)
        quadsim_cuda.render_yuv_y(
            y, self.flow, self.balls, self.cyl, self.cyl_h,
            self.voxels, self.R @ self.R_cam, self.R_old, self.p,
            self.p_old, self.drone_radius, self.n_drones_per_group,
            self._fov_x_half_tan)
        return y

    def render_main_luma_diff(self, fov_tensor, exposure, iso):
        """
        主相机亮度图 (Y) 渲染（可微相机参数路径）。
        训练时用于模拟 IMX477 RAW->ISP->YUV420 后仅取 Y 通道。
        该路径要求 CUDA 扩展提供原生可微 Y 渲染实现，不使用任何代理转换。
        (由于无人机使用固定焦距，我们去除了 focus 参数)
        返回:
            y: (B, H, W)
        """
        # ==================== 1) 几何层：深度 + 近似法线 + 材质先验 ====================
        fov_tensor = fov_tensor.contiguous()
        exposure = exposure.contiguous()
        iso = iso.contiguous()
        R_cam_world = (self.R @ self.R_cam).contiguous()
        pos = self.p.contiguous()

        depth = diff_render(
            fov_tensor,
            R_cam_world,
            pos,
            self.balls,
            self.cyl,
            self.cyl_h,
            self.voxels,
            self.n_drones_per_group,
            self.height,
            self.width,
        )
        depth = torch.clamp(depth, min=0.03, max=120.0)

        dir_world, _ = self._build_camera_rays(fov_tensor, R_cam_world)
        points_world = pos[:, None, None, :] + depth[..., None] * dir_world

        n_cam = self._estimate_normals_from_depth(depth)
        n_world = torch.einsum('bij,bhwj->bhwi', R_cam_world, n_cam)
        n_world = _safe_normalize(n_world, -1)

        albedo, specular_prior = self._material_prior(points_world, n_world)

        # ==================== 2) 光照层：环境光 + 主光源 + 阴影近似 + 大气散射 ====================
        L = self._cam_light_dir[:, None, None, :]
        ambient = self._cam_ambient[:, None, None]
        dir_int = self._cam_dir_intensity[:, None, None]

        ndotl = torch.clamp((n_world * L).sum(-1), min=0.0)

        light_cam = torch.einsum('bij,bj->bi', R_cam_world.transpose(1, 2), self._cam_light_dir)
        shadow = self._screen_space_shadow(depth, light_cam)

        # ==================== 3) 反射层：Lambert + 轻量镜面 ====================
        view_dir = _safe_normalize(-dir_world, -1)
        half_vec = _safe_normalize(L + view_dir, -1)
        ndoth = torch.clamp((n_world * half_vec).sum(-1), min=0.0)
        specular = specular_prior * (ndoth ** 24.0) if self.cam_enable_specular else torch.zeros_like(ndoth)

        irradiance = albedo * (ambient + dir_int * ndotl * shadow) + specular

        # 大气散射（airlight + Beer-Lambert）
        trans = torch.exp(-self._cam_fog_beta[:, None, None] * depth)
        irradiance = irradiance * trans + self._cam_airlight[:, None, None] * (1.0 - trans)
        irradiance = torch.clamp(irradiance, 0.0, 4.0)

        # ==================== 4) 镜头层：vignetting + 畸变 + flare ====================
        lens_y = self._apply_lens_model(irradiance)

        # ==================== 5) 传感器层：曝光 + shot/read + PRNU/DSNU ====================
        raw, iso01 = self._apply_sensor_model(lens_y, exposure, iso)

        # ==================== 6) ISP层：black/gain/tone/gamma/sharpen/denoise ====================
        y = self._apply_isp(raw, iso01)

        # ==================== 7) 时序层：AE状态机 + 运动模糊（rolling/global） ====================
        self._update_ae_state(y)
        y = self._apply_motion_blur(y)
        return torch.clamp(y, 0.0, 1.0)

    @torch.no_grad()
    def render_tof(self, ctl_dt, max_range=6.0, noise_std=0.01, return_meta=False):
        """
        训练阶段 ToF 近似观测。
        先复用几何渲染深度，再施加 ToF 风格量程截断与小噪声。

        Args:
            ctl_dt: 控制步长
            max_range: ToF 最大量程（米）
            noise_std: 高斯测距噪声标准差（米）
        Returns:
            tof_depth: (B, H_tof, W_tof)
            optional meta when return_meta=True:
                confidence: (B, H_tof, W_tof)
                valid_ratio: (B,)
                min_dist: (B,)
        """
        depth, _ = self.render(ctl_dt)
        if self.tof_downsample > 1:
            d = F.avg_pool2d(depth[:, None], self.tof_downsample, self.tof_downsample)
            depth = d[:, 0]
        if depth.shape[-2] != self.tof_height or depth.shape[-1] != self.tof_width:
            depth = F.interpolate(depth[:, None], size=(self.tof_height, self.tof_width), mode='nearest')[:, 0]
        # 量程截断
        tof_depth = depth.clamp(min=0.05, max=max_range)
        # 小幅噪声（远距离更不稳定）
        dist_scale = (tof_depth / max_range).clamp(0.0, 1.0)
        tof_depth = tof_depth + torch.randn_like(tof_depth) * noise_std * (0.5 + dist_scale)
        tof_depth = tof_depth.clamp(min=0.05, max=max_range)

        # ToF 置信度近似：距离越远、噪声越大，置信度越低
        confidence = torch.exp(-2.0 * (tof_depth / max_range)).clamp(0.0, 1.0)
        valid = (tof_depth < max_range - 1e-6).float()
        valid_ratio = valid.flatten(1).mean(-1)

        # 近场几何摘要（用于控制注入或统计）
        vec_to_pt = self.find_vec_to_nearest_pt()
        min_dist = torch.norm(vec_to_pt, 2, -1).min(0).values

        if return_meta:
            return tof_depth, confidence, valid_ratio, min_dist
        return tof_depth

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
        用于 G-DAC (Guided Differentiable Actor-Critic) 算法的内部优化循环 (Inner Loop)。
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
        用于 G-DAC 算法的内部优化循环。
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


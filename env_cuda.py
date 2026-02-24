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


# =============================================================================
# 2. 论文 §2.3 提出的可微相机传感器效应 (Optical Perception Potentials)
# =============================================================================

def apply_camera_effects(depth, exposure, iso, focus_dist):
    """
    将可微的相机传感器效应应用到渲染出的纯净深度图上。
    模拟真实相机的曝光、ISO 噪点和景深模糊。

    Args:
        depth: (B, H, W) 渲染器输出的原始深度图
        exposure: (B,) 曝光参数，范围 [0, 1] (通常由策略网络输出并经过 sigmoid)
        iso: (B,) ISO 参数，范围 [0, 1]
        focus_dist: (B,) 对焦距离参数，范围 [0, 1]
    Returns:
        (B, H, W) 带有传感器效应的深度图
    """
    # 将 [0,1] 的网络输出映射到物理范围
    exposure_phys = exposure * 10 + 0.5       # 曝光时间: [0.5, 10.5] 毫秒
    iso_phys = iso * 6400 + 100               # ISO 感光度: [100, 6500]
    focus_phys = focus_dist * 20 + 0.5        # 对焦距离: [0.5, 20.5] 米

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

    # 3. 景深模糊 (Focus distance): 模拟失焦区域的深度读取退化
    focus_phys = focus_phys[:, None, None]  # (B, 1, 1)
    dof_sigma = 4.0 # 景深范围参数
    # 计算每个像素的对焦权重：距离对焦平面越近，权重越接近 1
    focus_weight = torch.exp(-((depth.detach() - focus_phys) ** 2) / (2 * dof_sigma ** 2))
    # 焦内区域保留原深度；焦外区域与 detach 后的深度混合 (阻断梯度回传，从而鼓励网络主动对焦到障碍物上)
    depth = depth * focus_weight + depth.detach() * (1 - focus_weight)

    return depth


class Env:
    """
    无人机物理仿真环境类。
    负责管理无人机状态、障碍物生成、碰撞检测、可微渲染以及物理步进。
    支持大规模并行仿真 (Batch processing)。
    """
    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 single=False, gate=False, ground_voxels=False, scaffold=False, speed_mtp=1,
                 random_rotation=False, cam_angle=10,
                 wall_slit=False, ellipsoid_a=0.0, ellipsoid_c=0.0) -> None:
        self.device = device
        self.batch_size = batch_size
        self.width = width      # 渲染深度图的宽度
        self.height = height    # 渲染深度图的高度
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
        
        if wall_slit:
            self.single = True  # 狭缝穿越任务强制使用单机模式
            
        # 初始化环境状态
        self.reset()

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
        self.voxels[:, :, 1] *= (self.max_speed + 4) / scale
        self.balls[:, :, 1] *= (self.max_speed + 4) / scale
        self.cyl[:, :, 1] *= (self.max_speed + 4) / scale

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
        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        scale = torch.cat([
            scale,
            rd + 0.5,
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


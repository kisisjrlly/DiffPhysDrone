

在 main_cuda.py 中，`writer` (Tensorboard SummaryWriter) 记录了训练过程中的多项关键指标。主要分为**标量 (Scalars)**、**视频 (Videos)** 和 **图表 (Figures)** 三类。

以下是所有统计字段及其含义的详细说明：

### 1. 标量 (Scalars)
这些字段通过 `writer.add_scalar` 记录，主要用于监控损失函数项和性能指标。它们在 `smooth_dict` 函数中被收集。

#### Loss 相关 (用于优化)
*   **`loss`**: 总损失值。这是所有子项 loss 加权求和后的结果，优化器直接优化此目标。
*   **`loss_v`**: 速度跟踪误差。计算平均速度与目标速度之间的 Smooth L1 Loss。
*   **`loss_v_pred`**: 速度估计误差。模型预测的速度 (IMU/VIO 模拟) 与真实速度之间的 MSE Loss。
*   **`loss_obj_avoidance`**: 避障损失。基于障碍物向量场计算的 Barrier Loss，当接近安全边界时急剧增加。
*   **`loss_d_acc`**: 加速度平滑正则化。控制输出 (加速度) 的平方和，鼓励更平滑、更低能耗的动作。
*   **`loss_d_jerk`**: 加速度变化率 (Jerk) 正则化。惩罚加速度的剧烈变化，使飞行更平稳。
*   **`loss_collide`**: 碰撞惩罚。当距离小于 0 时产生的 Softplus 惩罚，接近障碍物时数值很大。
*   **`loss_ground_affinity`**: (Legacy) 地面吸附/高度限制损失。惩罚 `Z > 0` 的高度 (假设 Z 轴向上)，默认权重为 0。

#### Differentiable Camera 相关 (仅在 `--diff_cam` 开启时有效)
*   **`loss_cam_smooth`**: 相机参数平滑度。惩罚相机参数 (FOV, Exposure 等) 随时间的剧烈波动。
*   **`loss_fov_reg`**: FOV 正则化。防止视场角 (FOV) 偏离默认值太远。
*   **`loss_cam_range`**: 参数范围正则化。将相机参数约束在中心区域 (0.5 附近)，防止数值溢出或极端化。

#### 性能评估指标 (Metrics)
*   **`success`**: 成功率。当前 Batch 中未发生碰撞 (最小距离 > 0) 的轨迹比例。
*   **`max_speed`**: 最大速度。Batch 中所有无人机达到的最大速度均值。
*   **`avg_speed`**: 平均速度。Batch 中所有无人机的平均飞行速度。
*   **`ar`**: 平均奖励 (Average Reward) 或综合得分。计算公式为 `(success * avg_speed).mean()`，即考虑了生存率的平均速度 (撞毁的无人机即使速度快也会拖累此分数)。

### 2. 视频 (Videos)
*   **`demo`**: 深度图视频。
    *   可视化无人机视角的深度观测 (`depth`)。
    *   仅取 Batch 中的第 5 个样本 (`depth[4]`) 进行展示。
    *   用于直观评估感知模块是否正常工作。

### 3. 图表 (Figures)
这些是使用 Matplotlib 生成并记录的折线图，用于分析单次迭代中的轨迹详情。
*   **`p_history`**: 位置轨迹图 (X, Y, Z)。展示无人机在空间中的移动路径。
*   **`v_history`**: 速度轨迹图 (X, Y, Z)。展示速度随时间的变化。
*   **`a_reals`**: 动作/加速度输出图 (X, Y, Z)。展示神经网络输出的控制量。
*   **`cam_params`** (如果开启 `diff_cam`): 相机参数变化图。
    *   包含 4 个子图：FOV delta (视场角变化), Exposure (曝光), ISO (感光度), Focus (对焦距离)。

### 总结
*   **最核心指标**: `loss` (收敛情况), `success` (存活率), `avg_speed` (飞行效率), `ar` (综合表现)。
*   **调试用**: `demo` 视频看视觉输入，`p_history`看轨迹平滑度。

这三个 loss 项都是针对 **可微分相机参数 (Differentiable Camera Parameters)** 的正则化项，只有在开启 `--diff_cam` 参数时才会生效。

它们在代码中的计算逻辑如下（main_cuda.py 约第 283 行起）：

```python
# Camera parameter losses (differentiable perception)
if args.diff_cam and len(cam_params_history) > 1:
    cam_hist = torch.stack(cam_params_history)  # (T, B, 4)
    
    # Smoothness: penalize rapid camera parameter changes between timesteps
    cam_diff = cam_hist.diff(1, 0)
    loss_cam_smooth = cam_diff.pow(2).mean()

    # FOV regularization: keep FOV near default (fov_delta=0.5 → default FOV)
    fov_deltas = cam_hist[:, :, 0]
    loss_fov_reg = (fov_deltas - 0.5).pow(2).mean()

    # Range regularization: keep all params near center to avoid extreme values
    loss_cam_range = (cam_hist - 0.5).pow(2).mean()
```

具体的含义和作用如下：

### 1. `loss_cam_smooth` (相机参数平滑损失)
*   **含义**: 计算所有相机参数（FOV, 曝光, ISO, 对焦距离）在**连续时间步之间的变化量**的均方误差。
*   **作用**: **防止参数突变**。物理相机的变焦、光圈调整或对焦都需要时间，不可能在 1/15 秒内瞬间大幅改变。此损失强制模型预测出随时间平滑变化的参数曲线，使仿真更符合物理现实，也有助于训练稳定。

### 2. `loss_fov_reg` (视场角正则化损失)
*   **含义**: 计算预测的 FOV 调节因子 (`fov_delta`) 与 **0.5** 之间的差异。
*   **背景**: 代码中 FOV 的计算公式是 `cam_fov = base_fov * (0.5 + fov_delta)`。当 `fov_delta = 0.5` 时，相机的视场角等于默认的基础视场角 (`base_fov`)。
*   **作用**: **鼓励使用默认视场角**。它作为一个“软约束”，告诉模型除非有必要（例如为了看清远处物体而变焦），否则应该保持在标准的视场角附近，防止模型学习到一直使用极端的广角或长焦这种奇怪的策略。

### 3. `loss_cam_range` (参数范围正则化损失)
*   **含义**: 计算**所有**相机参数与 **0.5** 的均方误差。
*   **作用**: **防止参数饱和/极端化**。
    *   这些参数通常是由神经网络的 `Sigmoid` 激活函数输出的（范围 0~1）。
    *   如果参数长时间处于 0 或 1 的极端值，梯度会消失，导致网络难以继续学习。
    *   这个损失项鼓励参数保持在 0.5 的中间区域（即 Sigmoid 的线性敏感区），保证梯度流动的顺畅，防止数值溢出或死锁。

    Read [](file:///home/zhaoguodong/work/code/DiffPhysDrone/env_cuda.py#100-100), lines 100 to 300

Read [](file:///home/zhaoguodong/work/code/DiffPhysDrone/env_cuda.py#301-301), lines 301 to 403

`env.render` 和 `env.render_diff` 都是用于生成仿真环境的视觉观测（深度图），但它们的实现机制和用途有显著不同：

### 1. `env.render` vs `env.render_diff`

| 特性 | **env.render** (标准渲染) | **env.render_diff** (可微渲染) |
| :--- | :--- | :--- |
| **用途** | 用于基础训练、测试和可视化。 | 用于**可微感知 (Differentiable Perception)** 任务，即同时优化控制策略和相机参数（如 FOV）。 |
| **FOV 参数** | 使用 `self._fov_x_half_tan` (标量/固定值)。这是在 `env.reset()` 时随机初始化的，但在单次 `render` 调用中是固定不可导的。 | 接收 `fov_tensor` (Tensor) 作为输入。这是一个可学习的参数，梯度可以回传给它。 |
| **底层实现** | 调用 `quadsim_cuda.render`。这是一个标准的 CUDA 渲染核函数。 | 调用 `quadsim_cuda.render_diff_fov`。这是一个支持 FOV 梯度的定制 CUDA 核函数。 |
| **梯度流向** | **截断 (Non-differentiable)** w.r.t 相机参数。虽然通过 `self.R` 和 `self.p` 可以对无人机状态求导，但无法对相机本身的参数（如 FOV）求导。 | **全连通 (Fully Differentiable)**。梯度不仅可以回传给无人机状态，还可以通过 pixel 坐标通过链式法则回传给 FOV 等相机参数。 |

---

### 2. 相机可微 (Differentiable Camera) 是如何实现的？

相机的可微性主要通过 PyTorch 的自定义 `autograd.Function` 实现，它连接了 Python 层的自动微分图和 CUDA 底层的高效计算。

**实现代码位置**: env_cuda.py 中的 `DiffRenderFunction` 类 (第 39-66 行)。

#### 核心步骤：

1.  **前向传播 (Forward Pass)**:
    *   在 `forward` 函数中，调用专门的 CUDA 核函数 `quadsim_cuda.render_diff_fov`。
    *   该核函数不仅利用当前相机的 FOV 和位姿渲染出深度图 `canvas`，还会缓存必要的信息（如物体的位置、类型等）用于后续的反向传播。

2.  **反向传播 (Backward Pass)**:
    *   在 `backward` 函数中，接收上层（Loss）传来的梯度 `grad_output` (即 $\frac{\partial Loss}{\partial Image}$)。
    *   调用 CUDA 核函数 `quadsim_cuda.render_backward_fov`。
    *   **数学原理**: 这个核函数实现了光栅化或光线投射过程的解析梯度或近似梯度。它计算像素坐标相对于 FOV 的变化率。
        *   简单来说，改变 FOV 会导致物体在图像上的投影位置缩放。
        *   反向传播时，它根据图像边缘的梯度信息，推导出为了减少 Loss，FOV 应该变大还是变小，从而计算出 `grad_fov`。

3.  **参数优化**
    *   在 main_cuda.py 中，模型输出 `cam_params` (包含 FOV delta)。
    *   这个参数被用来计算当前的 FOV：`cam_fov = env._fov_x_half_tan * (0.5 + fov_delta)`。
    *   `cam_fov` 被传入 `env.render_diff(cam_fov)`。
    *   当 `loss.backward()` 执行时，梯度流经 `DiffRenderFunction.backward` -> `grad_fov` -> `fov_delta` -> 神经网络权重。

#### 总结
*   **`env.render`**: 只是“画”出一张图，就像拍了一张照片。
*   **`env.render_diff`**: 不仅“画”出了图，还记住了“每个像素是怎么由 FOV 决定的”。如果你觉得图片里的物体太小了（Loss 告诉你的），它可以通过梯度告诉你：“把 FOV 调小一点（变焦），物体就会变大”。这就是可微感知的核心。

if args.diff_cam and len(cam_params_history) > 0:
    cam_hist = torch.stack(cam_params_history)[:, 4].cpu()
    # [:, 4] 取 batch 中第 5 条轨迹（与速度/位置图一致）
    fig_cam, axes = plt.subplots(2, 2)      # 4 个子图
    labels = ['FOV delta', 'Exposure', 'ISO', 'Focus']
    for ci, (ax_c, lb) in enumerate(zip(axes.flatten(), labels)):
        ax_c.plot(cam_hist[:, ci].numpy())  # 画出该参数随时间步的变化
        ax_c.set_ylim(-0.05, 1.05)         # sigmoid 输出范围 [0, 1]


# DiffPhysDrone 程序参数说明

## 基础训练参数

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--resume` | str | None | 恢复训练：加载已保存的模型检查点路径 |
| `--batch_size` | int | 64 | 批处理大小（同时训练的轨迹数） |
| `--num_iters` | int | 50000 | 总训练迭代次数 |
| `--lr` | float | 1e-3 | 学习率（Adam优化器） |
| `--grad_decay` | float | 0.4 | 梯度衰减系数 |

## 损失函数系数（Loss Coefficients）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--coef_v` | 1.0 | 速度匹配损失（Smooth L1 loss：目标速度 vs 实际速度） |
| `--coef_v_pred` | 2.0 | 速度估计损失（MSE loss：无里程计情况下的速度预测） |
| `--coef_collide` | 2.0 | 碰撞避免损失（softplus loss：接近障碍物时惩罚） |
| `--coef_obj_avoidance` | 1.5 | 物体避让损失（二次清除损失） |
| `--coef_d_acc` | 0.01 | 控制加速度正则化（平滑性约束） |
| `--coef_d_jerk` | 0.001 | 控制抖动正则化（加速度变化平滑性） |
| `--coef_ground_affinity` | 0.0 | 地面亲和力损失（已弃用） |

## 可微相机参数（Differentiable Camera）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--diff_cam` | False | 启用可微感知模块（优化摄像头参数） |
| `--coef_cam_smooth` | 0.01 | 摄像头参数平滑度正则化 |
| `--coef_fov_reg` | 0.005 | FOV偏差正则化（保持视场角接近默认值） |
| `--coef_cam_range` | 0.001 | 摄像头参数范围正则化（防止极端值） |
| `--fov_x_half_tan` | 0.53 | 摄像头FOV参数（半视角的正切值） |

## 环境/模拟参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--timesteps` | 150 | 每条轨迹的时间步数 |
| `--speed_mtp` | 1.0 | 速度乘数系数 |
| `--cam_angle` | 10 | 摄像头安装角度（度） |

## 环境配置标志（Boolean Flags）

| 参数 | 含义 |
|------|------|
| `--single` | 单轨迹模式 |
| `--gate` | 启用门形障碍物 |
| `--ground_voxels` | 启用地面体素表示 |
| `--scaffold` | 启用脚手架/框架式障碍物 |
| `--random_rotation` | 随机旋转环境 |
| `--yaw_drift` | 启用偏航漂移模拟 |
| `--no_odom` | 禁用里程计（仅使用图像观测） |
| `--wandb_disabled` | 禁用Weights & Biases日志记录 |

## 用法示例

```bash
# 基础训练
python main_cuda.py --batch_size 32 --num_iters 50000

# 启用可微相机
python main_cuda.py --diff_cam --coef_cam_smooth 0.01

# 恢复训练
python main_cuda.py --resume checkpoint0001.pth

# 无里程计训练
python main_cuda.py --no_odom --coef_v_pred 2.0
```

以下为 **env_cuda.py 中环境逻辑**的结构化说明（基于你提供的源码）：

---

# 环境总体结构

`Env` 负责生成 **无人机状态 + 障碍物场景 + 相机参数**。核心在 `reset()`，每次迭代会重新随机化环境。

---

# 1. 无人机初始位置与目标位置

**固定基准点集（8个模板），按 batch 循环重复：**

## 初始位置 `self.p_init`
```text
[-1.5, -3, 1], [ 9.5, -3, 1],
[-0.5,  1, 1], [ 8.5,  1, 1],
[ 0.0,  3, 1], [ 8.0,  3, 1],
[-1.0, -1, 1], [ 9.0, -1, 1]
```

## 目标位置 `self.p_end`
```text
[8,  3, 1], [0,  3, 1],
[8, -1, 1], [0, -1, 1],
[8, -3, 1], [0, -3, 1],
[8,  1, 1], [0,  1, 1]
```

**实际位置计算方式：**
- 按组随机比例缩放 `scale`
- 添加噪声 `~ N(0, 0.1)`
```python
self.p = self.p_init * scale + noise
self.p_target = self.p_end * scale + noise
```

---

# 2. 环境中的障碍物类型

环境由 **球体 + 盒体 + 柱体**组成，并带随机参数：

| 类型 | 变量 | 形状 |
|------|------|------|
| 球体 | `self.balls` | `(B, 30, 4)` |
| 盒体 | `self.voxels` | `(B, 30, 6)` |
| 柱体 | `self.cyl` | `(B, 30, 3)` |
| 水平柱 | `self.cyl_h` | `(B, 2, 3)` |

**每种障碍物按均匀随机采样生成。**

---

# 3. 环境场景配置（可选）

由命令行参数启用：

## ✅ `--ground_voxels`
增加：
- 地面球体（模拟地面隆起）
- 地面体素块（近地障碍）

## ✅ `--gate`
生成一个“门框”，由 4 根体素柱组成，加入 `self.voxels`

## ✅ `--scaffold`
随机生成脚手架样式的格栅柱体（vertical / horizontal）

## ✅ `--random_rotation`
对场景整体施加随机偏航旋转（障碍物+起点+终点）

---

# 4. 其他环境随机化

| 项目 | 描述 |
|------|------|
| `self.max_speed` | 每组无人机最大速度随机 |
| `self.drone_radius` | 半径随机 0.1~0.15 |
| `self.v_wind` | 风速随机 |
| `self.pitch_ctl_delay / yaw_ctl_delay` | 控制延迟随机 |
| `self.drag_2` | 空阻系数随机 |
| `self._fov_x_half_tan` | 视场角随机扰动 |

---

# 5. 相机参数与姿态

- 相机角度 `cam_angle`（默认 10°）+ 噪声
- 生成 `self.R_cam`，用于渲染
- 若启用 `diff_cam`，则可对 FOV 等参数做可微优化

---

# 6. 运动状态初始化

| 状态 | 初始值 |
|------|--------|
| 速度 `self.v` | N(0, 0.2) |
| 控制 `self.act` | N(0, 0.1) |
| 加速度 `self.a` | 与 act 相同 |
| 噪声 `self.dg` | N(0, 0.2) |

---

如果你希望我进一步解释 **每种障碍物的具体几何含义**，或生成 **示意图/渲染示例**，请告诉我。



我先把与“可微感知”直接相关的代码完整梳理出来（env_cuda.py + model.py + main_cuda.py关键段），然后给你做一版“从零相机知识也能看懂”的逐行变量讲解。

Read [](file:///home/zhaoguodong/work/code/DiffPhysDrone/env_cuda.py)

Read [](file:///home/zhaoguodong/work/code/DiffPhysDrone/model.py)

Read [](file:///home/zhaoguodong/work/code/DiffPhysDrone/main_cuda.py)

Summarized conversation history我先把当前任务清单收口，确保状态清晰，然后给你做一版“零相机基础友好”的可微感知逐行级讲解。

Completed: *短跑验证稳定性* (3/8)

你这个问题问得非常好，而且非常“科研正确” 👍  
我会尽量把它讲成**你没学过相机也能懂**，同时保持和当前代码**一一对应**。

---

## 先给你一个总图（可微感知到底在干嘛）

当前项目的“可微感知”主链路是：

1. 策略网络输出相机参数（或相机参数增量）  
2. 环境用这些参数做可微渲染，生成主相机观测（亮度图 $Y$）  
3. 观测再喂回策略网络，影响下一步动作  
4. 训练时，损失对观测有梯度，梯度继续反传到相机参数（再到策略网络权重）

也就是：  
**相机参数 → 图像质量/视角 → 决策效果 → 损失 → 反传回相机参数控制策略**

核心文件：

- env_cuda.py：可微相机物理管线（重点）
- model.py：策略网络如何产生相机控制量
- main_cuda.py：训练循环里如何把相机参数接到可微渲染，再回传

---

## 相机基础知识（你完全新手版）

你只需要先记 3 个参数：

- `FOV`：视场角，越大看得越广（类似广角），越小看得更“望远”
- `Exposure`：曝光时间，越大画面越亮，但运动更容易糊
- `ISO`：增益，越大更亮，但噪声更大

项目里这 3 个参数对应张量：

- `cam_fov`
- `cam_exposure`
- `cam_iso`

并且在每个时间步都可能变化（主动感知）。

---

## env_cuda.py：可微相机逐层讲解（主战场）

下面按“真正执行顺序”讲。

### 1) 自动求导入口：`DiffRenderFunction`

在 env_cuda.py 里：

- `DiffRenderFunction.forward(...)` 调 `quadsim_cuda.render_diff_fov` 渲染深度图
- `DiffRenderFunction.backward(...)` 调 `quadsim_cuda.render_backward_fov` 算 $\partial \text{depth}/\partial \text{fov}$

关键变量：

- `fov_x_half_tan`: 每个 batch 的 FOV 参数（`tan(FOV/2)` 形式）
- `R_cam`, `pos`: 相机姿态和位置
- `balls/cyl/cyl_h/voxels`: 场景障碍物几何
- `canvas`: 输出深度图 `(B,H,W)`

这一步保证了：**几何渲染对 FOV 可导**。

---

### 2) 相机参数和状态初始化：`Env.__init__` + `_reset_camera_states`

在 `Env.__init__` 中定义了 7 层相机模型参数（你可以理解为“真实相机各子系统”）：

- 光照层：`_cam_light_dir`, `_cam_ambient`, `_cam_dir_intensity`, `_cam_fog_beta`, `_cam_airlight`
- 材质层：`_cam_mat_ground`, `_cam_mat_obstacle`, `_cam_mat_spec`
- 镜头层：`_cam_dist_k1`, `_cam_dist_k2`, `_cam_flare_strength`
- 传感器层：`_cam_prnu`, `_cam_dsnu`, `cam_read_noise`, `cam_base_gain`
- ISP层：`_cam_gamma`, `cam_sharpen_amount`
- 时序层：`_cam_prev_y`, `_cam_ae_log_t`, `_cam_ae_integral`, `_cam_use_rolling`

`_reset_camera_states()` 每个 episode 重置这些随机状态，模拟 domain randomization。

---

### 3) 主函数：`render_main_luma_diff(...)`（你最该盯的函数）

这是你问的“可微感知具体实现核心”。

#### 3.1 几何层（depth + normals + material）

- `depth = diff_render(...)`  
  用可微渲染得到深度图（对 FOV 可导）
- `dir_world, _ = _build_camera_rays(...)`  
  每个像素一条世界坐标光线方向
- `points_world = pos + depth * dir_world`  
  像素射线与深度组合成空间点
- `n_cam = _estimate_normals_from_depth(depth)`  
  用 Sobel 从深度估法线
- `n_world = R_cam_world * n_cam`  
  法线转世界系
- `albedo, specular_prior = _material_prior(...)`  
  按几何位置估计地面/障碍物反照率和镜面先验

#### 3.2 光照层（direct + ambient + shadow + fog）

- `L = _cam_light_dir`：主光方向
- `ndotl = clamp((n_world * L).sum(-1), min=0)`：Lambert 余弦项
- `light_cam = R^T * light_dir`：光方向转相机系
- `shadow = _screen_space_shadow(depth, light_cam)`：屏幕空间阴影近似
- `irradiance = albedo * (ambient + dir_int*ndotl*shadow) + specular`
- `trans = exp(-beta * depth)`：雾透过率（Beer-Lambert）
- `irradiance = irradiance*trans + airlight*(1-trans)`：空气光混合

#### 3.3 镜头层：`_apply_lens_model(y)`

- 暗角：`vignette = 1 - a*r^2 - b*r^4`
- 畸变：`scale = 1 + k1*r^2 + k2*r^4`，再 `grid_sample`
- flare：取高亮区 `relu(y-0.82)`，大核高斯扩散后加回

#### 3.4 传感器层：`_apply_sensor_model(irradiance, exposure, iso)`

- 网络输出先 `sigmoid` 到 `[0,1]`
- 有效曝光 `t_eff = t_cmd * t_ae`
- 光子电子数 `electrons = irradiance * t_eff * base_gain`
- ISO 增益 `iso_gain = 1 + 10 * iso^1.2`
- shot noise：`sqrt(electrons)` 近似
- read noise：`cam_read_noise * (1 + 2.5*iso)`
- 再叠加固定图样噪声：PRNU / DSNU

#### 3.5 ISP层：`_apply_isp(raw, iso01)`

- 黑电平：`relu(raw - black_level)`
- tone mapping：`x/(1+x)`（Reinhard）
- 降噪：高 ISO 时权重大
- 锐化：unsharp
- gamma：`x^(1/gamma)`

#### 3.6 时序层：AE + Motion Blur

- `_update_ae_state(y)`：PI 控制更新 `self._cam_ae_log_t`
- `_apply_motion_blur(y)`：
  - 全局快门模糊：当前帧与上一帧混合
  - 滚动快门模糊：按行权重混合（底部更“晚曝光”）

最后返回 `torch.clamp(y, 0, 1)`，即主相机亮度图。

---

## model.py：相机控制量是怎么从网络里出来的

关键在 `Model.forward(...)`：

1. `preprocess_sensor_inputs(...)`  
   - `depth` 模式和 `yuv` 模式做不同归一化
   - `yuv_tof` 时主相机和 ToF 双分支编码器融合
2. `img_feat + v_proj(v)` 融合视觉与状态
3. `GRUCell` 保持时序记忆
4. `fc` 输出控制

与相机相关两种模式：

- **统一控制** `use_unified_control=True`  
  `raw` 后半部分 `tanh` 成 `cam_deltas in [-1,1]`  
  对应每步增量（更像控制器）
- **传统 diff_cam**  
  `fc_cam` 后 `sigmoid` 得绝对参数 `[0,1]`  
  对应直接指定 FOV/曝光/ISO

数值护栏 `_finite` 会把 NaN/Inf 清理掉，避免训练炸掉。

---

## main_cuda.py：训练里是怎么“接线”的

每个时间步核心逻辑：

1. 先渲染观测：
   - 若 `use_cam`，调用  
     `env.render_main_luma_diff(cam_fov, cam_exposure, cam_iso)`
2. 把观测喂 `model(...)` 得到：
   - `act`
   - `cam_params`（增量或绝对值）
3. 更新相机参数：
   - unified：`cam_fov += delta_fov * scale * base_fov`（并 clamp）
   - legacy：直接映射成绝对值（并 clamp）
4. 下一步继续用新的相机参数渲染

所以它是个闭环：  
**策略改相机 → 相机改观测 → 观测改策略**。

---

## 梯度到底从哪到哪（非常关键）

可导主路径（简化）：

$$
\theta \rightarrow (cam\_fov, cam\_exp, cam\_iso) \rightarrow Y \rightarrow \text{policy output} \rightarrow \mathcal{L}
$$

再反传：

$$
\frac{\partial \mathcal{L}}{\partial \theta}
= \frac{\partial \mathcal{L}}{\partial Y}
\frac{\partial Y}{\partial (cam)}
\frac{\partial (cam)}{\partial \theta}
$$

其中：

- `diff_render` 提供 $\partial depth/\partial fov$
- lens/sensor/ISP 基本都是 PyTorch 张量算子，可自动求导
- 有些随机噪声项（`randn_like`）对随机数本身不可导，但对“加噪前变量”仍有梯度路径

---

## 你可以如何“读懂每个变量”（实战建议）

先只盯一个函数：`render_main_luma_diff`，按这个顺序看变量：

1. 几何：`depth`, `dir_world`, `points_world`, `n_world`
2. 光照：`L`, `ndotl`, `shadow`, `irradiance`
3. 镜头：`lens_y`
4. 传感器：`raw`, `iso01`
5. ISP：`y`
6. 时序：AE 更新 + blur

这 6 步理解了，项目可微感知你就已经吃透 80% 以上了。

---

## 更新后的待办状态

- [x] 定位非有限值来源  
- [x] 增加前向数值护栏  
- [x] 短跑验证稳定性  
- [x] 分析deterministic警告根因  
- [x] 给出消警告方案  
- [x] 同配置提速优化落地  
- [x] 梳理可微感知实现链路  
- [x] 讲解相机参数与变量

---

如果你愿意，我下一条可以直接给你一版：  
**“`render_main_luma_diff` 真正逐行注释版（按代码原顺序，每行都解释变量形状和物理含义）”**。  
这会非常长，但会是你以后看这套系统最稳的“说明书”。

我继续把 env_cuda.py 的可微感知部分做成“逐行变量字典式”讲解，这样你可以一边看代码一边对照每个张量的含义和形状。

Created 9 todos

好，我们直接进入“硬核逐行版” 👇  
这次我聚焦 env_cuda.py 的**主函数 `render_main_luma_diff`**，并把它依赖的关键子函数变量也串起来讲。你可以把这个回答当“配套注释手册”。

---

## `render_main_luma_diff` 逐行变量讲解（核心）

函数签名：

- `fov_tensor`: `(B,)`，每个环境一个 FOV 参数（`tan(FOV/2)`）
- `exposure`: `(B,)`，曝光控制（网络输出域）
- `iso`: `(B,)`，ISO控制（网络输出域）

返回：

- `y`: `(B,H,W)`，最终主相机亮度图（0~1）

---

### A. 输入整理与姿态拼装

`fov_tensor = fov_tensor.contiguous()`  
`exposure = exposure.contiguous()`  
`iso = iso.contiguous()`

- 作用：保证内存连续，底层 CUDA 和某些算子更稳定更快。

`R_cam_world = (self.R @ self.R_cam).contiguous()`

- `self.R`: 机体到世界旋转 `(B,3,3)`
- `self.R_cam`: 相机相对机体旋转 `(B,3,3)`
- 相乘后 `R_cam_world`: 相机到世界旋转 `(B,3,3)`

`pos = self.p.contiguous()`

- `self.p`: 无人机（相机）世界位置 `(B,3)`

---

### B. 几何层：可微深度 + 点云 + 法线 + 材质先验

`depth = diff_render(...)`

- 输出深度图 `(B,H,W)`，是可导的（尤其对 `fov_tensor`）。
- 这一步是可微感知的“几何根”。

`depth = torch.clamp(depth, min=0.03, max=120.0)`

- 防数值极端，避免后续除法/指数炸掉。

`dir_world, _ = self._build_camera_rays(fov_tensor, R_cam_world)`

- `dir_world`: `(B,H,W,3)`，每个像素一条单位视线（世界系）。
- 受 `fov_tensor` 影响，所以后续像素位置也间接受 FOV 控制。

`points_world = pos[:, None, None, :] + depth[..., None] * dir_world`

- 广播后得到每个像素对应空间点 `(B,H,W,3)`。
- 几何意义：相机中心 + 深度 × 光线方向。

`n_cam = self._estimate_normals_from_depth(depth)`

- 用 Sobel 在图像平面估计法线，输出 `(B,H,W,3)`（相机系）。

`n_world = torch.einsum('bij,bhwj->bhwi', R_cam_world, n_cam)`

- 法线转世界系 `(B,H,W,3)`。

`n_world = _safe_normalize(n_world, -1)`

- 单位化，保证后续点积有物理意义（`cos`）。

`albedo, specular_prior = self._material_prior(points_world, n_world)`

- `albedo`: `(B,H,W)`，漫反射反照率先验（地面 vs 障碍物）
- `specular_prior`: `(B,H,W)`，镜面强度先验（障碍物更强）

---

### C. 光照层：环境光 + 直射光 + 阴影 + 雾

`L = self._cam_light_dir[:, None, None, :]`

- 主光方向，扩展到像素维后 `(B,1,1,3)` 参与广播。

`ambient = self._cam_ambient[:, None, None]`
`dir_int = self._cam_dir_intensity[:, None, None]`

- 环境光强、主光强，均为 `(B,1,1)`。

`ndotl = torch.clamp((n_world * L).sum(-1), min=0.0)`

- Lambert 项：法线和光向量点积，得到 `(B,H,W)`。
- 小于0置0（背光面无直射）。

`light_cam = torch.einsum('bij,bj->bi', R_cam_world.transpose(1, 2), self._cam_light_dir)`

- 把光方向变到相机系 `(B,3)`，给屏幕空间阴影使用。

`shadow = self._screen_space_shadow(depth, light_cam)`

- `shadow`: `(B,H,W)`，值域大致 `[0.2,1]`。
- 光照被遮挡越多，`shadow` 越小。

---

### D. 反射层：漫反射 + 镜面

`view_dir = _safe_normalize(-dir_world, -1)`

- 视线反方向（从点到相机）`(B,H,W,3)`。

`half_vec = _safe_normalize(L + view_dir, -1)`

- Blinn-Phong 半角向量。

`ndoth = torch.clamp((n_world * half_vec).sum(-1), min=0.0)`

- 镜面核心项 `(B,H,W)`。

`specular = specular_prior * (ndoth ** 24.0) if self.cam_enable_specular else torch.zeros_like(ndoth)`

- 镜面高光，指数 24 让高光较尖锐。

`irradiance = albedo * (ambient + dir_int * ndotl * shadow) + specular`

- 入射亮度主表达式 `(B,H,W)`。

---

### E. 大气散射（雾）

`trans = torch.exp(-self._cam_fog_beta[:, None, None] * depth)`

- 透过率 `(B,H,W)`，深度越大越暗。

`irradiance = irradiance * trans + self._cam_airlight[:, None, None] * (1.0 - trans)`

- 经典 airlight 混合，远处偏雾白/灰。

`irradiance = torch.clamp(irradiance, 0.0, 4.0)`

- 限幅防止后续传感器层数值爆炸。

---

### F. 镜头层

`lens_y = self._apply_lens_model(irradiance)`

内部包含三件事：

1. 暗角：边缘变暗  
2. 径向畸变：`grid_sample` 重采样  
3. flare：亮部扩散光晕

输出 `lens_y: (B,H,W)`。

---

### G. 传感器层

`raw, iso01 = self._apply_sensor_model(lens_y, exposure, iso)`

这里是“电信号化”：

- `exposure` / `iso` 先 sigmoid 到 `(0,1)`
- 曝光有效时间 `t_eff`
- 电子数 `electrons`
- shot noise + read noise
- PRNU/DSNU 固定图样噪声

结果：

- `raw`: 含噪原始信号 `(B,H,W)`
- `iso01`: 归一化ISO `(B,)`（后续 ISP 还要用）

---

### H. ISP层

`y = self._apply_isp(raw, iso01)`

步骤：

1. 黑电平扣除
2. tone mapping (`x/(1+x)`)
3. 去噪（ISO 越高越强）
4. 锐化（unsharp）
5. gamma 校正

输出 `y: (B,H,W)`，范围 `[0,1]`。

---

### I. 时序层（自动曝光 + 运动模糊）

`self._update_ae_state(y)`

- PI 控制更新 `_cam_ae_log_t`，使平均亮度接近 `cam_ae_target`。
- 这里用 `no_grad`，相当于“状态机更新”，不是学习参数。

`y = self._apply_motion_blur(y)`

- 全局快门混合 `yg`
- 滚动快门混合 `yr`
- 按 `_cam_use_rolling` 混合输出
- 并把当前帧保存到 `_cam_prev_y` 做下一帧历史

`return torch.clamp(y, 0.0, 1.0)`

---

## 你最关心的“可微性断不断”总结

在这条链里：

- **连续可导主链**：几何、光照、镜头重采样、ISP大部分算子
- **噪声相关**：噪声采样本身不可导，但对输入变量仍有梯度通路（常见做法）
- **状态机更新（AE / prev frame）**：用 `no_grad` 更新内部状态，不通过这部分反传

所以训练时，梯度依然能有效传到相机控制输出（尤其 FOV/曝光/ISO 的策略输出）。

---

## 再补一个“变量速查表”（读代码时最实用）

- 几何：
  - `depth (B,H,W)`
  - `dir_world (B,H,W,3)`
  - `points_world (B,H,W,3)`
  - `n_world (B,H,W,3)`
- 材质/光照：
  - `albedo (B,H,W)`
  - `specular_prior (B,H,W)`
  - `ndotl, ndoth, shadow (B,H,W)`
  - `irradiance (B,H,W)`
- 相机层：
  - `lens_y (B,H,W)`
  - `raw (B,H,W)`
  - `y (B,H,W)`
- 参数：
  - `fov_tensor/exposure/iso (B,)`
  - `_cam_*` 多数是 `(B,)` 或 `(B,H,W)`

---

## todo 变更（本轮新增）

- [x] `逐行讲解相机主链路`

---

下一步如果你愿意，我可以继续做你要的“每行每变量”极致版本：  
我按 env_cuda.py 的**函数内真实代码顺序**，做成「行号 -> 该行数学含义 -> 张量形状 -> 影响哪个损失」四列表。  
这会更长，但会非常适合你直接拿来啃源码。

我继续把 `render_main_luma_diff` 依赖的子函数也按“逐行+变量形状+物理意义”拆开，这样你就能把整条可微感知链路从输入到输出完全吃透。

Created 10 todos

太好了，我们继续“源码啃透版”。  
这次把 `render_main_luma_diff` 背后的关键子函数逐行拆开。你把这部分看懂，基本就能独立改这个相机系统了。

---

## 1) `_build_camera_rays(fov_tensor, R_cam_world)`

这个函数做的事：**给每个像素生成一条世界坐标系单位光线**。

### 代码逻辑逐行（概念级）

- `B = R_cam_world.shape[0]`
  - batch 数量。
- `H, W = self.height, self.width`
  - 图像分辨率。
- `u = arange(H), v = arange(W)`
  - 像素网格行列索引。
- `uu, vv = meshgrid(u, v)`
  - 形成 `(H,W)` 网格，再扩展到 `(B,H,W)`。

- `fov = fov_tensor[:, None, None]`
  - 每个样本一个 FOV，广播到像素网格。
- `fov_y = fov / W * H`
  - 按宽高比换算纵向视场（近似 pinhole 的轴向缩放）。

- `fu = (2*(uu+0.5)/H - 1)*fov_y`
- `fv = (2*(vv+0.5)/W - 1)*fov`
  - 把像素坐标标准化到相机像平面坐标。

- `dir_cam = stack([1, -fv, -fu], -1)`
  - 在相机坐标里，前向是 x 轴，所以第一维固定是 1。
- `dir_cam = normalize(dir_cam)`
  - 每条射线单位化。

- `dir_world = einsum('bij,bhwj->bhwi', R_cam_world, dir_cam)`
  - 用相机旋转矩阵把光线从相机系旋转到世界系。
- `dir_world = normalize(dir_world)`

返回：

- `dir_world`: `(B,H,W,3)`
- `dir_cam`: `(B,H,W,3)`

---

## 2) `_estimate_normals_from_depth(depth)`

这个函数做的事：**从深度图估计法线（近似）**。

- `x = depth[:, None]`
  - 加通道维，变 `(B,1,H,W)`。

- 定义 `sobel_x`, `sobel_y`
  - Sobel 核，求深度在 x/y 方向梯度。

- `dx = conv2d(pad(x), sobel_x)[:,0]`
- `dy = conv2d(pad(x), sobel_y)[:,0]`
  - `dx,dy` 都是 `(B,H,W)`。

- `n = stack([-dx, -dy, ones_like(depth)], -1)`
  - 深度梯度转成法线近似，z 分量设为 1 保持朝向稳定。
- `normalize(n)`

输出：`(B,H,W,3)` 相机坐标法线。

---

## 3) `_material_prior(points_world, normals_world)`

这个函数做的事：**根据空间位置+法线猜材质**（地面 vs 障碍）。

- `z = points_world[...,2]`
- `nz = abs(normals_world[...,2])`
  - `z` 是高度；`nz` 是法线竖直程度。

- `near_ground = exp(-((z-ground_z)^2)/(2*band^2))`
  - 高斯权重：越接近地面高度，越像地面。
- `flatness = clamp((nz-0.55)/0.45,0,1)`
  - 越平坦（法线接近竖直）越像地面。
- `w_ground = clamp(near_ground * flatness,0,1)`

- `w_obs = 1 - w_ground`
- `albedo = w_ground * mat_ground + w_obs * mat_obstacle`
- `spec = w_obs * mat_spec`
  - 地面/障碍混合得到每像素漫反射和镜面先验。

输出：

- `albedo (B,H,W)`
- `spec (B,H,W)`

---

## 4) `_screen_space_shadow(depth, light_dir_cam)`

这个函数做的事：**屏幕空间阴影近似**（轻量，不做昂贵光线追踪）。

核心思路：沿光照方向在图像平面偏移采样，如果偏移处“更近”，说明当前点可能被挡光。

- `work_dtype = float16 if cuda else in_dtype`
  - 省显存加速（你最近优化点之一）。
- `depth_w = depth.to(work_dtype)`

- 构造基础网格 `gx, gy`（NDC 坐标 -1 到 1）。

- 从 `light_dir_cam` 取方向分量：
  - `lx, ly, lz`
  - `dir_u, dir_v` 代表图像平面采样方向。

- 循环 `t in (1.5, 3.0)` 两个采样半径：
  - 计算偏移网格 `sx, sy`
  - `d_shift = grid_sample(d, grid)`
  - `occ += sigmoid((depth_w - d_shift - 0.03)/0.02)`
    - 若 `d_shift` 比当前深度更小（更靠近相机），遮挡证据上升。

- `occ /= 2`
- `shadow = clamp(1 - 0.65*occ, 0.2, 1.0)`

返回：`shadow (B,H,W)`。

---

## 5) `_apply_lens_model(y)`

这个函数做的事：**镜头效应**（暗角+畸变+flare）。

### 暗角（vignette）

- 构造归一化半径 `r2 = x^2 + y^2`
- `vignette = 1 - a*r2 - b*r2^2`
- `y *= vignette`

### 畸变（radial distortion）

- `scale = 1 + k1*r2 + k2*r2^2`
- `sx = gx*scale, sy = gy*scale`
- `grid_sample` 重采样

### flare

- `bright = relu(y - 0.82)`
- `flare = gaussian_blur(bright, sigma=4)`
- `y += flare_strength * flare`

输出仍是 `(B,H,W)`。

---

## 6) `_apply_sensor_model(irradiance, exposure, iso)`

这个函数做的事：**从光照信号到传感器原始电信号**。

- `exposure01 = sigmoid(exposure)`, `iso01 = sigmoid(iso)`
  - 保证控制量有界。

- `t_cmd = 0.25 + 2.75*exposure01`
  - 用户曝光命令映射到有效时长比例。
- `t_ae = exp(_cam_ae_log_t)`
  - 自动曝光状态机乘子。
- `t_eff = clamp(t_cmd * t_ae, 0.15, 4.0)`

- `electrons = irradiance * t_eff * cam_base_gain`
- `iso_gain = 1 + 10*(iso01^1.2)`
- `electrons *= iso_gain`

噪声：

- shot noise 标准差：`sqrt(electrons)`
- read noise 标准差：`cam_read_noise*(1 + 2.5*iso01)`
- `noisy = electrons + randn*shot + randn*read`

固定图样：

- `noisy = noisy*(1+PRNU) + DSNU`

返回：

- `noisy (B,H,W)`
- `iso01 (B,)`

---

## 7) `_apply_isp(raw, iso01)`

这个函数做的事：**模拟 ISP 处理链**。

- `x = relu(raw - black_level)`：黑电平扣除
- `x = x / (1+x)`：tone mapping
- `denoise_strength = 0.08 + 0.28*iso01`
- `smooth = gaussian_blur(x)`
- `x = x*(1-denoise)+smooth*denoise`
- `x = x + sharpen_amount*(x - blur_small)`：unsharp
- `x = clamp(x,0,1)^(1/gamma)`

返回 `x`（亮度图）。

---

## 8) `_update_ae_state(y)` 与 `_apply_motion_blur(y)`

### AE（自动曝光 PI）

- `mean_luma = y.mean()`
- `err = target - mean_luma`
- `integral += err`（限幅）
- `log_t += kp*err + ki*integral`（限幅）

这是标准 PI 控制器思路。

### Motion blur

- `speed = norm(self.v)`
- `blur_alpha = clamp(speed * gain, 0, 0.72)`
- 全局快门：`yg = y*(1-a)+prev*a`
- 滚动快门：每行不同 `a_roll = a * row`
- `out = yg*(1-use_roll) + yr*use_roll`
- `prev_y = out.detach()`（作为下帧历史）

---

## 9) 和 main_cuda.py 的精确对接点（你训练时真正发生的）

在 rollout 里：

- 渲染调用：`main_obs = env.render_main_luma_diff(cam_fov, cam_exposure, cam_iso)`
- 网络输出：`act, cam_params, ... = model(...)`
- 参数更新：
  - unified 控制：增量更新 `cam_fov/exposure/iso`
  - legacy diff_cam：绝对赋值映射

也就是说，这些子函数不是“论文摆设”，而是在每个 timestep 被真实调用。

---

## 10) 你现在可以重点盯的 3 个“灵魂变量”

如果只盯 3 个，建议是：

1. `cam_fov`：影响几何投影与 blur 代价（通过有效焦距）
2. `cam_exposure`：影响亮度、噪声、运动模糊、AE耦合
3. `cam_iso`：影响噪声和ISP降噪行为

这 3 个变量就是主动感知策略在“视觉质量 vs 控制性能”之间平衡的拨杆。

---

## 本轮进度增量

- 新增完成：`逐行讲解依赖子函数` ✅

---

如果你点头，我下一条就给你做最终版：  
**把 `render_main_luma_diff` 全函数按“几乎每一行”写成带行号注释表**（含：变量形状、数值范围、梯度是否可回传、常见数值坑）。  
这版会非常适合你后续自己改代码和排查 bug。


我先把这次“仿真到真机对齐评估”的任务记到清单里并完成，然后给你一个可直接执行的落地路线（包含风险点、优先级和最小可行闭环）。

Created 12 todos

你这个问题非常专业，而且判断方向是对的。  
我先给结论，再给你“最快落地路线”。

## 结论（先说人话版）

- **能对齐，但不能指望“直接零改动一键上机=仿真效果原样复现”**。  
- 你当前项目的路线是正确的：**可微训练复杂、部署前向轻量**，这是工业上真正可落地的方式。  
- **不是越复杂越好**。对你这个阶段，复杂度应该服务于“可迁移误差最小化”，不是追求物理细节堆砌。  
- 你说的“点质（或简化）可微动力学模型反而更好学、上机更稳”——**大体成立**，前提是你把关键不可忽略误差做成“残差/随机化/安全兜底”。

---

## 你当前实现与真机的对齐程度（客观评估）

### 已经对齐得不错的点

- 有训练/部署解耦思路（你的 Paper.md 也写得很清楚）。
- 主相机可微链路包含了 FOV/曝光/ISO 对感知质量的耦合，这个方向是对的。
- ToF 作为几何安全层的角色定位正确（主从传感器分工合理）。

### 目前最可能导致“上机落差”的点

1. **IMX477 的“电子变焦”机制**  
   仿真里更像连续 FOV 参数；真机里是 ROI/crop + ISP 管线 + 帧时序影响，不完全等价。

2. **PMD flexx2 的真实误差模式**  
   真实 ToF 有多径、反射材质、边缘飞点、饱和区、温漂；仿真近似一般比真机“干净”。

3. **时延闭环**  
   相机寄存器写入延迟、曝光生效延迟、ToF 与 IMU 时间戳错位，会显著影响高速机动。

4. **动力学未建模项**  
   电机迟滞、气流干扰、推力不对称、机体振动耦合。  
   简化模型没问题，但需要残差补偿和安全边界。

---

## 回答你的核心问题：复杂模型一定更好吗？

**不是。**  
更准确地说：你要最小化的是

$$
\text{总误差} = \text{建模偏差} + \text{训练难度/不稳定性} + \text{部署实现误差}
$$

很多时候复杂模型降低了第一项，却把后两项抬得很高，结果更差。  
你这个项目当前阶段建议：

- 动力学：保持简化可微主干（你现在的做法对）
- 感知：保留关键一阶物理因子（曝光-噪声-模糊-FOV）
- 未建模误差：用随机化 + 残差 + 安全控制层补齐

这就是“灰盒最优点”。

---

## 你该怎么更快更稳地成功（按优先级）

### P0：先定义“成功标准”（一周内必须落地）

不要先追 SOTA 图，先追 4 个可测指标：

1. **穿越成功率**（无碰撞）  
2. **最小障碍距离分位数**（例如 P5）  
3. **速度-曝光反相关是否仍存在**  
4. **FOV 与障碍距离耦合趋势是否保持**

只要这 4 个在真机方向一致，你就赢了第一阶段。

---

### P1：做“硬件同构接口层”（最关键）

把仿真中的相机动作映射统一成真实寄存器语义：

- `sim_cam_fov` ↔ `IMX477 ROI/crop`（不是纯几何 FOV）
- `sim_exposure` ↔ 真机曝光时间/行时间约束
- `sim_iso` ↔ 模拟增益+数字增益分配

并且强制：

- 相同限幅
- 相同更新频率
- 相同生效延迟模型（至少一阶）

---

### P2：ToF 真机误差注入（立竿见影）

在训练中给 ToF 增加这几类扰动（比继续加光学特效更值）：

- 随机飞点/孔洞
- 边缘混叠
- 多径偏置（与材质、入射角相关的简单近似）
- 有效量程外饱和

这一步通常比“再加一层复杂镜头模型”更提升真机鲁棒性。

---

### P3：系统辨识 + 小残差模型（你现在非常需要）

做一个轻量 residual：

$$
a_{\text{real}} = a_{\text{sim}} + \Delta a_\phi(x,u)
$$

- `Δa` 用小 MLP 或线性残差即可
- 用真机短日志离线拟合
- 部署时前向补偿，不引入大计算负担

这会显著减小“仿真飞得好，真机飘”的问题。

---

### P4：分阶段上机，不要一步到位

1. 悬停+慢速避障（固定相机参数）  
2. 开启曝光自适应（只动曝光）  
3. 再开 ROI/FOV 调整  
4. 最后上高速窄缝机动

每阶段设“可回滚”开关，避免调参地狱。

---

## 你现在项目的建议架构（我建议你就这么干）

- **策略网络**：输出 intent + 相机动作（保持）
- **执行层**：dLQR/dMPC + ToF 安全兜底（保持）
- **部署**：纯前向，禁 Autograd（保持）
- **新增两件事**（最值）：
  1. 硬件参数映射一致性表（IMX477/PMD）
  2. 轻量残差动力学补偿

---

## 风险判断（帮你避坑）

- 最大风险不是“感知模型不够复杂”，而是**时序错位 + 参数映射不一致**。  
- 第二风险是“训练里学到的耦合，在真机被驱动链延迟打散”。  
- 所以你后面每次实验日志都要加：传感器时间戳、控制下发时间、相机寄存器生效帧号。

---

## 给你的最终建议（直球版）

你现在的方向很对，别被“必须上全复杂物理模型”绑架。  
你应该追求的是：

- 训练里：**可微 + 可学 + 关键机制不丢**
- 部署里：**接口同构 + 时序一致 + 安全兜底**
- 中间桥梁：**系统辨识 + 残差补偿 + 域随机化**

这样你会比“疯狂堆复杂模型”更快、更稳、更容易拿到真机展示效果。

---

如果你愿意，我下一步可以直接给你一份**“两周真机对齐执行清单（按天）”**，包括：
- 每天采什么日志
- 每天调哪些参数
- 到什么阈值才进入下一阶段  
让你按 checklist 推进，避免反复试错。


我继续给你最需要的“可执行落地版”：直接给出两周上机对齐计划（按天、带验收阈值和回滚条件），目标是最快把仿真效果稳定迁移到 IMX477 + PMD flexx2 真机。

Created 13 todos

很好，我们直接上“作战手册”版本。下面这套是给你**更快、更稳、可发表**的推进路径。

## 两周真机对齐计划（IMX477 + PMD flexx2）

### 总目标（14天后）
- 在真机上稳定复现 3 个核心现象中的至少 2 个：
  1) 速度-曝光反相关（visuo-motor interlocking）  
  2) FOV/ROI 随障碍密度动态变化（optical breathing）  
  3) 侧身通过窄缝（morphological phase transition）
- 安全指标达标：碰撞率、最小距离分位数、姿态稳定性不过阈。

---

## 第 1 周：先“对齐接口和时序”，再谈性能

### Day 1：硬件语义映射冻结（必须做）
把下面映射写成固定表（训练/评估/部署同一份）：

- `sim_fov` ↔ `IMX477 ROI crop ratio`
- `sim_exposure` ↔ `IMX477 exposure time`
- `sim_iso` ↔ `analog_gain + digital_gain`
- `tof_depth` ↔ PMD 量程裁剪 + invalid mask

**验收标准**：同一组归一化输入，仿真和真机驱动层输出的“物理参数”差异 < 5%。

---

### Day 2：时序链路标定（最容易被忽略、最致命）
记录时间戳：

- 相机帧到达时间
- ToF帧到达时间
- IMU时间
- policy输出时间
- dMPC输出时间
- 电机下发时间
- 相机寄存器生效帧号

**验收标准**：
- 控制总延迟均值 < 80ms，抖动（std）尽量 < 15ms
- 传感器对齐误差窗口可控（你文档里写的 $\pm 8$ms 是很好的目标）

---

### Day 3：ToF 真实误差建模回灌到训练
把真机日志抽样统计成扰动模型（轻量就行）：

- invalid/飞点比例
- 距离相关噪声曲线
- 边缘混叠强度
- 多径偏置（粗分段即可）

再把这些噪声注入仿真 ToF 训练支路。

**验收标准**：仿真 ToF 统计分布（均值/方差/invalid ratio）与真机误差 < 10–15%。

---

### Day 4：简化动力学 + 残差辨识
保持你现在简化模型不动，新增小残差项：

$$
a_{\text{deploy}} = a_{\text{policy}} + \Delta a_\phi(x,u)
$$

`Δa` 可先线性/小 MLP，别做复杂。

**验收标准**：短时轨迹预测误差下降 > 20%。

---

### Day 5：闭环安全壳联调（不追速度）
开启：

- policy前向
- dMPC/LQR
- ToF 安全约束兜底（近场硬约束）

但先不跑极限任务，只跑低速避障。

**验收标准**：
- 20 次回合 0 硬碰
- 控制链不断流，无时序异常峰值

---

### Day 6：只开曝光自适应（先单变量）
固定 ROI/FOV，不让它变；只让 exposure/iso 动。

看能否复现“暗场自动增曝+减速趋势”。

**验收标准**：
- speed 与 exposure 相关系数显著为负（例如 < -0.3）
- 成功率不下降

---

### Day 7：周复盘（必须做图）
出 4 张图：

1. 速度-曝光时序
2. 最小距离随时间
3. 控制延迟分布
4. 成功率/碰撞率条形图

并冻结 Week1 最优配置作为“基线”。

---

## 第 2 周：逐步开放动作自由度，逼近论文效果

### Day 8：开放 ROI/FOV（但限幅保守）
在 Week1 基线下启用 FOV/ROI 更新，限幅先收紧（防抖振）。

**验收标准**：不降低成功率前提下，出现可解释的 FOV 动态变化。

---

### Day 9：开启中速穿越任务
场景复杂度上一个台阶：稀疏→中密障碍，低速→中速。

**验收标准**：
- P5 最小距离不恶化
- 任务完成时间可接受

---

### Day 10：窄缝任务课程化
先宽缝再窄缝，不要一口吃成胖子。  
观察滚转角随缝宽变化曲线是否单调。

**验收标准**：
- 滚转角-缝宽曲线方向正确
- 至少在中等缝宽稳定通过

---

### Day 11：G-DAC 蒸馏小步快跑
Teacher 不要跑太深，先少步内循环，保证标签质量稳定。  
Student 先重 distill，物理损失低权重跟随。

**验收标准**：Student 在真机表现不劣于 teacher 轨迹统计。

---

### Day 12：Ablation 日（为论文做准备）
最少跑 3 组：

1. yuv_tof + dMPC（主方案）
2. yuv only + dMPC
3. direct action（无 dMPC）

**验收标准**：主方案在成功率/安全裕度上有清晰优势。

---

### Day 13：稳定性压力测试
不同光照、不同材质、不同起飞点，连续跑 50+ 回合。

**验收标准**：
- 无灾难性失控
- 性能方差可解释（不是随机崩）

---

### Day 14：最终 Demo 录制脚本
录三段固定流程：

1. 暗场互锁（speed-exposure）
2. 障碍丛林 optical breathing
3. 窄缝侧身通过

并固化一键实验参数。

---

## 关键原则（你问的“快而稳”核心）

1. **先时序，再模型复杂度**：时序不稳，任何 fancy 感知都白搭。  
2. **先单变量，再多变量耦合**：先 exposure，再 ROI，再全耦合。  
3. **简化主模型 + 残差补偿 > 全复杂模型硬上**。  
4. **ToF 是安全层，不要让它承担“炫技感知层”职责**。  
5. **每次只改一个自由度，必须有回滚点**。

---

## 你现在最应该立刻做的三件事（今天就能开始）

- 把 IMX477 参数映射表冻结成单一函数（训练/部署共用）
- 加完整时间戳日志链
- 用真机 ToF 日志反推噪声注入参数

这三件做完，你的 sim2real 成功率会立刻上一个台阶。

---

## 本轮待办增量

- [x] 制定两周上机计划

---

如果你愿意，我下一条可以直接给你一份**“实验记录模板（CSV 字段 + 画图脚本指标定义）”**，你照着填就能自动产出论文里的核心图（包括 speed-exposure correlation、roll-gap phase curve、success-safety Pareto）。
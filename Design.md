

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
*   **`loss_d_snap`**: Jerk 变化率 (Snap) 正则化。进一步的高阶平滑项 (标记为 legacy)。
*   **`loss_bias`**: 航向偏差损失。惩罚与目标方向垂直的速度分量，防止无人机虽然速度正确如果你偏离航向 (漂移)。
*   **`loss_speed`**: 速度幅值损失。仅关注速度大小是否匹配，不关注方向。
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
| `--coef_d_snap` | 0.0 | 控制震颤正则化（已弃用） |
| `--coef_speed` | 0.0 | 速度损失（已弃用） |
| `--coef_bias` | 0.0 | 偏差损失（已弃用） |
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
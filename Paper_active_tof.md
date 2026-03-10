---

# 论文标题 (Title)

**Spontaneous Emergence of Active-Sensory-Motor Intelligence: Unifying Differentiable Time-of-Flight Sensing and Agile Flight**
*(主动感知-运动智能的自发涌现：通过物理能量场统一可微ToF传感与敏捷飞行)*

---

# 摘要 (Abstract)

自然界的生物（如蝙蝠）展现出一种令人惊叹的协同能力：它们能在极速穿梭于极其复杂的环境时，精确地调节自身主动发出信号的强度、频次，并改变身体形态以穿过狭窄几何空间。然而，现代微型机器人系统通常将主动感知（如测距传感器的发射功率、曝光与增益）与受限的物理控制（姿态与运动）视为独立的模块。由于算力、电量与载荷的极度受限，在极端环境（如极暗、高速、极窄）下这种割裂的系统往往面临失效。

本文提出了一种 **“神经-物理流体 (Neuro-Physical Fluid)”** 框架，通过构建一个包含主动光学物理（光子散粒噪声、接收能量衰减、运动模糊）与非线性飞行力学的统一可微流形，将端到端神经网络的隐式表征能力与**可微模型预测控制 (dMPC)** 的物理严谨性深度耦合。我们将机器人的所有自由度——从主动 ToF（Time-of-Flight）深度相机的底层发射功率 (Power)、曝光时间 (Exposure)、接收增益 (Gain) 到无人机的飞行姿态——视为单一能量最小化问题的耦合变量。

在完全可微的仿真环境中进行端到端训练，我们证明了在无需人工启发式规则的情况下，无人机能自发涌现出“侧身穿缝”、“近场主动降功率”、“高速缩短曝光防模糊”等复杂的类生物行为。这一发现表明，极具适应性的具身智能并不需要复杂的规则堆砌，而是可以通过“黑盒感知意图”与“白盒物理定律”的结合，在统一的光电与运动约束场中顺梯度自然涌现。

---

# 1. 核心发现 (Main Results - The "Hook")

*在 Nature 风格中，我们先展示现象，再解释机理。*

### 1.1 动态“发声”与节能生存 (Spontaneous Energy-Adaptive Sensing)

* **现象**：在开阔或远距离障碍物前，网络会自动提高 ToF `power` 和 `gain` 以探测远景；而当靠近障碍物、即将穿过狭缝时，网络会自动且大幅度地降低 `power`。
* **物理意义**：证明了策略网络“理解”了能量平方反比定律。在近距离已有足够回波信号时，自动降低发射功率以最小化系统总能耗，体现了类似动物“节省体力”的生存本能。

### 1.2 主动感知-运动互锁 (Active Sensory-Motor Interlocking)

* **现象**：当无人机以极高速度飞行时，网络输出的 `exposure` (曝光时间) 自动下降；为了在低曝光下维持点云质量，网络会相应推高 `gain` 或 `power`。同时，一旦遇到不可穿越的高清障碍，dMPC 执行的飞行速度 $v$ 会下降，`exposure` 随之重新拉长。
* **物理意义**：证明了系统在隐空间中“理解”了 ToF 传感器的运动模糊（Flying Pixels）物理原理（Blur $\propto v \cdot t_{exp}$），它主动在“通过速度”与“曝光带来的光子信噪比”之间进行物理博弈。

### 1.3 形态相变 (Morphological Phase Transition)

* **现象**：随着前方缝隙宽度减小，无人机从平飞 ($\phi \approx 0^\circ$) 平滑过渡到侧身 ($\phi \approx 90^\circ$)，并在穿越期间维持高频的近场低功耗扫描。
* **物理意义**：策略网络结合底层的 dMPC 求解器，自发学会了利用机身体积的各向异性来最小化几何碰撞排斥势能。

---

# 2. 方法论细节 (Methods - Implementation Guide)

本系统将控制空间解耦为“网络意图”与“物理执行”两部分，打破光电传感与机电控制的界限。

**一条红线**:
> **训练 (Training)** 与 **部署 (Deployment)** 必须严格解耦。训练阶段在云端可微仿真中完成全链路的反向传播；部署阶段在机载边缘设备上仅执行前向推理与基于 C++ 的快速求解。

## 2.1 状态与动作解耦

* **物理状态 $\mathbf{x}_t$**: $\mathbf{x}_t = [\mathbf{p}, \mathbf{v}, \mathbf{R}, \mathbf{\omega}]$ (由 IMU 与状态估计提供)。
* **网络输出 (黑盒意图) $\mathbf{y}_{net}$**: $\mathbf{y}_{net} = [\mathbf{x}_{ref}, \text{Power}, \text{Exposure}, \text{Gain}]$。包含提供给 dMPC 的抽象参考轨迹 $\mathbf{x}_{ref}$，以及直接下发给 ToF 传感器的底层主动参数。
* **dMPC 输出 (白盒控制) $\mathbf{u}_{cmd}$**: $\mathbf{u}_{cmd} = [\mathbf{a}_{cmd}, \dot{\psi}_{cmd}]$。三维加速度指令与偏航角速度指令。

## 2.2 可微主动深度相机算子 (Differentiable Active ToF)

我们实现了一个纯 PyTorch 编写的可微 ToF 渲染器。从传感器参数 $(P, t_{exp}, g)$ 映射到带噪声与运动模糊的深度置信度场 $(D, C)$：

### A. 能量衰减与置信度 (Energy & Confidence)
接收能量 $E_{recv}$ 遵循物理衰减：
$$ E_{recv} \propto \frac{P \cdot t_{exp} \cdot g}{D^2 + \epsilon} $$
深度图置信度 $C = \tanh(\alpha \cdot E_{recv})$。
如果接收能量低于阈值，网络将无法获得可信深度。

### B. 运动模糊势 (Motion Blur Penalty)
当无人机处于高速时，ToF 会产生拖影（Flying Pixels）：
$$ \text{Blur Factor} \propto ||\mathbf{v}|| \cdot t_{exp} $$
模糊会直接导致深度图置信度 $C$ 下降和深度拉伸失真。

### C. 噪声注入 (Noise Injection)
采用重参数化技巧，信号越弱（或增益 $g$ 越大），深度的方差 $\sigma_{noise}^2$ 越大。

## 2.3 物理能量场损失 (Physical Energy Potentials)

训练阶段使得整个系统倾向于能量最低的状态。总损失由以下各项构成：

1. **碰撞排斥势 (Geometric Repulsion)**: $\mathcal{L}_{collision}$。基于隐式 SDF 以及飞行包络线计算。迫使系统发生侧身行为。
2. **功耗势能 (ToF Power Penalty)**: $\mathcal{L}_{power} \propto P^2$。惩罚盲目开大功率，促使其只在远距离或关键时刻耗电。
3. **模糊与失真惩罚**: $\mathcal{L}_{blur}$。促使系统在高速下缩短曝光。
4. **控制代价的拉格朗日量**: $\mathcal{L}_{control}$ (来自 LQR/dMPC 的代价)。

## 2.4 可微预测控制 (dMPC) 与微分平坦

利用多旋翼的微分平坦特性，我们将 $\mathbf{a}_{cmd}$ 解析为目标旋转矩阵 $\mathbf{R}$：
$$ \mathbf{z}_b = \frac{\mathbf{a}_{cmd} + \mathbf{g}}{||\mathbf{a}_{cmd} + \mathbf{g}||} $$
此步骤 `requires_grad=True`，使得 $\mathcal{L}_{collision}$ 产生的梯度能够从空间 SDF，穿过旋转矩阵，完美传导至加速度指令和策略网络。

## 2.5 非对称训练-部署架构 (Asymmetric Sim-to-Real Pipeline)

* **训练时**: `可微 ToF 渲染 → Policy (Intent) → 可微 dMPC → 动力学积分 → 光电/几何损失 → Autograd 反传`
* **部署时**: `真实 ToF 数据流 → Policy (Intent) → I2C 写 ToF 寄存器 (P/exp/gain) + LQR/dMPC 求解 (1ms) → 飞控指令`

---

# 3. 实验配置与架构落地

根据代码库的规范，当前可微 ToF 对应的配置方案如下：

### 3.1 命令行变量与主实验开关
请在运行时采用如下配置验证：
* `--vision_mode=active_tof`：启用可微分主动深度相机作为唯一视觉输入。
* `--use_dmpc`（或依赖统称开启联合控制的状态）：确保物理执行器由 LQR/MPC 代理。
* `--diff_cam` / `--paper_unified_control`：必须为 True，使得 `(power, exposure, gain)` 可以作为网络输出变量并接入反向传播计算图。
* `--coef_tof_power` 与 `--coef_tof_blur`：调节主动传感优化目标的关键惩罚项权重。

### 3.2 网络结构
* **ToF 编码器**：使用 2D CNN 处理带置信度掩码的深度输入 $(D, C)$。
* **状态编码器**：MLP 编码自身物理状态（位置、速度、姿态）。
* **融合与输出**：通过门控循环单元 (GRU) 融合时序特征，分别有两个输出 Head：
   - 运动意图（Navigation Reference）
   - 传感器意图（`Power`, `Exposure`, `Gain`）

### 3.3 预期对比基线 (Baselines)
为了彰显本方案的极高鲁棒性与能效比：
1. **[被动/固定参数 ToF] + [纯 RL]**: 功率、增益常开最大，极易发生运动模糊导致近距离碰撞，功耗极大。
2. **[启发式适应]**: 写死规则（如：速度过快则减小曝光；距离太近则降低功率）。难以完美耦合复杂碰撞约束下的瞬时机动。
3. **[Ours: Active-ToF + dMPC]**: 物理驱动的全知联合优化，零规则，自发平衡各项指标。

---

# 4. 给同行的主要卖点 (The Pitch)

在论文推介或答辩时，请强调以下亮点：

1. **将“光电参数”转化为可微变量**：这是具身智能首次将主动传感器底层的**发射功率**与**曝光配置**，与飞行器的牛顿力学共同放在一个可微分的大熔炉中求解，打破了机器人软硬协同的次元壁。
2. **超越“信息堆砌”的能量生存本能**：我们的多模态并不是“越多越好”。无人机学会了“够用就行”——近场主动收敛功率省电，高速主动变曝光保真。展现出惊人的生物相仿性。
3. **消除启发式代码**：急停、侧身、调光、增益，全部是神经网络为了最小化“感知质量-能量功耗-运动惩罚”这一混合标量函数的纯数学必然。

---

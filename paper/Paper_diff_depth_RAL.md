# 逆光退化下四旋翼导航的可微主动深度感知

> RAL 稿件草案。  
> 范围：本文仅描述当前 `diff_depth` 分支，对应 `configs/slit_active_sensing.args`。  
> 本版本不包含：teacher-student learning、dMPC、policy intent output、TBPTT-enabled training、复杂多场景叙事，以及更宽泛的 Nature 风格场景故事。

## 摘要

深度相机在小型空中机器人中被广泛使用，因为它们能为避障提供紧凑的几何观测。然而，Intel RealSense D 系列这类实用主动深度传感器并非被动测量设备：其输出强烈依赖可控物理寄存器，包括激光功率、曝光时间以及模拟/数字增益。传统视觉运动策略通常将深度图视为外生观测，并把这些传感参数固定或交由手工设计的自动曝光例程控制。当机器人进入“传感过程本身就是任务难点”的光照条件时，这种“感知—控制分离”会变得脆弱。

本文研究一种用于四旋翼导航的可微主动深度感知框架。核心思想是将主动深度传感器参数暴露给策略，并通过深度采集过程的可微近似，对感知—控制闭环进行端到端训练。每个控制步中，CUDA 几何渲染器先根据当前无人机位姿和固定障碍图生成理想深度图；随后，可微深度传感器模型将理想几何深度变换为退化深度观测，该变换由归一化激光功率、曝光和增益共同决定。策略以深度图、本体状态、目标相对状态以及当前传感器参数为输入，同时输出飞行指令与下一时刻主动感知参数。得到的闭环为

$$
\text{geometry} \rightarrow \text{differentiable depth sensor}
\rightarrow \text{policy network}
\rightarrow \text{camera update and control}
\rightarrow \text{quadrotor dynamics}
\rightarrow \text{loss and backpropagation}.
$$

该方法在两个固定地图场景中评估：基础障碍场景与逆光 Sun Glare 场景。基础场景用于测试可微主动深度控制是否保持标准避障性能；Sun Glare 场景用于测试策略在进入强逆光区域、局部深度质量退化时是否能自适应传感参数。实验协议将提出的可微主动深度策略与不可微感知基线、固定相机策略以及 ego-planner 风格基线进行比较。结果通过导航成功率、碰撞率、轨迹效率、深度填充质量、眩光区域局部可靠性以及学习到的相机参数统计进行报告。本文档给出完整算法表述与实验结构；数值结果有意保留为占位，待后续训练与评估填充。

## 关键词

可微感知，主动深度感知，四旋翼导航，深度相机仿真，端到端视觉运动控制，传感器参数自适应，可微物理。

## 1. 引言

小型空中机器人在感知、算力和控制上都受到严格约束。四旋翼需要在有限机载传感硬件条件下，以高频完成近距离障碍感知、安全运动选择与控制执行。深度相机特别适合这一场景，因为其直接提供几何结构，避免了从单目图像推断尺度。在许多学习型导航流程中，深度相机被建模为固定观测函数。策略接收深度图 $D_t$ 并输出控制命令 $u_t$。传感器被视作外部黑箱：

$$
D_t = \mathcal{S}(x_t),
\qquad
u_t = \pi_\theta(D_t, o_t),
$$

其中 $x_t$ 表示机器人状态，$o_t$ 表示低维本体或目标相对状态，$\pi_\theta$ 为神经策略。

对于主动深度相机，这种视角并不完整。RealSense 风格主动双目深度传感器依赖激光发射功率、曝光时间、增益等物理与固件层参数。这些参数影响信噪比、运动模糊、无效深度空洞、量程上限以及对环境红外照明的敏感性。因此，更合理的观测模型应为

$$
D_t = \mathcal{S}(x_t, c_t; \phi),
\qquad
c_t = [p_t, e_t, g_t],
$$

其中 $p_t$ 为激光功率，$e_t$ 为曝光，$g_t$ 为增益，$\phi$ 为传感器与场景参数。在困难光照下，传感器参数并非“附带项”，它们决定机器人是否看得到足够几何信息来避障。

传统系统通常通过固定相机参数、外置自动曝光启发式或离线调参处理该问题。这些方案简单，但有两点局限。第一，启发式自动曝光目标未必与下游导航目标一致。视觉上“更亮”的深度图不一定对避障更可靠。第二，正确的感知动作依赖机器人运动状态。长曝光在静态场景可能增强弱信号深度，但对高速四旋翼会带来运动模糊。高激光功率在某些条件下能增强主动双目结构，但在另一些条件下可能浪费能量或加剧镜面伪影。

本文探索另一种表述：策略同时直接控制运动与感知，并将主动深度传感器放入可微训练闭环中。我们不以手工标注动作监督相机参数，而是使用导航目标与感知可靠性目标训练策略，使其自行发现何时调整激光功率、曝光或增益能改进任务表现。

本文刻意聚焦于窄范围问题，不声称覆盖真实世界深度相机的所有效应，而是在最小、可复现设置下隔离一个核心问题：

**端到端可微主动深度流水线是否能学习到有用的传感器参数自适应，以支持四旋翼避障，尤其是在进入导致局部深度可靠性下降的逆光区域时？**

为回答该问题，我们实现了固定小地图仿真器，包含两个实验场景：

1. **基础场景：** 固定 $10\,\mathrm{m} \times 10\,\mathrm{m}$ 障碍场，含 6 根高方柱，布局要求横向绕障。
2. **Sun Glare 场景：** 固定逆光场景中，当无人机进入指定区域后，强光源投影进入视野，造成局部深度退化。

策略使用当前 `diff_depth` 流水线训练。配置采用直接动作输出、CUDA 几何渲染后接 Python/Torch 可微深度传感器模型、基于 CNN 的循环策略，并在 rollout 上进行完整 BPTT。本文版本中 teacher-student learning、dMPC 和 policy-intent output 均关闭。

### 1.1 贡献

本文稿的贡献如下：

1. **可微主动深度观测模型。**  
   我们构建 D455 风格主动深度传感器近似模型，通过可微 Torch 算子将理想渲染深度转换为含噪声、无效像素、运动模糊和光照退化的深度观测。模型显式依赖激光功率、曝光和增益。

2. **闭环主动感知—控制联合训练。**  
   我们训练一个循环视觉运动策略，同时输出飞行命令与主动深度传感参数。梯度可穿过传感器模型、神经网络、控制解码和可微四旋翼动力学。

3. **最小化逆光评估环境。**  
   我们定义紧凑的基础障碍场与 Sun Glare 变体，用于检验在逆光导致局部深度观测不可靠时，主动感知自适应是否有助于导航。

4. **面向不可微感知与规划基线的评估协议。**  
   我们提供完整实验框架，对比可微主动深度控制、固定相机学习、不可微传感自适应与 ego-planner 风格基线。

5. **透明的损失与指标设计。**  
   我们完整描述训练目标，包括任务损失、避碰、相机平滑、功率正则、能耗/模糊/噪声代理、深度填充率正则和 Sun Glare 局部可靠性项。

## 2. 相关工作

### 2.1 基于深度的四旋翼导航

深度相机因提供度量几何观测而被广泛用于空中导航。经典流水线通常先构建局部占据图或欧式符号距离场，再通过采样或优化规划安全轨迹。这类系统将感知、建图、规划和控制解耦，在结构化场景中有效，但对传感器失效敏感。若深度图存在空洞、飞点或局部无效测量，规划器可能过于保守，或错误表达障碍物。

学习式深度导航将其中部分模块替换为神经策略。策略可输入深度图与本体状态，输出速度、加速度或航路点命令。相较经典流水线，神经策略可通过训练提升对噪声与部分可观测性的鲁棒性。然而，大多数此类策略默认深度图来自固定相机模型。尽管训练中可能随机化图像噪声，传感器控制寄存器通常并不属于策略动作空间。

我们的工作将深度相机参数纳入闭环动作。机器人不仅“响应”深度图，还会主动改变未来深度图的采集方式。

### 2.2 主动视觉与传感器参数控制

主动视觉研究智能体如何改变自身感知过程以提升任务性能，包括相机运动、注视、变焦、曝光、照明和视角选择。在机器人中，主动感知往往聚焦于视角规划或 next-best-view 探索。对于主动深度相机，另一关键维度是物理传感器控制：发射功率、曝光和增益。

手工自动曝光算法常优化低层图像统计，如平均亮度或饱和比例。对主动双目深度而言，这通常不足够：亮图不一定是可靠深度图，而可靠深度图也不一定适合高速飞行。例如，提高曝光可增强信号，但也会加重运动模糊：

$$
\text{blur} \propto \|v_t\|\,t_{\mathrm{exp}},
$$

其中 $\|v_t\|$ 为相机速度，$t_{\mathrm{exp}}$ 为有效曝光时间。因此，传感控制应与机器人运动及下游安全目标耦合。

Intel RealSense D400/D455 这类实用主动双目深度相机本身也提供固件级自动曝光。根据官方文档，深度立体模组在默认模式下会自动调节 exposure 与 gain，以使红外图像强度逼近某个预设 intensity set-point；若用户显式设置 exposure 或 gain，自动曝光会被关闭。官方文档还指出，自动曝光的结果受整幅图像平均亮度及 region-of-interest 设定影响，并且在太阳或强反射进入视野时，整幅图像平均值会被高亮区域主导，从而使其余区域“变黑”；因此在户外或逆光条件下，常需缩小自动曝光 ROI，例如仅使用图像下半部分。换句话说，D455 的内置 AE 主要是一个面向红外图像强度的固件控制回路，而不是面向导航成功率、局部障碍边缘可靠性或任务风险的任务级控制器。

这一区分与本文工作直接相关。本文对比实验中的“启发式自动曝光”基线，不应被理解为对 Intel 固件 AE 的逐寄存器精确复现；更准确地说，它是一个 **D455-style global auto-exposure baseline**，即模仿其“根据全局观测统计调节 exposure/gain”的基本思想，并保持实现透明、可复现、便于公平比较。相比之下，本文方法直接优化任务损失与局部深度可靠性，并将传感器参数控制并入策略闭环。

本文方法将传感器参数作为策略输出，并通过可微任务损失而非手工曝光规则进行优化。

### 2.3 可微渲染与可微仿真

可微渲染可将图像空间目标的梯度回传到场景、相机或材质参数。机器人中的可微仿真同样可将任务损失梯度回传到控制动作与策略参数。本文流水线以工程可行方式结合两者：CUDA kernel 提供快速几何深度与四旋翼动力学，Torch 可微算子建模传感器退化过程。

该流水线不要求对全部场景几何进行 fully differentiable mesh rendering。当前目标是提供从感知可靠性损失到策略控制相机参数的梯度，以及从轨迹损失到策略控制飞行动作的梯度。该设计足以支持主动感知策略优化：

$$
\frac{\partial \mathcal{L}}{\partial \theta}
=
\sum_t
\frac{\partial \mathcal{L}}{\partial D_t}
\frac{\partial D_t}{\partial c_t}
\frac{\partial c_t}{\partial \theta}
+
\sum_t
\frac{\partial \mathcal{L}}{\partial x_{t+1}}
\frac{\partial x_{t+1}}{\partial u_t}
\frac{\partial u_t}{\partial \theta}
+ \cdots .
$$

### 2.4 端到端视觉运动策略

端到端视觉运动控制直接将传感输入映射到控制动作。由于导航具有部分可观测性（机器人无法一次看全障碍，且传感器可能暂时退化），循环网络很常见。我们的实现中，卷积深度编码器与低维状态编码器融合后输入 GRUCell。策略头输出直接加速度域命令，相机头输出主动感知参数。

不同于需要独立 planner/teacher 的方法，本文 RAL 版本仅关注直接端到端分支，不包含 teacher-student distillation，也不使用可微 MPC 模块。

## 3. 方法

### 3.1 问题表述

我们考虑四旋翼从固定起点导航到固定目标并避开障碍。时刻 $t$ 的状态为

$$
x_t =
\{p_t, v_t, R_t, a_t, c_t\},
$$

其中 $p_t \in \mathbb{R}^3$ 为位置，$v_t \in \mathbb{R}^3$ 为速度，$R_t \in SO(3)$ 为姿态，$a_t$ 为上一时刻已施加的类加速度命令，且

$$
c_t = [p^{\mathrm{cam}}_t, e_t, g_t] \in [0,1]^3
$$

为主动深度相机状态（归一化激光功率、曝光、增益）。为避免与位置 $p_t$ 冲突，我们将相机功率记为 $p^{\mathrm{cam}}_t$。

每个时间步，环境生成深度观测：

$$
D_t, Q_t = \mathcal{R}_{\phi}(x_t, c_t, \mathcal{M}),
$$

其中 $D_t \in \mathbb{R}^{H \times W}$ 为退化深度图，$Q_t \in [0,1]^{H \times W}$ 为可微质量图，$\mathcal{M}$ 为障碍图，$\phi$ 为传感器与场景参数。

策略为

$$
(y_t, \hat{c}_{t+1}, h_{t+1})
=
\pi_\theta(D_t, s_t, h_t),
$$

其中 $y_t \in \mathbb{R}^6$ 含原始飞控输出和辅助速度预测，$\hat{c}_{t+1}\in[0,1]^3$ 为原始下一时刻相机命令，$s_t$ 为低维状态向量，$h_t$ 为循环隐藏状态。

相机状态通过指数滑动平均更新：

$$
c_{t+1}
=
\alpha c_t + (1-\alpha)\hat{c}_{t+1},
\qquad
\alpha = 0.7,
$$

实现中对 $c_t$ 使用 stop-gradient，以保证相机更新数值稳定，同时保留从 $\hat{c}_{t+1}$ 到策略的当前步梯度。

飞行动作解码为类加速度命令 $u_t$，并由可微四旋翼动力学推进状态：

$$
x_{t+1} = f_{\mathrm{quad}}(x_t, u_t, \Delta t_t).
$$

训练目标最小化加权 rollout 损失：

$$
\min_{\theta}
\mathbb{E}_{\mathcal{M}, \phi}
\left[
\sum_{t=0}^{T-1}
\mathcal{L}_{\mathrm{task}}(x_t, u_t)
+
\mathcal{L}_{\mathrm{cam}}(c_t, D_t, Q_t)
\right].
$$

当前配置中，$T=80$，控制频率约为 $15\,\mathrm{Hz}$，并采用全 rollout 反向传播；TBPTT 关闭。

### 3.2 固定地图环境

仿真器使用固定 $10\,\mathrm{m} \times 10\,\mathrm{m}$ 小地图，以原点为中心。起点与终点为

$$
p_{\mathrm{start}} = [-5, 0, 1.5]^\top,
\qquad
p_{\mathrm{goal}} = [5, 0, 1.5]^\top .
$$

基础场景包含 6 根高方柱，每根半宽 $0.25\,\mathrm{m}$、半高 $1.5\,\mathrm{m}$。其在中心线两侧交替布置，迫使无人机进行非平凡绕障：

$$
\begin{aligned}
(-3.80,  0.10, 1.5),\quad
(-2.20, -0.80, 1.5),\quad
(-0.60,  0.50, 1.5),\\
( 1.00, -0.80, 1.5),\quad
( 2.60,  0.50, 1.5),\quad
( 4.20, -0.50, 1.5).
\end{aligned}
$$

Sun Glare 场景采用简化逆光环境，不追求复现阳光全部光学性质，而是构造一个最小主动深度失效模式：当无人机进入指定 $x$ 区域后，亮源投影到相机图像中，增加局部环境红外项、冲洗惩罚与有效性阈值。光源由世界坐标锚点表示：

$$
p_{\mathrm{sun}} = [7.2, 0.0, 1.8]^\top .
$$

投影到图像空间的眩光掩码为

$$
M_{\mathrm{sun}}(u,v)
=
\exp
\left(
-\frac{1}{2}
\left[
\left(\frac{u-u_s}{\sigma_u}\right)^2
+
\left(\frac{v-v_s}{\sigma_v}\right)^2
\right]
\right),
$$

其中 $(u_s,v_s)$ 为 sun anchor 的投影坐标。空间门控在无人机进入逆光区时激活效应：

$$
G_{\mathrm{zone}}(x)
=
\sigma
\left(
\frac{x - x_{\mathrm{enter}}}{\tau_{\mathrm{zone}}}
\right).
$$

最终眩光强度为

$$
M_{\mathrm{glare}} = M_{\mathrm{sun}}\,G_{\mathrm{zone}}\,\mathbb{I}_{\mathrm{visible}} .
$$

该场景刻意简化，使其真实世界对应实验可由固定 D455、小量障碍和目标方向附近强逆光/红外光源构建。

### 3.3 相机参数语义

策略输出归一化相机参数：

$$
p^{\mathrm{cam}}, e, g \in [0,1].
$$

它们被映射为语义传感量。曝光映射为有效曝光时间：

$$
t_{\mathrm{exp}}(e)
=
\mathrm{clip}
\left(
t_{\min} + t_{\mathrm{span}} e,\,
t_{\mathrm{eff,min}},\,
t_{\mathrm{eff,max}}
\right).
$$

当前配置：

$$
t_{\min}=0.25,\quad
t_{\mathrm{span}}=2.75,\quad
t_{\mathrm{eff,min}}=0.25,\quad
t_{\mathrm{eff,max}}=3.0.
$$

增益映射为语义 ISO-like 增益：

$$
G(g)
=
G_0 + G_s g^\gamma,
$$

其中

$$
G_0=1.0,\quad
G_s=10.0,\quad
\gamma=1.2 .
$$

激光功率已使用归一化表示。默认 D455-like 标称功率设为

$$
p^{\mathrm{cam}}_0 = 0.416667,
$$

对应用户测得硬件范围中的比例 $150/360$。

### 3.4 可微深度传感器模型

渲染流水线分两阶段。

首先，CUDA 几何渲染器计算理想深度：

$$
Z_t = \mathcal{G}_{\mathrm{cuda}}(x_t, \mathcal{M}).
$$

其次，可微 Torch 传感器模型将 $Z_t$ 转换为退化深度观测：

$$
D_t,Q_t
=
\mathcal{S}_{\phi}
\left(
Z_t,\,
c_t,\,
x_t,\,
\mathcal{E}
\right),
$$

其中 $Q_t$ 为可靠性/质量图，$\mathcal{E}$ 含场景相关光照和材质效应。

该模型不是像素级精确 D455 仿真器，其目的在于保留策略学习所需的主要因果关系：

1. 更高激光功率提升主动信号与量程，但增加能耗。
2. 更长曝光增强信号，但增加运动模糊。
3. 更高增益增强放大，但增加噪声。
4. 强环境红外会冲洗主动双目图案。
5. 深度不连续处更易出现飞点。
6. 低质量会提高无效深度概率。

#### 3.4.1 边缘与正视代理

对每张深度图 $Z$，通过 max pooling 估计局部近/远深度：

$$
Z_{\mathrm{far}}
=
\mathrm{MaxPool}(Z),
\qquad
Z_{\mathrm{near}}
=
-\mathrm{MaxPool}(-Z).
$$

边缘强度为

$$
E
=
\mathrm{clip}
\left(
\frac{k_E (Z_{\mathrm{far}}-Z_{\mathrm{near}})}
{Z + b_E},
0,
1.5
\right).
$$

正视代理为

$$
F
=
\exp(-k_F E).
$$

该项会在深度不连续附近降低主动双目可靠性。

#### 3.4.2 环境红外与材质项

基础环境红外水平建模为

$$
A
=
\left(
0.12
+0.55 A_{\mathrm{amb}}
+0.25 A_{\mathrm{dir}}
+0.18 A_{\mathrm{air}}
\right)
\left(
1+1.5\beta_{\mathrm{fog}}
\right).
$$

材质反照率代理为

$$
\rho
=
\mathrm{clip}(0.25 + 0.75\rho_{\mathrm{obs}}, 0.1, 1.0).
$$

镜面性表示为

$$
\kappa = \rho_{\mathrm{spec}}.
$$

在 Sun Glare 场景中，环境项被投影眩光掩码调制：

$$
A'
=
A + a_{\mathrm{glare}} M_{\mathrm{glare}}.
$$

主动信号乘子同样被调制：

$$
\mu_{\mathrm{active}}
=
\mathrm{clip}_{\min}
\left(
1
-d_{\mathrm{active}}M_{\mathrm{glare}}
+r_{\mathrm{active}}p^{\mathrm{cam}}M_{\mathrm{glare}},
0.05
\right).
$$

这表达了预期物理权衡：强逆光会削弱主动图案，但更高激光功率可部分恢复主动信号。

#### 3.4.3 主动与被动信号

主动深度信号为

$$
S_{\mathrm{active}}
=
\frac{
k_a
p^{\mathrm{cam}}
t_{\mathrm{exp}}
\rho
F
\exp(-\beta_{\mathrm{fog}} Z)
}
{Z^2 + b_a}
\mu_{\mathrm{active}} .
$$

被动信号为

$$
S_{\mathrm{passive}}
=
t_{\mathrm{exp}}
A'
\left(b_p + k_p E\right)
\left(b_\rho + k_\rho \rho\right)
\sqrt{G(g)}
\mu_{\mathrm{passive}} .
$$

主动量程门为

$$
R_{\mathrm{active}}
=
r_0
+Z_{\max}
\left(
\eta_0+\eta_1\sqrt{p^{\mathrm{cam}}t_{\mathrm{exp}}}
\right)
+\eta_g \log G(g),
$$

被动量程门为

$$
R_{\mathrm{passive}}
=
r_p
+Z_{\max}
\left(
\eta_e+\eta_A t_{\mathrm{exp}}A'
\right).
$$

软量程掩码为

$$
\Gamma_a(Z)
=
\sigma
\left(
\frac{R_{\mathrm{active}}-Z}{w_a}
\right),
\qquad
\Gamma_p(Z)
=
\sigma
\left(
\frac{R_{\mathrm{passive}}-Z}{w_p}
\right).
$$

组合信号为

$$
S
=
S_{\mathrm{active}}\Gamma_a
+\lambda_p S_{\mathrm{passive}}\Gamma_p.
$$

#### 3.4.4 冲洗、SNR 与质量

强环境红外会冲洗主动双目图案。我们将冲洗建模为

$$
W
=
\frac{A'}{S_{\mathrm{active}} + b_W}.
$$

镜面 bloom 为

$$
B_{\mathrm{spec}}
=
\kappa p^{\mathrm{cam}}
\left(0.6 + 0.4 A'\right)
\left(1+E\right).
$$

运动模糊代理为

$$
M
=
\mathrm{clip}
\left(
\|v_t\|\,t_{\mathrm{exp}}\,k_M,
0,
1.25
\right).
$$

传感器 SNR 代理为

$$
\mathrm{SNR}
=
\frac{S}
{
0.08
+\lambda_A A'
+\lambda_G G(g)
+\lambda_B B_{\mathrm{spec}}
+\lambda_M M
}.
$$

质量图为

$$
Q
=
\mathrm{clip}
\left(
\sigma
\left[
k_Q\mathrm{SNR}
+k_P S_{\mathrm{passive}}
-k_W W
-k_B B_{\mathrm{spec}}
-k_M M E
-k_R R_{\mathrm{far}}
\right]
+\Delta Q_{\mathrm{scene}},
0,
1
\right),
$$

其中

$$
R_{\mathrm{far}}
=
\max
\left(
\frac{Z}{R_{\mathrm{active}}+\epsilon}-0.9,
0
\right).
$$

在 Sun Glare 中，局部质量调整包含退化项和功率恢复项：

$$
\Delta Q_{\mathrm{glare}}
=
-k_{\mathrm{glare}} P_{\mathrm{glare}}
+k_{\mathrm{rescue}} R_{\mathrm{power}},
$$

其中

$$
P_{\mathrm{glare}}
=
M_{\mathrm{glare}}
\frac{b_g + k_e t_{\mathrm{exp}}}
{b_p + k_p p^{\mathrm{cam}}},
$$

$$
R_{\mathrm{power}}
=
M_{\mathrm{glare}}
\frac{p^{\mathrm{cam}}}
{b_r + k_r t_{\mathrm{exp}}}.
$$

这并非直接奖励高功率。只有当高功率提升局部质量目标时其才“有用”；同时损失仍惩罚过高功率。

#### 3.4.5 运动模糊、飞点、噪声与无效深度

方向性模糊通过邻域深度估计。设 $Z_h$、$Z_v$ 为水平/垂直局部模糊近似。相机坐标系速度权重为

$$
w_h
=
\frac{|v_y^{\mathrm{cam}}|}
{|v_y^{\mathrm{cam}}|+|v_z^{\mathrm{cam}}|+\epsilon},
\qquad
w_v
=
\frac{|v_z^{\mathrm{cam}}|}
{|v_y^{\mathrm{cam}}|+|v_z^{\mathrm{cam}}|+\epsilon}.
$$

方向模糊深度为

$$
Z_{\mathrm{dir}}
=
w_h Z_h + w_v Z_v.
$$

融合后深度为

$$
Z_{\mathrm{blur}}
=
(1-\alpha_M)Z + \alpha_M Z_{\mathrm{dir}},
\qquad
\alpha_M = \mathrm{clip}(k_{\mathrm{blend}}M,0,\alpha_{\max}).
$$

飞点概率建模为

$$
F_{\mathrm{fly}}
=
\mathrm{clip}
\left[
\left(b_f + k_f(1-Q)\right)
E
\left(b_m + k_m(M+B_{\mathrm{spec}})\right),
0,
1
\right].
$$

受损（加噪前）深度为

$$
Z_{\mathrm{corr}}
=
Z_{\mathrm{blur}}
+F_{\mathrm{fly}}
\left(
Z_{\mathrm{far}}-Z_{\mathrm{blur}}
\right).
$$

深度噪声标准差为

$$
\sigma_Z
=
\mathrm{clip}
\left(
\sigma_{\mathrm{read}}(1+\lambda_G G)
+
\frac{k_{\mathrm{sig}}(1+\lambda_R (Z/Z_{\max})^2)}
{S+b_\sigma}
+
k_{\mathrm{mot}}M(b_E+k_EE)
+
k_{\mathrm{spec}}B_{\mathrm{spec}},
\sigma_{\min},
\sigma_{\max}
\right).
$$

加噪后深度为

$$
\tilde{Z}
=
Z_{\mathrm{corr}}
+\epsilon_Z\sigma_Z,
\qquad
\epsilon_Z\sim\mathcal{N}(0,1).
$$

有效性掩码为可微形式：

$$
V
=
\sigma
\left(
\frac{Q-\tau_{\mathrm{valid}}}{s_{\mathrm{valid}}}
\right).
$$

最终深度与质量为

$$
D = \mathrm{clip}(\tilde{Z}, Z_{\min}, Z_{\max})V,
\qquad
Q_{\mathrm{out}} = QV.
$$

### 3.5 策略输入的深度预处理

策略接收处理后的单通道深度张量。无效深度不应被当作近障碍。设 $D$ 为渲染深度图，$Z_{\min}, Z_{\max}$ 为有效深度范围。有效掩码为

$$
\Omega(i,j)=\mathbb{I}[D(i,j)\ge Z_{\min}].
$$

安全深度为

$$
D_{\mathrm{safe}}
=
\begin{cases}
\mathrm{clip}(D,Z_{\min},Z_{\max}), & \Omega=1,\\
Z_{\max}, & \Omega=0.
\end{cases}
$$

逆深度归一化输入为

$$
I
=
\frac{
D_{\mathrm{safe}}^{-1} - Z_{\max}^{-1}
}
{
Z_{\min}^{-1}-Z_{\max}^{-1}
}
\Omega.
$$

图像通过自适应最大池化变换到神经网络输入分辨率：

$$
I_{\mathrm{nn}}
=
\mathrm{AdaptiveMaxPool}(I, H_{\mathrm{nn}},W_{\mathrm{nn}}).
$$

当前配置：

$$
H\times W = 48\times 64,
\qquad
H_{\mathrm{nn}}\times W_{\mathrm{nn}} = 24\times 32.
$$

最后：

$$
\bar{I}=2I_{\mathrm{nn}}-1.
$$

### 3.6 策略网络

策略有三类输入：

1. 处理后深度图 $\bar{I}_t$；
2. 低维状态 $s_t$；
3. 循环隐藏状态 $h_t$。

状态向量包含局部速度、目标相对速度、机体上方向、安全裕度和当前相机状态：

$$
s_t =
\left[
v_t^{\mathrm{local}},
v_{\mathrm{target},t}^{\mathrm{local}},
R_t[:,2],
m,
2c_t-1
\right].
$$

在启用里程计且包含相机状态时，状态维度为 $13$。

#### 3.6.1 深度编码器

深度图经三层卷积编码：

$$
\begin{aligned}
F_1 &= \phi(\mathrm{Conv}_{3\times3}^{1\rightarrow 32}(\bar{I})),\\
F_2 &= \phi(\mathrm{Conv}_{3\times3,s=2}^{32\rightarrow 64}(F_1)),\\
F_3 &= \phi(\mathrm{Conv}_{3\times3}^{64\rightarrow 128}(F_2)).
\end{aligned}
$$

特征池化并线性投影：

$$
z_D
=
W_D
\mathrm{Flatten}
\left(
\mathrm{AdaptiveAvgPool}_{3\times6}(F_3)
\right)
\in \mathbb{R}^{192}.
$$

#### 3.6.2 状态编码器与门控融合

状态编码器：

$$
z_s = W_s s_t \in \mathbb{R}^{192}.
$$

深度与状态特征均做层归一化：

$$
\hat{z}_D=\mathrm{LN}(z_D),
\qquad
\hat{z}_s=\mathrm{LN}(z_s).
$$

学习到的门控平衡视觉与状态特征：

$$
\lambda_t
=
\sigma
\left(
W_2 \phi(W_1[\hat{z}_D,\hat{z}_s])
\right),
$$

$$
z_t
=
\phi
\left(
\lambda_t\odot \hat{z}_D
+
(1-\lambda_t)\odot \hat{z}_s
\right).
$$

该机制可避免训练全程由单一模态支配。

#### 3.6.3 循环记忆

循环更新为

$$
\tilde{h}_{t+1}
=
\mathrm{GRUCell}(z_t,h_t).
$$

并施加残差稳定器：

$$
h_{t+1}
=
\mathrm{LN}
\left(
\tilde{h}_{t+1}
+0.1 f_{\mathrm{res}}(\tilde{h}_{t+1})
\right).
$$

当深度观测局部无效、或机器人需记忆近期几何信息时，循环状态尤为重要。

#### 3.6.4 飞行头与相机头

飞行头输出 6 维向量：

$$
y_t = W_u\phi(h_{t+1})\in\mathbb{R}^6.
$$

经局部坐标变换 $R_{\mathrm{local}}$ 后重排为两个 3D 向量：

$$
(a_t^{\mathrm{pred}}, v_t^{\mathrm{pred}})
=
R_{\mathrm{local}}\,
\mathrm{reshape}(y_t).
$$

直接加速度命令为

$$
u_t
=
\mathrm{clip}
\left(
(a_t^{\mathrm{pred}}-g_{\mathrm{std}})\eta_{\mathrm{thr}}
+g_{\mathrm{std}},
-u_{\max},
u_{\max}
\right),
$$

其中 $u_{\max}=20.0$。

相机头输出

$$
\hat{c}_{t+1}
=
\sigma(W_c\phi(h_{t+1}))
\in [0,1]^3.
$$

三个通道分别对应激光功率、曝光与增益。

### 3.7 闭环可微训练

rollout 循环如下：

1. 渲染可微深度：

   $$
   (D_t,Q_t)=\mathcal{R}_{\phi}(x_t,c_t,\mathcal{M}).
   $$

2. 由 $Q_t$ 计算软填充率与局部可靠性指标。
3. 构建状态向量 $s_t$。
4. 评估策略：

   $$
   (y_t,\hat{c}_{t+1},h_{t+1})=\pi_\theta(D_t,s_t,h_t).
   $$

5. 更新相机状态：

   $$
   c_{t+1}=0.7c_t+0.3\hat{c}_{t+1}.
   $$

6. 解码飞行动作 $u_t$。
7. 推进可微四旋翼动力学：

   $$
   x_{t+1}=f_{\mathrm{quad}}(x_t,u_t,\Delta t_t).
   $$

8. 累积损失并通过完整 rollout 反向传播。

有效控制步长包含曝光相关延迟代理：

$$
\Delta t_t
=
\Delta t_{\mathrm{base}}
+0.01\,t_{\mathrm{exp}}(e_t).
$$

这使感知延迟与控制耦合。

完整梯度含两条关键路径：

$$
\frac{\partial \mathcal{L}}{\partial \theta}
\supset
\frac{\partial \mathcal{L}_{\mathrm{perception}}}{\partial Q_t}
\frac{\partial Q_t}{\partial c_t}
\frac{\partial c_t}{\partial \theta},
$$

以及

$$
\frac{\partial \mathcal{L}}{\partial \theta}
\supset
\frac{\partial \mathcal{L}_{\mathrm{task}}}{\partial x_{t+1}}
\frac{\partial x_{t+1}}{\partial u_t}
\frac{\partial u_t}{\partial \theta}.
$$

因此策略可利用任务级反馈同时优化运动与感知。

### 3.8 算法总结

算法 1 总结了当前 `diff_depth` 分支的训练过程。关键点在于：主动深度传感器在每步策略动作计算前、于 rollout 内部执行；传感质量损失与导航损失共同累积后再反向传播。

#### 算法 1：可微主动深度 rollout 训练

**输入：** 固定障碍图 $\mathcal{M}$、场景 profile $\phi$、策略参数 $\theta$、rollout 长度 $T$、batch 大小 $B$、初始相机状态 $c_0$、初始循环状态 $h_0$。  
**输出：** 更新后的策略参数 $\theta$。

1. 初始化批量四旋翼状态：

   $$
   x_0^{1:B} \leftarrow p_{\mathrm{start}}, v_0, R_0 .
   $$

2. 初始化相机参数：

   $$
   c_0^{1:B} \leftarrow [p_0,0.5,0.5],
   \qquad
   p_0=0.416667 .
   $$

3. 对于 $t=0,\ldots,T-1$：

   $$
   Z_t \leftarrow \mathcal{G}_{\mathrm{cuda}}(x_t,\mathcal{M})
   $$

   $$
   D_t,Q_t,\Psi_t
   \leftarrow
   \mathcal{S}_{\phi}(Z_t,c_t,x_t),
   $$

   其中 $\Psi_t$ 表示辅助可微传感统计量，如填充率、空洞率、局部眩光掩码、局部眩光质量、模糊代理和噪声代理。

4. 预处理退化深度：

   $$
   \bar{I}_t \leftarrow \mathrm{PreprocessDepth}(D_t).
   $$

5. 构建低维状态：

   $$
   s_t \leftarrow
   [
   v_t^{\mathrm{local}},
   v_{\mathrm{target},t}^{\mathrm{local}},
   R_t[:,2],
   m_t,
   2c_t-1
   ].
   $$

6. 评估循环策略：

   $$
   y_t,\hat{c}_{t+1},h_{t+1}
   \leftarrow
   \pi_\theta(\bar{I}_t,s_t,h_t).
   $$

7. 解码飞行命令并更新相机状态：

   $$
   u_t \leftarrow \mathrm{DecodeAction}(y_t),
   $$

   $$
   c_{t+1}
   \leftarrow
   0.7\,\mathrm{stopgrad}(c_t)+0.3\,\hat{c}_{t+1}.
   $$

8. 推进动力学：

   $$
   x_{t+1}\leftarrow f_{\mathrm{quad}}(x_t,u_t,\Delta t_t).
   $$

9. 累积每步损失：

   $$
   \mathcal{L}
   \leftarrow
   \mathcal{L}
   +\mathcal{L}_{\mathrm{task}}(x_t,u_t)
   +\mathcal{L}_{\mathrm{cam}}(c_t,\hat{c}_t)
   +\mathcal{L}_{\mathrm{depth}}(\Psi_t).
   $$

10. 对展开计算图反向传播：

    $$
    \theta
    \leftarrow
    \mathrm{AdamW}
    \left(
    \theta,
    \nabla_\theta \mathcal{L}
    \right).
    $$

该算法相较传统 depth-policy 训练循环有两点不同：第一，深度观测分布不是固定输入分布，而是策略控制相机寄存器的可微函数；第二，相机命令没有目标轨迹监督，仅通过导航与感知可靠性目标优化。

### 3.9 训练目标

总损失为

$$
\mathcal{L}
=
\lambda_v\mathcal{L}_v
+\lambda_{\mathrm{avoid}}\mathcal{L}_{\mathrm{avoid}}
+\lambda_{\mathrm{coll}}\mathcal{L}_{\mathrm{coll}}
+\lambda_{\mathrm{acc}}\mathcal{L}_{\mathrm{acc}}
+\lambda_{\mathrm{jerk}}\mathcal{L}_{\mathrm{jerk}}
+\lambda_{\mathrm{cam}}\mathcal{L}_{\mathrm{cam}}
+\lambda_{\mathrm{depth}}\mathcal{L}_{\mathrm{depth}} .
$$

当前配置：

$$
\lambda_v=2.5,\quad
\lambda_{\mathrm{avoid}}=4.0,\quad
\lambda_{\mathrm{coll}}=10.0,\quad
\lambda_{\mathrm{acc}}=0.1,\quad
\lambda_{\mathrm{jerk}}=0.2.
$$

#### 3.9.1 速度跟踪

设 $\bar{v}_t$ 为窗口平均速度：

$$
\bar{v}_t
=
\frac{1}{K}
\sum_{k=0}^{K-1} v_{t+k}.
$$

速度损失为 smooth L1：

$$
\mathcal{L}_v
=
\mathrm{SmoothL1}
\left(
\|\bar{v}_t-v^{\mathrm{target}}_t\|_2,
0
\right).
$$

#### 3.9.2 动作平滑

加速度正则为

$$
\mathcal{L}_{\mathrm{acc}}
=
\frac{1}{TB}
\sum_{t,b}
\|u_{t,b}\|_2^2.
$$

jerk 正则为

$$
\mathcal{L}_{\mathrm{jerk}}
=
\frac{1}{TB}
\sum_{t,b}
\left\|
\frac{u_{t,b}-u_{t-1,b}}{\Delta t}
\right\|_2^2.
$$

实现中按名义控制频率缩放动作差分。

#### 3.9.3 避障与碰撞损失

设 $d_{t,b}$ 为机器人 $b$ 到最近障碍表面的符号距离减去安全裕度。软避障势垒为

$$
\mathcal{L}_{\mathrm{avoid}}
=
\mathbb{E}
\left[
w_{t,b}
\left(
\max(0,1-d_{t,b})
\right)^2
\right].
$$

碰撞损失为

$$
\mathcal{L}_{\mathrm{coll}}
=
\mathbb{E}
\left[
w_{t,b}
\mathrm{softplus}(-32d_{t,b})
\right].
$$

当机器人快速接近障碍时，权重 $w_{t,b}$ 增大。

#### 3.9.4 相机平滑与范围正则

设 $\hat{c}_t=[\hat{p}_t,\hat{e}_t,\hat{g}_t]$ 为相机头原始输出。平滑项为

$$
\mathcal{L}_{\mathrm{cam,smooth}}
=
\mathbb{E}_t
\left[
\|\hat{c}_t-\hat{c}_{t-1}\|_2^2
\right].
$$

功率正则在标称 D455-like 值 $p_0$ 周围采用 deadband：

$$
\mathcal{L}_{\mathrm{power,reg}}
=
\mathbb{E}
\left[
\max
\left(
0,\,
|\hat{p}_t-p_0|-\delta_p
\right)^2
\right].
$$

当前配置：

$$
p_0=0.416667,\qquad
\delta_p=0.18.
$$

曝光与增益范围正则为

$$
\mathcal{L}_{\mathrm{cam,range}}
=
\mathbb{E}
\left[
(\hat{e}_t-0.5)^2
+(\hat{g}_t-0.5)^2
\right].
$$

相机平滑、功率正则和范围项权重目前为

$$
\lambda_{\mathrm{cam,smooth}}=100,\quad
\lambda_{\mathrm{power,reg}}=100,\quad
\lambda_{\mathrm{cam,range}}=1.
$$

#### 3.9.5 深度传感器损失

能耗代理：

$$
\mathcal{L}_{\mathrm{power}}
=
\mathbb{E}
\left[
\max(0,p^{\mathrm{cam}}_t-p_{\mathrm{thr}})^2
\right],
$$

其中

$$
p_{\mathrm{thr}}=0.416667.
$$

模糊代理：

$$
\mathcal{L}_{\mathrm{blur}}
=
\mathbb{E}
\left[
\left(
\|v_t\|_2 t_{\mathrm{exp}}(e_t)
\right)^2
\right].
$$

噪声代理：

$$
\mathcal{L}_{\mathrm{noise}}
=
\mathbb{E}
\left[
g_t^2
\right].
$$

软填充率损失使用可微填充代理：

$$
F_t
=
\frac{1}{HW}
\sum_{i,j}
\sigma
\left(
\frac{Q_t(i,j)-q_{\min}}{\tau_q}
\right).
$$

填充损失为

$$
\mathcal{L}_{\mathrm{fill}}
=
\max(0,F_{\min}-F_t)^2.
$$

当前配置：

$$
F_{\min}=0.25.
$$

对 Sun Glare，在眩光掩码内部计算局部质量项：

$$
\bar{Q}_{\mathrm{glare}}
=
\frac{
\sum_{i,j} M_{\mathrm{glare}}(i,j) Q(i,j)
}
{
\sum_{i,j} M_{\mathrm{glare}}(i,j)+\epsilon
}.
$$

局部 Sun Glare 可靠性损失为

$$
\mathcal{L}_{\mathrm{glare}}
=
\max
\left(
0,\,
Q_{\mathrm{target}}-\bar{Q}_{\mathrm{glare}}
\right)^2.
$$

当前配置为

$$
Q_{\mathrm{target}}=0.1,
\qquad
\lambda_{\mathrm{glare}}=30.0.
$$

该项并不监督具体相机动作，不会显式要求“提高功率”或“降低曝光”。它只定义局部可靠性目标。若仿真物理中功率在眩光下确实有用，梯度会鼓励策略使用；若功率无效，功率惩罚会抑制不必要高功率。

深度相关系数为：

$$
\lambda_{\mathrm{power}}=20,\quad
\lambda_{\mathrm{blur}}=0.1,\quad
\lambda_{\mathrm{noise}}=5,\quad
\lambda_{\mathrm{fill}}=30.
$$

### 3.10 优化与运行配置

当前 `slit_active_sensing.args` 分支使用：

| 类别 | 取值 |
|---|---|
| Batch size | 当前 args 文件中为 $150$ |
| Rollout 长度 | $80$ steps |
| Optimizer | AdamW |
| Learning rate | $5\times 10^{-5}$ |
| Scheduler | Cosine annealing |
| AMP | 默认开启 |
| Depth render 尺寸 | $64\times48$ |
| Policy depth input 尺寸 | $32\times24$ |
| Depth range | $0.3\,\mathrm{m}$ 到 $6.0\,\mathrm{m}$ |
| Camera angle | $20^\circ$ |
| Sensor backend | `diff_depth=python` |
| 训练中启用场景 | `glare/specular/dark` |
| Direct control | enabled |
| dMPC | disabled |
| Policy intent output | disabled |
| Teacher-student training | disabled |
| TBPTT | disabled |

若 GPU 显存受限，可在不改变方法本身的前提下降低 batch size。算法描述不依赖该硬件相关设置。

### 3.11 实现对应关系

实现按论文模块一一对应组织，这对可复现性很关键，因为本文方法不仅是网络结构，更是完整可微闭环系统。

| 论文模块 | 实现文件 | 主要职责 |
|---|---|---|
| 运行参数与配置 | `config.py`, `configs/slit_active_sensing.args` | 定义 rollout 长度、深度分辨率、相机语义、传感模型参数、损失权重与启用场景 |
| 环境构建 | `train_utils.py` | 基于所选场景 profile 创建训练与评估环境 |
| 固定地图与场景效应 | `env_cuda.py` | 定义固定障碍布局、Sun Glare 场景效应、材质代理和场景局部掩码 |
| CUDA 几何渲染 | `src/quadsim.cpp`, `src/quadsim_kernel.cu`, `env_cuda.py` | 计算理想深度和四旋翼仿真基础量 |
| 可微传感器模型 | `env_cuda.py` | 将理想深度转换为退化主动深度、质量图、无效掩码与传感统计 |
| 传感 rollout 辅助 | `rollout_ops.py` | 初始化/更新相机状态，渲染传感器，计算共享代理与填充统计 |
| 神经策略 | `model.py` | 实现 CNN 深度编码器、状态编码器、门控融合、GRU 记忆、飞行头、相机头 |
| 损失计算 | `losses.py` | 计算任务损失、相机正则、深度可靠性损失与 Sun Glare 局部质量损失 |
| 训练循环 | `trainer.py`, `main_cuda.py` | 展开仿真、累积损失、记录统计并更新策略参数 |
| 评估与可视化导出 | `eval.py`, `eval.sh`, `rerun_vis.py` | 评估 checkpoint 并导出轨迹、深度、质量、场景掩码与相机时序 |

当前版本使用 CUDA 加速几何深度与动力学，但可微主动传感退化在 PyTorch 中实现。因此，相机头所需传感器梯度来自 `env_cuda.py` 中的 PyTorch 运算，而非对 CUDA 几何渲染器进行场景几何梯度求导。该分离是有意设计：本文研究的是主动寄存器自适应，而不是几何优化。

### 3.12 “端到端可微”包含与不包含的内容

“end-to-end differentiable” 容易产生歧义。本文中它表示：训练用梯度可在闭环计算中从损失流向策略参数，包括深度质量对相机参数的可微依赖。更具体地，以下路径是激活的：

$$
\mathcal{L}_{\mathrm{depth}}
\rightarrow
Q_t,D_t
\rightarrow
c_t
\rightarrow
\hat{c}_t
\rightarrow
\theta ,
$$

$$
\mathcal{L}_{\mathrm{task}}
\rightarrow
x_{t+1}
\rightarrow
u_t
\rightarrow
y_t
\rightarrow
\theta .
$$

本文不声称以下路径：

$$
\frac{\partial Z_t}{\partial \mathcal{M}},
\qquad
\frac{\partial Z_t}{\partial \text{obstacle geometry}},
\qquad
\frac{\partial \mathcal{L}}{\partial \text{real camera firmware}} .
$$

几何深度 $Z_t$ 被视作由当前状态和固定地图渲染得到的观测。主动感知所需可微部分是从理想几何与相机寄存器到退化深度可靠性的映射：

$$
(Z_t,c_t,x_t,\phi)
\mapsto
(D_t,Q_t,\Psi_t).
$$

这种区分使方法更实用。若需完整可微地模拟 D455（含双目匹配固件、投影图案物理、传感器饱和、rolling shutter、红外材质 BRDF 等），复杂度会显著增加且仍需标定。本文 surrogate 聚焦于最相关的因果导数：

$$
\frac{\partial Q}{\partial p^{\mathrm{cam}}},
\qquad
\frac{\partial Q}{\partial e},
\qquad
\frac{\partial Q}{\partial g}.
$$

## 4. 实验

本节定义实验框架。数值结果留空，待运行训练与评估脚本后填充。

### 4.1 研究问题

实验旨在回答四个问题：

**Q1：可微主动深度感知是否能在干净基础场景中保持避障性能？**  
主动相机不应破坏标准导航能力。

**Q2：在逆光 Sun Glare 退化下，可微主动深度感知是否改进导航？**  
策略应在进入眩光区域后保持深度可靠性并避免碰撞。

**Q3：策略是否学到了有意义的相机参数自适应？**  
在 Sun Glare 中，预期功率、曝光、增益会出现与眩光区域及局部深度质量相关的非平凡变化。

**Q4：与不可微感知或经典规划基线相比，可微性是否关键？**  
提出方法应优于无法通过主动传感器模型获取梯度信息的方法。

### 4.2 场景

#### 4.2.1 基础场景（Base）

Base 场景使用固定 6 柱障碍场。无人机从 $(-5,0,1.5)$ 出发，目标为 $(5,0,1.5)$。该场景评估基础导航、障碍净空、轨迹平滑性，以及主动传感控制是否引入不必要相机波动。

预期定性行为：

1. 无人机沿 S 型轨迹绕过柱体；
2. 深度填充率保持稳定；
3. 除非策略发现小幅有益调整，相机参数应保持在标称附近。

#### 4.2.2 逆光场景（Sun Glare）

Sun Glare 场景引入逆光区域。当无人机接近出口方向时，投影光源会在深度质量图上造成局部退化，用于测试主动感知是否能恢复可靠几何。

预期定性行为：

1. 无人机穿过逆光区，而非在入口前停下；
2. 曝光可能下降以抑制环境冲洗和运动模糊；
3. 若更强主动信号能改善局部眩光质量，功率可能上升；
4. 增益会依据噪声-信号权衡发生变化。

### 4.3 对比方法

建议评估包含以下方法：

#### 方法 A：本文方法（可微主动深度）

即第 3 节完整方法。策略同时控制飞行和相机参数。传感器模型对功率、曝光、增益可微。策略以导航损失与感知可靠性损失训练。

#### 方法 B：固定相机深度策略

网络结构相同，但相机参数固定：

$$
c_t = [0.416667,0.5,0.5].
$$

策略仅控制飞行。该基线用于检验是否确实需要主动相机控制。

#### 方法 C：不可微主动深度策略

策略可输出相机参数，但对传感器模型的梯度断开：

$$
\frac{\partial D_t}{\partial c_t}=0,
\qquad
\frac{\partial Q_t}{\partial c_t}=0.
$$

该基线用于检验可微性对学习有用传感自适应是否必要。它仍可通过延迟任务回报间接学习相机行为，但缺少直接传感质量梯度。

#### 方法 D：启发式自动曝光基线

手工控制器根据全局深度填充率或图像质量统计调整曝光/增益。例如：

$$
e_{t+1}
=
\mathrm{clip}
\left(
e_t + k_e(F^\star-F_t),
0,
1
\right),
$$

并固定功率。该基线用于检验简单传感启发式是否可替代端到端主动感知。

#### 方法 E：Ego-Planner 风格基线

ego-planner 风格方法基于当前深度图构建局部障碍表示，并在固定相机设置下规划避碰轨迹。该基线代表模块化几何导航。

为公平比较，所有基线应使用相同地图、起点、终点、传感分辨率和深度范围。

### 4.4 基线实现细节

基线应尽量只改变要研究的科学变量。目标不是做“弱基线”，而是识别性能来源。

#### 4.4.1 固定相机策略

固定相机策略应保持与提出方法相同的网络主体，仅调整相机分支。可接受两种实现：

1. 保留相机头但 rollout 时忽略其输出；
2. 删除相机头，仅保留相同循环飞控主体。

第一种改动更少。rollout 时：

$$
c_t=[p_0,0.5,0.5],
\qquad
\forall t .
$$

由于无可学习相机动作，应移除相机平滑与相机范围项：

$$
\lambda_{\mathrm{cam,smooth}}
=
\lambda_{\mathrm{power,reg}}
=
\lambda_{\mathrm{cam,range}}
=0.
$$

深度传感损失仍可作为指标报告，但不应更新相机参数。该基线回答“是否需要主动相机分支”。

#### 4.4.2 不可微主动深度策略

不可微基线保留相机输出，但对传感响应 detach：

$$
D_t^{\mathrm{detach}} = \mathrm{stopgrad}(D_t),
\qquad
Q_t^{\mathrm{detach}} = \mathrm{stopgrad}(Q_t)
$$

相对于 $c_t$。

实现中应允许策略继续接收深度观测：

$$
\bar{I}_t = \mathrm{PreprocessDepth}(D_t^{\mathrm{detach}}),
$$

但梯度

$$
\frac{\partial Q_t}{\partial c_t}
$$

应为零。该基线强于固定相机，因为策略仍可改变相机参数，但只能通过延迟导航结果学习其作用。

预期失效模式是相机自适应弱或噪声大。若该方法与提出方法性能接近，论文应保守表述“在测试环境下可微性可能非必要”。

#### 4.4.3 启发式自动曝光与自动增益

这里需明确区分两层概念。第一层是 **D455 原生固件自动曝光**：它运行在 RealSense 立体模组固件中，核心目标是把红外图像强度维持在预设 set-point 附近，主要通过自动调节 exposure 与 gain 实现，并可受 ROI 设定影响。第二层是本文实验中的 **启发式 AE 基线**：它并不是对 Intel 私有固件控制器的逐项复刻，而是一个受其思想启发、但在仿真中完全公开可复现的全局统计控制器，用于公平比较“手工传感规则”与“端到端可微主动感知”。

因此，论文正文与图表中若继续使用“启发式自动曝光”一词，建议在首次出现时注明：

> 启发式 AE（D455-style global exposure/gain controller, not exact firmware reproduction）

在本文实验中，一个合理且可复现的全局填充控制器为：

$$
e_{t+1}
=
\mathrm{clip}
\left(
e_t
+k_e(F^\star-F_t),
0,
1
\right),
$$

$$
g_{t+1}
=
\mathrm{clip}
\left(
g_t
+k_g(F^\star-F_t),
0,
1
\right),
$$

$$
p^{\mathrm{cam}}_{t+1}=p_0 .
$$

其设计动机与 D455 原生 AE 的共性在于：二者都根据全局观测统计做逐帧相机参数调节，而不直接优化轨迹成功、碰撞代价或局部障碍可见性。不同之处在于：D455 固件 AE 作用对象是红外强度统计，且控制逻辑封装在设备固件与 advanced controls 中；本文基线则直接基于仿真可得的深度 fill / quality 统计构造，目的是获得一个公开、稳定、可消融的比较对象。

更强变体还可控功率：

$$
p^{\mathrm{cam}}_{t+1}
=
\mathrm{clip}
\left(
p^{\mathrm{cam}}_t
+k_p(F^\star-F_t),
0,
1
\right).
$$

但论文需明确采用哪个变体。若启发式控制功率，应使用相同能耗惩罚或可比饱和上限，避免其通过长期最大功率获得不公平优势。

全局启发式的主要局限是仅观测标量填充统计，无法区分无效像素位于局部眩光区、障碍边界附近，还是无关背景。这正是本文使用学习策略与局部可靠性项的原因。

#### 4.4.4 Ego-Planner 风格基线

ego-planner 风格基线应代表模块化导航栈：

$$
D_t
\rightarrow
\text{local occupancy}
\rightarrow
\text{trajectory optimization}
\rightarrow
\text{tracking control}.
$$

在当前仿真中公平比较时，规划器应与固定相机基线一样使用退化深度观测和固定相机参数。规划器可对障碍采用保守膨胀半径。若规划器使用了真值障碍位置，必须单独报告为 oracle planner，且不应与感知基线直接比较。

预期局限：当 Sun Glare 降低局部深度有效性时，固定深度的模块化规划可能过保守。若障碍附近局部图不完整，规划器可能停住、选不安全路径，或对未知空间过度膨胀。

### 4.5 指标

#### 导航指标

| 指标 | 定义 |
|---|---|
| 成功率 | 无碰撞到达目标的 episode 比例 |
| 碰撞率 | 发生任意障碍碰撞的 episode 比例 |
| 最小净空 | rollout 中到障碍的最小符号距离 |
| 到达时间 | 到达目标所需步数或秒数 |
| 路径长度 | 累计行进距离 |
| 平均速度 | 平均 $\|v_t\|_2$ |
| 控制代价 | $\sum_t \|u_t\|_2^2$ |
| Jerk | $\sum_t \|u_t-u_{t-1}\|_2^2$ |

#### 感知指标

| 指标 | 定义 |
|---|---|
| 深度填充率 | 有效/可靠深度像素的比例（或软比例） |
| 空洞率 | $1-$ fill rate |
| 质量均值 | 深度质量 $Q$ 的均值 |
| 无效率 | 无效概率均值 |
| 局部眩光质量 | 眩光掩码内 $Q$ 的均值 |
| 局部眩光无效率 | 眩光掩码内无效概率均值 |

#### 主动相机指标

| 指标 | 定义 |
|---|---|
| 功率均值/方差/最小/最大 | $p^{\mathrm{cam}}_t$ 统计 |
| 曝光均值/方差/最小/最大 | $e_t$ 统计 |
| 增益均值/方差/最小/最大 | $g_t$ 统计 |
| 能耗代理 | $\mathbb{E}[p_t^2]$ |
| 模糊代理 | $\mathbb{E}[(\|v_t\|t_{\mathrm{exp}})^2]$ |
| 噪声代理 | $\mathbb{E}[g_t^2]$ |
| 相机平滑性 | $\mathbb{E}[\|c_t-c_{t-1}\|^2]$ |

#### 事件对齐相机指标

对于 Sun Glare，仅看全局统计不足够。策略可能因与逆光无关的原因产生较高功率标准差。因此，核心指标应围绕“进入眩光区域时刻”进行事件对齐。

设

$$
t_{\mathrm{entry}}
=
\min\{t\mid x_t>x_{\mathrm{enter}}\}.
$$

定义入区前后窗口：

$$
\mathcal{T}_{\mathrm{pre}}
=
[t_{\mathrm{entry}}-K_{\mathrm{pre}},t_{\mathrm{entry}}),
$$

$$
\mathcal{T}_{\mathrm{post}}
=
[t_{\mathrm{entry}},t_{\mathrm{entry}}+K_{\mathrm{post}}].
$$

报告：

$$
\Delta p
=
\mathbb{E}_{t\in\mathcal{T}_{\mathrm{post}}}[p_t^{\mathrm{cam}}]
-
\mathbb{E}_{t\in\mathcal{T}_{\mathrm{pre}}}[p_t^{\mathrm{cam}}],
$$

$$
\Delta e
=
\mathbb{E}_{t\in\mathcal{T}_{\mathrm{post}}}[e_t]
-
\mathbb{E}_{t\in\mathcal{T}_{\mathrm{pre}}}[e_t],
$$

$$
\Delta g
=
\mathbb{E}_{t\in\mathcal{T}_{\mathrm{post}}}[g_t]
-
\mathbb{E}_{t\in\mathcal{T}_{\mathrm{pre}}}[g_t].
$$

这使结果更可解释：只有当相机自适应在时序上与物理退化事件对齐，并提升局部感知或导航表现时，才有意义。

### 4.6 实验协议

对每种方法和每个场景：

1. 用对应训练配置训练；
2. 在固定随机种子（可选再加传感模型随机化）上评估；
3. 记录完整轨迹、深度图、质量图、无效掩码、场景掩码与相机参数；
4. 报告评估 episodes 上均值与标准差；
5. 可视化代表轨迹和相机参数时序。

对 Sun Glare，按入区事件对齐时序：

$$
t_{\mathrm{entry}}
=
\min\{t\mid x_t > x_{\mathrm{enter}}\}.
$$

再将相机参数绘制为

$$
\Delta t = t-t_{\mathrm{entry}}.
$$

这能显示相机自适应是否确实发生在进入逆光区域时。

#### 4.6.1 训练协议

主训练运行应使用 `configs/slit_active_sensing.args` 中激活配置。为保证论文比较干净，每个基线都应仅做必要最小改动并从头训练。推荐协议：

1. 学习类方法使用同一组随机种子；
2. 使用相同 rollout 长度 $T=80$、深度分辨率 $64\times48$、网络深度输入 $32\times24$；
3. 使用相同优化器、学习率、调度器与梯度裁剪；
4. 通过验证成功率或固定迭代数选 checkpoint，避免按可视化“挑图”；
5. 若算力允许，每个学习方法至少报告 3 个种子。

若首稿仅有单种子，应明确标注为初步结果，避免做强统计结论。

#### 4.6.2 评估协议

除非专门研究鲁棒性，否则评估应采用确定性设置。推荐：

1. 固定障碍布局；
2. 固定 Sun Glare anchor 与 zone gate；
3. 主表关闭训练期传感参数随机化；
4. 额外给一张小幅随机化鲁棒性表；
5. 导出所有评估 rollout 用于可视化。

每个 episode 记录：

$$
\{x_t,u_t,c_t,D_t,Q_t,M_{\mathrm{glare},t},F_t\}_{t=0}^{T-1}.
$$

判定 episode 成功需满足：

1. 无人机进入目标半径阈值；
2. 未发生碰撞；
3. 末态速度不过高；
4. 未在眩光区前永久停滞。

“眩光前停车”指标很重要，因为策略可能通过拒绝进入困难区域来降低损失。可用定义：

$$
\mathrm{StopBeforeGlare}=1
$$

当

$$
\max_t x_t < x_{\mathrm{enter}}+\epsilon_x
$$

且

$$
\frac{1}{K}\sum_{t=T-K}^{T-1}\|v_t\| < v_{\mathrm{stop}} .
$$

#### 4.6.3 Rerun 可视化协议

定性图应来自与表格一致的评估 rollout。对每个选定 rollout，Rerun 可视化应包含：

1. 固定地图俯视轨迹；
2. 无人机位姿与朝向；
3. 6 根障碍柱；
4. Sun Glare anchor 与眩光锥/投影掩码；
5. 退化深度图；
6. 质量图 $Q_t$；
7. 无效/空洞图；
8. 局部眩光掩码 $M_{\mathrm{glare}}$；
9. 功率、曝光、增益、速度、填充率和局部眩光质量时序。

期望视觉证据不是 3D 视图“看起来像照片那样变亮”，而是仿真状态中存在正确场景局部退化：进入逆光区时眩光掩码出现、局部深度可靠性下降、学习到的相机参数在同一事件附近变化。

### 4.7 主量化表格

#### 表 1：Base 场景导航

| 方法 | 成功率 ↑ | 碰撞率 ↓ | 最小净空 ↑ | 到达时间 ↓ | 路径长度 ↓ | 控制代价 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 本文：可微主动深度 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 固定相机深度策略 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 不可微主动深度 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 启发式自动曝光 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Ego-Planner 风格基线 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |

预期解读：在 Base 场景中，提出的主动传感策略应达到或超过固定相机导航性能，同时不产生不必要高能耗。

#### 表 2：Sun Glare 场景导航

| 方法 | 成功率 ↑ | 碰撞率 ↓ | 眩光前停车率 ↓ | 局部眩光质量 ↑ | 局部眩光无效率 ↓ | 到达时间 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 本文：可微主动深度 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 固定相机深度策略 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 不可微主动深度 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 启发式自动曝光 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Ego-Planner 风格基线 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |

预期解读：在逆光退化下，固定相机与模块化基线可能失去局部深度可靠性或变得保守。可微主动方法应提升局部眩光质量并保持导航成功。

#### 表 3：Sun Glare 中的相机自适应

| 方法 | 功率均值 | 功率方差 | 功率最大值 | 曝光均值 | 曝光方差 | 增益均值 | 能耗代理 | 模糊代理 | 噪声代理 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 本文方法 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 固定相机 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 不可微主动 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 启发式 AE | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |

预期解读：提出方法应呈现与环境事件相关的相机变化，而非恒定参数。关键并非“功率尽量大”，而是局部可靠性、能耗、模糊与噪声之间的权衡。

#### 表 4：Sun Glare 中事件对齐相机响应

| 方法 | $\Delta$ 功率 | $\Delta$ 曝光 | $\Delta$ 增益 | $\Delta$ 局部质量 | $\Delta$ 填充率 | 成功率 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| 本文方法 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 固定相机 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 不可微主动 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| 启发式 AE | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |

预期解读：主动感知最有力证据不是全局标准差大，而是在事件对齐后，局部质量和导航成功率出现正向提升。

### 4.8 消融实验

#### 消融 A：移除可微传感梯度

将 $D_t$、$Q_t$ 相对于 $c_t$ 的梯度断开，检验“通过传感器模型回传梯度”是否关键。

预期结果：相机自适应变弱，或与眩光入区的相关性变差。

#### 消融 B：固定功率

设置

$$
p^{\mathrm{cam}}_t=p_0
$$

仅允许曝光和增益变化。

预期结果：若功率在 Sun Glare 中确有作用，则相较完整模型，局部眩光质量或成功率应下降。

#### 消融 C：移除局部眩光质量损失

设置

$$
\lambda_{\mathrm{glare}}=0.
$$

预期结果：策略可能更依赖全局填充率，或学会绕开眩光区域。该消融用于判定局部感知可靠性项是否改善了主动感知行为。

#### 消融 D：从观测中移除相机状态

不再向低维状态拼接 $2c_t-1$。

预期结果：由于策略看不到当前传感器状态，相机控制可能更不稳定。

#### 消融 E：移除传感随机化

关闭训练期分组传感参数的小随机化，测试学习到的相机策略是否过拟合窄传感模型。

预期结果：训练表现可能提升，但对标定偏差的鲁棒性可能下降。

#### 消融 F：移除传感物理中的功率恢复项

移除 Sun Glare 主动信号恢复项：

$$
r_{\mathrm{active}}p^{\mathrm{cam}}M_{\mathrm{glare}}.
$$

预期结果：若功率在眩光下不再提升局部质量，学习到的策略应停止提高功率。这是关键合理性检查：验证功率变化来自“建模物理效用”，而非日志伪影或直接动作奖励。

#### 表 5：消融总结

| 变体 | 成功率 ↑ | 碰撞率 ↓ | 局部眩光质量 ↑ | $\Delta$ 功率 | 能耗代理 ↓ | 解读 |
|---|---:|---:|---:|---:|---:|---|
| 完整模型 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | 主结果 |
| 无传感梯度 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | 检验可微性 |
| 固定功率 | `<待填>` | `<待填>` | `<待填>` | `0` | `<待填>` | 检验功率效用 |
| 无局部眩光损失 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | 检验局部可靠性项 |
| 无相机状态观测 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | 检验可观测性 |
| 无传感随机化 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | 检验鲁棒性 |
| 无功率恢复物理项 | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | 功率机制合理性检查 |

### 4.9 定性图

最终论文建议包含以下图：

#### 图 1：系统总览

模块图：

$$
\text{CUDA geometry}
\rightarrow
\text{differentiable sensor}
\rightarrow
\text{depth preprocessing}
\rightarrow
\text{CNN-GRU policy}
\rightarrow
\text{flight and camera heads}
\rightarrow
\text{quadrotor dynamics}
\rightarrow
\text{loss}.
$$

#### 图 2：基础场景轨迹

展示所有对比方法的俯视轨迹。提出方法应沿 6 柱周围形成平滑、无碰撞路径。

#### 图 3：Sun Glare 场景可视化

展示：

1. 无人机轨迹；
2. 眩光区域；
3. 障碍布局；
4. 代表性深度图；
5. 质量图；
6. 无效掩码。

#### 图 4：眩光入区附近的相机参数

绘制

$$
p^{\mathrm{cam}}_t,\quad e_t,\quad g_t
$$

相对于 $t-t_{\mathrm{entry}}$ 的曲线。该图是证明主动感知自适应的核心证据。

#### 图 5：局部质量与导航结果

绘制局部眩光质量与无效率随时间变化，对比 fixed-camera 与可微主动深度策略。

#### 图 6：功率效用消融

展示有/无“功率恢复项”的 Sun Glare rollout。若解释正确，移除功率物理效用应显著减弱或消除学习到的功率上升。

### 4.10 预期结论与谨慎解读

最强且科学上可辩护的结论是：

> 可微主动深度模型使策略能够将传感器参数与运动联合优化，从而在逆光避障场景中产生与任务相关的相机自适应。

论文应避免声称：

1. 仿真器是完美 D455 digital twin；
2. 学习行为无需标定即可保证迁移；
3. 局部 Sun Glare 损失直接证明“通用感知策略”自发涌现；
4. 高功率总是正确响应。

正确解读应为：

1. 模型捕捉了物理上有意义的因果趋势；
2. 策略未获得相机参数直接动作监督；
3. 相机行为由可微传感模型、导航目标和感知可靠性损失共同涌现；
4. Sim-to-real 迁移需通过 D455 标定数据和真实场景测试验证。

## 5. 讨论

### 5.1 为什么可微性重要

在不可微流水线中，策略仍可能通过试错发现相机自适应，但 credit assignment 很困难。功率或曝光变化会影响未来深度，未来深度影响未来控制，进而影响未来碰撞/成功，梯度信号间接且延迟。

在本文流水线中，局部深度质量损失提供直接梯度：

$$
\frac{\partial \mathcal{L}_{\mathrm{glare}}}{\partial p^{\mathrm{cam}}}
=
\frac{\partial \mathcal{L}_{\mathrm{glare}}}{\partial \bar{Q}_{\mathrm{glare}}}
\frac{\partial \bar{Q}_{\mathrm{glare}}}{\partial Q}
\frac{\partial Q}{\partial p^{\mathrm{cam}}}.
$$

同理，

$$
\frac{\partial \mathcal{L}_{\mathrm{blur}}}{\partial e}
=
2\|v\|^2 t_{\mathrm{exp}}
\frac{\partial t_{\mathrm{exp}}}{\partial e}.
$$

这些梯度让策略更容易学习到在可靠性、模糊、噪声、能耗之间进行权衡的传感控制。

### 5.2 为什么 Sun Glare 使用局部可靠性

全局填充率可能掩盖局部失效。若只有小而关键区域受眩光影响，全局填充率仍可能看似可接受：

$$
F_{\mathrm{global}}
=
\frac{1}{HW}
\sum_{i,j} Q(i,j).
$$

但避障取决于任务相关障碍边界是否可见。因此在眩光掩码内测量局部可靠性：

$$
\bar{Q}_{\mathrm{glare}}
=
\frac{\sum M_{\mathrm{glare}}Q}{\sum M_{\mathrm{glare}}+\epsilon}.
$$

该项不是动作标签，不编码“提高功率”。它编码的是“在传感退化区域维持可用局部深度可靠性”。策略可在可微传感模型下选择最优寄存器组合以满足该目标。

### 5.3 局限性

当前方法有以下局限：

1. 几何深度渲染器虽然高效，但模型简化；
2. D455 风格传感模型是 surrogate，不是完整物理仿真；
3. Sun Glare 场景刻意最小化，未覆盖全部真实阳光条件；
4. 当前局部质量损失依赖场景掩码。更通用方案应在多种退化类型下使用统一任务局部可靠性项；
5. 当前策略为直接动作输出，RAL 版本未使用 model-predictive planning；
6. 真实迁移需要对曝光、增益、激光功率、深度噪声、无效深度与逆光响应进行标定。

### 5.4 面向真实验证的路径

真实验证应复现实验中的最小 Sun Glare 搭建：

1. D455 固定在无人机或运动平台上；
2. 逆光区域附近放置简单障碍；
3. 在目标方向附近布置强可见光或红外光源；
4. 对激光功率、曝光、增益做受控扫描；
5. 测量深度填充率、无效率和局部障碍边界可靠性。

随后调整仿真参数，使其与硬件定性响应一致：

$$
\frac{\partial \bar{Q}_{\mathrm{glare}}}{\partial p^{\mathrm{cam}}},
\quad
\frac{\partial \bar{Q}_{\mathrm{glare}}}{\partial e},
\quad
\frac{\partial \bar{Q}_{\mathrm{glare}}}{\partial g}
$$

在工作区间内应与真实传感器具有相同符号和近似相对量级。

### 5.5 有效性威胁

本节列出主要有效性威胁及论文应对方式。

#### 5.5.1 传感器模型保真度

最大威胁是可微传感模型与真实 D455 不一致。RealSense D455 深度图由主动双目系统与专有固件共同产生，涉及双目匹配、投影图案交互、曝光控制、无效化逻辑和滤波等。本文模型并未复现完整流程。

因此论文应避免像素级真实性主张，正确表述应是“在工作区间内的因果真实性”：

$$
\mathrm{sign}
\left(
\frac{\partial Q_{\mathrm{sim}}}{\partial c}
\right)
\approx
\mathrm{sign}
\left(
\frac{\partial Q_{\mathrm{real}}}{\partial c}
\right),
$$

其中

$$
c\in\{p^{\mathrm{cam}},e,g\}.
$$

若仿真能保持“在决策边界附近哪些相机调整会提升/降低局部深度可靠性”，它就对策略学习有价值。可通过逆光条件下的 D455 静态标定数据进行检验，比较：

$$
\Delta Q_{\mathrm{real}}(p,e,g)
\quad\text{vs.}\quad
\Delta Q_{\mathrm{sim}}(p,e,g).
$$

#### 5.5.2 场景特定局部损失

Sun Glare 局部质量损失使用场景掩码，审稿人可能认为引入了场景知识。正确回应为：

1. 掩码不监督相机动作；
2. 掩码仅标识该合成场景中“感知可靠性重要”的区域；
3. 策略仍通过优化自行选择功率、曝光与增益；
4. 论文报告了移除局部质量损失的消融以量化其影响。

在 RAL 版本中，这可作为受控诊断实验接受。对于更宽范围论文，应将局部可靠性项推广到任务相关区域，如障碍边界、预测碰撞走廊或规划器注意力图：

$$
\bar{Q}_{\mathrm{task}}
=
\frac{
\sum_{i,j} W_{\mathrm{task}}(i,j)Q(i,j)
}
{
\sum_{i,j} W_{\mathrm{task}}(i,j)+\epsilon
}.
$$

#### 5.5.3 Reward Hacking 与保守停车

学习策略可能通过在困难眩光区前停下而降低损失，但这不等于成功导航。因此评估中包含“眩光前停车率”和“到达时间”。训练目标也应保证“朝目标推进”仍重要。

论文应显式报告失败模式：

1. 在眩光区内碰撞；
2. 眩光区前停车；
3. 障碍附近振荡；
4. 到达目标但能耗过高；
5. 相机参数改变但感知质量未改善。

#### 5.5.4 功率变化的解读

目标不是证明功率总会上升，而是证明“与任务相关的自适应”。功率上升只有在满足以下条件时才有意义：

$$
\Delta p>0,
\qquad
\Delta \bar{Q}_{\mathrm{glare}}>0,
\qquad
\Delta \mathrm{Success}>0,
$$

相对基线或消融成立。若功率变化而局部质量与成功率不提升，不应将其作为主动感知证据。

移除 Sun Glare 传感物理中“功率效用”的消融非常关键，因为它可验证：当功率不再有帮助时，学习到的功率变化是否随之消失。这是科学控制实验，而非形式化测试。

#### 5.5.5 Sim-to-Real 迁移

即使仿真结果强，真实部署仍需：

1. 将归一化曝光映射到 D455 曝光微秒；
2. 将归一化功率映射到 D455 激光功率寄存器；
3. 选择不会导致饱和或深度不稳定的增益范围；
4. 在逆光下验证深度无效化趋势；
5. 测量延迟与丢帧；
6. 对功率、速度和障碍间隙施加安全限制。

若硬件实验尚未完成，本文可将真实标定作为未来工作。若纳入硬件结果，应作为独立验证章节报告，而非与仿真指标混合。

### 5.6 可复现性清单

最终投稿建议包含以下细节：

| 项目 | 需要给出的细节 |
|---|---|
| 代码分支 | Commit hash 和分支名 |
| 配置 | 完整 `slit_active_sensing.args` 快照 |
| 硬件 | GPU 型号、CUDA 版本、PyTorch 版本 |
| 训练预算 | 迭代数、batch 大小、rollout 长度、墙钟时间 |
| 随机种子 | 各报告 run 的种子 |
| Checkpoint 选择 | 最后一个 checkpoint 或验证集最佳 checkpoint |
| 场景几何 | 起点、终点、障碍中心和障碍尺寸 |
| Sun Glare 参数 | Sun anchor、zone gate、掩码宽度、环境增量、主动衰减/恢复 |
| 传感语义 | 曝光映射、增益映射、标称功率 |
| 损失权重 | 全部任务/相机/深度/眩光系数 |
| 评估脚本 | 精确 `eval.sh` 命令和 checkpoint 路径 |
| 可视化 | Rerun 导出设置和代表性 rollout ID |

该清单尤为重要，因为本文研究的是闭环系统。传感损失、相机平滑或障碍布局的小变化，都可能改变最终学习行为。

## 6. 结论

本文提出了一种面向四旋翼导航的可微主动深度感知框架。该方法将 D455 风格深度传感器模型置于端到端训练闭环中，使循环视觉运动策略能够同时控制飞行动作与主动深度参数。可微传感模型刻画了激光功率、曝光、增益、环境红外、运动模糊、深度噪声、无效像素与局部质量之间的关键因果关系。策略通过导航、避碰、控制平滑、能耗、模糊、噪声、填充率与局部 Sun Glare 可靠性损失联合训练。

本 RAL 版本有意聚焦于最小实验设定：固定基础障碍图与 Sun Glare 变体。这一窄范围有价值，因为它隔离了核心科学问题：在没有手工标注相机动作的前提下，可微主动感知是否能产生与任务相关的传感器自适应。实验框架将本文方法与固定相机策略、不可微主动感知、启发式自动曝光和 ego-planner 风格基线对比。量化结果将在训练与评估完成后填充。

预期结果并非“策略只是提高功率”。更准确目标是：策略仅在能够在能耗、模糊和噪声权衡下提升局部深度可靠性与导航性能时，才调节功率、曝光与增益。若该结论在仿真与 D455 硬件测试中均得到验证，将支持更广泛观点：主动传感寄存器应被视为机器人控制闭环的一部分，而非固定相机设置。

## 附录 A. 当前配置摘要

| 参数 | 取值 |
|---|---|
| `--scenarios` | `glare specular dark` |
| `--batch_size` | `150` |
| `--num_iters` | `5000` |
| `--timesteps` | `80` |
| `--depth_width` | `64` |
| `--depth_height` | `48` |
| `--depth_nn_width` | `32` |
| `--depth_nn_height` | `24` |
| `--depth_min_valid` | `0.3` |
| `--depth_max_range` | `6.0` |
| `--include_camera_state_in_obs` | enabled |
| `--diff_sensor_impl` | `diff_depth=python` |
| `--use_dmpc` | disabled |
| `--policy_output_intent` | disabled |
| `--enable_teacher_student_training` | disabled |
| `--tbptt_enable` | disabled |
| `--coef_v` | `2.5` |
| `--coef_obj_avoidance` | `4.0` |
| `--coef_collide` | `10.0` |
| `--coef_cam_smooth` | `10` |
| `--cam_power_baseline` | `0.55` |
| `--coef_diff_depth_power` | `5` |
| `--coef_diff_depth_blur` | `0.1` |
| `--coef_diff_depth_noise` | `5` |
| `--coef_diff_depth_fill` | `30` |
| `--diff_depth_min_fill_rate` | `0.25` |
| `--coef_sun_glare_local_quality` | `30` |
| `--sun_glare_local_quality_target` | `0.1` |

## 附录 B. 符号表

| 符号 | 含义 |
|---|---|
| $x_t$ | 机器人状态 |
| $p_t$ | 机器人位置 |
| $v_t$ | 机器人速度 |
| $R_t$ | 机器人姿态 |
| $c_t$ | 相机状态 |
| $p^{\mathrm{cam}}_t$ | 归一化激光功率 |
| $e_t$ | 归一化曝光 |
| $g_t$ | 归一化增益 |
| $D_t$ | 退化深度观测 |
| $Q_t$ | 可微深度质量图 |
| $Z_t$ | 理想几何深度 |
| $\mathcal{M}$ | 障碍物地图 |
| $\pi_\theta$ | 神经策略 |
| $u_t$ | 飞行控制动作 |
| $h_t$ | 循环隐藏状态 |
| $M_{\mathrm{glare}}$ | Sun Glare 图像掩码 |
| $F_t$ | 软深度填充率 |
| $\bar{Q}_{\mathrm{glare}}$ | 眩光区域局部质量 |

## 附录 C. 待填结果占位

投稿前需补充：

1. 总损失与损失占比训练曲线；
2. Base 场景成功率/碰撞率表；
3. Sun Glare 场景成功率/碰撞率表；
4. 相机参数统计；
5. 眩光入区对齐的功率/曝光/增益曲线；
6. 深度、质量、无效掩码和场景掩码可视化；
7. 固定相机、不可微主动感知、启发式 AE 与 ego-planner 的基线实现细节；
8. 若包含硬件验证，需补充真实 D455 标定细节。

## 附录 D. 结果段落模板（建议）

以下段落可在实验完成后填充。

### D.1 Base 场景结果模板

在 Base 场景中，提出的可微主动深度策略取得 `<待填>` 成功率，碰撞率为 `<待填>`。其最小净空为 `<待填>`，平均到达时间为 `<待填>`。与固定相机策略相比，提出方法在相机参数保持标称工作区附近的同时，实现了 `<待填>` 的导航表现。这表明在清洁条件下引入主动相机控制不会破坏导航稳定性。

### D.2 Sun Glare 场景结果模板

在 Sun Glare 场景中，可微主动深度策略成功率为 `<待填>`，而固定相机策略为 `<待填>`，不可微主动基线为 `<待填>`。在眩光入区附近，局部眩光质量从 `<待填>` 提升到 `<待填>`，眩光前停车率从 `<待填>` 降至 `<待填>`。这些结果表明，可微传感反馈有助于在逆光退化下提升任务相关感知。

### D.3 相机自适应模板

事件对齐分析显示策略在眩光入区事件附近调整了相机参数。平均功率变化为 $\Delta p=$<`待填`>，平均曝光变化为 $\Delta e=$<`待填`>，平均增益变化为 $\Delta g=$<`待填`>。这些变化同时伴随 `<待填>` 的局部质量提升和 `<待填>` 的成功率提升。因此，相机响应并非全局随机波动，而是与物理退化区域在时序上对齐。

### D.4 消融模板

移除可微传感梯度使 `<待填>` 下降，并削弱事件对齐相机响应。固定功率使局部眩光质量下降 `<待填>`，说明在该场景中功率对感知恢复有贡献。移除局部眩光质量项导致 `<待填>`，表明仅靠全局填充率不足以让策略聚焦任务相关退化区域。

## 附录 E. 最小 LaTeX 转换说明

面向 IEEE RAL 投稿，可将该 Markdown 草案按以下结构转换为 LaTeX：

1. `\section{Introduction}`
2. `\section{Related Work}`
3. `\section{Method}`
4. `\section{Experiments}`
5. `\section{Discussion}`
6. `\section{Conclusion}`

较长实现表格和结果模板应移入附录或补充材料。正文建议保留：

1. 系统总览图；
2. 与相机控制最相关的传感模型方程；
3. 一个算法框；
4. 两个主结果表；
5. 一个事件对齐相机响应图；
6. 一个消融表。

最终 RAL 版本应在保留关键方程的前提下压缩方法部分：

$$
D,Q=\mathcal{S}_{\phi}(Z,c,x),
\qquad
c_{t+1}=0.7c_t+0.3\hat{c}_{t+1},
$$

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{task}}
+\mathcal{L}_{\mathrm{cam}}
+\mathcal{L}_{\mathrm{depth}}
+\mathcal{L}_{\mathrm{glare}}.
$$

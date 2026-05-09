# 面向四旋翼穿缝导航的可微主动深度感知

> RAL 稿件草案，当前版本对齐 `DiffPhysDrone` 项目截至 2026-05-08 的实现与实验结果。  
> 主实验 checkpoint: `checkpoint/2026-05-08-03-04-28/checkpoint0014.pth`。  
> 主结果目录: `paper/experiment/results/final_semantics_v3_eval_20260508`。  
> 本稿只讨论仿真实验与当前可复现流程。

## 摘要

主动深度相机不是固定观测设备。以 RealSense D 系列为代表的主动双目深度传感器，其深度质量受到激光功率、曝光时间和增益等寄存器影响。传统视觉导航策略通常将深度图视为外生输入，固定相机参数或使用与任务无关的自动曝光规则。这在穿缝、逆光、低反射/红外吸收材质和镜面伪深度等条件下会暴露出一个核心问题：机器人失败不一定是因为控制策略不会绕障，而可能是因为传感参数使关键几何线索不可见或不可靠。

本文研究一种面向四旋翼穿缝导航的可微主动深度感知方法。我们构建一个单墙狭缝任务，飞行器需要从墙前穿过随机横向位置的窄缝，并在三类传感退化下完成导航：强光眩光、镜面伪深度和低反射/红外吸收材质退化。仿真器先由 CUDA 几何渲染生成理想深度，再由可微主动深度模型根据相机功率、曝光和增益生成退化深度观测。策略以深度图、低维状态和当前相机状态为输入，同时输出飞行动作和下一步相机参数。

直接端到端训练会受到闭环分布偏移影响：离线预训练的相机头能拟合 teacher 标签，但在 learned-camera closed-loop 中可能无法稳定地区分低反射材质和眩光退化。为解决这一问题，本文采用两阶段流程。第一阶段在 learned-camera 在线状态分布上运行可微 teacher 优化器，对访问到的状态重新标注相机参数，形成 DAgger-style camera relabel 数据集，并监督预训练相机头。第二阶段冻结相机控制分支和共享视觉 stem，仅对飞行控制相关层进行 flight-only 适配。最终策略在 3 个场景各 100 次评估中取得 0.717 的总体成功率，相比固定相机 0.673、随机固定相机 0.627 和不可微相机策略 0.697 均有提升；同时深度填充率从固定相机的 0.861 提升到 0.981。episode-aggregated 相机行为分析显示，最终策略在狭缝附近对低反射材质和眩光采用明显不同的曝光/增益：dark near 为 0.591/0.613/0.586，glare near 为 0.741/0.068/0.110，glare-dark near camera L1 达到 0.390。

## 关键词

可微感知，主动深度相机，四旋翼导航，穿缝导航，传感器参数控制，闭环模仿学习。

## 1. 引言

深度相机在小型空中机器人导航中很常见，因为深度图提供了紧凑的局部几何信息。许多学习式视觉运动策略可写为

$$
u_t = \pi_\theta(D_t, s_t),
$$

其中 $D_t$ 是深度观测，$s_t$ 是本体状态或目标相对状态。这个表达隐含了一个强假设：深度观测由环境和位姿决定，而非策略动作的一部分。

对于主动双目深度相机，这个假设并不完整。实际深度质量还取决于激光功率、曝光和增益等寄存器：

$$
D_t = \mathcal{S}(x_t, c_t; \phi),
\qquad
c_t = [p_t, e_t, g_t] \in [0,1]^3 .
$$

其中 $p_t$、$e_t$ 和 $g_t$ 分别表示归一化激光功率、曝光和增益。在低反射/红外吸收材质区域，提高曝光和增益可能改善有效深度；在眩光下，高曝光和高增益可能放大退化；在镜面区域，高功率或高增益可能诱发错误深度。也就是说，相机参数不是附加控制量，而是决定导航观测分布的动作。

本文关注一个刻意简化但可复现的问题：四旋翼从墙前飞向墙后目标，必须穿过单墙狭缝。狭缝横向位置随机，飞行器初始朝向随机旋转，使策略不能只记忆固定世界方向。任务中的关键困难不是复杂地图构建，而是在靠近狭缝时维持足够可靠的局部深度线索。

本文问题可以概括为：

**可微主动深度模型能否帮助学习一个任务相关的相机控制策略，使四旋翼在低反射材质、眩光和镜面伪深度退化下更可靠地穿过狭缝？**

### 1.1 贡献

本文当前版本的贡献如下。

1. **主动深度穿缝导航基准。**  
   我们实现了一个最小化单墙狭缝任务，包含随机狭缝位置、随机飞行朝向和三类传感退化场景，便于隔离主动深度感知对导航的影响。

2. **可微相机参数到深度质量的闭环模型。**  
   仿真器将理想几何深度转换为由功率、曝光和增益控制的退化深度，并输出填充率、局部质量、运动模糊和噪声代理，供 teacher 优化和策略训练使用。

3. **闭环相机 teacher relabel 流程。**  
   我们发现固定分布上的 camera pretrain 会在 learned-camera 在线分布中出现 dark/glare 混淆。因此引入 DAgger-style relabel：用 learned camera 访问在线状态，再用可微 teacher 优化器重标注这些状态。

4. **冻结相机的 flight-only 适配。**  
   在相机头预训练后，冻结相机控制分支和共享视觉 stem，只训练飞行控制相关层，使策略在保留相机场景区分能力的同时适配闭环动力学。

5. **可复现实验脚本与结果表。**  
   `tools/run_checkpoint_eval_suite.py` 对主方法和 baselines 统一运行每场景 100 次评估，并导出 raw trace、episode metrics、phase summary 和定性轨迹图。

## 2. 相关工作

### 2.1 深度视觉四旋翼导航

经典深度导航系统通常将深度图转换为局部占据图或距离场，再执行轨迹规划和控制。学习式导航方法则直接将深度图和低维状态输入神经策略，输出速度、加速度或姿态命令。这两类方法多数将深度相机视为固定观测函数，即使在训练中加入噪声随机化，也很少将相机寄存器放入策略动作空间。

本文与这些方法的区别在于，策略不仅利用深度图，还控制未来深度图的采集方式。

### 2.2 主动感知和相机参数控制

主动视觉通常研究智能体如何选择视角、注视方向、曝光、照明或传感配置。对于主动深度相机，功率、曝光和增益之间存在任务相关权衡：曝光增大可改善弱信号，但会带来运动模糊；增益增大可放大信号，也会放大噪声；激光功率可改善主动图案，但在镜面或眩光条件下可能不是单调有益。

传统自动曝光常优化全局亮度统计，而导航关心的是局部几何可靠性，特别是障碍边缘和狭缝区域。本文不复现某个具体硬件固件，而是研究任务级主动深度控制：相机参数由策略根据当前状态和深度观测决定，teacher 优化目标也直接作用于深度可靠性代理。

### 2.3 可微感知与闭环学习

可微渲染和可微仿真允许将任务损失梯度传回场景、传感器或控制动作。本文使用 CUDA 几何渲染获得理想深度，再用 Torch/CUDA 可微深度模型模拟主动深度退化。关键不是获得像素级真实传感器仿真，而是保留相机参数对深度有效性、模糊、噪声和场景退化的主要因果关系，使 teacher 优化器和学习策略能够利用这些关系。

## 3. 方法

### 3.1 任务和状态

飞行器从墙前起点飞向墙后目标：

$$
x_{\mathrm{start}}=-1.5,\qquad x_{\mathrm{goal}}=1.5 .
$$

墙位于 $x=0$，狭缝中心 $y$ 在 $[-0.75,0.75]$ 内随机采样，狭缝半宽为 $0.15$ m。狭缝有效高度中心为 $z=1.5$ m，有效半高为 $0.75$ m。评估时启用随机旋转，最大角度为 $360^\circ$，因此策略必须在局部坐标下处理穿缝任务。

状态包括飞行器位置、速度、姿态、上一时刻动作和当前相机参数。相机状态为

$$
c_t = [p_t,e_t,g_t],
$$

其中三个通道均归一化到 $[0,1]$，分别表示功率、曝光和增益。

### 3.2 传感退化场景

实验使用三个场景：

1. **glare:** 狭缝附近存在强光眩光 halo，降低关键区域深度可靠性。  
   主要参数包括 `simple_glare_halo_width_y=0.22`、`simple_glare_halo_extra_half_z=0.30` 和 `simple_glare_halo_strength=0.50`。

2. **specular:** 狭缝附近出现镜面伪深度和误导性深度线索。  
   主要参数包括 `simple_specular_false_depth_strength=0.85`。

3. **dark:** 狭缝两侧为低反射/红外吸收材质，使主动深度信号在关键区域变弱。  
   主要参数包括 `simple_key_cue_degrade_strength=0.92`。

三类场景共享同一个几何任务，因此导航差异主要来自传感退化和相机控制。

### 3.3 可微主动深度模型

每个控制步先渲染理想几何深度：

$$
Z_t = \mathcal{G}_{\mathrm{cuda}}(x_t,\mathcal{M}).
$$

然后可微主动深度模型根据相机状态和场景 profile 生成退化深度：

$$
D_t,Q_t,\Psi_t =
\mathcal{S}_\phi(Z_t,c_t,x_t).
$$

其中 $D_t$ 是策略输入深度图，$Q_t$ 是质量或有效性代理，$\Psi_t$ 包含填充率、patch CVaR 填充、模糊代理、噪声代理和场景退化统计。

模型保留以下因果关系：

1. 高功率通常增强主动信号，但受到功率正则约束。
2. 高曝光可提升弱信号，也会增加运动模糊。
3. 高增益可放大弱信号，也会增加噪声。
4. 眩光、低反射材质和镜面退化会以不同方式影响关键狭缝区域。
5. 深度质量不足会产生空洞或错误深度，从而影响策略输入。

策略使用的深度分辨率为 $96\times72$，神经网络输入分辨率为 $48\times36$。深度有效范围为 $0.05$ m 到 $6.0$ m。传感器后端为 `diff_depth=cuda`。

### 3.4 策略结构

策略由深度编码器、状态编码器、共享 GRU、飞行头和相机头组成。输入为

$$
(\bar{D}_t, s_t, c_t, h_t),
$$

其中 $\bar{D}_t$ 是预处理后的深度图，$s_t$ 包含局部速度、目标相对状态、姿态和安全裕度，$c_t$ 是当前相机状态，$h_t$ 是循环隐藏状态。

策略输出飞行动作和下一时刻相机目标：

$$
(u_t,\hat{c}_{t+1},h_{t+1}) =
\pi_\theta(\bar{D}_t,s_t,c_t,h_t).
$$

相机状态使用指数滑动平均更新：

$$
c_{t+1} = 0.7 c_t + 0.3\hat{c}_{t+1}.
$$

飞行控制使用 direct action，最大加速度命令为 `2.2`。rollout 长度为 80 步，基础控制频率为 15 Hz。

### 3.5 可微 teacher 相机标注

单纯通过强化式任务损失训练相机头不稳定。本文使用可微 teacher 优化器为在线状态生成相机目标。给定访问到的状态和深度渲染上下文，teacher 直接优化相机目标 $c$，目标包括：

$$
\mathcal{L}_{\mathrm{teacher}}
=
\lambda_{\mathrm{fill}}\mathcal{L}_{\mathrm{fill}}
\lambda_{\mathrm{blur}}\mathcal{L}_{\mathrm{blur}}
\lambda_{\mathrm{noise}}\mathcal{L}_{\mathrm{noise}}
\lambda_{\mathrm{smooth}}\mathcal{L}_{\mathrm{smooth}}
\lambda_{\mathrm{nominal}}\mathcal{L}_{\mathrm{nominal}} .
$$

最终采用的 DAgger relabel 参数为：

| 参数 | 取值 |
|---|---:|
| rollouts per scene | 4 |
| batch size | 12 |
| timesteps | 80 |
| teacher steps | 120 |
| teacher lr | 0.08 |
| rollout camera mode | learned |
| teacher camera EMA | disabled |
| coef nominal when healthy | 0.075 |
| nominal fill margin | 0.25 |
| coef fill | 50 |
| coef power | 0.0 |
| coef blur | 0.00015 |
| coef noise | 0.0007 |
| coef camera smooth | 0.005 |

这里 `coef power=0.0` 是有意选择：功率约束留给训练配置和其他基线，不让 teacher 的功率惩罚与 nominal recovery 过度重合。teacher 的主要职责是找到能恢复局部深度健康的相机参数，同时在健康区域回到 nominal。

### 3.6 DAgger-style relabel

最初的 teacher 数据是在 fixed camera rollout 上采集。虽然离线 camera pretrain 能拟合标签，但在线 learned-camera rollout 会改变状态分布，导致低反射材质和眩光的相机响应变得相似。为此我们采用 DAgger-style 流程：

1. 用当前 learned camera checkpoint 闭环 rollout，收集在线访问状态。
2. 在这些状态上运行可微 teacher 优化器，重新生成相机目标。
3. 用 relabeled dataset 再次预训练 camera head。
4. 评估 pretrained camera 的在线相机分离度。
5. 若 online dark/glare near L1 充分恢复，再进入 flight-only 适配。

最终 DAgger 数据集为：

| 项目 | 数值 |
|---|---:|
| sequences | 144 |
| timesteps | 80 |
| samples | 11520 |
| teacher mean p/e/g | 0.584/0.408/0.403 |
| teacher std p/e/g | 0.164/0.197/0.181 |

camera pretrain 最佳结果：

| 项目 | 数值 |
|---|---:|
| best epoch | 119 |
| validation loss | 0.000715 |
| MAE p/e/g | 0.0094/0.0126/0.0131 |

对应数据和 checkpoint：

- dataset: `paper/experiment/results/closed_loop_teacher_camera_policy_semantics_v3_dagger/camera_teacher_dataset.pt`
- camera checkpoint: `checkpoint/closed_loop_teacher_camera_policy_semantics_v3_dagger/camera_head_pretrained_best.pth`

### 3.7 Flight-only 适配

DAgger camera checkpoint 本身只保证相机响应合理，并不保证飞行控制在该相机闭环下已经适配。因此第二阶段从 camera checkpoint 恢复模型，并启用 `--train_flight_only`：

```bash
RESUME_CKPT=checkpoint/closed_loop_teacher_camera_policy_semantics_v3_dagger/camera_head_pretrained_best.pth \
RUN_TRAIN=1 MODES="flightonly" bash run_train_modes.sh
```

实现中 `train_flight_only` 会冻结：

- camera control 分支；
- camera 相关 recurrent/adapter 参数；
- shared visual stem。

训练期间相机仍参与闭环前向，飞行控制层在该固定相机策略下适配。最终 checkpoint 为：

```text
checkpoint/2026-05-08-03-04-28/checkpoint0014.pth
```

## 4. 实验设置

### 4.1 对比方法

评估包含以下方法：

| 方法 | 说明 | checkpoint |
|---|---|---|
| `flightonly` | 本文主方法，DAgger camera pretrain 后冻结相机并适配飞行控制 | `checkpoint/2026-05-08-03-04-28/checkpoint0014.pth` |
| `fixed` | 固定相机参数 0.5/0.5/0.5 | `checkpoint/2026-05-08-00-44-22/checkpoint0014.pth` |
| `randfix` | 每个 episode 使用随机固定相机参数训练得到的策略 | `checkpoint/2026-05-08-01-17-21/checkpoint0014.pth` |
| `nondiff` | learned camera，但传感器梯度 detached | `checkpoint/2026-05-08-01-49-59/checkpoint0014.pth` |
| `zero` | blind/zero-depth 下界 | `checkpoint/2026-05-08-02-31-46/checkpoint0014.pth` |
| `pretrained` | 仅 camera pretrain，未 flight-only 适配，用作诊断 | `checkpoint/closed_loop_teacher_camera_policy_semantics_v3/camera_head_pretrained_best.pth` |
| `dagger` | DAgger camera pretrain，未 flight-only 适配，用作诊断 | `checkpoint/closed_loop_teacher_camera_policy_semantics_v3_dagger/camera_head_pretrained_best.pth` |

### 4.2 指标

导航指标：

- success rate: episode 是否到达目标并避免碰撞；
- collision rate: episode 是否碰撞；
- final goal distance: episode 结束时到目标距离；
- average speed: 平均速度。

感知指标：

- fill rate: 深度有效填充率；
- phase-wise camera mean: 按局部 $x$ 位置划分为 before、near 和 after，统计相机参数均值；
- scene separation: 比较不同场景 near 阶段相机参数差异，重点为 glare-vs-dark near L1。

phase 划分为：

$$
\mathrm{before}: x_{\mathrm{local}} < -0.25,\quad
\mathrm{near}: |x_{\mathrm{local}}|\le 0.25,\quad
\mathrm{after}: x_{\mathrm{local}} > 0.25 .
$$

### 4.3 评估脚本

主评估使用：

```bash
bash run_exp_eval.sh
```

该脚本导出：

- `combined_report.md`: 主汇总表；
- `summary_by_method_scene.csv`: 方法和场景级指标；
- `camera_phase_summary.csv`: before/near/after 相机行为统计；
- `raw/*_episodes.csv` 和 `raw/*_trace.csv`: 原始 episode 与逐步 trace；
- `figures/*_camera_and_trajectories.png`: 相机曲线和局部轨迹图；
- `figures/scene_metrics.png`: 场景级指标图。

## 5. 结果

### 5.1 场景、地图和实验展示顺序

结果部分按“任务是否清楚、训练是否可靠、闭环是否有效、机制是否可信”的顺序组织。Figure 1 首先展示单墙狭缝地图、三类退化场景和 relabel-and-adapt 流程。当前主图使用二维/流程化示意；如果需要更接近 `rerun` 中 `student_3d` 窗口的效果，最合适的补充是一张 3D 场景渲染图，包含墙体、狭缝、目标点、无人机轨迹以及 glare/dark/specular 退化区域。该图应作为 Figure 1a 或 Extended Data 的视觉任务设定图，而不是替代后续定量结果。

Figure 2 展示新加入的训练曲线，包括 loss、success rate 和 collision rate。它的作用不是报告最终性能，而是说明各训练模式进入稳定训练区间，避免“最终评估只是未收敛 checkpoint 偶然结果”的疑问。最终性能仍以独立 closed-loop eval 为准。

Figure 3 给出主导航指标，回答主动相机是否改善闭环穿缝成功率和深度有效填充率。Figure 4 展示无人机局部轨迹、相机参数轨迹和 near-slit 参数响应，回答提升是否来自场景相关的主动感知。Figure 5 进一步展示同一组 near-slit 位姿下的 raw/geometric depth 和不同方法相机参数产生的 observed depth，给出直观观测证据。最后 Figure 6 展示 DAgger relabel 诊断，解释为什么仅做 camera pretrain 不够，以及为什么需要在线状态分布上的 relabel 和 flight-only 适配。

Figure 5 由 `tools/export_journal_depth_sequences.py` 生成。脚本固定 scene、far-right slit slot 和随机种子，先运行各方法的真实 closed-loop rollout，再在主方法轨迹的连续 near-slit 位姿上重新渲染 raw/geometric depth 和不同方法相机参数对应的 observed depth。为保证主文图清晰，Figure 5 只展示 fixed、random fixed 和本文方法；non-diff 仍保留在量化表和 Figure 3/Extended Data 的指标对比中。对应原始数组、相机参数和局部填充率保存在 `journal_assets/qualitative_depth/`。

### 5.2 训练曲线显示各对比方法进入稳定区间

WandB 导出的三条训练指标曲线位于：

```text
paper/experiment/results/final_semantics_v3_eval_20260508/raw/
```

对应文件为 `wandb_export_2026-05-08_loss.csv`、`wandb_export_2026-05-08_success_rate.csv` 和 `wandb_export_2026-05-08_collision_rate.csv`。Figure 2 将这些曲线统一绘制为训练收敛诊断。主方法 `flightonly` 在训练末端 success rate 达到约 0.771，collision rate 降至约 0.229；`fixed`、`randfix` 和 `nondiff` 也进入非零成功率区间，末端 success rate 分别约为 0.653、0.607 和 0.668。zero/blind 作为无有效深度输入的下界，训练末端 success rate 仍约为 0.033。

这些训练曲线有两个用途。第一，它们说明主要对比方法不是完全未训练状态，`randfix` 和 `nondiff` 都学到了一定飞行能力。第二，它们也提示训练曲线本身不能替代最终评估：训练日志中的 rolling/online 指标与统一评估中的随机场景采样、checkpoint 加载和 episode 统计口径不同。因此后续所有性能结论均以每场景 100 次的统一 closed-loop eval 为准。

### 5.3 主动相机同时改善导航和有效深度

每个方法在 glare、dark 和 specular 三个场景各评估 100 次，共 300 个 closed-loop episode。主方法取得 0.717 的总体成功率，固定相机、随机固定相机和不可微 learned camera 分别为 0.673、0.627 和 0.697。相对 fixed camera，成功率提升为 +0.043；相对 nondiff camera，提升为 +0.020。碰撞率也从 fixed camera 的 0.327 降至 0.283。

更稳定的优势体现在观测质量。主方法的深度 fill rate 为 0.981，而 fixed、randfix 和 nondiff 分别为 0.861、0.840 和 0.854。也就是说，主动相机带来的主要收益不是单纯“飞得更激进”，而是在闭环中维持了更健康的深度观测。Figure 3a,b 和 Table 1 给出总体估计和 95% 置信区间。

分场景结果也支持这一点。主方法在 glare、dark 和 specular 中的成功率分别为 0.670、0.740 和 0.740，分别与或高于对应 fixed camera 的 0.670、0.670 和 0.680。glare 和 dark 中 fill rate 从 fixed camera 的 0.730 和 0.874 提升到 0.967 和 0.989；specular 中 fixed 已经有较高 fill rate，主方法仍维持 0.988 的有效深度填充率。Figure 3c 展示了相对 fixed camera 的分场景成功率增益，Figure 3d 展示终点距离分布。

### 5.4 提升来自场景相关的相机响应和闭环轨迹

如果 learned camera 只是输出任意非 nominal 参数，那么相机 trace 不应在 dark 和 glare 之间形成可解释差异。结果恰好相反。最终策略在 near-slit 阶段对 glare 的相机参数为 0.741/0.068/0.110，对 dark 为 0.591/0.613/0.586，对 specular 为 0.485/0.568/0.300。glare 中 exposure/gain 被主动压低，dark 中 exposure/gain 保持较高，specular 则采用另一组降低 gain 的参数。

这种行为和任务退化机制一致。dark 场景需要增强低反射区域的弱信号，glare 场景则需要避免高 exposure/gain 放大退化。episode-aggregated glare-dark near camera L1 达到 0.390，fixed、randfix 和 nondiff 分别约为 0.000、0.013 和 0.000。Figure 4a,b 和 Table 3 给出 near-slit 相机响应；Figure 4d 显示 exposure/gain 随 local wall distance 的连续变化；Figure 4e 以成功轨迹包络说明这些相机行为发生在穿缝闭环内，而不是离线标签统计。

Figure 5 中的 matched-pose 深度序列沿用 Figure 4 的机制解释：同一位姿下 raw/geometric depth 不随相机参数改变，而 observed depth 会因功率、曝光和增益而改变。为让几何挑战更直观，Figure 5 固定使用 `far_right` slit。在 dark 场景的 matched-pose 序列中，主方法的局部有效填充率为 0.997，高于 fixed 的 0.770，并与 randfix 的 1.000 接近；在 glare 场景中，主方法的局部有效填充率为 0.983，高于 fixed 的 0.094 和 randfix 的 0.112；在 specular 场景中，fixed 的几何线索本身已经较完整，主方法仍保持 0.983 的局部有效填充率，略高于 fixed 的 0.975，并高于 randfix 的 0.840。该图因此提供了比标量 fill rate 更直观的证据：主动相机参数改变了策略实际看到的狭缝附近深度图，同时也显示 specular 是相对容易保持可观测性的场景，而不是主要的视觉塌缩来源。

### 5.5 Camera relabel 和 flight-only 适配互补

原始 camera pretrain 在 supervised validation 上可以拟合 teacher 标签，但进入 learned-camera closed loop 后需要检查 dark/glare 分离是否仍能保持。当前语义修正版中，online pretrained camera 已能形成清晰分离，glare-dark near L1 约为 0.417，说明低反射材质和眩光退化在当前 teacher 数据和在线状态中均可被相机头区分。

DAgger-style relabel 用 learned camera 访问在线状态，再对这些状态运行可微 teacher 优化器重新标注。经过 relabel 后，DAgger camera 的 online glare-dark near L1 约为 0.430；flight-only 适配后仍保持 0.390。没有 flight-only 适配时，DAgger checkpoint 的导航成功率仍低于最终策略，说明相机语义本身不足以带来导航成功，飞行控制层还必须在该相机闭环分布下适配。最终策略同时保留相机分离能力和较高成功率。Figure 6 对这一训练诊断进行汇总。

### 5.6 论文图表组织

最终投稿版本不使用 `paper_assets` 中的逐方法诊断图。这些图适合检查轨迹和相机 trace，但信息组织不符合主文证据链。当前主文图表统一使用：

```text
paper/experiment/results/final_semantics_v3_eval_20260508/journal_assets
```

Figure 1 作为任务和方法图，展示地图、传感退化和训练流程。Figure 2 作为训练诊断图，展示 loss、success 和 collision 的训练过程。Figure 3 作为主导航证据图，展示 success、depth fill、相对 fixed camera 的分场景成功率增益，以及终点距离经验分布。Figure 4 作为机制图，展示 near-slit 相机参数指纹、exposure-gain 响应平面、退化强度、沿墙距离的相机剖面和成功轨迹包络。Figure 5 作为 qualitative observation 图，展示 raw/geometric depth 和相机参数改变后的 observed depth。Figure 6 作为 relabel 诊断图，说明相机语义和 flight-only adaptation 是互补条件：相机分离能力需要保留，飞行控制层也必须在该闭环观测分布下适配。

主文建议只保留 Table 1 作为主要数值表；Table 2 和 Table 3 可根据版面放入 Methods/Extended Data。完整结果矩阵放入 Extended Data Fig. 1，避免主文被工程化矩阵淹没。

## 6. 讨论

### 6.1 当前结果支持什么

当前结果支持一个谨慎但清晰的结论：在本文构造的主动深度穿缝任务中，任务相关的相机控制能够显著改善深度填充率，并带来小到中等幅度的导航成功率提升。与固定相机和随机固定相机相比，主方法不仅成功率更高，而且在 glare/dark 退化中学到了可解释的不同相机响应。

### 6.2 当前结果不应如何解读

这些结果不应被解释为主动相机策略已经在导航指标上碾压所有 baseline。fixed、randfix 和 nondiff 仍有 0.65 左右的成功率，说明当前任务中飞行控制和几何随机性仍占相当大比重。主方法的优势主要体现在两个方面：

1. 深度填充率显著更高；
2. 相机参数具有场景相关的可解释分化；
3. 成功率在所有场景中稳定高于 baselines，但幅度不是压倒性的。

因此论文表述应避免夸大为“解决了复杂真实导航”，更适合写成“在受控仿真基准中验证可微主动深度感知的有效性和可解释行为”。

### 6.3 为什么 pretrained alone 失败

`pretrained` 和 `dagger` 在没有 flight-only 适配时成功率仍低于最终策略。这不是相机头完全无效，而是因为飞行控制层尚未适配 learned camera 闭环观测分布。相机策略改变了深度输入分布，飞行控制必须在该分布下重新适配。最终 `flightonly` 保留了 DAgger 相机分离能力，同时将总体成功率提高到 0.717。

### 6.4 有效性威胁

1. **传感器模型是任务级近似。**  
   可微深度模型保留了功率、曝光、增益、模糊、噪声和退化之间的主要关系，但不是像素级硬件仿真器。

2. **场景仍然较窄。**  
   单墙狭缝任务便于隔离主动感知因素，但不能覆盖复杂地图导航。

3. **baseline 仍有优化空间。**  
   fixed、randfix 和 nondiff 的成功率接近 0.65，说明任务没有被设计成只有主动相机才能通过。后续若要强化论文对比，可以增加更难的缝宽、退化强度或跨 seed 评估，但当前主结果已经可用于初版论文。

4. **成功率方差需要更多 seed 验证。**  
   当前主表基于每场景 100 次评估，但训练 checkpoint 仍是单 seed 结果。若时间允许，最有价值的补充不是更复杂场景，而是 2 到 3 个训练 seed 的均值和标准差。

## 7. 结论

本文提出并实现了一个可微主动深度感知流程，用于四旋翼在传感退化条件下的穿缝导航。通过在 learned-camera 闭环分布上进行 DAgger-style teacher relabel，并在相机预训练后冻结相机分支进行 flight-only 适配，最终策略在 glare、specular 和 dark 三类场景上取得 0.717 的总体成功率和 0.981 的深度填充率，优于固定相机、随机固定相机和不可微 learned camera baselines。相机 trace 进一步显示，策略在狭缝附近对 dark 和 glare 学到不同的曝光/增益控制模式，说明提升并非仅来自任意参数扰动，而是来自任务相关的主动感知行为。

## 附录 A. 关键配置摘要

| 类别 | 取值 |
|---|---|
| base config | `configs/slit_active_sensing.args` |
| scenarios | `glare specular dark` |
| depth backend | `diff_depth=cuda` |
| depth render size | `96 x 72` |
| policy depth size | `48 x 36` |
| timesteps | 80 |
| base control frequency | 15 Hz |
| camera mode, main | learned |
| camera EMA alpha | 0.7 |
| sensor grad, flight-only | detached |
| include camera state | enabled |
| random rotation | enabled, max 360 deg |
| collision clearance | 0.0011 m |
| final main checkpoint | `checkpoint/2026-05-08-03-04-28/checkpoint0014.pth` |

## 附录 B. 复现实验命令

### B.1 DAgger camera relabel

```bash
bash run_camera_dagger_relabel_pipeline.sh
```

主要输出：

```text
paper/experiment/results/closed_loop_teacher_camera_policy_semantics_v3_dagger
checkpoint/closed_loop_teacher_camera_policy_semantics_v3_dagger/camera_head_pretrained_best.pth
```

### B.2 Flight-only 训练

```bash
RESUME_CKPT=checkpoint/closed_loop_teacher_camera_policy_semantics_v3_dagger/camera_head_pretrained_best.pth \
RUN_TRAIN=1 MODES="flightonly" bash run_train_modes.sh
```

当前主 checkpoint：

```text
checkpoint/2026-05-08-03-04-28/checkpoint0014.pth
```

### B.3 统一评估

```bash
bash run_exp_eval.sh
```

### B.4 相机分离度诊断

```bash
/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python tools/diagnose_pretrain_camera_trace.py \
  --eval_dir paper/experiment/results/final_semantics_v3_eval_20260508 \
  --dataset paper/experiment/results/closed_loop_teacher_camera_policy_semantics_v3_dagger/camera_teacher_dataset.pt \
  --pretrained_ckpt checkpoint/closed_loop_teacher_camera_policy_semantics_v3_dagger/camera_head_pretrained_best.pth \
  --offline_method_label dagger
```

输出：

```text
paper/experiment/results/final_semantics_v3_eval_20260508/diagnosis.md
```

## 附录 C. 主文图表组织

| 图表 | 使用文件 | 用途 |
|---|---|---|
| Figure 1 | `journal_assets/figures/fig1_system_protocol.pdf` | 任务、可微主动深度闭环和 relabel-and-adapt 流程 |
| Figure 2 | `journal_assets/figures/fig2_training_convergence.pdf` | 训练过程：loss、success rate、collision rate |
| Figure 3 | `journal_assets/figures/fig3_navigation_performance.pdf` | 主导航证据：success、fill、分场景增益和终点距离分布 |
| Figure 4 | `journal_assets/figures/fig4_active_camera_mechanism.pdf` | 主动相机机制：near-slit 参数、曝光/增益响应、退化强度、轨迹包络 |
| Figure 5 | `journal_assets/figures/fig5_depth_observation_sequence_glare.pdf`, `journal_assets/figures/fig5_depth_observation_sequence_dark.pdf`, `journal_assets/figures/fig5_depth_observation_sequence_specular.pdf` | 同一 near-slit 位姿下 raw depth 和不同相机参数的 observed depth |
| Figure 6 | `journal_assets/figures/fig6_dagger_relabel_diagnosis.pdf` | DAgger relabel 如何恢复 online 相机语义并与 flight-only 适配互补 |
| Extended Data Fig. 1 | `journal_assets/figures/extended_data_fig1_full_matrix.pdf` | 完整 method-by-scene 指标矩阵 |
| Extended Data Fig. 2 | `journal_assets/figures/extended_data_fig2_terminal_distance.pdf` | 终点距离分布 |
| Extended Data Fig. 3 | `journal_assets/figures/extended_data_fig3_method_depth_sequences_glare.pdf`, `journal_assets/figures/extended_data_fig3_method_depth_sequences_dark.pdf`, `journal_assets/figures/extended_data_fig3_method_depth_sequences_specular.pdf` | 各方法沿自身 closed-loop 轨迹看到的连续 depth 序列 |
| Table 1 | `journal_assets/tables/table1_primary_navigation.tex` | 主量化结果，含 95% CI 和相对 fixed 的提升 |
| Table 2 | `journal_assets/tables/table2_scene_breakdown.tex` | 分场景 success/fill |
| Table 3 | `journal_assets/tables/table3_camera_response.tex` | episode-aggregated near-slit 相机行为和 dark/glare 分离 |

完整 caption 草案位于：

```text
paper/experiment/results/final_semantics_v3_eval_20260508/journal_assets/caption_drafts.md
```

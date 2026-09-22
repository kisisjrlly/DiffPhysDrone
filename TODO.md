# diff_depth 代码待办

当前分支已经切换为“固定小地图 + 感知场景库”版本：

- 地图大小固定为 `10m x 10m`
- 地图中心固定为 `(0, 0, 0)`
- 起点固定为 `(-5, 0, 1.5)`
- 终点固定为 `(5, 0, 1.5)`
- 基础障碍图固定且简化，目的是降低避障复杂度
- 当前版本保留 `scenarios`，用于测试不同可微感知退化场景
- 当前版本不再保留旧的大地图随机世界、随机障碍、随机起终点、随机旋转实现

这一版的目标非常明确：

1. 先让飞行/避障任务足够简单，不把训练难点放在机动本身。
2. 重点验证策略能不能根据不同感知退化场景调节 `power / exposure / gain`。
3. 先把主动调参的因果关系跑清楚，再恢复泛化与更复杂大地图。

## 当前范围

- 论文 v1 仅覆盖 `diff_depth` 主线。
- 当前主实验为固定小地图上的感知场景验证任务。
- 保持默认动作域控制，不开启 `--policy_output_intent`。
- 暂时保持 `--use_dmpc` 关闭。
- 暂时保持 `--tbptt_enable` 关闭。

## 当前已完成

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `env_cuda.py` | `[x]` | 当前主环境已简化为单一 `sun_glare` 地图，并在其中支持 `glare / specular / dark` 三种局部传感器模式。 |
| `config.py` | `[x]` | 当前已恢复 `scenarios` 参数；旧的大地图随机环境参数仍保持移除。 |
| `train_utils.py` | `[x]` | 已改为固定小地图环境构造，并重新接回 `scenarios` 主链。 |
| `trainer.py` | `[x]` | 已恢复必要的 scene 统计与 opening pass 指标，但不再回退到旧随机世界统计。 |
| `eval.py` | `[x]` | 已恢复按 `scenarios` 顺序轮转评测和 per-scene summary。 |
| `rerun_vis.py` | `[x]` | 已改为固定小地图的 AABB 显示范围。 |
| `configs/slit_active_sensing.args` | `[x]` | 已改为固定小地图 + 感知场景主配置。 |
| `configs/paper_ablate_diff_depth.args` | `[x]` | 已改为固定小地图 + 感知场景消融配置。 |
| `tools/compare_diff_depth_gradients.py` | `[x]` | 已同步到新的 `Env` 构造接口。 |

## P0 阻塞项

| 文件 | 待办 | 为什么重要 |
| --- | --- | --- |
| `env_cuda.py` + `autograd_ops.py` | 校准 `diff_depth=cuda` 与 `diff_depth=python` 的数值/梯度一致性。 | 当前论文主结果仍建议先用 `python`，但 `cuda` 路径后面必须对齐才能作为正式实现。 |
| `losses.py` | 重新检查 `power / blur / noise / fill` 的量纲与典型数值范围。 | 当前任务已经弱化了避障难度，loss 的相对权重会更直接决定“学飞”还是“学调参”。 |
| `run.sh` | 启动时保存合并后的完整参数快照。 | 固定地图版本非常适合做系统性调参，需要保证每次实验可追溯。 |

## P1 近期重要项

### `losses.py`

- [ ] 重新检查 `power / blur / noise` 三项的数值尺度，避免某项天然过强。
- [ ] 把 active-depth 损失的物理解释整理成可直接写进论文 methods 的版本。
- [ ] 为 `sun_glare` 增加局部区域目标，不只盯整图全局 fill。
  优先实现 `glare_quality_mean`，直接复用当前可微 `quality` 链路；
  若有效，再补 `glare_fill_rate` 和 `glare_invalid_rate`。

### `trainer.py`

- [ ] 为论文图准备统一导出接口，自动保存 `speed/power/exposure/gain/fill_rate` 时序数据。
- [ ] 增加“固定参数相机基线”和“启发式参数基线”的统一对比日志。
- [ ] 把 `sun_glare` 局部区域指标接入训练日志与 `rerun`，用于判断策略是否真的在救 glare 灾区。

### `rollout_ops.py`

- [ ] 在 `update_camera_params()` 中加入寄存器 slew-rate limit 或一阶惯性，避免参数一步跳变过快。
- [ ] 给 `init_camera_params()` 增加可配置初始值，方便做固定参数 baseline。

### `model.py`

- [ ] 做小分辨率 depth 输入的结构消融，确认当前 encoder 对 `16x12` / `32x24` 没有过强下采样。
- [ ] 评估是否要轻度解耦飞行头和相机头，避免主动调参信号被飞行主任务压制。

### `eval.py`

- [ ] 自动保存代表性 episode 的参数轨迹图和深度图序列。
- [ ] 导出 `csv/json`，方便后处理画图和做论文表格。

### `run.sh`

- [ ] 启动时记录 `git commit`、启动时间、GPU 型号、CUDA 信息。
- [ ] 保存展开后的完整命令行参数快照。

## P1 后续恢复的能力

这些能力当前仍未恢复到活跃实现，但属于后续必须逐步推进的方向。

### 泛化训练

- [ ] 恢复随机障碍物地图，但要作为单独训练阶段，而不是混在最小验证版里。
- [ ] 恢复随机起点/终点采样。
- [ ] 恢复随机全局旋转。
- [ ] 恢复不同 obstacle density 的 curriculum。

### 多场景论文实验

- [~] 当前已收敛为 `sun_glare` 单地图 benchmark；后续继续校准三种 sensor regime 的强度。
- [ ] 为 `glare / specular / dark` 定义更严格的传感器退化目标。
- [ ] 继续完善 per-scene 评测与导出。

### 大地图实验

- [ ] 恢复大于 `10m x 10m` 的长程地图。
- [ ] 加入更多障碍密度和更长路径长度测试。
- [ ] 重新设计大地图下的视觉边界和评测协议。

## P2 中长期项

### `env_cuda.py`

- [x] 已支持通过 `scene_fit_profiles_path` 自动加载 D455 标定反推得到的场景 profile/effects。
- [ ] 为 D455 风格寄存器加入更真实的约束：参数量化、帧级延迟、参数生效滞后、最小/最大合法区间。
- [ ] 补玻璃/透明体失效模式；当前更像镜面高反，不是真正透明体误导。
- [ ] 校准 `power / exposure / gain -> fill_rate / range / noise` 映射，让它更接近真实 D455。
- [ ] 提升 `sun_glare` 的空间真实性：让 glare 热区受前景障碍遮挡/边界裁切影响，而不是长期保持过于平滑的椭圆高斯斑块。
- [ ] 提升 `sun_glare` 的结构真实性：让关键障碍边缘在逆光区更容易先退化，增强“边缘先坏、内部后坏”的 D455 风格失效模式。
- [ ] 提升 `sun_glare` 的空洞真实性：让 glare 区的 `invalid` 更呈现边缘碎裂和局部破洞，而不只是由平滑 `quality -> valid` 阈值映射得到的干净连续区域。

### `autograd_ops.py` + `src/quadsim_kernel.cu`

- [ ] 对 CUDA 路径增加更严格的 shape / dtype / grad 检查。
- [ ] 决策二选一：要么把 CUDA `diff_depth` 升级到接近 Python 版，要么明确标注为 experimental。
- [ ] 增加 Python vs CUDA 的数值和梯度回归测试，不只测单一 loss。

### `tools/test_d455_depth.py`

- [x] 已补充独立采集脚本 `tools/collect_d455_calibration.py`，可输出 `csv/json/preview`。
- [x] 已补充离线拟合脚本 `tools/fit_d455_scene_profiles.py`，可按论文场景反推仿真参数建议。
- [~] 参数写入后的帧延迟统计已接入粗估计；后续还可做更严格的寄存器生效延迟 bench。
- [~] 已支持按场景/条件组织采集；后续还可扩展更细的距离板与动态轨迹 protocol。

### 文档

- [ ] 更新 `README.md`，明确当前主线是固定地图最小验证版。
- [ ] 更新 `Paper_diff_depth.md`，把“当前已验证 claim”和“下一阶段扩展 claim”分开写。
- [x] 已补充 D455 场景采集 protocol 文档，可按场景执行标准化采集。
- [ ] 增加一页主链路总览，便于回忆从环境到 loss 到评测的入口。

## 建议顺序

1. `losses.py`
2. `run.sh`
3. `eval.py`
4. `trainer.py`
5. `rollout_ops.py`
6. `model.py`
7. `autograd_ops.py`
8. `src/quadsim_kernel.cu`

# 当前git提交(c3c033dff03b350927eeaeabcfbe531452cfc295)对应的TODO

下一步不要继续盲跑三场景 end-to-end。我建议按这个顺序做：

1. **先把问题拆开验证**
   - 只训练 `glare` 一个场景。
   - 起点混合：一部分正常起点，一部分直接从 occluder 后 / gate 前开始。
   - 目标：确认 learned camera 至少能在单场景下学到合理相机参数，并成功穿 gate。

2. **把 camera head 改成“模式选择 + 小残差”**
   不再让网络直接连续搜索 `power/exposure/gain`。改成：
   ```python
   cam = softmax(mode_logits) @ preset_table + small_delta
   ```
   例如：
   - `glare`: 高 power、低 exposure、低 gain
   - `specular`: 低/中 power、低 exposure、低 gain
   - `dark`: 高 power、高 exposure、高 gain

   这样网络学的是“当前像哪个场景，我该选哪个模式”，比从 3 个连续值里自己摸索容易很多。

3. **加入 reset curriculum**
   训练时不要全从最远起点开始。建议早期：
   - 30% 正常起点
   - 40% occluder 后方，已经能看到 opening cue
   - 30% gate approach，马上需要看清 gate edge

   后期再逐渐提高正常起点比例。否则相机策略的有效梯度太晚出现，模型容易先学成“停在 occluder 前比较安全”。

4. **单场景通过后，再做三场景**
   顺序：
   - `glare only`
   - `specular only`
   - `dark only`
   - `glare + specular`
   - `glare + specular + dark`

   每一步都看 camera 参数是否符合预期，而不是只看 success rate。

5. **训练指标先看这几个**
   - `goal_dist` 是否下降
   - `collision_rate` 是否真实下降
   - `cam_power/exposure/gain` 是否分场景分化
   - `diff_depth_fill / blur / noise` 是否真的影响 camera
   - episode 轨迹里无人机是否穿 gate，而不是绕逻辑漏洞

6. **固定参数 baseline 要重新定义**
   fixed/randfix 不能参与 diff-depth 相机质量 loss，否则它们会通过飞行速度、姿态间接优化 sensor loss，和 learned camera 的对比会混掉。fixed/randfix 应主要优化飞行任务本身。

我的建议是下一步直接实现两个核心改动：

- `camera mode selector + residual`
- `reset curriculum`

然后先跑 `glare only`。只要单场景都学不会，就不要继续三场景。当前最关键的问题不是再调一点 loss，而是把学习问题从“太晚、太耦合、太连续”改成“早看到、可分类、可优化”。


有，而且我检索下来发现：**你这个方向其实已经有一条非常清晰的“前人路线”可以借鉴**。尤其有两个项目/论文和你现在 DiffPhysDrone 的问题高度相关：**SAPIEN + SimSense**，以及 ICCV 2021 的 **Physics-Based Differentiable Depth Sensor Simulation (DDS)**。

更重要的是，我刚刚也读取了你 `active-sensing-4f-tools-2b-core` 分支里的 `env_cuda.py` 和 `autograd_ops.py`。结合你当前代码，我现在反而觉得你**没有必要把整个 Isaac Sim / D455 渲染器改成可微**。你的任务可以简单很多。

### 先说结论

如果让我现在替你选技术路线，我会选：

> **SAPIEN/SimSense 当“真实感教师模型” + DDS 的 soft stereo 思路 + 你现有 DiffPhysDrone 的 CUDA/PyTorch 框架。**

而不是继续手工写现在 `glare/specular/dark` 那一大坨经验公式，也不是直接去魔改 Isaac Sim。

你目前的核心目标实际上只需要：

$$
\boxed{
\frac{\partial D_{\text{obs}}}
{\partial (P,E,G)}
}
$$

也就是：

> projector power / exposure / gain 改变后，最终观测深度怎么变。

你**不一定需要**

$$
\frac{\partial D}{\partial \text{geometry}}
$$

更不需要把整个四旋翼世界、光线追踪器、Isaac Sim 都放进 PyTorch autograd。

而你自己的代码实际上已经在这么干了：`ActiveSensingSensorFunction` 的注释就明确写了，输入是已经渲染好的 raw depth + scene mask，backward **只返回 power/exposure/gain 的梯度，geometry depth 故意不回传梯度**。所以你现在真正缺的不是“一个完全可微的世界模拟器”，而是一个**可信的、通用的、参数驱动的 sensor model**。

---

## 1. 最值得你研究：SAPIEN + SimSense

这个我认为和你的 D455 方向**最接近**。

SAPIEN 有一套专门针对**主动双目深度相机**的 realistic depth pipeline。对应论文是：

**Close the Optical Sensing Domain Gap by Physics-Grounded Active Stereo Sensor Simulation**

它做的流程基本就是：

$$
\text{IR projector}
\rightarrow
\text{ray-traced IR}
\rightarrow
\text{left/right IR images}
\rightarrow
\text{IR noise}
\rightarrow
\text{stereo matching}
\rightarrow
\text{depth}
$$

而且论文明确考虑了：

* 物体材质；
* IR 投影图案；
* specular / transparent material；
* 光线传播；
* IR camera noise；
* stereo matching；
* 深度 invalid hole。

这已经比你现在手工定义的 `glare/specular/dark mask` 更接近“真正的传感器形成过程”。([arxiv.org][1])

它的立体匹配部分单独开源成了 **SimSense**。SimSense 接收左右 IR 图，然后走：

`IR noise → rectification → Census transform → stereo matching → uniqueness test → LR consistency → median filter → disparity → depth`

而且 README 直接拿 **RealSense D435** 真机深度和 SimSense 输出做对照。它使用类似 SGM/SGBM 的 GPU 实现，README 给出的 848×480 benchmark 在 RTX 2080 Ti 上可以达到几十到两百 FPS。([GitHub][2])

项目：

[SimSense GitHub](https://github.com/angli66/simsense?utm_source=chatgpt.com)

[SAPIEN GitHub](https://github.com/haosulab/SAPIEN?utm_source=chatgpt.com)

这对于你非常有价值。

但注意：

**SimSense 本身不是可微的。**

因为它内部存在很多东西：

```text
Census / Hamming
argmin
uniqueness threshold
left-right check
median filter
invalid decision
```

这些天然就是离散操作。

所以我的建议不是：

> 把 SimSense CUDA 一行一行改成 backward。

而是：

> **拿 SimSense 的 forward pipeline 当物理参考，把其中真正与你的 camera action 有关系的部分做成 differentiable approximation。**

这就会简单很多。

---

## 2. 更关键：其实已经有人做过“可微深度传感器”

这个工作和你的问题几乎正面撞上了：

**Benjamin Planche & Rajat Vikram Singh, Physics-Based Differentiable Depth Sensor Simulation, ICCV 2021**

它简称可以叫 DDS。

论文明确提出了：

> end-to-end differentiable simulation pipeline for realistic depth scans

整个 pipeline 可以对**sensor parameters 和 scene parameters 求导**。([CVF Open Access][3])

它的结构基本是：

```text
3D geometry
     ↓
differentiable ray tracing
     ↓
projected structured-light image
     ↓
sensor noise
     ↓
differentiable stereo matching
     ↓
depth
```

而且 supplemental 给出了很重要的实现细节：

* PyTorch
* `redner`
* differentiable ray tracing
* differentiable block matching
* **softargmax**
* sensor intrinsic/extrinsic/baseline 都可以是参数。

他们甚至给出了诸如：

```text
baseline = 75 mm
block size = 9 px
softargmax temperature β = 15
subpixel refinement = 2
```

这样的实现参数。([CVF Open Access][4])

这里最值得你借鉴的不是 ray tracer，而是：

> **把传统 stereo 的 argmin 换成 softargmin。**

传统方法：

$$
d=\arg\min_d C(d)
$$

不可微。

改成：

$$
p(d)
=
\frac{\exp(-\beta C(d))}
{\sum_j\exp(-\beta C(j))}
$$

然后：

$$
\hat d
=
\sum_d d\,p(d)
$$

就可以反向传播。

最后：

$$
Z=\frac{fB}{\hat d+\epsilon}
$$

于是：

$$
P,E,G
\rightarrow I_L,I_R
\rightarrow C(d)
\rightarrow \hat d
\rightarrow Z
$$

整条链就是可微的。

**这个思路非常适合你。**

---

## 3. Isaac Sim 的 D455：有，而且官方直接提供

你刚才猜得没错。

最新 Isaac Sim 官方文档直接提供：

**RealSense D455 digital twin**

资产为类似：

`Realsense/D455/rsd455.usd`

包括：

* 左 IR camera
* 右 IR camera
* RGB camera
* IMU
* 相机间真实几何位置
* focal length
* aperture
* FOV 等。

([Isaac Sim 文档][5])

但是这里有一个**非常非常重要的坑**。

Isaac Sim 官方自己明确说：

> `Camera_Pseudo_Depth` 只是 depth firmware 输出的替代物。

它**没有真的跑 D455 的 stereo matching algorithm**。

实际做的是：

```text
场景
 ↓
直接获得真实几何深度
 ↓
Pseudo Depth
```

而不是：

```text
IR projector
 ↓
left IR
right IR
 ↓
stereo matching
 ↓
D455 depth
```

官方甚至明确说明：“如果真正从 stereo 重建并使用 RealSense 相同算法，才会产生包括 artifacts 在内的相同结果。”([Isaac Sim 文档][6])

所以：

### Isaac Sim D455 很适合做

```text
camera geometry
baseline
extrinsics
intrinsics
ROS interface
RGB/IR cameras
sensor placement
```

但**不适合直接拿来证明你的 active sensing gradient**。

新版本 Omniverse 其实还有一个更有意思的 generic depth sensor，它会：

```text
scene depth
→ disparity
→ stereo reprojection
→ occlusion holes
→ disparity noise
```

比 `Pseudo Depth` 更真实，但依然不是 D455 firmware，也不是 PyTorch autograd pipeline。([NVIDIA Docs][7])

---

## 4. 如果不用深度相机，RGB/灰度相机反而更容易

如果你的目标真的是：

> 先证明“可微感知能够改善导航”

那普通 grayscale camera 是最容易搞的。

例如仿真输出一个干净灰度图：

$$
I_{\rm clean}
$$

你的 camera action 是：

$$
a=(E,G)
$$

也就是：

* exposure
* gain

然后定义：

$$
I_{\rm sensor}
=
\operatorname{clip}
\left(
G\cdot E\cdot I_{\rm clean}
+
n(E,G)
\right)
$$

其中噪声可以用 reparameterization：

$$
n=\sigma(E,G)\epsilon,\qquad
\epsilon\sim\mathcal N(0,1)
$$

全部 PyTorch 写。

于是：

```text
exposure/gain
      ↓
brightness / saturation / noise
      ↓
grayscale image
      ↓
navigation network
      ↓
navigation loss
```

反向传播：

```text
navigation loss
       ↓
policy
       ↓
pixels
       ↓
exposure/gain
```

这一套实现难度大概只有你现在 D455 模型的 **1/10**。

而 Kornia 里的大量图像操作本身都是 PyTorch differentiable operation，包括滤波、warp、颜色变换等。([Kornia][8])

真机也很好做：

```text
USB/global-shutter camera
    ↓
manual exposure/gain
    ↓
navigation
```

所以如果只是 first proof-of-concept，我甚至会考虑它。

不过从你的论文连续性来看，我还是更倾向继续 D455。

---

# 5. 我认为你现在真正应该采用的方案

我看完你现在这版代码以后，最大的感受是：

你现在的 `_sensor_reference()` 已经开始变成一个**人为设计的 expert system** 了。

比如里面现在实际上有：

```python
active_signal
passive_signal
ambient_ir
motion
washout
noise_proxy
snr
```

然后 glare 又有：

```python
overexp
gain_sat
gain_exposure_sat
rescue
rescue_window
joint_sat
under_power
```

specular 又有：

```python
power_quad
power_knee
exposure_quad
exposure_bloom
gain_quad
gain_bloom
spec_safe
...
```

dark 又有：

```python
exposure_lift
gain_lift
projector_lift
dark_rescue
...
```

最后再：

```text
sigmoid
threshold
straight-through estimator
false depth
edge drift
body dropout
...
```

这个模型当然可以调到 work。

但是论文审稿人很容易问：

> 为什么这些系数是 0.38、0.74、0.22、0.26？

> 为什么 glare 是这个 sigmoid？

> 为什么 specular 是另一个 sigmoid？

> 这些现象是 RealSense D455 的真实响应吗？

这是你当前路线最大的隐患。

---

# 6. 我会把你的 sensor model 改成下面这样

保留你现在的：

```text
quadrotor simulator
      ↓
raw geometric depth
```

**这里完全不用改。**

然后增加一个：

```text
Differentiable Active Stereo Sensor
```

输入：

$$
D_{\rm gt},P,E,G
$$

输出：

$$
D_{\rm sensor},Q
$$

内部第一版甚至不用真正渲染左右 IR。

可以做成：

```text
raw depth
   ↓
compute geometry features
   ├── distance
   ├── depth edge
   ├── incidence proxy
   └── material / illumination feature
          ↓
sensor response network
(P,E,G,features)
          ↓
σ_depth
p_valid
bias
false-depth probability
          ↓
soft sensor depth
```

也就是：

$$
\sigma=f_\theta(D,\nabla D,P,E,G,m)
$$

$$
p_{\rm valid}
=
f_\theta(D,\nabla D,P,E,G,m)
$$

$$
b
=
f_\theta(D,\nabla D,P,E,G,m)
$$

再：

$$
D_{\rm obs}
=
p_{\rm valid}
(D+b+\sigma\epsilon)
$$

全部 PyTorch。

---

## 关键来了：\(f_\theta\) 从哪来？

不是手填。

### 用三种数据监督它：

第一来源：

**真实 D455**

```text
真实场景
P,E,G sweep
↓
记录 IR-left
IR-right
depth
```

第二来源：

**SAPIEN + SimSense**

```text
virtual scene
↓
ray-traced active IR
↓
SimSense
↓
depth
```

第三来源：

**ideal simulator**

用于获得：

```text
ground-truth depth
geometry
surface normal
material ID
```

于是你实际上是在学：

$$
f_\theta:
(D_{gt},P,E,G,\text{scene})
\rightarrow D_{D455}
$$

这个网络本身就是天然可微的。

这样就不需要把 RealSense firmware 可微化。

---

# 7. 如果你想更加“物理”，就使用 DDS 路线

第二版可以升级成：

```text
                projector power P
                       ↓
IR pattern × P
                       ↓
            differentiable renderer
              ↙                ↘
          left IR             right IR
             ↓                   ↓
      exposure + gain      exposure + gain
              ↘               ↙
                 cost volume
                      ↓
                 softargmin
                      ↓
                  disparity
                      ↓
                 fB / disparity
                      ↓
                    depth
```

这里可以直接借 DDS。

而且对你的课题来说有一个很漂亮的梯度：

$$
\frac{\partial L_{\rm nav}}{\partial P}
=
\frac{\partial L_{\rm nav}}{\partial D}
\frac{\partial D}{\partial d}
\frac{\partial d}{\partial C}
\frac{\partial C}{\partial I_{L,R}}
\frac{\partial I_{L,R}}{\partial P}
$$

同理：

$$
\frac{\partial L_{\rm nav}}{\partial E},
\quad
\frac{\partial L_{\rm nav}}{\partial G}
$$

这才真正是你标题里所说的：

> **differentiable sensing for navigation**

而不是“我人工设计了一个 quality function，然后对 quality function 求梯度”。

这两者的论文说服力差很多。

---

# 8. 几个项目放在一起看

| 项目                      |   真实感 | D455相关 |    可微 | 我建议的用途                         |
| ----------------------- | ----: | -----: | ----: | ------------------------------ |
| Isaac Sim D455          |     中 |  ★★★★★ |     ✗ | D455几何、标定、ROS接口                |
| Gazebo RealSense        |     低 |    ★★★ |     ✗ | ROS接口测试                        |
| Habitat + Redwood Noise |     中 |      ★ |     ✗ | 简单 depth noise baseline        |
| SimKinect               |     中 |      ✗ | 部分容易改 | 最简单 depth noise baseline       |
| **SAPIEN + SimSense**   | **高** |   ★★★★ |     ✗ | **forward teacher / 仿真真值**     |
| **DDS ICCV 2021**       | **高** | 主动深度相机 | **✓** | **可微算法设计蓝本**                   |
| redner                  |     高 |     通用 | **✓** | differentiable light transport |
| Mitsuba 3               |    很高 |     通用 | **✓** | 更物理的最终版本                       |
| nvdiffrast              |     中 |     通用 | **✓** | 快速可微几何渲染                       |
| PyTorch3D               |     中 |     通用 | **✓** | 快速 prototype                   |

SAPIEN 那篇论文自己也指出，DDS 是此前直接进行 end-to-end differentiable depth simulation 的工作。([arXiv][1])

---

# 9. 对你这个项目，我现在最推荐的不是“SAPIEN 替换 DiffPhysDrone”

而是：

```text
                    ┌──────────────┐
                    │   SAPIEN     │
                    │ + SimSense   │
                    └──────┬───────┘
                           │
                    realistic samples
                           │
Real D455 ────────────────→│
                           ↓
              ┌────────────────────────┐
              │ differentiable sensor │
              │       surrogate        │
              └───────────┬────────────┘
                          │
                          ↓
raw geometric depth → sensor model → observed depth
                          ↑
                       P,E,G
                          ↑
                    camera policy
                          ↑
                     navigation
```

也就是说：

### 飞行仿真

继续用你现在的 DiffPhysDrone。

### 几何深度

继续用你现在 CUDA renderer。

### 相机响应

把现在 `_sensor_reference()` 那个越来越复杂的手工模型逐渐替换掉。

### ground truth

用：

```text
SAPIEN / SimSense
+
real D455 sweep
```

### 可微结构

借：

```text
DDS
```

这样你的工程量会骤降。

---

## 我甚至觉得，这可能比你之前那条路线更适合重新组织论文

你论文的核心就可以变得非常干净：

$$
\boxed{
\text{task loss}
\rightarrow
\text{navigation}
\rightarrow
\text{depth}
\rightarrow
\text{camera parameters}
}
$$

然后做三个实验：

```text
Fixed
Learned + no sensor gradient
Learned + sensor gradient
```

区别真正只在：

```python
# ours
depth = differentiable_sensor(raw_depth, camera_params)

# no-diff
depth = differentiable_sensor(
    raw_depth,
    camera_params.detach()
)
```

这样如果 ours 赢了，

你的核心因果关系就真的非常清楚了。

---

而且还有一个非常重要的好消息：**你完全不需要一开始就复现 D455 的全部 ASIC / firmware。** RealSense 自己公开的资料也说明 D400 系列本质上是双目视差 + IR texture projector，投影器在低纹理/困难照明条件下用于增加可匹配纹理。([GitHub][9])

所以对你来说，一个：

**IR projector + stereo + softargmin**

的“D455-like differentiable active stereo camera”，已经有非常充分的合理性。

如果让我继续推进，我下一步最想做的是：**直接结合你现在这个 GitHub 分支，给你设计 `DiffPhysDrone Sensor V2` 的代码架构**——具体到应该删除/保留 `env_cuda.py` 哪些部分，新建哪些 `.py/.cu` 文件，以及先做一个 **64×48 可微双目主动深度相机**，让 Codex 能直接照着实现。这个现在已经可以设计得很具体了。

[1]: https://arxiv.org/abs/2201.11924 "Close the Optical Sensing Domain Gap by Physics-Grounded Active Stereo Sensor Simulation"
[2]: https://github.com/angli66/simsense "GitHub - angli66/simsense: A Real-Time Depth Sensor Simulator with GPU Acceleration · GitHub"
[3]: https://openaccess.thecvf.com/content/ICCV2021/html/Planche_Physics-Based_Differentiable_Depth_Sensor_Simulation_ICCV_2021_paper.html?utm_source=chatgpt.com "ICCV 2021 Open Access Repository"
[4]: https://openaccess.thecvf.com/content/ICCV2021/supplemental/Planche_Physics-Based_Differentiable_Depth_ICCV_2021_supplemental.pdf?utm_source=chatgpt.com "Physics-based Differentiable Depth Sensor Simulation"
[5]: https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/isaacsim_sensors_camera.html?utm_source=chatgpt.com "Camera Sensors — Isaac Sim Documentation"
[6]: https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/isaacsim_sensors_camera.html "Camera Sensors — Isaac Sim Documentation"
[7]: https://docs.omniverse.nvidia.com/kit/docs/omni.sensors.nv.camera/1.0.0/index.html?utm_source=chatgpt.com "Omniverse Camera Extension — Omniverse Kit"
[8]: https://kornia.readthedocs.io/en/latest/get-started/differentiability.html?utm_source=chatgpt.com "Differentiability — Kornia"
[9]: https://github.com/realsenseai/librealsense/blob/master/doc/depth-from-stereo.md?utm_source=chatgpt.com "librealsense/doc/depth-from-stereo.md at master · realsenseai/librealsense · GitHub"

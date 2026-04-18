> **免责声明**：当前 md 中的描述以当前代码版本为准；若后续实现变动，真实行为请优先以代码为准。

# `rerun` 可视化说明

本文说明当前项目中 `rerun` 的实现位置、数据流、各个视图与指标的含义，以及它适合验证什么、不适合验证什么。

---

## 1. `rerun` 在当前项目中的作用

当前 `rerun` 不是一个“照片级渲染器”，也不是 Unity / Isaac Sim 那种基于光照和材质的真实图形引擎。

它的定位是：

1. 可视化无人机、目标点、障碍物和相机姿态；
2. 可视化 `diff_depth` 深度观测的退化结果；
3. 可视化当前场景中的“现象证据”，用于检查仿真是否真的触发了对应场景机制；
4. 可视化策略是否根据这些现象调整了 `power / exposure / gain`。

所以它最适合回答的问题是：

- `sun_glare` 是否真的在出口附近触发了逆光退化？
- `specular_trap` 是否真的在镜面板附近触发了高光/反射问题？
- `vantablack_gap` 是否真的让门框区域的深度变差？
- `dark_morphing` 是否真的在狭缝和暗光下造成更强的感知困难？
- 无人机是否因此调整了 `power / exposure / gain`？

它不适合直接回答的问题是：

- “我拖动 3D 视角后，画面是否会像真实太阳那样刺眼？”
- “玻璃表面是否有照片级镜面反射效果？”

当前 `student_3d` 只能做**几何与语义层面的场景解释**，不能做真实图形渲染。

---

## 2. 相关代码文件

当前 `rerun` 链路主要分布在以下文件：

- `rerun_vis.py`
  - 定义 `RerunVis`
  - 负责发送 blueprint
  - 负责记录 3D 场景、2D 图像、时间序列标量
- `trainer.py`
  - 训练阶段调用 `vis.log_environment(...)` 与 `vis.log_step(...)`
- `eval.py`
  - 评测阶段调用 `vis.log_environment(...)` 与 `vis.log_step(...)`
- `env_cuda.py`
  - `render_diff_depth(...)` 内部生成并缓存当前帧的场景调试信息
  - `export_last_diff_depth_debug(...)` 将调试图和标量导出给 `rerun`

---

## 3. 数据流

当前的 `rerun` 数据流如下：

1. `env.render_diff_depth(power, exposure, gain)` 被调用；
2. `env_cuda.py` 中的 `_apply_diff_depth_sensor_model(...)` 计算：
   - `noisy_depth`
   - `quality`
   - `valid`
   - 各种场景相关的局部退化量
3. 同一帧里，环境把调试结果缓存到 `self.last_diff_depth_debug`；
4. 训练或评测循环调用 `env.export_last_diff_depth_debug(env_idx)`；
5. `trainer.py` / `eval.py` 把这些图和标量传给 `RerunVis.log_step(...)`；
6. `rerun_vis.py` 把它们记录到：
   - `student_3d`
   - `depth`
   - `quality`
   - `invalid`
   - `scene_effect`
   - `power / exposure / gain`
   - step 级标量曲线

---

## 4. 如何开启

需要在参数中启用：

- `--vis_enable`
- `--vis_backend rerun`

通常还会搭配：

- `--vis_spawn`
- `--vis_student`
- `--vis_teacher`

训练与评测都可以使用 `rerun`：

- 训练入口：`main_cuda.py`
- 评测入口：`eval.py`

---

## 5. 当前面板说明

当前默认重点观察的是 `student` 命名空间。

### 5.1 `student_3d`

这是三维几何视图，主要显示：

- 无人机本体和轨迹；
- 目标点；
- 体素障碍物；
- 深度相机的视锥；
- 当前场景的关键语义对象。

它的用途是：

- 看无人机走了什么轨迹；
- 看障碍物位置是否合理；
- 看当前场景的关键触发位置在哪里。

它不是一个真实光照渲染窗口。

---

### 5.2 `depth`

这个窗口显示的是 `depth_aux`，也就是**为了方便观察而做过对比度拉伸的深度图**。

显示规则：

- 近处更亮；
- 远处更暗；
- 无效像素显示为暗灰；
- 它更适合“观察结构和空洞”，不是做定量读数。

注意：

- 它不是原始深度数值的直接灰度映射；
- 论文图和调试时看它很有用，但如果要做严格数值分析，还是要回到 tensor 本身。

---

### 5.3 `quality`

`quality` 是当前每个像素的**深度可靠性软评分**，范围大致在 `[0, 1]`。

可以把它理解成：

- 越接近 `1`：这个像素的深度更可信；
- 越接近 `0`：这个像素的深度更不可信。
- 看哪些区域“本来就不可信”
- 越亮表示越可靠

它并不是简单的“亮度”或“对比度”，而是综合下面这些因素得到的：

- 主动信号强弱；
- 被动信号强弱；
- 环境光淹没（washout）；
- 镜面高光（specular bloom）；
- 运动模糊；
- 超量程惩罚。

因此：

- 在 `sun_glare` 中，逆光强的区域通常会让 `quality` 降低；
- 在 `specular_trap` 中，镜面板附近高功率时 `quality` 会下降；
- 在 `vantablack_gap` 和 `dark_morphing` 中，黑框 / 狭缝附近也可能出现 `quality` 降低。

---

### 5.4 `invalid`

`invalid` 表示当前深度像素的**无效程度**。

在代码语义里，先根据 `quality` 生成一个 `valid`，再得到：

- `invalid = 1 - valid`

可以把它理解成：

- 越接近 `1`：这个像素更像“空洞 / 无返回 / 无效深度”；
- 越接近 `0`：这个像素更像“有效深度”。

因此它和 `quality` 的关系通常是负相关的：

- `quality` 低的区域，经常对应 `invalid` 高；
- 如果某个区域大量变亮，说明该区域正在丢失有效深度。
- 看哪些区域已经变成空洞/无效深度
- 越亮表示越无效

---

### 5.5 `scene_effect`

这是当前最重要的“场景现象证据图”。

它不是通用深度图，而是**当前场景特有的触发区域或退化强度图**。不同场景含义不同：

- `sun_glare`
  - 显示逆光 / 眩光惩罚强度；
  - 越亮说明当前位置和视角下，`sun_glare` 影响越强。

- `specular_trap`
  - 显示镜面板区域在当前激光功率下的危险程度；
  - 越亮说明高功率反射问题越强。

- `vantablack_gap`
  - 显示黑洞门框区域；
  - 越亮说明当前视图中低反照率门框区域占比越大。

- `dark_morphing`
  - 显示暗狭缝相关的敏感区域；
  - 越亮说明当前图像里“暗光 + 狭缝 + 运动”耦合影响越强。

- `scene_effect` 是最关键的“场景现象证据图”
  - `sun_glare` 下亮的地方就是逆光/炫光真正生效的区域
  - `specular_trap` 下亮的地方就是镜面陷阱主要影响区
  - `vantablack_gap` / `dark_morphing` 下亮的地方就是黑洞门框/狭缝相关的退化区域

如果你的目标是验证“场景是否真的被仿真出来”，`scene_effect` 是第一优先级观察对象。

---

## 6. 3D 场景语义 overlay

为了让 `student_3d` 不只是“看几何盒子”，当前还会额外画出场景关键对象：

- `sun_glare`
  - 太阳锚点 `SUN`
  - 太阳方向箭头 `SUN_RAY`

- `specular_trap`
  - 中央镜面板 `SPECULAR_PANEL`

- `vantablack_gap`
  - 黑洞门洞区域 `VANTABLACK_GAP`

- `dark_morphing`
  - 狭缝区域 `DARK_SLIT`

这些 overlay 的意义是：

- 帮助你知道场景事件“在世界坐标里”发生在哪里；
- 再和 `depth / quality / invalid / scene_effect` 对起来看。

---

## 7. 时间序列指标含义

### 7.1 相机控制量

这些是最直接的策略输出：

- `power`
  - 激光发射功率控制量；
  - 越高说明主动照明更强。

- `exposure`
  - 曝光控制量；
  - 越高通常意味着更长的有效曝光，也更容易带来运动模糊。

- `gain`
  - 接收增益控制量；
  - 越高说明在暗环境下更依赖增益放大，但也更容易放大噪声。

如果你的目标是验证“策略是否对现象做出响应”，这三条曲线必须和 `scene_effect` 一起看。

---

### 7.2 通用场景调试标量

当前 `env_cuda.py` 会导出如下场景调试标量：

- `scene_mask_mean`
  - 当前视图中，场景触发区域平均占比；
  - 越大说明当前画面里“事件区域”越多。

- `scene_effect_mean`
  - 当前视图中，场景效应强度的平均值；
  - 越大说明该场景现象当前越强。

- `quality_mean`
  - 当前帧 `quality` 的平均值；
  - 越高说明整体深度越可靠。

- `invalid_rate`
  - 当前帧 `invalid` 的平均值；
  - 越高说明整体空洞 / 无效深度越多。

- `ambient_ir_mean`
  - 当前帧平均环境红外水平；
  - 对 `sun_glare` 尤其重要。

- `signal_active_mean`
  - 当前帧平均主动测距信号强度；
  - 大致反映激光 + 曝光 + 反照率带来的主动有效信号。

- `signal_passive_mean`
  - 当前帧平均被动信号强度；
  - 更像环境光和纹理给双目匹配带来的帮助。

- `spec_bloom_mean`
  - 平均镜面高光强度；
  - 对 `specular_trap` 尤其重要。

- `motion_blur_mean`
  - 平均运动模糊强度；
  - 对 `dark_morphing` 和高速穿越场景尤其重要。

- `washout_mean`
  - 平均环境淹没程度；
  - 越高说明主动信号越被环境光压制。

- `far_override_mean`
  - 平均远距覆盖程度；
  - 越高说明越多像素被“拉向远距/失真”。

- `scene_id`
  - 当前场景的编号，仅用于快速区分场景，不表示物理量。

---

### 7.3 评测阶段额外标量

`eval.py` 中还会记录一些飞行状态量：

- `speed_mps`
  - 当前速度大小
- `angular_speed_rps`
  - 当前角速度大小
- `thrust_norm_mps2`
  - 当前控制推力幅值
- `accel_norm_mps2`
  - 当前加速度幅值
- `dist_to_goal_m`
  - 当前到目标点的距离

这些量主要用于回答：

- 无人机是因为场景太难而减速了吗？
- 调参之后它是否保持了稳定飞行？
- 什么时候它开始接近目标？

---

## 8. 如何用 `rerun` 判断场景是否实现正确

下面给出四个场景的推荐观察方式。

### 8.1 `sun_glare`

推荐同时观察：

- `student_3d`
  - 看 `SUN` 和出口位置关系
- `scene_effect`
  - 看逆光效应是否只在合理区域变亮
- `quality`
  - 看逆光区域是否明显下降
- `invalid`
  - 看是否出现更多空洞
- `power / exposure`
  - 看策略是否提高 `power`、降低 `exposure`

如果只看到 `scene_effect` 变亮，但 `quality`、`invalid`、相机参数完全没反应，说明仿真或策略还没真正打通。

---

### 8.2 `specular_trap`

推荐同时观察：

- 3D 中的 `SPECULAR_PANEL`
- `scene_effect`
- `quality`
- `invalid`
- `power`

理想现象是：

- 面板进入视野后，`scene_effect` 上升；
- 高功率时 `quality` 变差，`invalid` 或远距失真增加；
- 策略随后降低 `power`。

---

### 8.3 `vantablack_gap`

推荐同时观察：

- 3D 中的 `VANTABLACK_GAP`
- `scene_effect`
- `quality`
- `invalid`
- `gain / exposure`

理想现象是：

- 门框附近是主要退化区域；
- 黑色门框比普通墙更容易导致感知变差；
- 策略会尝试用 `gain` 和 `exposure` 做权衡。

---

### 8.4 `dark_morphing`

推荐同时观察：

- 3D 中的 `DARK_SLIT`
- `scene_effect`
- `motion_blur_mean`
- `quality`
- `exposure / gain`
- `speed_mps`

理想现象是：

- 狭缝附近在高速时更难看清；
- `motion_blur_mean` 上升；
- 策略可能降低速度或限制曝光，避免拖影。

---

## 9. 当前限制

当前 `rerun` 的限制主要有三点：

### 9.1 不是照片级真实渲染

`student_3d` 目前只能显示：

- 几何结构
- 场景关键对象
- 相机视锥

不能显示：

- 真实太阳炫光 bloom
- 真实玻璃镜面反射
- 真实材质 BRDF

---

### 9.2 当前调试图主要依赖 `python` 版 `diff_depth`

目前：

- `diff_depth=python`
  - 会导出 `quality / invalid / scene_effect` 等调试图
- `diff_depth=cuda`
  - 当前只返回 `noisy_depth`
  - 不单独导出 `quality`
  - 也不会生成上述完整调试缓存

因此如果你的目标是**验证仿真现象是否实现正确**，当前更推荐使用：

- `--diff_sensor_impl diff_depth=python`

---

### 9.3 `depth` 窗口不是原始数值图

`depth_aux` 为了可读性做了拉伸和着色：

- 适合看结构；
- 不适合直接读真实深度数值。

如果要做数值分析，应该回到 tensor 本身。

---

## 10. 最推荐的使用方法

如果你想验证“场景实现是否正确、策略是否响应”，建议固定只看下面几项：

1. `student_3d`
2. `depth`
3. `quality`
4. `invalid`
5. `scene_effect`
6. `power / exposure / gain`

然后围绕一个最简单的问题来观察：

- 当前场景现象有没有触发？
- 触发后深度是否真的坏了？
- 坏了以后策略是否真的调参了？

只要这三步闭环成立，`rerun` 就真正发挥作用了。


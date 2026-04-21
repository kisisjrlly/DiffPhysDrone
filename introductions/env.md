> **免责声明**：当前md中的描述并不完全等价项目中的代码实现，真实的实现以代码为准。

# `env_cuda.py` 场景创建与仿真机制详解

本文专门回答：`env_cuda.py` 里到底创建了什么场景、怎么创建、以及这些场景如何被仿真。

---

## 1. 这个环境本质上是什么

`Env` 是一个 **diff_depth-only** 的固定小地图环境（不是旧版大随机世界）。

核心特征：

- 地图坐标边界：`x,y ∈ [-5, 5]`，`z` 约在 `[0, 3]`
- 默认起点：`(-5, 0, 1.5)`
- 默认终点：`( 5, 0, 1.5)`
- 几何障碍以 **体素盒（voxels）** 为主，`balls/cyl/cyl_h` 在当前任务中通常为空
- 感知使用可微主动深度相机链路（`render_diff_depth`）
- 动力学推进使用 `autograd_ops.run(...)` + CUDA 姿态更新

环境的设计理念是：

1. 机动几何适度简化（避免把问题都变成飞控难题）；
2. 把难点放在感知退化（光照/材质/反射/极暗）和传感控制策略上；
3. 场景可切换、可批量并行、可微训练友好。

---

## 2. 支持的场景类型（以及别名）

`supported_scenarios` 固定为 5 种：

1. `base`
2. `sun_glare`
3. `specular_trap`
4. `vantablack_gap`
5. `dark_morphing`

另外在 `_normalize_scenarios` 里做了兼容别名：

- `random_base` / `random` / `random_scene` → `base`
- `black_gap` → `vantablack_gap`
- `dark_slit_lite` → `dark_morphing`

如果 `reset(scene_name=None)`，会在 `self.scenarios` 中随机选一个场景；
如果显式传 `scene_name`，就固定重置到指定场景。

---

## 3. 场景“几何”是怎么创建的

### 3.1 几何表示形式

当前障碍主数据是 `self.voxels`，每个 voxel 一行 6 个数：

`[cx, cy, cz, hx, hy, hz]`

- `(cx, cy, cz)`：盒中心
- `(hx, hy, hz)`：半尺寸（half extents）

`reset()` 时会把单场景几何复制到 batch：

- `self.voxels = voxels.unsqueeze(0).repeat(B, 1, 1)`

也就是说，**同一个 batch 内所有并行环境共享同一张几何图**（但传感器扰动参数可随机化）。

---

### 3.2 各场景几何构造函数

#### A) `base`：轻量 slalom 基础图

由 `_build_base_voxel_layout()` 生成 6 个交错立方体柱：

- 沿 x 轴从左到右排布
- y 方向交错正负（形成蛇形绕行）
- 高度统一（`hz≈1.5`）

目的是让飞行有基础避障需求，但不过度复杂。

---

#### B) `sun_glare`：最小逆光关键障碍场景

由 `_build_sun_glare_voxel_layout()` 构造：

- 不再使用走廊墙、门框或复杂开口
- 只保留少量固定高柱体
- 其中有一个关键柱体位于逆光区域中心线附近
- 在 `goal` 与光源之后还放置一块较窄的背景墙，用来形成更真实的“出口后背景面”感，但不会做成封死全场的大墙
- 该场景使用 6m 小地图语义：起点 `x=-3`，终点 `x=2`，光源 `x=2.8`，背景墙 `x=3.0`

这使得无人机从较正常观测区域飞向目标方向时，在进入逆光区后必须重新调节
`power / exposure / gain` 才更容易继续看清并绕开关键障碍。

---

#### C) `specular_trap`：镜面/反光陷阱结构

由 `_build_specular_trap_layout()` 构造：

- 前后放置几根柱状障碍
- 中央放置一个“薄面板”式结构（用于触发强反射行为）

其几何配合后续材质/高光参数，形成“高功率激光反而害自己”的情况。

---

#### D) `vantablack_gap`：黑洞门洞

由 `_build_opening_wall(...)` 生成“带孔墙”：

- 墙体由四块大盒拼出（上、下、左、右）
- 中间留一个缝隙（gap）作为可穿越通道

该场景还会降低速度与安全边距：

- `max_speed = 1.35`
- `margin = 0.12`

代表“暗环境 + 门洞穿越”任务。

---

#### E) `dark_morphing`：更窄的极暗狭缝

同样用 `_build_opening_wall(...)`，但缝隙更窄：

- gap 更小（相比 `vantablack_gap`）
- 任务更苛刻

并进一步收紧动力学任务参数：

- `max_speed = 0.95`
- `margin = 0.08`

用于模拟“极暗 + 高机动精度”场景。

---

### 3.3 几何参数速查（关键数值）

| 场景 | 几何构造方式 | 关键几何参数 | 任务参数改动 |
|---|---|---|---|
| `base` | `_build_base_voxel_layout()` | 6 个柱体，`x≈[-3.4, -2.0, -0.6, 0.8, 2.2, 3.6]`，`y=±0.95` 交错，`hz=1.5` | `max_speed=1.8`, `margin=0.18`（默认） |
| `sun_glare` | `_build_sun_glare_voxel_layout()` | 4 个 0.5m 宽高柱体 + 1 块位于 `x=3.0` 的窄背景墙；起点 `x=-3`，终点 `x=2`，光源 `x=2.8`，关键柱体位于 `x≈1.45, y≈0.0` 的逆光区域中心附近 | 速度与 margin 保持默认 |
| `specular_trap` | `_build_specular_trap_layout()` | 中央薄面板 `x=0.0, hy≈0.95, hz≈1.15` + 前后柱体 | 速度与 margin 保持默认 |
| `vantablack_gap` | `_build_opening_wall(...)` | 墙面 `x=0.0`，门洞中心 `y=0.85,z=1.5`，`gap_half_w=0.58`, `gap_half_h=0.95` | `max_speed=1.35`, `margin=0.12` |
| `dark_morphing` | `_build_opening_wall(...)` | 墙面 `x=0.0`，狭缝中心 `y=-0.80,z=1.5`，`gap_half_w=0.32`, `gap_half_h=0.88` | `max_speed=0.95`, `margin=0.08` |

---

## 4. 场景“传感条件”是怎么创建的

几何之外，环境还会设置每个 batch 的光照/材质随机参数：

- `_cam_ambient`
- `_cam_dir_intensity`
- `_cam_fog_beta`
- `_cam_airlight`
- `_cam_mat_obstacle`
- `_cam_mat_spec`

流程分两层：

1. `_reset_camera_states()`：先按当前 preset（low/medium/high/ultra）给一组基础随机分布；
2. `_apply_scene_sensor_profile(scene_name)`：再按场景覆盖到更有针对性的范围。

注意：`base` 场景不会进入 `if/elif` 覆盖分支，直接使用 `_reset_camera_states()` 的基础随机结果。

各场景覆盖范围（来自 `_apply_scene_sensor_profile`）：

| 场景 | ambient | dir_intensity | fog_beta | airlight | mat_obstacle | mat_spec | 含义 |
|---|---:|---:|---:|---:|---:|---:|---|
| `base` | 由 preset 决定 | 由 preset 决定 | 由 preset 决定 | 由 preset 决定 | 由 preset 决定 | 由 preset 决定 | 基础随机退化 |
| `sun_glare` | 0.10~0.18 | 0.35~0.75 | 0.010~0.030 | 0.12~0.25 | 0.52~0.78 | 0.04~0.10 | 配合局部太阳 mask 触发逆光惩罚 |
| `specular_trap` | 0.08~0.16 | 0.18~0.42 | 0.006~0.018 | 0.05~0.12 | 0.45~0.72 | 0.18~0.38 | 高镜面材质，放大高光陷阱 |
| `vantablack_gap` | 0.02~0.06 | 0.05~0.16 | 0.003~0.015 | 0.02~0.08 | 0.30~0.48 | 0.00~0.02 | 极暗低反射门洞 |
| `dark_morphing` | 0.006~0.020 | 0.015~0.070 | 0.004~0.020 | 0.005~0.020 | 0.22~0.38 | 0.00~0.01 | 更极端暗光+狭缝 |

---

## 5. 场景效果如何注入到“图像级”仿真

几何 + 全局参数还不够，`env_cuda.py` 还做了图像空间局部调制：

- `_scene_sensor_adjustments(...)`

这一步会：

1. 把场景中的关键世界点（如太阳锚点、反光板中心、门洞中心）投影到相机像平面；
2. 在图像上生成高斯/盒状 mask；
3. 按 mask 局部修改传感器模型中的项（ambient/albedo/spec/motion/quality/far_override 等）。

对应场景逻辑大意：

- `sun_glare`：在“目标方向太阳投影 + 关键障碍局部区域”制造逆光退化，并把局部质量统计聚焦到必须看清的关键障碍附近
- `specular_trap`：高功率下高光惩罚增强，并可能触发远距覆盖（深度失真）
- `vantablack_gap`：门框区域反照率/被动信号下降，运动模糊影响上升
- `dark_morphing`：全局更暗，狭缝边缘更敏感，对曝光与运动更苛刻

这使场景不是“只改常数”，而是**在图像局部有结构化退化**。

---

## 6. `reset()` 到仿真的完整创建流程

`reset(scene_name)` 可以概括为以下步骤：

1. 选场景（指定或随机）
2. 设置相机外参 `R_cam`（由 `cam_angle` 生成固定俯仰）
3. 调 `_build_scene_geometry(scene_name)` 得到：
	- `voxels`
	- `start`
	- `goal`
	- `max_speed`
	- `margin`
	- `effects`（给图像级场景调整用）
4. 写入动力学参数：`drag_2, pitch/yaw delay, margin, max_speed ...`
5. 初始化状态：`p, v, act, a, dg, v_wind, R, R_old, p_old`
6. 初始化并应用传感器场景配置：
	- `_reset_camera_states()`
	- `_apply_scene_sensor_profile(scene_name)`

到此，场景和仿真状态都就绪。

---

## 7. 场景如何被“仿真”出来

### 7.1 感知仿真（深度渲染）

入口：`render_diff_depth(power, exposure, gain)`

- 后端选择：
  - `python`：几何深度 + Python/Torch 可微传感器链（推荐训练用）
  - `cuda`：CUDA 后端（质量图可能不单独输出）

python 路径核心：

1. `quadsim_cuda.render_depth(...)` 先出几何深度
2. `_apply_diff_depth_sensor_model(...)` 注入：
	- 主动/被动信号
	- 反射与高光
	- 运动模糊
	- 飞点与噪声
	- 无效化（valid gating）
3. 输出 `(noisy_depth, quality)`

这一步真正把场景光学特性变成深度观测退化。

---

### 7.2 动力学仿真（机体状态推进）

入口：`run(act_pred, ctl_dt, v_pred)`

- 使用 `autograd_ops.run(...)` 做可微动力学推进
- 更新 `p, v, a, act`
- 再根据控制延迟更新姿态 `R`

因此每个场景都在同一动力学内核运行，不同的是：

- 几何障碍布局
- 场景速度/安全边距参数
- 感知退化模型的局部和全局参数

---

## 8. 场景创建机制总结（最简版）

`env_cuda.py` 的场景机制可概括为三层叠加：

1. **几何层**：体素障碍布局（走廊、反光板、门洞/狭缝）
2. **统计层**：场景化光照/材质参数分布
3. **图像层**：基于投影 mask 的局部退化调制

最终在统一可微链路里，把“场景”变成“深度可见性与控制难度”的差异。

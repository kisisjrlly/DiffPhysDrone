> **免责声明**：当前md中的描述并不完全等价项目中的代码实现，真实的实现以代码为准。

# DiffPhysDrone 数据闭环全链路说明（渲染 → Model → 控制 → 仿真 → 反向传播）

## 0. 先回答你最关心的一句

`Env.render_diff_depth(power, exposure, gain)` 的三个输入都是 **[0,1] 的归一化控制档位**，**不是 D455 SDK 的真实寄存器值**。

- `power`：归一化主动光功率档位（语义上 0=关，1=满档）
- `exposure`：归一化曝光档位，内部映射为有效曝光时间尺度
- `gain`：归一化增益档位，内部映射为 ISO 风格增益尺度

当前默认映射（见 `camera_semantics.py`）：

- $t_{eff}=\mathrm{clamp}(0.25 + 2.75\cdot exposure01,\ 0.15,\ 4.0)$
- $gain\_scale=1.0 + 10.0\cdot gain01^{1.2}$
- $power01=\mathrm{clamp}(power01,0,1)$（直接作为主动光强比例参与传感器模型）

> 所以：日志里 `power=0.8` 表示“仿真归一化档位 0.8”，不是“D455 硬件寄存器值 0.8”。
> 若要上真机，需要额外加一层 `[0,1] -> D455 实际参数` 的标定映射。

> 本文描述 **训练主链路（`trainer.py::student_rollout`）**，并补充 teacher/eval 的差异。  
> 重点回答：每一步函数的输入/输出是什么？数值是归一化量还是物理语义量？梯度如何流动？

---

## 1. 一眼看懂：单步闭环总览

在第 $t$ 步，代码的**实际执行顺序**是：

1. 用当前相机状态 `power_t/exposure_t/gain_t` 渲染深度；
2. 用缓冲中的历史动作 `a_{t-1}` 执行 `env.run(...)` 推进动力学；
3. 基于推进后的状态构造 `state_t`，送入 `Model`，得到 `act_t/cam_params_t`；
4. 用 `cam_params_t` 更新下一步相机状态 `power_{t+1}/exposure_{t+1}/gain_{t+1}`；
5. 将 `act_t` 解码为物理命令并写入动作缓冲（供后续步使用）；
6. 累积 loss，反向更新网络参数；
7. 进入下一步，形成闭环。

注意：代码里 `env.run(act_buffer[t], ...)` 在本步前半段执行，因此存在“动作缓冲/控制延迟”机制（更接近真实控制链路）。

可把它理解为：**本步感知用当前相机状态，动力学执行用上一拍动作，网络输出用于下一拍执行与下一帧感知。**

---

## 1.1 闭环时序图（Mermaid）

```mermaid
sequenceDiagram
	autonumber
	participant Cam as CameraState(power/exposure/gain)
	participant Rend as render_sensors / render_diff_depth
	participant Env as Env Dynamics
	participant Pol as Model.forward
	participant Upd as update_camera_params
	participant Dec as decode_action_*
	participant L as losses + aggregate_loss
	participant Opt as backward + optimizer.step

	Cam->>Rend: (power_t, exposure_t, gain_t) in [0,1]
	Rend-->>Pol: depth_obs_t (meter-like), quality_t in [0,1]
	Note over Env,Pol: step t dynamics consumes buffered a_{t-1}
	Env->>Env: run(a_{t-1}, ctl_dt, target_v_raw)
	Env-->>Pol: state_t (mixed units), hx_{t-1}
	Pol-->>Upd: cam_params_t in [0,1], h_t
	Upd-->>Cam: (power_{t+1}, exposure_{t+1}, gain_{t+1}) via EMA
	Pol-->>Dec: act_raw_t (or intent_t)
	Dec-->>Env: a_t (m/s^2), appended to action buffer
	Env-->>L: trajectories / speeds / distances
	Cam-->>L: power/exposure/gain histories + fill proxy
	L-->>Opt: scalar loss
	Opt-->>Pol: update policy weights
```

### 图例：每条主链路上的数值域

- `power/exposure/gain`, `cam_params`: **归一化控制量**，范围 `[0,1]`
- `depth_obs`: **深度观测**（米语义），但含噪声/空洞
- `quality`: **像素可用性/置信度代理**，范围 `[0,1]`
- `state_t`: **混合量纲**（m、m/s、方向余弦、归一化相机状态）
- `a_t`: **物理加速度命令**（m/s²）
- `loss`: **无量纲标量目标**（多项加权和）

---

## 2. 关键状态与数值域（先建立统一语义）

### 2.1 相机控制三元组（网络输出 + 仿真控制）

- `power`：归一化控制量，范围 `[0,1]`，语义是“主动光发射功率档位”。
- `exposure`：归一化控制量，范围 `[0,1]`，语义是“曝光档位”。
- `gain`：归一化控制量，范围 `[0,1]`，语义是“接收增益档位”。

它们是 **归一化绝对控制量**（不是增量），随后在传感器模型中映射到更“物理语义”的内部量。

### 2.2 深度/质量图

- `depth_obs`: `(B,H,W)`，单位近似米（m），但包含噪声、空洞（无效像素常为 0）。
- `quality`: `(B,H,W)`，`[0,1]`，表示该像素的观测质量/有效性（确定性、可微，python 后端可用）。

### 2.3 动力学量

- `env.p` 位置，m
- `env.v` 速度，m/s
- `act_final` 推力加速度命令，m/s²（含重力补偿/估计误差后再限幅）

---

## 3. 闭环逐步展开（函数级 I/O + 数值语义）

下面按 `student_rollout` 的时间步顺序说明。

---

### Step A. 控制步长生成（受曝光影响）

函数/位置：`trainer.py` 中

- `base_dt = normalvariate(1/base_control_freq, 0.1/base_control_freq)`
- `exposure_delay = diff_depth_exposure_to_time(exposure.mean()) * 0.01`
- `ctl_dt = base_dt + exposure_delay`

涉及函数：

- `rollout_ops.diff_depth_exposure_to_time(exposure01)`
	- 输入：`exposure01`（归一化 `[0,1]`）
	- 输出：`exp_phys`（统一“有效曝光时间尺度”）
	- 实现：`CameraSemantics.exposure_to_time`

数值语义：

- `exposure01` 是归一化控制值；
- `exp_phys` 是内部统一时间尺度（用于 blur proxy 与 dt 延时）；
- 最终 `ctl_dt` 是物理时间（秒）。

---

### Step B. 传感器渲染（从仿真几何到深度观测）

函数：`rollout_ops.render_sensors(env, ctl_dt, power, exposure, gain, differentiable=True)`

- 输入：
	- `env`: 当前仿真状态（位置/姿态/障碍物）
	- `power/exposure/gain`: 当前相机状态，`(B,)`，归一化 `[0,1]`
	- `differentiable=True`: 训练时保留梯度
- 输出：
	- `depth_obs`: `(B,H,W)` 噪声深度图
	- `quality`: `(B,H,W)` 质量图（python 后端），cuda 后端可能为 `None`

下钻调用：

1. `Env.render_diff_depth(power, exposure, gain)`
2. `Env._render_diff_depth_python(...)`（默认）
3. `quadsim_cuda.render_depth(...)` 先做几何深度
4. `Env._apply_diff_depth_sensor_model(depth, power, exposure, gain)` 叠加 D455 风格退化

`_apply_diff_depth_sensor_model` 内关键映射：

- `power01 = clamp(power, 0,1)`
- `exposure_s = cam_sem.exposure_to_time(exposure)`
- `gain_scale = cam_sem.iso_to_gain(gain)`

因此：

- 输入控制仍是归一化；
- 渲染时转换为“物理语义”内部量（曝光时间尺度、ISO 增益尺度）；
- 输出 `depth_obs`（m 语义）和 `quality`（0~1 语义）。

---

### Step C. Fill-rate 统计（训练里优先 quality）

函数：`compute_depth_fill_rate(depth_like, min_valid_depth, softness=None)`

- 输入：`depth_like`（训练中优先 `quality`，否则退化到 `depth_obs`）
- 输出：标量 fill rate

训练当前逻辑：

- `fill_src = depth_quality if depth_quality is not None else depth_obs.detach()`
- `fill_rate_t = compute_depth_fill_rate(fill_src, ...)`
- `fill_rate_soft_t = compute_depth_fill_rate(fill_src, ..., softness=...)`

数值语义：

- fill rate 是概率/占比（0~1）；
- 用 `quality` 时梯度可稳定回到相机控制分支（避免 `randn` 噪声污染）；
- fallback 到 `depth_obs.detach()` 时仅作统计，不再反传到渲染链路。

---

### Step D. 仿真推进（使用动作缓冲）

函数：`env.run(act_buffer[t], ctl_dt, target_v_raw)`

- 输入：
	- `act_buffer[t]`: 历史动作（物理加速度命令）
	- `ctl_dt`: 秒
	- `target_v_raw`: 当前目标速度方向量
- 输出（更新 `env` 内部状态）：`p,v,a,R,act,...`

数值语义：

- `act_pred` 是 m/s² 语义的控制命令；
- `run` 内部结合动力学扰动、阻力、控制延迟推进状态；
- 这是“模型输出反过来影响仿真”的关键闭环环节（虽有一拍延迟）。

---

### Step E. 构造模型输入状态向量

#### E1) `build_local_frame(env)`

- 输入：`env.R`
- 输出：`R`（仅 yaw 对齐的局部坐标系）

#### E2) `compute_target_velocity(target_v_raw, env)`

- 输入：原始目标向量
- 输出：`target_v`（限幅到 `env.max_speed`，m/s）

#### E3) `build_state_vector(...)`

- 输入：`env, target_v, R, power, exposure, gain, no_odom, include_camera_state`
- 输出：
	- `state`: `(B, obs_dim)`
	- `local_v`: `(B,3)`

`state` 组成：

- `local_v`（可选，m/s）
- `tv_local`（m/s）
- `env.R[:,2]`（方向余弦，[-1,1]）
- `env.margin`（m）
- 相机状态（可选）：`power/exposure/gain` 从 `[0,1]` 映射到 `[-1,1]`

所以 `state` 是“混合量纲向量”：既有物理量（m、m/s），也有无量纲方向量和归一化控制量。

---

### Step F. 策略网络前向（`model.forward`）

调用：`act, cam_params, h = model(state, h, depth_obs=depth_obs, add_noise=...)`

输入：

- `v=state`: `(B,obs_dim)`，混合语义
- `hx=h`: `(B,192)` 或 `None`
- `depth_obs`: `(B,H,W)`，m 语义

输出：

- `act`: `(B,dim_action)`，动作域原始输出（默认 dim_action=6）
- `cam_params`: `(B,3)`，经 sigmoid 后 `[0,1]`
- `h`: 新隐状态
- （可选）`intent`：dLQR 模式

内部预处理（关键归一化）：

1. `preprocess_depth_input` / `_depth_pipeline`
2. 有效深度阈值：`depth >= depth_min_valid`
3. 反深度映射到 `[0,1]`
4. 再线性映射到 `[-1,1]`

因此：

- 网络看到的深度不是原始米值，而是 **反深度归一化特征**；
- 网络输出 `cam_params` 是归一化绝对控制量；
- 网络输出 `act` 还不是最终物理命令，需要后处理解码。

---

### Step G. 相机参数更新（输出反作用到下一帧渲染）

函数：`update_camera_params(cam_params, power, exposure, gain, env)`

- 输入：
	- `cam_params`: `(B,3)`，归一化绝对控制量（本步网络输出）
	- `power/exposure/gain`: `(B,)`，当前相机状态
- 输出：
	- 新状态 `power/exposure/gain`：用于下一步渲染
	- `cam_hist_entry`：原始 `cam_params`（用于 camera loss）

更新公式（`alpha=0.7`）：

$$
x_{t+1}=0.7\,\mathrm{sg}(x_t)+0.3\,\hat{x}_t,\quad \hat{x}_t=\mathrm{clamp}(cam\_params_t,0,1)
$$

其中 $x\in\{power,exposure,gain\}$，`sg` 表示对历史状态 `detach`。

意义：

- 让传感器控制有时间连续性（不瞬变）；
- 历史状态不反传，控制图不无限增长；
- 同时保留原始 `cam_params` 供 `loss_cam_smooth` 完整回传。

---

### Step H. 动作解码（网络输出 → 物理控制）

有两条路径：

#### H1) 直接动作域：`decode_action_direct(act_raw, R, env, B, max_acc_cmd)`

- 输入：`act_raw`（网络原始动作）
- 输出：
	- `a_final`: `(B,3)`，m/s²，限幅后的物理命令
	- `v_pred`: `(B,3)`，辅助速度预测

流程：

1. `act_raw.reshape(B,3,-1)`（默认 6 维会得到 3x2）
2. 通过 `R @ ...` 从局部系转世界系
3. 叠加重力/推力估计误差校正
4. clamp 到 `[-max_acc_cmd, max_acc_cmd]`

#### H2) 意图+dLQR：`decode_action_lqr(intent, ...)`

- `intent[:3]` → `v_ref_local`（tanh 后乘 `max_speed`，m/s）
- `intent[3:6], intent[6:9]` → `Q/R` 对角
- dLQR 求 `u_local`（m/s²），再转世界系并限幅

最终均得到 `a_final`，写入 `act_buffer.append(a_final)`，供后续 `env.run` 使用。

---

### Step I. Loss 组装与反向传播（闭环学习）

#### I1) 物理损失：`compute_physics_losses(...)`

- 输入：`v,target_v,act,vec_to_pt,p,margin,...`
- 输出：`loss_v, loss_avoid, loss_collide, loss_d_acc, loss_d_jerk, loss_ground`

语义：主要在 m、m/s、m/s² 空间度量。

#### I2) 相机损失：`compute_camera_losses(...)`

- 输入：`cam_hist, power/exposure/gain 序列, speed_seq, fill_rate_seq`
- 输出：
	- `loss_cam_smooth`（相机输出平滑）
	- `loss_diff_depth_power`（power 高于 `cam_power_baseline` 的能耗成本）
	- `loss_diff_depth_blur/noise/fill`

特别注意：

- `loss_diff_depth_blur = (speed * exp_phys)^2`，其中 `speed` 是 m/s，`exp_phys` 是内部有效曝光尺度；
- `loss_diff_depth_fill` 使用 fill rate 与阈值差（0~1 空间）。

#### I3) 汇总：`aggregate_loss(...)`

- 输入：physics + camera +（可选 distill）
- 输出：总损失 `loss` 与分量字典

然后执行：

- TBPTT chunk 内 `chunk_loss.backward()` + 分段 `optimizer.step()`；
- 或 full-BPTT 末尾 `loss.backward()` + `optimizer.step()`。

这一步把误差信号反传回：

1. 动作头参数（`fc` / `fc_intent`）
2. 相机头参数（`fc_cam`）
3. 融合与GRU主干
4. 深度预处理前端（通过可微渲染路径可间接影响相机控制策略）

从而在下一 iteration 改变同一闭环的行为。

---

## 3.x 闭环函数清单（按调用链顺序速查）

> 下面这张表用于“逐函数对源码”，与上面的 Step A~I 是一一对应关系。

| 函数 | 主要输入 | 主要输出 | 数值语义（归一化/物理） |
|---|---|---|---|
| `init_camera_params` | `B, device` | `power/exposure/gain` | 输出 `[0,1]` 初值（power 用 `cam_power_baseline`，exposure/gain 用 0.5） |
| `diff_depth_exposure_to_time` | `exposure01` | `exp_phys` | `[0,1]` → 有效曝光时间尺度 |
| `render_sensors` | `env, power, exposure, gain` | `depth_obs, quality` | 输入归一化控制，输出深度(m语义)+质量(0~1) |
| `Env.render_diff_depth` | `power, exposure, gain` | `(noisy_depth, quality)` 或 `(noisy_depth,None)` | 后端分派（python/cuda） |
| `Env._render_diff_depth_python` | 仿真几何+相机控制 | `noisy_depth, quality` | 几何深度 + 传感器退化 |
| `Env._render_diff_depth_cuda` | 同上 | `noisy_depth, None` | CUDA fused 路径，质量图不可用 |
| `Env._apply_diff_depth_sensor_model` | `depth, power, exposure, gain` | `noisy_depth, quality` | 内部把控制量映射为曝光/增益尺度并注入噪声/空洞 |
| `compute_depth_fill_rate` | `depth_like, min_valid, softness` | fill rate 标量 | 输出 `[0,1]` 比例 |
| `Env.find_vec_to_nearest_pt` | `env state` | `vec_to_pt` | 距离向量（m） |
| `build_local_frame` | `env.R` | `R_local` | 局部坐标系（无量纲旋转） |
| `compute_target_velocity` | `target_v_raw` | `target_v` | m/s，限幅到 `max_speed` |
| `build_state_vector` | `env, target_v, R, power/exposure/gain` | `state, local_v` | 混合量纲；相机状态可映射到 `[-1,1]` |
| `Model.preprocess_depth_input` | `depth_obs` | `x_depth` | 深度(m语义)→反深度归一化 `[-1,1]` |
| `Model.forward` | `state, hx, depth_obs` | `act, cam_params, hx(,intent)` | `cam_params` 是 `[0,1]` 归一化绝对值 |
| `update_camera_params` | `cam_params, old power/exposure/gain` | `new power/exposure/gain, hist` | EMA 更新后的执行状态 + 原始网络输出 |
| `decode_action_direct` | `act_raw, R` | `a_final, v_pred` | 输出 `a_final` 为 m/s² 物理命令 |
| `decode_action_lqr` | `intent, R, local_v, ...` | `a_final, v_pred` | 意图域 → dLQR → m/s² 命令 |
| `Env.run` | `act_pred, ctl_dt, v_pred` | 更新 `p,v,a,R,act` | 动力学推进（物理状态更新） |
| `compute_physics_losses` | `v,tv,act,vec,p,...` | 物理损失字典 | 主要基于 m/m/s/m/s² |
| `compute_camera_losses` | `cam_hist,power,exposure,gain,speed,fill` | 相机损失字典 | 归一化控制 + 物理 proxy 混合 |
| `aggregate_loss` | physics + camera (+distill) | `loss, all_losses` | 标量损失用于反传 |

---

## 4. “归一化 vs 物理语义”速查表

| 变量/张量 | 位置 | 形状 | 数值域 | 语义 |
|---|---|---:|---|---|
| `power/exposure/gain` | 相机控制状态 | `(B,)` | `[0,1]` | 归一化绝对控制量（仿真中的相机档位） |
| `cam_params` | `model` 输出 | `(B,3)` | `[0,1]` | 归一化绝对控制量（非增量） |
| `exposure_s` | 传感器模型内部 | `(B,)` | `[eff_min, eff_max]` | 有效曝光时间尺度（内部物理语义） |
| `gain_scale` | 传感器模型内部 | `(B,)` | `>=1` | ISO 风格增益尺度（内部物理语义） |
| `depth_obs` | 渲染输出 | `(B,H,W)` | `0` 或 `[min_valid,max_range]` | 深度（m语义）+ 噪声/空洞 |
| `quality` | 渲染输出 | `(B,H,W)` | `[0,1]` | 像素质量/可观测性 |
| `x_depth` | `model` 输入特征 | `(B,1,h,w)` | `[-1,1]` | 反深度归一化特征（非原始米值） |
| `state` | `model` 输入 | `(B,obs_dim)` | 混合 | 速度/姿态/margin/可选相机状态 |
| `act_raw` | `model` 输出 | `(B,6)` 常见 | 无界 | 动作域原始网络输出 |
| `a_final` | 解码后控制 | `(B,3)` | 限幅到 `[-max_acc_cmd,max_acc_cmd]` | 物理加速度命令（m/s²） |
| `fill_rate` | 统计/损失 | 标量 | `[0,1]` | 有效像素占比 |

---

## 5. Teacher / Student / Eval 三条链路区别

### 5.1 Student（主训练闭环）

- 渲染：`differentiable=True`
- 前向：`model(...)`
- 控制：`update_camera_params + decode_action_* + env.run`
- 学习：physics/camera/distill loss 反传更新网络

### 5.2 Teacher（内循环优化）

- 先用当前模型 rollout 得到初值；
- 再把动作序列/意图序列/相机序列当可优化变量直接优化；
- 得到 `u_star / y_star / u_star_cam` 给 student 蒸馏。

### 5.3 Eval（推理闭环）

- 同样渲染→模型→控制→仿真闭环；
- 但 `no_grad`，不反向、不更新参数；
- 记录成功率、碰撞率、fill rate、camera stats 等指标。

---

## 6. 两个最容易误解的点（务必记住）

1. **`cam_params` 是归一化绝对值，不是增量**。  
	 网络每步直接给“目标档位”，再由 EMA 形成平滑执行值。

2. **网络看到的深度不是原始米值图像**。  
	 进入 `Model` 前会做有效性掩码 + 反深度映射 + 归一化到 `[-1,1]`。

---

## 7. 最终一句话总结

这个系统是一个标准“感知-决策-控制-再感知”闭环：  
**相机控制（归一化）影响渲染质量（物理语义）→ 渲染结果经归一化后进入网络 → 网络输出动作与相机控制 → 动作推进动力学、相机控制改变下一帧观测 → loss 反传更新网络，闭环持续收敛。**

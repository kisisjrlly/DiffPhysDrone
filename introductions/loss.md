> **免责声明**：当前md中的描述并不完全等价项目中的代码实现，真实的实现以代码为准。

# DiffPhysDrone Loss 说明（`full_bptt_losses` 全量详解）

> 对应代码：`trainer.py`、`losses.py`、`rollout_ops.py`、`train_utils.py`  
> 目标：把 `full_bptt_losses` 里所有 loss 的计算方法、含义、聚合方式、梯度流向讲清楚，方便精准调参。

---

## 1. `full_bptt_losses` 在训练中的位置

`train()` 里 student 分支有两种路径：

- **TBPTT**：在 `student_rollout()` 内按 chunk 计算并反传；
- **Full-BPTT**：rollout 结束后调用 `full_bptt_losses()`，一次性计算全时域损失，再 `backward()`。

这份文档聚焦 Full-BPTT。

---

## 2. 输入张量（`full_bptt_losses` 开头）

从 `rollout` 中堆叠出的关键张量：

- `p_history`: `[T, B, 3]`，位置
- `v_history`: `[T, B, 3]`，速度
- `target_v_history`: `[T, B, 3]`，目标速度
- `vec_to_pt_history`: `[T, B, 3]`，到最近障碍点向量
- `v_preds`: `[T, B, 3]`，策略辅助速度预测
- `act_history`: `[T, B, 3]`，执行动作（控制加速度）
- `prev_act_tail`: 上一时刻动作，用于 jerk 连续性

其中 `T = timesteps`，`B = batch_size`。

---

## 3. 计算流程总览

`full_bptt_losses()` 的执行顺序：

1. 计算 `loss_v_pred`
2. 调 `compute_physics_losses(...)` 得到物理组损失
3. 计算 `distance` 与 `speed_history`（日志/统计）
4. 调 `compute_camera_losses(...)` 得到相机组损失
5. （可选）调 `compute_distill_loss(...)`
6. 调 `aggregate_loss(...)` 按系数聚合总损失
7. 返回 `(loss, loss_dict)`

---

## 4. 每个 loss 的方法与含义（逐项）

## 4.1 `loss_v_pred`（辅助速度预测）

代码：

- `loss_v_pred = F.mse_loss(v_preds, v_history.detach())`

公式（等价写法）：

- `L_v_pred = MSE(v_pred, stopgrad(v_true))`

含义：

- 让网络学会“预测当前/下一步速度”的辅助任务；
- 提升表征质量与优化稳定性；
- `detach` 表示该项不会把梯度回到目标端。

---

## 4.2 物理组损失（`compute_physics_losses`）

### 4.2.1 `loss_v`（速度跟踪主损失）

实现思路：

1. 对速度做窗口均值（`win=30`）；
2. 与对齐后的目标速度比较；
3. 使用 SmoothL1。

等价表达：

- `v_avg[i] = mean(v[i : i+win])`
- `L_v = SmoothL1(norm(v_avg - tv_ref), 0)`

含义：

- 强调“时域趋势跟踪”，不是逐帧硬对齐。

---

### 4.2.2 `loss_d_acc`（动作幅值正则）

等价表达：

- `L_d_acc = mean(sum(act^2, dim=-1))`

含义：

- 惩罚动作幅值过大，减少激进控制。

---

### 4.2.3 `loss_d_jerk`（动作变化率正则）

实现：

- 把 `prev_act_tail` 与 `act_history` 拼接，做一阶差分，乘 15。

等价表达：

- `jerk[t] = (act[t] - act[t-1]) * 15`
- `L_d_jerk = mean(sum(jerk^2, dim=-1))`

含义：

- 抑制高频抖动，让控制更平滑。

---

### 4.2.4 `loss_obj_avoidance`（代码中 `loss_avoid`）

核心中间量：

- `dist = norm(vec_to_pt) - margin`

屏障函数：

- `barrier(dist) = relu(1 - dist)^2`（代码还做 `clamp(max=5)` 稳定梯度）

最终：

- `L_avoid = mean(v_to * barrier(dist))`

其中 `v_to` 是基于距离变化构建的权重（当前实现使用 `torch.diff(dist, dim=1)`）。

含义：

- 在“尚未碰撞但接近障碍”阶段给连续惩罚，强迫保留安全余量。

---

### 4.2.5 `loss_collide`

等价表达：

- `L_collide = mean(v_to * softplus(-32 * dist))`

实现细节：

- `dist` 会先 `clamp(min=-3)`，防止数值过激。

含义：

- 对碰撞区/穿透区提供陡峭惩罚，是硬约束近似。

---

### 4.2.6 `loss_ground_affinity`（代码中 `loss_ground`）

等价表达：

- `L_ground = mean(relu(p_z)^2)`

含义：

- 高度偏好项（是否生效取决于 `coef_ground_affinity`）。

---

## 4.3 相机组损失（`compute_camera_losses`）

Full-BPTT 路径传入：

- `cam_hist`: 相机头原始输出历史（power/exposure/gain）
- `power_seq`, `exposure_seq`, `gain_seq`, `speed_seq`
- `fill_rate_seq = depth_fill_soft_history`

### 4.3.1 `loss_cam_smooth`

- `L_cam_smooth = mean((cam_t - cam_t-1)^2)`

含义：相机参数时序平滑。

---

### 4.3.2 `loss_power_reg`

- `L_power_reg = mean((power - 0.5)^2)`

含义：约束功率不过度偏离中心。

---

### 4.3.3 `loss_cam_range`

- `L_cam_range = mean((exposure - 0.5)^2 + (gain - 0.5)^2)`

含义：抑制 exposure / gain 贴边策略。

---

### 4.3.4 `loss_diff_depth_power`

- `L_dd_power = mean(relu(power - 0.5)^2)`

含义：只惩罚高功率，鼓励节能但不强制关灯。

---

### 4.3.5 `loss_diff_depth_blur`

先做曝光语义映射：

- `exp_phys = diff_depth_exposure_to_time(exposure)`

再计算：

- `L_dd_blur = mean((speed * exp_phys)^2)`

含义：惩罚“高速 + 长曝光”导致的拖影风险。

---

### 4.3.6 `loss_diff_depth_noise`

- `L_dd_noise = mean(gain^2)`

含义：抑制高增益噪声放大。

---

### 4.3.7 `loss_diff_depth_fill`

- `fill_gap = relu(min_fill_rate - fill_rate)`
- `L_dd_fill = mean(fill_gap^2)`

含义：当有效像素覆盖不足时惩罚，避免“黑屏/空洞化”。

> 注意：`full_bptt_losses` 使用的是 soft fill 序列（`depth_fill_soft_history`）。

---

## 4.4 蒸馏损失（可选）

仅当 `enable_teacher_student_training=True` 且 teacher 目标存在时启用：

- intent 蒸馏：`MSE(student_intent, teacher_intent)`
- action 蒸馏：`MSE(student_action, teacher_action)`
- camera 蒸馏：`MSE(student_cam, teacher_cam)`

否则 `loss_distill = 0`。

---

## 5. 总损失聚合（`aggregate_loss`）

先做基础加权和：

- `L_base = sum_i (coef_i * loss_i)`

包含：

- `loss_v`
- `loss_obj_avoidance`
- `loss_d_acc`
- `loss_d_jerk`
- `loss_collide`
- `loss_ground_affinity`
- `loss_v_pred`
- `loss_cam_smooth`
- `loss_power_reg`
- `loss_cam_range`
- `loss_tilt`
- `loss_diff_depth_power`
- `loss_diff_depth_blur`
- `loss_diff_depth_noise`
- `loss_diff_depth_fill`

若 teacher-student 打开，再做：

- `L = distill_coef_iter * loss_distill + student_physics_coef * L_base`

---

## 6. 你要的“梯度流向参数表”（核心）

参数组（按模型结构）：

- 视觉编码：`stem`
- 状态融合：`v_proj`, `fuse_gate`, `img_norm`, `v_norm`
- 时序记忆：`gru`, `gru_residual`, `hx_norm`
- 动作头：`fc`
- 相机头：`fc_cam`
- 意图头：`fc_intent`（仅 intent/dmpc 模式）

### 6.1 loss -> 梯度主去向

| loss | 直接依赖 | 主梯度终点 | 说明 |
|---|---|---|---|
| `loss_v` | `v_history`, `target_v_history` | `fc`（direct）或 `fc_intent`（dmpc） + 主干 | 通过动力学链路反传 |
| `loss_obj_avoidance` | `vec_to_pt_history`, `act_history` | 同上 | 安全余量驱动 |
| `loss_collide` | `distance` | 同上 | 碰撞强惩罚 |
| `loss_d_acc` | `act_history` | `fc`/`fc_intent` + 主干 | 限动作幅值 |
| `loss_d_jerk` | `act_history`, `prev_act_tail` | `fc`/`fc_intent` + 主干 | 限动作变化率 |
| `loss_ground_affinity` | `p_history[...,2]` | `fc`/`fc_intent` + 主干 | 系数为 0 时无效 |
| `loss_v_pred` | `v_preds`, `v_history.detach()` | 主要到 `fc` + 主干 | dmpc 路径该项较弱 |
| `loss_cam_smooth` | `cam_hist` | `fc_cam` + 主干 | 相机头直接监督 |
| `loss_power_reg` | `cam_hist[:,:,0]` | `fc_cam` + 主干 | power 通道 |
| `loss_cam_range` | `cam_hist[:,:,1:]` | `fc_cam` + 主干 | exposure/gain 通道 |
| `loss_diff_depth_power` | `power_seq` | `fc_cam` + 主干 | 节能约束 |
| `loss_diff_depth_blur` | `speed_seq`, `exposure_seq` | `fc_cam` + 主干（主） | 曝光-速度耦合 |
| `loss_diff_depth_noise` | `gain_seq` | `fc_cam` + 主干 | 增益噪声约束 |
| `loss_diff_depth_fill` | `fill_rate_seq` | `fc_cam` + 主干（主） | 抑制空洞/黑屏 |
| `loss_tilt` | 默认 0 | 通常无梯度 | 预留项 |
| `loss_distill` | student/teacher 序列 | `fc`/`fc_intent`/`fc_cam` + 主干 | 仅 teacher-student 模式 |

### 6.2 相机头对“避障主损失”的间接梯度

是存在的：

1. `fc_cam` 决定 `power/exposure/gain`；
2. 这些参数影响下一步 `depth_obs/quality`；
3. 深度输入影响后续动作与轨迹；
4. 所以 `loss_v`, `loss_avoid`, `loss_collide` 也会间接推动 `fc_cam`。

因此相机头同时受到：

- 直接监督：`loss_cam_*` + `loss_diff_depth_*`
- 间接监督：物理主损失（通过感知-决策耦合）

---

## 7. 名称/权重/日志键对应（`active_loss_term_specs`）

- `v` -> `loss_v` -> `coef_v`
- `obj_avoidance` -> `loss_obj_avoidance` -> `coef_obj_avoidance`
- `d_acc` -> `loss_d_acc` -> `coef_d_acc`
- `d_jerk` -> `loss_d_jerk` -> `coef_d_jerk`
- `v_pred` -> `loss_v_pred` -> `coef_v_pred`
- `collide` -> `loss_collide` -> `coef_collide`
- `ground_affinity` -> `loss_ground_affinity` -> `coef_ground_affinity`
- `tilt` -> `loss_tilt` -> `coef_tilt`
- `cam_smooth` -> `loss_cam_smooth` -> `coef_cam_smooth`
- `power_reg` -> `loss_power_reg` -> `coef_power_reg`
- `cam_range` -> `loss_cam_range` -> `coef_cam_range`
- `diff_depth_power` -> `loss_diff_depth_power` -> `coef_diff_depth_power`
- `diff_depth_blur` -> `loss_diff_depth_blur` -> `coef_diff_depth_blur`
- `diff_depth_noise` -> `loss_diff_depth_noise` -> `coef_diff_depth_noise`
- `diff_depth_fill` -> `loss_diff_depth_fill` -> `coef_diff_depth_fill`
- `distill` -> `loss_distill` -> `distill_coef_iter`

日志常见键：

- `loss_raw/...`（需 `wandb_log_raw_loss_terms=true`）
- `loss_contrib/...`（加权贡献）
- `loss_share/...`（贡献占比）

---

## 8. 调参指南（按现象定位）

### 8.1 碰撞多

- 先加：`coef_collide`、`coef_obj_avoidance`
- 再查：`coef_v` 是否过大（只追速度）

### 8.2 动作抖

- 先加：`coef_d_jerk`、`coef_d_acc`
- 若相机控制也抖：加 `coef_cam_smooth`

### 8.3 深度黑屏/空洞多

- 先加：`coef_diff_depth_fill`
- 联动看：`diff_depth_min_fill_rate`、`coef_diff_depth_power`

### 8.4 噪声大

- 加：`coef_diff_depth_noise`
- 与 `coef_diff_depth_fill`、`coef_diff_depth_power` 共同平衡

### 8.5 太保守（慢但不撞）

- 适当加 `coef_v`，或略降 `coef_collide` / `coef_obj_avoidance`

---

## 9. 代码语义注意点（务必看）

1. Full-BPTT 相机填充率监督用的是 **soft fill**，不是 hard fill。  
2. `loss_v_pred` 目标侧 `detach`，不会把梯度回到目标端。  
3. `compute_physics_losses` 里 `v_to` 目前基于 `torch.diff(dist, dim=1)`；若你期望“时间差分”，请复核设计意图。  
4. `loss_tilt` 在当前 full-BPTT 默认是 0（占位）。

---

## 10. 一句话总结

`full_bptt_losses` = 轨迹控制目标（跟踪+平滑+安全） + 成像可用性目标（功耗/模糊/噪声/填充） + 可选蒸馏约束。  
它同时决定“飞得好不好”和“看得见不见”。

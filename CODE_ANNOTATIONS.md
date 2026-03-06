# DiffPhysDrone 代码导读（面向第一次阅读）

> 目标：让你在不熟悉项目的情况下，快速理解“从参数 -> 环境 -> 模型 -> rollout -> 损失 -> 更新”的完整链路。

## 1. 项目整体数据流

每轮训练（`main_cuda.py`）大致是：

1. 初始化环境 `Env` 与策略网络 `Model`
2. 重置环境，运行 `timesteps` 步 rollout（渲染传感器、网络推理、物理步进）
3. 记录轨迹，构造多项损失（速度、避障、碰撞、光学等）
4. 反向传播并更新参数

在当前版本中，支持两种时序训练方式：

- **完整 BPTT**：整段 `timesteps` 后统一反传
- **TBPTT**：按 `tbptt_chunk_steps` 分段反传，边界处 `detach`

以及一种混合调度：

- 大多数迭代：`TBPTT`（高吞吐）
- 每隔 `hybrid_full_bptt_every` 轮：完整 `BPTT`（长程校准）

---

## 2. `main_cuda.py` 核心阅读顺序

### 2.1 参数区

重点关注这几组：

- **训练规模**：`--batch_size`, `--timesteps`, `--num_iters`
- **TBPTT/混合**：
  - `--tbptt_enable`
  - `--tbptt_chunk_steps`
  - `--tbptt_chunk_accum`
  - `--hybrid_full_bptt_every`
  - `--hybrid_full_bptt_batch_size`
- **视觉模式**：`--vision_mode in {depth, yuv, yuv_tof}`
- **相机主动感知**：`--paper_unified_control`, `--paper_cam_obs`, `--paper_optical_loss`

### 2.2 环境与模型初始化

- `build_env(batch_size)`：构建环境实例
- `env_train` / `env_full`：分别用于 TBPTT 主训练和低频完整 BPTT 校准
- `Model(...)`：根据 `vision_mode` 构建单分支或双分支编码器

### 2.3 调度器步数估计

`estimate_optimizer_steps()` 会根据 TBPTT chunk 与混合频率估计真实 `optim.step()` 次数，避免 LR 时间轴错位。

### 2.4 主循环

每个 `iter` 先决定本轮模式：

- `use_hybrid_full=True`：完整 BPTT
- 否则：若 `tbptt_enable=True`，则 TBPTT

### 2.5 student rollout + loss

rollout 每步做：

1. 渲染观测（主相机 Y 或 depth + 可选 ToF）
2. 组装状态向量（速度、姿态、margin、可选相机状态）
3. 网络前向输出动作/意图/相机控制
4. 物理步进 `env.run(...)`
5. 缓存轨迹用于损失

损失主要包括：

- `loss_v`：平滑速度跟踪
- `loss_v_pred`：网络速度预测监督（目标速度 detach）
- `loss_obj_avoidance` / `loss_collide`：避障与碰撞
- `loss_d_acc`, `loss_d_jerk`：控制平滑
- 光学项：`loss_blur`, `loss_noise`（若启用）

---

## 3. TBPTT 的三个关键点（你重点关心）

### 3.1 状态衔接 + 边界断图

chunk 边界会做：

- `h = h.detach()`（GRU 隐状态）
- `detach_env_graph(env)`（环境动力学状态）
- 动作缓存/相机状态 detach

这样保留数值连续性，但截断梯度链，显存显著下降。

### 3.2 跨段窗口损失

`loss_v` 依赖 30-step 平滑窗口。TBPTT 下通过 `v_roll/tv_roll` 保留窗口前缀（detached），再与当前 chunk 拼接计算，避免窗口损失在 chunk 边界断裂。

### 3.3 chunk 级优化更新

- 每个 chunk backward 一次
- 每 `tbptt_chunk_accum` 个 chunk 执行一次 `optim.step()`

这是“chunk 级累积梯度”的实现。

---

## 4. `model.py` 结构说明

`Model` 由三部分组成：

1. **视觉编码器**
   - `depth/yuv`：单分支 `stem`
   - `yuv_tof`：`stem_main + stem_tof + fuse`
2. **状态融合**
   - `v_proj(state)` 与视觉特征相加
3. **时序与输出**
   - `GRUCell`
   - 动作头 `fc`
   - 可选意图头 `fc_intent`
   - 可选相机头（统一控制或 diff_cam）

传感器预处理在 `preprocess_sensor_inputs`：

- 深度映射：`3/depth - 0.6`
- 亮度映射到 `[-1,1]`
- ToF 深度与置信度拼接（若启用 `use_tof_conf`）

---

## 5. `env_cuda.py` 结构说明

### 5.1 自动求导封装

- `RunFunction`：可微物理步进（CUDA 前向/反向）
- `DiffRenderFunction` / `DiffRenderYuvYFunction`：可微渲染

### 5.2 `Env` 的关键职责

- 随机场景生成（球、体素、圆柱、墙缝）
- 状态重置 `reset()`
- 传感器渲染：
  - `render()` 深度
  - `render_main_luma()` 主相机亮度
  - `render_tof()` ToF 近似
- 物理步进 `run(...)`
- 最近障碍查询 `find_vec_to_nearest_pt()`
- 状态快照 `save_state/restore_state()`（G-DAC 与 TBPTT 都会用）

---

## 6. `eval_wall_slit.py` 说明

评估脚本与训练保持相同观测和控制路径，但关闭梯度：

- 加载 checkpoint
- 跑 N 个 episode
- 输出通过率、碰撞率、滚转角等统计

这是对 wall-slit 任务最直接的离线评估入口。

---

## 7. 实践建议（首次上手）

1. 先用短跑验证：
   - 小 `num_iters`、小 `batch_size`
2. 再开 TBPTT：
   - `tbptt_chunk_steps=30~40`
3. 再加混合完整 BPTT：
   - `hybrid_full_bptt_every=20~50`
   - `hybrid_full_bptt_batch_size=1~2`

这样可以先保稳定，再慢慢提性能。

---

## 8. 常见疑问速答

- **Q: `timesteps` 是不是每步都更新参数？**
  - 不是。完整 BPTT 是整段后更新；TBPTT 是每个 chunk 更新。

- **Q: TBPTT 是否完全等价完整 BPTT？**
  - 不完全等价。它是工程折中：更省显存、更高吞吐，但有截断偏差。

- **Q: 为什么 `loss_v_pred` 的目标要 `detach()`？**
  - 这是把它当监督目标，避免通过真实速度分支反向污染动力学链路。

---

如果你愿意，我可以下一步继续补一版“按函数逐行讲解”的文档（例如专门拆 `main_cuda.py` 的 rollout 循环每 10 行解释一次）。

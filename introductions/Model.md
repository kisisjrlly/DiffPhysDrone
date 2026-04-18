> **免责声明**：当前md中的描述并不完全等价项目中的代码实现，真实的实现以代码为准。

# DiffPhysDrone `model.py` 结构详解（输入/输出/含义）

> 目标：详细说明当前项目里 `Model` 网络的**结构、输入输出、每个分支的语义**，并解释输出在训练/推理中如何被消费。

---

## 1. 模型定位与职责

`model.py` 中的 `Model(nn.Module)` 是策略网络主干，职责是：

1. 编码可微深度图观测（`depth_obs`）；
2. 编码物理状态向量（`v`）；
3. 融合两种模态 + GRU 时序记忆；
4. 输出：
	- 飞行控制头 `act`（动作域输出）
	- 相机控制头 `cam_params`（Power/Exposure/Gain）
	- 可选意图头 `intent_raw`（用于 dLQR 模式）

这是一个 **diff_depth-only** 设计：`depth_obs` 是必填输入。

---

## 2. 构造参数与运行时常用配置

构造函数：

`Model(dim_obs=9, dim_action=4, include_camera_state_in_obs=False, use_policy_intent=False, intent_dim=9, depth_nn_width=16, depth_nn_height=12, depth_use_pipeline=True, depth_min_valid=0.3, depth_max_range=6.0)`

在本项目实际调用（`main_cuda.py` / `eval.py`）通常是：

- `dim_action = 6`
- `intent_dim = 9`
- `obs_dim = 7 if no_odom else 10`

因此默认训练场景里动作头输出 6 维（3 维加速度 + 3 维速度预测语义）。

---

## 3. 输入是什么？含义是什么？

`forward` 签名：

`forward(v, hx=None, return_intent=False, depth_obs=None, add_noise=False)`

### 3.1 `depth_obs`（必填）

- 语义：可微主动深度相机输出（来自 `render_sensors -> env.render_diff_depth`）
- 形状支持：
  - `(B, H, W)`
  - `(B, 1, H, W)`
- 数值语义：米制深度，`< depth_min_valid` 视为无效深度（空洞）

若传 `None`，会抛异常：`diff_depth-only 模型需要 depth_obs 输入`。

---

### 3.2 `v`（物理状态向量）

- 语义：由 `rollout_ops.build_state_vector(...)` 构建的状态观测
- 形状：`(B, obs_dim_actual)`

组成（按代码顺序）：

1. （可选）`local_v`，3 维（若 `no_odom=False`）
2. `tv_local`，3 维（目标速度在局部坐标系）
3. `env.R[:,2]`，3 维（机体 z 轴方向）
4. `env.margin`，1 维（安全边距）
5. （可选）相机状态 `co`，3 维（Power/Exposure/Gain 从 `[0,1]` 映射到 `[-1,1]`）

因此：

- `no_odom=False` 时，基础 `obs_dim=10`
- `no_odom=True` 时，基础 `obs_dim=7`
- 若 `include_camera_state_in_obs=True`，再 +3 维

---

### 3.3 `hx`（GRU 隐状态）

- 形状：`(B, 192)` 或 `None`
- 语义：时序记忆状态

注意：模型内部 `reset()` 为空，隐藏状态由外部循环维护并传入。

---

### 3.4 `return_intent` 与 `add_noise`

- `return_intent=True`：若模型启用了 `use_policy_intent`，会返回 `intent_raw`
- `add_noise=True`：在深度预处理阶段对归一化深度加 `N(0, 0.01)` 噪声（训练用）

---

## 4. 深度输入预处理（`preprocess_depth_input`）

模型有两种预处理分支：

---

### 4.1 `depth_use_pipeline=True`（默认推荐）

调用 `_depth_pipeline`，步骤：

1. 统一输入形状到 `(B,H,W)`；
2. 构造有效掩码 `valid = d >= depth_min_valid`；
3. 无效像素先替换为 `max_depth`，做反深度归一化：
	- `inv = 1 / d_valid`
	- 归一化到 `[0,1]`
	- 再乘 `valid.float()`，保证无效像素回到 0
4. 若原图小于目标尺寸，先最近邻上采样到至少 `(depth_nn_height, depth_nn_width)`；
5. `adaptive_max_pool2d` 到固定输入尺寸 `(depth_nn_height, depth_nn_width)`；
6. 可选加噪声并 clamp 到 `[0,1]`；
7. 最终线性映射到 `[-1,1]`。

输出形状：`(B,1,depth_nn_height,depth_nn_width)`。

---

### 4.2 `depth_use_pipeline=False`

仍做反深度归一化与无效掩码处理，但不走固定尺寸 pipeline；最后只确保通道维存在：

- 输出形状：`(B,1,H,W)`

由于后续有 `AdaptiveAvgPool2d((3,6))`，网络仍可接收可变分辨率输入。

---

## 5. 网络结构（逐层）

## 5.1 视觉编码分支 `stem`

输入：`(B,1,H,W)`（通常 `H=12, W=16`）

结构：

1. `Conv2d(1 -> 32, k=3, stride=1, pad=1, bias=False)`
2. `LeakyReLU(0.05)`
3. `Conv2d(32 -> 64, k=3, stride=2, pad=1, bias=False)`
4. `LeakyReLU(0.05)`
5. `Conv2d(64 -> 128, k=3, stride=1, pad=1, bias=False)`
6. `LeakyReLU(0.05)`
7. `AdaptiveAvgPool2d((3,6))`
8. `Flatten()`
9. `Linear(128*3*6=2304 -> 192, bias=False)`

输出：`img_feat`，形状 `(B,192)`。

---

## 5.2 状态编码分支 `v_proj`

- `Linear(actual_obs_dim -> 192)`
- 输出：`v_feat`，形状 `(B,192)`

初始化上，`v_proj.weight` 被乘 `0.5`，降低初期状态特征对视觉特征的压制。

---

## 5.3 多模态门控融合 `fuse_gate`

先规范化：

- `img_feat = LayerNorm(192)`
- `v_feat = LayerNorm(192)`

门控网络：

1. 输入拼接：`cat(img_feat, v_feat)` -> `(B,384)`
2. `Linear(384 -> 192)`
3. `LeakyReLU(0.05)`
4. `Linear(192 -> 192)`
5. `Sigmoid()` -> `gate` in `(0,1)`

融合公式：

- `x = LeakyReLU(gate * img_feat + (1-gate) * v_feat)`

含义：动态平衡视觉与状态信息，避免某一模态长期统治。

---

## 5.4 时序记忆 `GRUCell + residual`

1. `hx = GRUCell(input=192, hidden=192)(x, hx)`
2. 残差稳态头：
	- `Linear(192->192, bias=False)`
	- `LeakyReLU`
	- `Linear(192->192, bias=False)`
3. 归一化更新：
	- `hx = LayerNorm(hx + 0.1 * gru_residual(hx))`

含义：

- GRU 提供 POMDP 记忆；
- residual + LN 提升时序稳定性，缓解后期抖动。

---

## 5.5 输出头

统一先过 `act(hx)`（`LeakyReLU`）再分头。

### A) 动作头 `fc`

- `Linear(192 -> dim_action)`
- 输出：`act`（代码里名为 `raw`）

在项目默认配置 `dim_action=6` 时，后续 `decode_action_direct` 会按 `(B,3,2)` 解释：

- 第一列：加速度意图 `a_pred`
- 第二列：速度预测 `v_pred`

### B) 相机头 `fc_cam`

- `Linear(192 -> 3)` 后接 `sigmoid`
- 输出：`cam_params`，范围 `[0,1]`
- 通道语义：`(power, exposure, gain)`

### C) 意图头 `fc_intent`（可选）

- 仅 `use_policy_intent=True` 时创建
- `Linear(192 -> intent_dim)`，默认 `intent_dim=9`
- 输出：`intent_raw`

---

## 6. forward 输出形式（必须明确）

`forward(...)` 有两种返回签名：

1. 若 `return_intent=True` 且 `use_policy_intent=True`：

- 返回 `(act, cam_params, hx, intent_raw)`

2. 其他情况：

- 返回 `(act, cam_params, hx)`

形状（默认配置）：

- `act`: `(B,6)`
- `cam_params`: `(B,3)`，值域 `[0,1]`
- `hx`: `(B,192)`
- `intent_raw`: `(B,9)`（仅意图模式）

---

## 7. 输出在下游中的真实含义

## 7.1 `act` 如何被用

### 直接动作模式（常见）

- 在 `decode_action_direct(...)` 中：
  - `act.reshape(B,3,-1)` 后解包为 `a_pred, v_pred, ...`
  - `a_pred` 经坐标变换、重力补偿、误差标定、限幅得到最终推力命令

### 意图 + dLQR 模式

- 控制主要来自 `intent_raw` 经 `decode_action_lqr(...)`
- `act` 仍可用于日志/蒸馏分支

---

## 7.2 `cam_params` 如何被用

在 `update_camera_params(...)` 中：

- 先 clamp 到 `[0,1]`；
- 再对物理传感器状态做 EMA 更新（`alpha=0.7`）；
- 历史里保留原始 `cam_params`（便于相机损失稳定回传）。

即：模型输出的是**相机绝对控制参数**（不是增量）。

---

## 8. 典型输入输出维度例子

设 `B=64`，`depth_nn_height=12`，`depth_nn_width=16`。

### 情况 A：`no_odom=False` 且 `include_camera_state_in_obs=False`

- `v` 维度：10
- `depth` 输入：`(64,1,12,16)`（pipeline 后）
- 输出：
  - `act`: `(64,6)`
  - `cam_params`: `(64,3)`
  - `hx`: `(64,192)`

### 情况 B：`no_odom=False` 且 `include_camera_state_in_obs=True`

- `v` 维度：13（多 3 维相机状态）
- 其他同上。

### 情况 C：`no_odom=True` 且 `include_camera_state_in_obs=False`

- `v` 维度：7

### 情况 D：`no_odom=True` 且 `include_camera_state_in_obs=True`

- `v` 维度：10

---

## 9. 关键实现细节与注意事项

1. `Model.reset()` 为空：GRU 隐状态由外部维护。  
2. `depth_obs` 不能为空：这是 diff_depth-only 路径。  
3. 动作头默认被下游按 6 维解释；若改 `dim_action`，需同步检查 `decode_action_direct` 的 reshape 语义。  
4. `cam_params` 经过 `sigmoid`，天然在 `[0,1]`，与传感器控制接口语义一致。  
5. 文件里 `from utils import g_decay` 当前未实际使用，属于历史遗留导入，不影响前向。

---

## 10. 总结（一句话）

当前 `Model` 是一个“**深度感知 + 状态融合 + GRU 记忆 + 多头输出**”的策略网络：

- 输入：`depth_obs` + 物理状态 `v` + 可选历史 `hx`；
- 输出：飞行动作 `act`、相机控制 `cam_params`、可选意图 `intent_raw`；
- 通过下游解码与传感器闭环，联合决定“怎么飞”和“怎么看”。

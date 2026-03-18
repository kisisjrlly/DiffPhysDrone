# Naming Guide (参数命名统一说明)

本文只描述当前有效命名，不包含兼容层语义。

## 1) 传感模式 `--sensor_mode`

可选值与语义：

- `depth`：深度渲染输入（不走主动发射模型）
- `camera_luma`：可微亮度相机输入
- `camera_luma_plus_depth`：亮度主视觉 + 深度辅助通道
- `diff_depth`：可微主动深度输入（发射功率/曝光/增益与策略联动）

## 2) 相机动作模式 `--camera_action_mode`

可选值：

- `off`：不输出相机动作
- `absolute`：策略输出相机绝对参数
- `incremental`：策略输出相机增量动作（推荐）

### 控制通道物理含义（严格按 `sensor_mode`）

当前策略网络的传感器控制头固定输出 3 个通道，其物理解释由 `sensor_mode` 决定：

| `sensor_mode` | 通道1 | 通道2 | 通道3 |
|---|---|---|---|
| `camera_luma` / `camera_luma_plus_depth` | FOV | Exposure | ISO |
| `diff_depth` | Power | Exposure | Gain |

说明：
- 在 `diff_depth` 下，代码里历史变量名可能仍叫 `cam_fov/cam_iso`，但其物理语义分别对应 `power/gain`。
- `include_camera_state_in_obs` 拼接的第 1 通道也遵循同样语义：`camera_luma*` 为 `fov_norm`，`diff_depth` 为 `power_norm`。

## 3) 相机状态是否入观测

- `--include_camera_state_in_obs`
- `--no-include_camera_state_in_obs`

启用后会将当前相机参数拼接到状态向量。

## 4) 相机质量损失

- `--enable_camera_quality_loss`

相关权重：

- `--coef_blur`
- `--coef_noise`
- `--coef_cam_smooth`
- `--coef_fov_reg`
- `--coef_cam_range`

## 5) 教师-学生训练

- `--enable_teacher_student_training`
- `--teacher_inner_steps`
- `--teacher_inner_lr`
- `--coef_distill`
- `--student_physics_coef`
- `--teacher_tbptt_chunk_steps`
- `--distill_final_ratio`
- `--student_noise_mode`

## 6) 推荐配置骨架

- 传感与控制：`--sensor_mode camera_luma_plus_depth --camera_action_mode incremental --include_camera_state_in_obs`
- 损失：`--enable_camera_quality_loss` + 对应系数
- 训练：按需开启 `--enable_teacher_student_training`

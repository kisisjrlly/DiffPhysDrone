> **免责声明**：当前md中的描述并不完全等价项目中的代码实现，真实的实现以代码为准。

# diff_depth Design Notes

## Branch Goal

本分支只围绕 `diff_depth` 维护一条主链：

- 可微深度传感
- `power / exposure / gain` 主动调参
- 飞行控制与感知联合训练

## Policy Input

- 视觉输入：单通道深度
- 状态输入：局部目标速度、姿态上方向、安全 margin
- 可选附加：当前 `power / exposure / gain`

## Policy Output

- 飞行动作头：加速度命令与速度预测
- 传感器头：`power / exposure / gain`
- 可选意图头：保留给后续 dMPC 研究，但不属于当前主线

## Sensor Model

[env_cuda.py](/home/zhaoguodong/work/code/DiffPhysDrone/env_cuda.py) 中的 `render_diff_depth()` 当前是主入口：

1. `diff_render()` 生成几何深度
2. D455-inspired `diff_depth` 传感器模型注入：
   `power`
   `exposure`
   `gain`
   ambient washout
   passive fallback
   flying pixels
   motion blur
   specular failure

## Main Loss Terms

- 物理控制：
  `loss_v`
  `loss_obj_avoidance`
  `loss_collide`
  `loss_d_acc`
  `loss_d_jerk`
- 传感器控制：
  `loss_cam_smooth`
  `loss_diff_depth_power`
  `loss_diff_depth_blur`
  `loss_diff_depth_noise`

## Logging Focus

- 任务结果：
  `success_rate`
  `collision_rate`
  `time_to_goal`
- 深度质量：
  `diff_depth_fill_rate`
  `diff_depth_hole_rate`
- 主动感知行为：
  `speed_exposure_corr`
  `power_obstacle_corr`

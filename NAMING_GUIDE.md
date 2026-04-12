# diff_depth Naming Guide

当前分支只保留 `diff_depth` 语义。

## Sensor Mode

- 唯一有效值：`diff_depth`

## Control Channels

策略相机头固定输出 3 个通道：

1. `power`
2. `exposure`
3. `gain`

## Historical Names

当前主干代码里的运行时变量和损失名已经统一到 `power / exposure / gain`。

## Observation State

如果启用 `--include_camera_state_in_obs`，拼入状态向量的 3 个量依次为：

1. `power_norm`
2. `exposure_norm`
3. `gain_norm`

## Loss Names

- `loss_diff_depth_power`
- `loss_diff_depth_blur`
- `loss_diff_depth_noise`

这三个是当前分支唯一保留的传感器质量损失。

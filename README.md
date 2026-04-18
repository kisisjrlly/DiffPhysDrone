> **免责声明**：当前md中的描述并不完全等价项目中的代码实现，真实的实现以代码为准。

# DiffPhysDrone diff_depth-only

本分支已经收敛为 `diff_depth` 的单主线版本。

## Scope

- 仅支持 `diff_depth`
- 传感器控制语义固定为 `power / exposure / gain`
- 策略输入固定为单通道深度

## Current Pipeline

1. `config.py`
   只保留 `diff_depth` 主线参数
2. `env_cuda.py`
   使用 `render_diff_depth(power, exposure, gain)` 生成可微深度
3. `model.py`
   只接收 `depth_obs`
4. `rollout_ops.py`
   只维护 `power / exposure / gain` 更新和 `diff_depth` 渲染
5. `trainer.py`
   只统计 `diff_depth` 相关损失与指标
6. `eval.py`
   只走 `diff_depth` 推理链路

## Main Losses

- `loss_v`
- `loss_obj_avoidance`
- `loss_collide`
- `loss_d_acc`
- `loss_d_jerk`
- `loss_cam_smooth`
- `loss_power_reg`
- `loss_cam_range`
- `loss_diff_depth_power`
- `loss_diff_depth_blur`
- `loss_diff_depth_noise`

## Main Metrics

- `diff_depth_fill_rate`
- `diff_depth_hole_rate`
- `speed_exposure_corr`
- `power_obstacle_corr`

## Running

训练：

```bash
bash run.sh
```

评估：

```bash
bash eval.sh
```

默认主配置：

- [configs/paper_final_full.args](/home/zhaoguodong/work/code/DiffPhysDrone/configs/paper_final_full.args)

默认不再隐式叠加 `CAM_PROFILE`。如需叠加，请显式传入：

```bash
CAM_PROFILE=low bash run.sh
```

## Notes

- 当前论文主线建议使用 `diff_sensor_impl diff_depth=python`
- `diff_depth=cuda` 仍可用于对照与梯度检查，但论文主结果默认使用 `python`
- 本分支的目标是围绕 `diff_depth` 论文主线继续精简与强化
- 当前版本render_depth是不可微的，如果你想做的是：动作改变位姿
位姿改变下一帧看到的东西
再把“未来看见什么”的梯度反传回动作
那 render_depth 就必须至少对 pos / R 近似可微，否则这条“通过改变视角改善感知”的梯度链会断掉。

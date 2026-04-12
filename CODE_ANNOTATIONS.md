# diff_depth-only Code Map

## Main Files

- [config.py](/home/zhaoguodong/work/code/DiffPhysDrone/config.py)
  只保留 `diff_depth` 主线参数
- [main_cuda.py](/home/zhaoguodong/work/code/DiffPhysDrone/main_cuda.py)
  训练入口
- [eval.py](/home/zhaoguodong/work/code/DiffPhysDrone/eval.py)
  评估入口
- [model.py](/home/zhaoguodong/work/code/DiffPhysDrone/model.py)
  单深度分支策略网络
- [rollout_ops.py](/home/zhaoguodong/work/code/DiffPhysDrone/rollout_ops.py)
  `power / exposure / gain` 更新、状态构造、动作解码
- [trainer.py](/home/zhaoguodong/work/code/DiffPhysDrone/trainer.py)
  rollout、TBPTT、Full-BPTT、日志
- [losses.py](/home/zhaoguodong/work/code/DiffPhysDrone/losses.py)
  统一损失定义
- [train_utils.py](/home/zhaoguodong/work/code/DiffPhysDrone/train_utils.py)
  WandB 过滤、调度辅助、环境构造
- [env_cuda.py](/home/zhaoguodong/work/code/DiffPhysDrone/env_cuda.py)
  物理环境与 `diff_depth` 传感器仿真
- [autograd_ops.py](/home/zhaoguodong/work/code/DiffPhysDrone/autograd_ops.py)
  CUDA autograd 封装，重点看 `diff_render` 与 `diff_render_diff_depth`
- [src/quadsim.cpp](/home/zhaoguodong/work/code/DiffPhysDrone/src/quadsim.cpp)
  C++ 绑定入口
- [src/quadsim_kernel.cu](/home/zhaoguodong/work/code/DiffPhysDrone/src/quadsim_kernel.cu)
  CUDA kernel 实现

## Runtime Dataflow

1. `main_cuda.py` 解析参数并创建 `Env` 与 `Model`
2. `trainer.py` 每步调用 `rollout_ops.render_sensors()`
3. `env_cuda.py` 通过 `render_diff_depth(power, exposure, gain)` 生成深度图
4. `model.py` 消费 `depth_obs + state`
5. `rollout_ops.update_camera_params()` 更新 `power / exposure / gain`
6. `trainer.py` 聚合物理损失与 `diff_depth` 光学损失
7. `train_utils.py` 只记录当前有效的 `diff_depth` loss 与指标

## Key Semantics

- 运行时控制通道统一为 `power / exposure / gain`
- 正则项统一命名为 `loss_power_reg`

# 真实灰度网络前置检查 — 2026-11-03

**结论：固定灰度飞行网络未通过前置检查，按协议停止；尚不能检验拍摄与运动是否有额外联合收益。**

## 本轮问题

先确认同一个实际神经网络在 joint_slit 中能否使用图像完成飞行，再决定是否运行拍摄/运动干预及学习信号检查。此前模板读图器的 8/8 是带真实姿态、光照、相机状态先验的诊断，不能替代本轮网络结果。

协议见 [固定执行协议](POLICY_SENSING_DECISION_PROTOCOL_2026_11_03.md)。原始 checkpoint 严格载入既有 legacy 架构，1,640,714 个参数；未转换成 intent_shared。原文件只读，SHA256 为 `748d5e11fcf4ed66af462e80c2787d7aa798ae25e37227aa31b955fcdb621b7a`。

开发只评分历史编号 1/2/3/12，每个布局分别 dark / bright_to_dark，共 8 次。完整保留历史 32 元素初态与逐控制步扰动；图像噪声由原布局编号及帧编号确定，额外渲染不会改变动力学随机序列。它们来自历史参考跟踪成功布局，**不是新独立测试样本**。

保持现有 joint_slit、15Hz/6秒、96×72→48×36、两步动作队列、执行器、积分/噪声/延迟、原预处理。固定 E=.10/G=0，与这个固定相机 checkpoint 的设置一致。`calibrated: false` 不变。

## 结果

| 模型 | 正常灰度图安全到达 | 同模型零图像安全到达 |
| --- | ---: | ---: |
| 冻结原 checkpoint | 0/8 | 0/8 |
| 固定相机适配 final600 | 0/8 | 0/8 |

正常图像至少 4/8 且领先零图像至少 2 次的门槛未通过。原模型 8 次均在全机身过墙前碰撞；适配后正常图像为 4 次碰撞、4 次超时，零图像也是 4 次碰撞、4 次超时。避免部分碰撞并未转化为 6 秒内安全到达。适配的 600 个训练批次也均未成功；这不是联合学习结果。

600 次有效更新全部完成，损失与梯度均有限，飞行权重 hash 从 `5240ca588cd523088bab0a17d7cde7224a37c45118f4869b3f693ecf1a8de161` 变为 `d7d55c185d497e133af1d8d9ea98453cc1e6591f1d1ffe255c0ac459417aa656`。相机参数 hash 始终为 `49802754d337658b87ec7b1dee0c11a622dac41bba115b55ebdf753120871376`。说明更新实际发生、冻结也有效，但不代表学会了任务。有效适配模型 SHA256 为 `9ec16d9a2995d82b23c8cee78aef2aa1bcba0cdfa84fc2823b0cae94ad6b358d`。

前 50 / 后 50 批次平均导航损失为 3.463 / 3.822；均值涉及不同训练样本，不能据此单独判断收敛。**已知的是这个固定训练预算和配置没有恢复成功飞行；未知的是具体由训练预算、旧反传/损失，还是场景与成像迁移中的哪个因素造成。** 当前 joint_slit 相比历史成功链路有起点、成像和时域积分等变化，这次没有逐项隔离它们。

拍摄/运动分叉、复核候选锁定、经过图像的运动作用诊断以及有限差分梯度检查，**全部未启动**。不能用随机状态下的非零梯度替代这些检查，也不能把未找到收益说成“联合方法无效”。此前特权模板 8/8 与当前神经网络 0/8 说明两种诊断不能互相替代；不证明相机图像完全没有信息。

**下一项唯一优先事项：定位并恢复当前窄缝中可靠的固定灰度神经网络飞行基线。** 应以历史成功链路为参照，先查清迁移失败发生在哪里，再决定学习方式；本轮不继续训练，不做联合优势或算法比较。

[结论图](policy_sensing_decision_20261103/comparison.png) · [逐 episode、时间线与哈希](policy_sensing_decision_20261103/summary.json)

测试 **179 passed**，仅既有 PyTorch TypedStorage 弃用提示。新增检查覆盖 checkpoint 严格载入、相机冻结、原损失及唯一训练预算、完整保存点恢复、相机命令只能影响后续帧、动作队列与限幅、隐藏缝位置没有直接进入网络，以及延迟队列深拷贝。当前没有声称后续尚未运行的干预/梯度阶段通过验收。

## 适配与审计

只进行固定相机飞行适配，没有相机策略或联合策略训练。原 checkpoint 初始化、600 更新、batch=8、lr=1e-5、seed=42、四类光照各两条、AMP 关闭；legacy 全段 BPTT、grad_decay=.3、图像位姿梯度关闭。沿用旧损失权重：速度 10.5、碰撞 6、额外避障 0、动作变化 .05、jerk .05、相机平滑 0、额外终点损失 0。非有限损失或梯度会立即中止，不跳过后续继续训练。只评估 final600，不选择中间最优 checkpoint。

**实施偏差：**最初启动时遗漏显式设置 `coef_obj_avoidance=0`，错误继承默认 .5；发现后在完成 46 次更新时停止并作废。该运行未产生选用模型，没有被混入有效结果。更正参数并增加训练前逐项核对后，从原始 checkpoint 开始一次有效 600 更新。因此总计算量包含额外作废的 46 步，不能称总共只花了 600 步。原错误参数、指标、停止记录全部保留。

## 可复现命令

在独立工作树 `/home/zhaoguodong/.codex/worktrees/imx900-intent-joint/DiffPhysDrone` 中运行；新目录可以避免覆盖已完成结果。解释器为 `/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python`。

```bash
/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python tools/audit_policy_sensing_decision.py \
  --out_dir logs/policy_sensing_decision_new --stage baseline
# 只有 baseline 判为 needs_flight_baseline，才允许下面这一有效适配。
/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python tools/audit_policy_sensing_decision.py \
  --out_dir logs/policy_sensing_decision_new --stage adapt
/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python tools/audit_policy_sensing_decision.py \
  --out_dir logs/policy_sensing_decision_new --stage evaluate_adapted
/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python tools/summarize_policy_sensing_decision.py \
  --run_dir logs/policy_sensing_decision_new \
  --destination /tmp/policy_sensing_decision_summary
```

适配目录一旦存在，工具拒绝重新启动该训练。已完成评估只能在 checkpoint 和源码校验匹配时复用。原始评估之后仅更正了训练配置和训练前检查；其旧源码 hash 与兼容说明保留，未重选样本或重新挑结果。

完整运行目录及源快照保留在 `logs/policy_sensing_decision_20261103/`；原运行路径为 `outputs/policy_sensing_decision_20261103/`，全部计算结束后才归档，原始参数中的路径未篡改，映射见 archive_manifest.json。交付包含精简逐 episode、压缩全时间线、有效/作废训练参数及指标、校准快照、源码/模型 hash。checkpoint 留在本地，不把新适配 checkpoint 误作冻结主实验或真实飞行版本。

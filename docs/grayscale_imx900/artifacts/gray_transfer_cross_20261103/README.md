# 交叉检查与球形碰撞证据

本目录配套 [中文结论](../../GRAY_TRANSFER_CROSS_2026_11_03.md) 与 [事先锁定计划](../../GRAY_TRANSFER_CROSS_PLAN_2026_11_03.md)。两组新增闭环条件共32次，不训练、不增加布局；48个固定状态图像检查不计为飞行成功结果。

## 复现

在研究工作树 `codex/imx900-intent-closed-loop` 中运行：

```bash
PYTHON=/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python
$PYTHON tools/audit_gray_transfer_cross.py --stage cross
$PYTHON tools/audit_gray_transfer_cross.py --stage offline
$PYTHON tools/audit_gray_transfer_cross.py --stage summary
$PYTHON -m pytest -q
```

默认输出为 `logs/gray_transfer_cross_20261103`；也可用 `--out_dir` 指定。`--stage all` 依次执行上述三阶段。闭环结果按既有数值worker校验配置、依赖和checkpoint后恢复，不重新运行已完成episode；离线阶段每次复算固定48个快照。需要前一轮完整逐步记录 `logs/gray_chain_regression_20261103`，缺失或hash不匹配即停止，不静默换样本或重算旧模型矩阵。

## 固定配置与来源

模型A：`/home/zhaoguodong/work/code/DiffPhysDrone/checkpoint/2026-09-24-11-35-14/final.pth`，SHA256 `87875d118803503ab66bab855f07f542de4fc8460d4c2918fb38f27c6d344168`，严格legacy加载。calibration为同一provisional快照，仍未实测。

四组均为旧起点−1.5m、dt固定1/15s、90步、原始96×72/网络48×36、AMP开启、平均亮度自动曝光、初始E=.10/G=0，两步飞行队列和原相机更新流程。四类光照按历史CSV顺序循环；seed30000–30015，不按成功筛选。

- OO：复用上轮M02；旧场景/成像、旧运动算法，8/16。
- WO：本轮新增；新场景/成像、旧运动算法，3/16。
- OD：本轮新增；旧场景/成像、新动力学/碰撞，6/16。
- WD：复用上轮reverse_long_approach；新场景/成像、新运动算法，1/16。

场景组包含过渡位置、saturation、走廊几何、射线求交、表面光照、曝光积分；运动组包含动力学/姿态和碰撞算法。新增几何对渲染和碰撞同时生效，所以不能把WO称为纯粹换照片。原有Components、NetworkRun及生产源码均未修改。原源码、模型和旧记录均已按上轮manifest/inventory核对。

`matching.json` 核对共同初态、扰动、布局和共同步的传感器/动力学随机状态，全部一致；也记录首次图像、动作和位置差异。动作比较采用AMP容差1e-3，位置FP32采用atol1e-6/rtol1e-5。OD共同段最大位置差约0.352mm；这不涵盖其终止之后未执行的运动。

## 固定状态读图检查

16条OO轨迹各取首帧、首次x≥−1m、首次x≥−0.5m，共48帧；未增加飞行轨迹。前后两帧均按对应历史位姿/速度/角速度和当时E/G生成，复用原逐步种子。网络状态向量、循环记忆保持原记录值。原始OO帧、预处理输入和动作首先与存档逐一匹配；全部通过后才解释WO图像的动作响应。每次forward只预处理一次，sensor noise仍只在积分后施加一次。

固定E/G意味着这里没有重新让AE调整；它隔离的是当下观测变化的直接动作效应，不是新场景AE闭环结果。不同状态的平均最大动作分量变化：首帧0.0566m/s²，−1m处0.1372m/s²，−0.5m处0.0783m/s²。不能据此断言某一场景表面导致所有失败，也不能证明新策略学不会。

图中示例按episode顺序取第一个OO成功、WO失败的30003；对两幅传感器图使用相同的显示gamma1/2.2，以展示暗部。保存的NPZ、误差计算和网络输入均保持原值，没有对训练/评估图像做gamma修改。

## 碰撞核对及界限

当前 `swept_box_collision` 将障碍物沿每轴扩张半径，以包围球的箱体作保守扫掠。`box_clearance` 则是球形距离。墙角处两者不等价；代码原有注释已明确保守近似，本轮将这一差异量化，不掩盖其设计来源。

只对增加运动组后相对其参考组新增失败的四例核对终止段：OD相对OO的30005/30015，WD相对WO的30001/30008；没有按间隙大小挑选例子。球形半径r=.12m，间隙阈值.0011m。

球形机体到箱体的有符号间隙是1-Lipschitz函数。线段17个等距样本的最小间隙为d，长度为L，每个连续位置到最近样本不超过L/32，故连续最小间隙≥d−L/32−1e-6m（末项是数值保护）。使用CPU FP64计算，下界分别11.396/10.234/9.301/26.777mm，均大于1.1mm。没有改终止算法或阈值。

参考完整轨迹中三例也被该下界证实清晰避开阈值；WO的30001完整路径下界不足以作此保证，故保留未决，不把“下界不足”解释为实际碰撞。这一额外信息不影响四个已终止运动段的证书。没有模拟停止之后的轨迹，所以不宣布任何修正后的成功数。

下一项优先处理连续碰撞几何一致性，再复评同一批样本；世界成像变化造成的5例损失仍保留，不能把全部问题归咎于碰撞近似。

## 文件

- `protocol.json` / `cell_inputs.json`：四组锁定定义、全配置、源码/模型/扩展hash及原CSV行；只授权WO/OD各16次新执行。
- `cross_results.json` / `episodes.csv` / `paired.csv`：64行四组结果（其中32行复用）、配对损失与恢复编号。`priority=world_first` 是最初只按成功损失的排序；后续几何证据使最终优先级更新为 `decision.json` 的 `collision_geometry_first`，不覆盖原始结果。
- `matching.json`：初态/随机对齐及首次差异；`offline_results.json`：48次同状态检查、原始回放容差与动作响应；`terminal_geometry.json`：四例连续间隙证书。
- 三个 `30003_*.npz`：示例的旧/新原始两帧、网络输入与动作；`WO_30003_trace.npz` / `OD_30005_trace.npz`：本轮两组的代表性完整逐步记录。
- 完整32次新轨迹、48次离线图像位于工作树 `logs/gray_transfer_cross_20261103`。`trace_inventory.json` 给出全部80个原始NPZ hash；旧组trace hash在前轮交付中。
- `tool_versions.json`：开发过程中追加分析功能的工具版本；数值执行均复用未修改的前轮worker，按其实际源码与依赖校验，未用文件改名冒充相同版本。
- `delivery_manifest.json` / `tests.json`：交付源码与环境、模型/calibration校验、测试记录。

本轮唯一实现错误是只读证书工具在Env.reset前取drone_radius，调用失败后补上reset；未影响已执行的32次闭环条件，没有丢弃或重选episode。

交付验证：完整pytest **198 passed**，6条原有告警（TypedStorage、CUDA median），7.84秒。新增8项测试通过；生产算法本轮没有更改。

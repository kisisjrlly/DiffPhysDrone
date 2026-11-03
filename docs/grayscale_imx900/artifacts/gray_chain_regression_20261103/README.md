# 灰度链路回归：完整证据与复现说明

**结论：旧实验能完整复现，当前执行程序在旧条件下也一致。已证实把起点移远会使这套旧策略失败；但它不能独自解释完整新场景的失败。本轮决定属于“多项迁移共同影响，限定检查尚未完全隔离”。**

## 已证实什么

| 使用各自历史条件 | 历史记录 | 旧源码重算 | 当前标准评估 | 当前神经网络诊断 |
|---|---:|---:|---:|---:|
| N：常规光照模型，固定 E=.10/G=0，正常图像 | 69/100 | 69/100 | 69/100 | 69/100 |
| 同一 N 模型，网络图像置零 | 1/100 | 1/100 | 1/100 | 1/100 |
| A：混合光照模型，平均亮度自动曝光 | 66/100 | 66/100 | 66/100 | 66/100 |

这不是仅核对总数：900 次执行均保存逐步记录；各 runner 的逐 episode 成功、碰撞、停止步与原因一致；初态、图像、网络输入/记忆/动作、E/G、执行命令、随机状态及后续运动在规定容差内一致。N 关闭 AMP，A 开启 AMP。旧源码按存档 manifest 校验，两个 checkpoint 严格加载且 SHA256 未变。没有训练或使用此前的适配模型。

随后固定 A＋自动曝光，用历史 `30000–30015`（每类光照四次）依次迁移：

| 步骤 | 本步新增变化 | 成功/16 |
|---|---|---:|
| M0 | 完整旧条件 | 9 |
| M1 | 取消控制周期抖动 | 8 |
| M2 | 最多 60→90 步 | 8 |
| M3 | 起点 −1.5→−2.5 m | **0** |
| M4 | 光照过渡位置 0→−.65 m | 0 |
| M5 | soft→hard forward saturation | 0 |
| M6 | 加入走廊侧壁、地面、顶面 | 0 |
| M7 | CUDA→箱体射线求交 | 0 |
| M8 | CUDA→当前研究场景动力学/姿态 | 0 |
| M9 | 原碰撞→扫掠碰撞 | 0 |
| M10 | 观察者全局光照→世界表面光照 | 0 |
| M11 | 盒式模糊→九次曝光积分 | 0 |

M2→M3 的八个成功全部变成碰撞，**不是飞得更远导致超时**。它证明的是：在其余条件固定时，这套旧策略对接近过程的变化不稳健；尚未证明具体是循环记忆、图像尺度还是动作反馈导致。M0 与独立旧条件诊断一致；M11 与未经适配层修改的生产 `joint_slit` 标准评估逐步一致。各阶段基础布局、初始扰动及共同步的图像/动力学随机状态均核对通过。

## 尚未定位什么

按预先规定的排名，锁定“起点”和“周期抖动”两个因素，仅做三项回退：

| 完整新链路的回退 | 成功/16 |
|---|---:|
| 只恢复旧起点 | 1 |
| 只恢复旧周期抖动 | 0 |
| 两者共同恢复 | 1 |

所以“把起点改回去就好了”是错误结论。保留旧起点、零抖动时，M2 为 8/16，而其余 M4–M11 全部引入后仅 1/16；**这组剩余迁移仍造成损失，尚未分解**。M3 以后已全部失败，后续各步的 0/16 不能说明那些变化无影响。新增走廊的动作差异最大，但没有做其回退，不能指定它为根因。

连接最新诊断的八个固定历史条件（编号 1/2/3/12，各 dark、bright_to_dark），A 模型的“自动曝光＋正常图像”“自动曝光＋网络零图像”“固定低曝光＋正常图像”均 **0/8，全部碰撞**。因此，合适的历史模型加自动曝光也未恢复当前八例；不能再把此前 0/8 全部解释为选错模型或低曝光。自动曝光仍读取真实帧，只有飞行网络被置零。这八例沿用原 32 元素历史批次，并非新的独立样本。

## 下一项唯一工作

**在旧起点下，隔离 M4–M11 这组剩余迁移为什么使灰度飞行能力下降。** 本轮不继续搜索组合，不选新训练算法，不改网络/控制结构。待该差异清楚后，再决定是否需要适配，以及如何检查拍摄与运动的配合。

本轮结果既不证明也不否定可微联合学习；确认的是历史灰度视觉飞行能力及其迁移问题。已知不合适的固定曝光失败，不是研究相机控制的准入门槛。`calibrated:false` 保持不变，无真机结论。

## 复现命令

在 `codex/imx900-intent-closed-loop` 工作树运行，使用既定 CUDA Python。主项目中的 checkpoint 与 logs 只读。四阶段可分别恢复；源码、配置或模型校验不匹配会拒绝旧缓存。

```bash
PYTHON=/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python
$PYTHON tools/audit_gray_chain_regression.py --stage historical --out_dir logs/gray_chain_regression_20261103
$PYTHON tools/audit_gray_chain_regression.py --stage parity --out_dir logs/gray_chain_regression_20261103
$PYTHON tools/audit_gray_chain_regression.py --stage migration --out_dir logs/gray_chain_regression_20261103
$PYTHON tools/audit_gray_chain_regression.py --stage summary --out_dir logs/gray_chain_regression_20261103
$PYTHON -m pytest -q
```

`historical` 每个 arm 在独立解释器中加载归档源码；`parity` 分别调用当前标准评估和可注入配置的 NetworkRun。`migration` 用只在诊断工具内存在的 Components 组合已有子系统，不使用场景名称一次切换所有物理行为。每个迁移阶段缓存独立，M0/M11 验证失败即停止归因。最后八例的三个 arm 不作候选选择。

### 配置、来源和数值规则

- N：`2026-09-24-10-33-44/final.pth`，SHA256 `748d5e11fcf4ed66af462e80c2787d7aa798ae25e37227aa31b955fcdb621b7a`；N 的配置来自原 evaluation manifest。
- A：`2026-09-24-11-35-14/final.pth`，SHA256 `87875d118803503ab66bab855f07f542de4fc8460d4c2918fb38f27c6d344168`；配置来自对应 args.json 与原 evaluation.log 的命令，AMP 保留开启。
- 来源源码为 `logs/imx900_formal_20260924_103343/source`，已按其父 manifest 的 source_sha256 全量校验；没有仅靠 Git revision 代替历史未提交代码。
- 旧起点 −1.5m、60步、15Hz，dt 抖动标准差 .1/15s；N 固定曝光，A 平均亮度 AE；旧 soft saturation、全局光照、盒式模糊与两步飞行队列保持。
- episode seed 决定初态；每步 image/network phase seed `(episode_seed*1000003+3*step+1) mod 2^32`，dynamics phase 使用 `+2`。没有通过额外渲染或采样记录数据。
- FP32 连续值 atol=1e-6/rtol=1e-5；AMP 网络中间值 atol=rtol=1e-3；离散终止结果与随机状态精确一致。CUDA median 有确定性告警，逐步比较实际通过，告警不隐藏。
- 使用同一已安装 `quadsim_cuda` 扩展，路径/二进制 hash 在 protocol.json；新旧 CUDA 源码三文件 hash 一致。旧结果在该扩展下复现，不声称它就是历史机器当时的原二进制。
- `runtime` 文件记录实际模块路径、完整配置与 TF32 状态。确定性函数开启 cudnn deterministic、关闭 benchmark、设置 CUBLAS_WORKSPACE_CONFIG=:4096:8，并启用 warn-only deterministic algorithms；函数源码在已校验快照中。运行环境为 PyTorch 2.1.0+cu121 / RTX4080，完整版本见 delivery_manifest.json。

### 文件索引

- `episodes.csv`：三个 runner 的 900 行、12 个累积迁移的192行、生产一致性16行、三项回退48行，共1156行；最后八例×三 arm 的24行在 `endpoint_results.json`。没有新增历史布局或训练 seed。
- `protocol.json`、`*_runtime.json`：初始协议和实际配置；`source_versions.json` 保存诊断开发期间的源码版本变化；`tool_source_at_historical.py` 保留最初历史复算时的工具。后续 worker 的数值代码经 AST 校验相同，新增迁移/汇总功能不冒充原始工具版本。
- `historical_comparison.json`：旧 CSV 对照；`parity_comparison.json`：全部300次旧条件的双 runner 逐步比较，无差异。
- `migration_validation.json`：M0 与独立旧条件、M11 与生产评估的逐步一致性。
- `migration_inputs.json`：各组合完整配置、源文件 hash、worker 源码与冻结模型 hash；`migration_results.json`：全部逐 episode 结果；`reverse_lock.json`：回退前锁定的排序和三个检查。
- `migration_first_differences.json`：每个相邻迁移的首次差异（包括预期的初态/几何差异）；不能把有意改变因素的首次差异写成代码 bug。
- `matched_random_process.json`：全部迁移基础布局/扰动与逐步 RNG 对齐；`submission_queue_validation.json`：从所存 raw_action 和 pre_R 解码世界坐标提交命令，与两步后执行命令一致。
- 四个 `*_representative.npz`：第一个 M2 成功、M3 失败的旧 episode 30001；含图像、输入、记忆、动作、命令、dt、E/G、状态与随机状态。导出的 submitted_command 是从记录解码的派生字段，未额外调用策略或仿真。
- 其余完整逐步 NPZ、endpoint 的32元素逐步时间线保存在工作树 `logs/gray_chain_regression_20261103`（约2.3GB，不全部提交）；`trace_inventory.json` 保存原始1156个 NPZ 的 SHA256。代表性文件在导出时补充提交命令，与原始文件 hash 有所不同。
- `delivery_manifest.json`：最终诊断源码、模型 args、calibration 和历史输入 hash、版本和结论；`chain_regression.png/.pdf`：静态对照图。

原始灰度帧只加一次 shot/read noise，网络预处理一次；九次曝光积分不重复加噪。模型图像置零不影响 AE 的真实图像输入。新增回归测试检查上述规则、记录不改轨迹、两个命令延迟、M0/M11 精确前向一致、AMP/配置来源及人工注入图像/队列/dt 差异定位。

### 运行过程记录与限制

逐步比对最初发生一次压缩读取 CRC 异常；对全部 NPZ 的 ZIP 校验未发现损坏。改为每个文件一次解压后，全量比对通过。随后汇总器遇到不同几何数组形状未预先处理的问题，已补形状检查与测试。这两项是记录分析工具问题，没有改模型、物理、碰撞阈值或执行条件，也未丢弃失败 episode。

本轮未训练、未调参数、未扩大样本、未运行完整 OOD 或新视频。先前 nominal 模型固定低曝光在暗光的失败仍保留，但不能当作整个研究的停止门槛。当前 A+AE 仍失败，说明替换合适的历史模型本身也不足以解决迁移。此处不把新观测认定为“合理可用”，也不据此宣称可微感知无效。

交付检查：完整 pytest **190 passed**（6条已披露告警，8.12s），git diff --check 通过。

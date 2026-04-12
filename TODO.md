# diff_depth 代码待办

## 范围

- 论文 v1 仅覆盖 `diff_depth` 主线。
- 保持默认动作域控制，不开启 `--policy_output_intent`。
- 暂时保持 `--use_dmpc` 关闭。
- 暂时保持 `--policy_output_intent` 关闭。
- 暂时保持 `--tbptt_enable` 关闭。

## P0 阻塞项

| 文件 | 待办 | 为什么重要 |
| --- | --- | --- |
| `trainer.py` | `[x]` 将 `loss_diff_depth_noise` 接入 Full-BPTT 和 TBPTT 两条路径。 | 配置里已经暴露了 `--coef_diff_depth_noise`，主训练循环现已真正优化它。 |
| `trainer.py` | `[x]` 将 `loss_diff_depth_blur` 与 `losses.py` 统一，并使用 `(speed * exp_t)^2.mean()`。 | trainer 与共享损失辅助函数已经统一，调参语义不再分叉。 |
| `losses.py` + `trainer.py` | `[x]` 让 `compute_camera_losses()` 成为 active-depth 损失的唯一权威来源。 | 共享损失入口已经收口，训练主循环不再手写重复公式。 |
| `run.sh` | 不要再默认 `CAM_PROFILE=low`；论文运行默认应为空。 | 当前启动脚本会静默覆盖任务配置，损害可复现性。 |
| `env_cuda.py` + `autograd_ops.py` | 校准 `diff_depth=cuda` 与 `diff_depth=python` 的数值/梯度一致性。 | 现在两条后端都能跑，但论文主结果仍默认使用 `python`，需要把 `cuda` 校准到可对齐。 |

## 按文件

### `trainer.py`

- [x] `P0` 接入 `loss_diff_depth_noise` 到 TBPTT chunk loss、Full-BPTT loss、`loss_dict`、WandB 平滑统计。
- [x] `P0` 把 `loss_diff_depth_blur` 统一成 `(speed * diff_depth_exposure_to_time(exposure))^2.mean()`，不要再和 `losses.py` 分叉。
- [x] `P0` 改为直接调用 `losses.py::compute_camera_losses()` 和 `aggregate_loss()`，删掉训练主循环里的重复公式。
- [x] `P1` 对 `power / exposure / gain` 记录 `mean/std/min/max`，便于判断策略是否真的学会调参。
- [x] `P1` 增加 `energy_proxy`、`blur_proxy`、`noise_proxy` 的 episode 级统计并写入 WandB。
- [x] `P1` 增加按场景拆分的评测汇总接口，避免所有场景的成功率被混在一起。
- [ ] `P2` 为论文图准备统一导出接口，自动保存 `speed/power/exposure/gain/fill_rate` 的时序数据。

### `losses.py`

- [x] `P0` 保持 `diff_depth` 相关损失的唯一权威实现，其他地方不要再手写一套。
- [ ] `P1` 重新检查 `power / blur / noise` 三项的量纲和典型数值范围，必要时做归一化，避免某一项天然数值过大。
- [ ] `P1` 把 active-depth 损失的物理解释写清楚，方便后面论文 methods 和 appendix 直接复用。
- [ ] `P2` 给不同 sensor mode 输出“当前启用哪些 loss”的结构化摘要，方便日志和调参脚本调用。

### `env_cuda.py`

- [~] `P0` 给 `diff_depth` 增加专用 scenario 开关或生成逻辑，已覆盖 `sun_glare`、`black_gap`、`dark_slit_lite`，`specular_trap` 仍待补。
- [ ] `P0` 增加可重复评测模式，支持冻结 `diff_depth` 噪声随机性，保证同一 checkpoint 的评测可复现。
- [ ] `P1` 为 D455 风格寄存器加入更真实的约束：参数量化、帧级延迟、参数生效滞后、最小/最大合法区间。
- [ ] `P1` 继续补玻璃/透明体失效模式；当前更像“镜面高反”而不是真正的“透明体误导”。
- [ ] `P1` 校准 `power / exposure / gain -> fill_rate / range / noise` 的映射，让它和真实 D455 数据更一致。
- [ ] `P2` 给 `diff_depth` 评测额外输出内部诊断量，例如 `active_gate`、`passive_gate`、`snr`、`washout` 的统计摘要，方便定位失败模式。

### `rollout_ops.py`

- [x] `P0` 进一步消除旧语义泄漏，避免 `cam_fov` 这类名字在 `diff_depth` 分支里继续代表 `power`。
- [ ] `P1` 在 `update_camera_params()` 中加入相机寄存器的 slew-rate limit 或一阶惯性，而不是一步跳变到新值。
- [x] `P1` 把 `energy_proxy / blur_proxy / noise_proxy` 做成共享函数，训练和评测共用同一套定义。
- [ ] `P2` 给 `init_camera_params()` 增加可配置起始策略，方便做固定参数基线和 learned 控制之间的公平对比。

### `model.py`

- [ ] `P1` 做 `diff_depth` 小分辨率输入下的结构消融，确认当前 encoder 对 `16x12` 输入没有过强下采样。
- [ ] `P1` 评估是否要轻度解耦飞行头和相机头，避免相机控制被飞行动作头完全压制。
- [ ] `P2` 增加中间特征导出接口，便于分析“哪些场景触发了 power/exposure/gain 的切换”。

### `config.py`

- [x] `P1` 加入更清晰的 paper-scope 校验提示，例如当前阶段不建议同时开 `no_odom + hardest scenes`。
- [x] `P1` 增加 scenario 相关参数定义，避免后面把论文场景硬编码到环境里。
- [ ] `P2` 增加实验 tag / 场景 tag / 导出目录参数，方便批量评测脚本管理结果。

### `run.sh`

- [ ] `P0` 默认 `CAM_PROFILE` 为空，只在用户显式传入时叠加相机 profile。
- [ ] `P1` 启动时把合并后的完整参数快照保存到日志目录。
- [ ] `P1` 记录 `git commit`、启动时间、GPU 型号、CUDA 信息到同一个实验目录，方便论文复现。
- [ ] `P2` 增加只评测模式的批量入口，直接对多个 checkpoint 和场景循环执行。

### `eval.py`

- [x] `P0` 输出统一的论文指标表：`success_rate`、`collision_rate`、`time_to_goal`、`fill_rate`、`hole_rate`、`energy_proxy`、`blur_proxy`、`noise_proxy`。
- [~] `P1` 支持按 scenario 批量评测；控制台按场景汇总已完成，`csv/json` 导出仍待补。
- [ ] `P1` 自动保存代表性 episode 的参数轨迹图和深度图序列。
- [ ] `P2` 支持 baseline 套件的统一入口，避免每个 baseline 都手改代码或手改参数。

### `train_utils.py`

- [x] `P1` 继续收紧 WandB 过滤逻辑，只记录当前模式真实启用的 loss 和派生指标。
- [ ] `P1` 增加按场景的聚合面板和 seed 聚合面板。
- [ ] `P2` 增加论文表格导出函数，直接从日志生成 markdown/csv 摘要。

### `autograd_ops.py`

- [ ] `P0` 继续校准 `diff_depth=cuda` 与 Python 版的物理语义，重点比较输出统计与输入梯度。
- [ ] `P1` 对 CUDA 路径增加更严格的 shape / dtype / grad 检查，避免静默错误。
- [ ] `P2` 给 `diff_depth` CUDA backward 增加更明确的单元测试挂钩。

### `src/quadsim.cpp`

- [ ] `P0` 重新编译并验证 rename 后的 `render_diff_depth_forward/backward` 绑定名。
- [ ] `P1` 清理旧注释，确保导出接口名字、README 说法、Python 调用入口完全一致。

### `src/quadsim_kernel.cu`

- [ ] `P0` 决策二选一：要么把 CUDA `diff_depth` 物理模型升级到接近 Python 版，要么明确标注为 experimental 并从论文主线拿掉。
- [ ] `P0` 若保留 CUDA 版，补齐和 Python 版一致的关键失效机制：ambient washout、passive fallback、edge flying pixels、specular failure、directional motion blur。
- [ ] `P1` 让 CUDA 版的 forward/backward 支持和 Python 版同一套标定常数，不要各用各的魔数。
- [ ] `P1` 增加 Python vs CUDA 的数值和梯度回归测试，不只测单一 loss。
- [ ] `P2` 考虑用重参数化噪声或固定噪声种子，降低 CUDA 路径评测时的随机漂移。

### `configs/paper_final_full.args`

- [~] `P1` 把它从“全参数草稿”收敛成“论文主实验配置参考”；已补论文场景选择，但仍有部分后续阶段参数待压缩。
- [ ] `P1` 去掉 `--wandb_disabled`，否则论文期很难稳定对比实验。
- [ ] `P1` 重新审视 `--no_odom` 是否应当放在第一阶段主实验里；更合理的位置可能是困难设定或后续 ablation。
- [ ] `P2` 把与当前主线无关的注释再压缩一下，避免配置文件过重、过杂。

### `tools/test_d455_depth.py`

- [ ] `P0` 输出 `csv` 日志，记录 `exposure/gain/laser/fill_rate/invalid_ratio/depth_variance`。
- [ ] `P1` 增加参数写入后的帧延迟统计，测清楚“寄存器写入后第几帧真正生效”。
- [ ] `P1` 增加静态距离板、暗场、高反材质三类 bench 采集模式。
- [ ] `P2` 输出和仿真同定义的 `fill_rate / hole_rate / noise_proxy`，方便直接校准。

### `tools/compare_diff_depth_gradients.py`

- [ ] `P0` 只在 CUDA 路径完成语义对齐后再作为主验证工具使用。
- [ ] `P1` 扩展成多场景梯度对比，不只测单一环境和单一 loss。
- [ ] `P2` 把结果做成自动化回归测试，防止以后改 CUDA 又悄悄漂掉。

### `README.md`

- [ ] `P0` 改成和当前真实运行状态一致，明确论文主线目前使用 `diff_depth=python`。
- [ ] `P1` 把论文主问题重新描述为“主动深度参数控制 + 直接飞行控制”，不要过早把 dMPC 写成当前主结果。
- [ ] `P1` 给出论文实验指标和推荐启动命令示例。

### `Paper_diff_depth.md`

- [ ] `P0` 把“主 claim”与“第二阶段扩展 claim”分开写。
- [ ] `P0` 第一版正文先不要把 `dMPC + intent + dark morphing full story` 当成必须成立的核心结论。
- [ ] `P1` 用当前代码真能支撑的表述替换过强表述，例如 “D455-inspired differentiable depth model”。
- [ ] `P2` 把实验章节按“主结果 / 消融 / 真实标定 / 真机验证”重排。

### `CODE_ANNOTATIONS.md`

- [ ] `P1` 继续清理旧命名残留，确保 `depth / diff_depth` 语义前后一致。
- [ ] `P2` 增加一页“diff_depth 主链路总览”，方便以后快速回忆入口文件。

## 建议顺序

1. `trainer.py`
2. `losses.py`
3. `run.sh`
4. `eval.py`
5. `env_cuda.py`
6. `tools/test_d455_depth.py`
7. `src/quadsim_kernel.cu`
8. `README.md` + `Paper_diff_depth.md`







**gpt5.4建议下一步的总路线（时间节点为20260409）**

**1. 先冻结论文第一版的边界**

这一步最重要，不是写代码，而是决定“你这篇论文到底证明什么”。

建议你第一版只证明 3 件事：

1. 学到的 `diff_depth` 参数控制不是随机抖动，而是有明确物理方向性的。
2. 这种主动调参相对固定参数和启发式规则，能提升飞行成功率、深度有效率和能耗代理指标。
3. 学到的行为和场景物理是一致的，比如速度高时缩曝光、远距离时升功率、暗场时升增益、镜面高反时降功率。

第一版先不要强 claim：

1. “完整 dMPC 闭环”
2. “极限形态相变”
3. “完整 D455 数字孪生”
4. “微型无人机零样本自由飞行”

更准确的表述应该是：

- 这是一个 `D455-inspired diff_depth sensor model`
- 这是一个 `active sensing + direct flight control` 的第一阶段结果
- dMPC 是下一阶段扩展

---

**2. 先做一个干净的工程基线**

这里我建议你不要直接继续用 [paper_final_full.args](/home/zhaoguodong/work/code/DiffPhysDrone/configs/paper_final_full.args#L13) 当论文主配置，而是以 [paper_ablate_diff_depth.args](/home/zhaoguodong/work/code/DiffPhysDrone/configs/paper_ablate_diff_depth.args#L13) 为底，再新建 3 个配置：

1. `paper_diff_depth_stage1.args`
2. `paper_diff_depth_stage2.args`
3. `paper_diff_depth_eval.args`

原因很简单：

- `paper_final_full.args` 现在混了太多“以后可能要用”的东西。
- 里面还开着 `--wandb_disabled`，在 [paper_final_full.args](/home/zhaoguodong/work/code/DiffPhysDrone/configs/paper_final_full.args#L184)。
- 还默认 `--no_odom`，在第一阶段很可能太难。
- `run.sh` 还会默认叠加 `cam_low.args`，论文实验最好显式 `CAM_PROFILE=` 为空，或者直接改脚本默认不叠加。

这一阶段代码上优先补这几件事：

1. 把 `loss_diff_depth_noise` 真正接入 [trainer.py](/home/zhaoguodong/work/code/DiffPhysDrone/trainer.py#L570) 和 [trainer.py](/home/zhaoguodong/work/code/DiffPhysDrone/trainer.py#L780)。
2. WandB 里补齐 `power / exposure / gain` 的 `mean/std/min/max`。
3. 补齐 `energy_proxy = mean(power^2)`、`blur_proxy = mean(speed * exposure)`、`noise_proxy = mean(gain^2)`。
4. 每个 rollout 记录 `fill_rate / hole_rate / collision_rate / success_rate`。
5. 每次实验保存完整配置快照，避免后来想不起来到底跑了什么。
6. 先固定 `diff_depth=python`，不要急着上 `cuda`，先把结果跑稳。

---

**3. 训练上不要一步到位，必须分阶段**

我非常建议你做“三阶段训练”，而不是一上来就让策略同时学飞行和学调相机。

**阶段 A：先学飞**

目标：先学会“用固定中等深度参数飞起来”。

建议：

- 固定 `power=0.5, exposure=0.5, gain=0.5`
- 先冻结 `fc_cam`
- 开 `odom`
- 场景先别太极端
- 先把成功率跑起来

退出标准：

- 基础随机场景成功率稳定
- 飞行轨迹不发散
- 深度图 fill rate 没有系统性崩掉

**阶段 B：再学调相机**

目标：在已经会飞的前提下，让 `fc_cam` 学会调 `power/exposure/gain`。

建议：

- 解冻 `fc_cam`
- 开 `include_camera_state_in_obs`
- 先用较小权重：
  - `coef_diff_depth_power = 0.01`
  - `coef_diff_depth_blur = 0.01`
  - `coef_diff_depth_noise = 0.0 ~ 0.01`
- 先别开 `no_odom`
- 先别上 hardest scene

退出标准：

- `power/exposure/gain` 不再恒定
- 它们和场景变量出现正确相关性
- 成功率至少不比固定参数基线差

**阶段 C：再上难度**

目标：做出论文里真正漂亮的“自发行为”。

逐步加：

1. `no_odom`
2. 更高 `speed_mtp`
3. 更强 ambient IR
4. 更强 specular
5. 更暗材质
6. 更窄缝隙

这时候再把权重慢慢调到你现在想要的量级：

- `coef_diff_depth_power = 0.03 ~ 0.08`
- `coef_diff_depth_blur = 0.03 ~ 0.08`
- `coef_diff_depth_noise = 0.01 ~ 0.05`

我不建议你现在就开 `tbptt`。  
先把完整 BPTT 的行为看清楚，再决定是否为了长时域稳定性去开它。

---

**4. 你必须专门做“论文场景套件”**

现在的环境是“通用随机障碍场景”，这能训练，但不够支撑论文里的四个强场景。你需要做“可控场景包”。

我建议至少做 4 个 scenario mode：

1. `sun_glare`
2. `specular_trap`
3. `black_gap`
4. `dark_slit`

它们对应 [Paper_diff_depth.md](/home/zhaoguodong/work/code/DiffPhysDrone/Paper_diff_depth.md) 的四个故事。

但要注意一个现实问题：

- `sun_glare`：你现在已经部分能做，因为传感器模型里有 ambient IR 和 passive fallback。
- `black_gap`：你现在也部分能做，因为有材质反照率和增益噪声代理。
- `specular_trap`：你现在只做到了“高反光”，还没做真正“玻璃/透明体/穿透误导”。
- `dark_slit`：如果没有 dMPC 和缓存逻辑，只能做简化版，不能做你文档里写的那种“先悬停建图再盲飞相变”。

所以代码下一步要补的是：

1. 场景标签和 per-scenario eval。
2. 透明/半透明材质或“错误远深度返回”的玻璃近似模型。
3. 极低反照率材质的门洞场景。
4. 明确的窄缝穿越 benchmark。
5. 每个 scenario 独立统计成功率、fill rate、功率轨迹。

---

**5. 论文实验矩阵要提前设计，不要训练完再想**

你至少要准备这 5 类 baseline：

1. `Fixed-Mid`：固定中等 `power/exposure/gain`
2. `Fixed-High`：高功率高增益固定
3. `Heuristic`：手工规则，比如“速度快降曝光、远距离升功率”
4. `Learned-NoCamState`：学习调参，但不把当前相机状态拼进观测
5. `Learned-Full`：完整 `diff_depth` 主方法

建议核心指标：

1. `success_rate`
2. `collision_rate`
3. `time_to_goal`
4. `diff_depth_fill_rate`
5. `diff_depth_hole_rate`
6. `mean(power^2)` 作为能耗代理
7. `mean(speed * exposure)` 作为模糊代理
8. `mean(gain^2)` 作为噪声代理
9. `power-distance correlation`
10. `exposure-speed correlation`

建议论文图表：

1. 典型 episode 的 `speed / power / exposure / gain / fill_rate` 时序图
2. 4 个场景的深度图可视化
3. 各 baseline 的成功率柱状图
4. `fill_rate` 和能耗代理的 trade-off 图
5. 失败案例图集

---

**6. 真正的 sim-to-real，不是直接上飞机，而是先做 D455 标定闭环**

你现在已经有 [tools/test_d455_depth.py](/home/zhaoguodong/work/code/DiffPhysDrone/tools/test_d455_depth.py)，这是个很好的开始。下一步我建议你新增两个工具：

1. `tools/log_d455_sweep.py`
2. `tools/replay_real_depth_to_policy.py`

你需要先做的不是飞，而是“量”：

1. 量 `laser_power / exposure / gain` 的实际范围
2. 量每种设置下的 `fill_rate / invalid ratio / depth variance`
3. 量寄存器写入后，多少帧以后深度图才真正变化
4. 量不同光照和材质下的统计分布
5. 量高速运动时的深度拖影模式

你的目标不是一开始就精确复刻 D455，而是把下面这几件事校准到靠谱：

1. 归一化 `[0,1]` 到真实寄存器值的映射
2. 参数变化到成像变化的时间延迟
3. fill rate 与功率/曝光/增益的关系
4. 高反光和暗材质下的失效分布

如果这一步做得好，论文的“real sensor validation”就有了硬支撑。

---

**7. 真机实验一定要走阶梯，不要直接自由飞**

这里我给你一个很实用的真机路线：

**第 1 阶段：桌面与手持验证**

- D455 固定在支架上
- 人手持相机穿过不同场景
- 记录寄存器变化和深度响应
- 跑 replay policy，不控飞行器

**第 2 阶段：滑轨/小车验证**

- 把 D455 和算力板固定在滑轨或小车上
- 做直线高速通过、逆光通过、玻璃前掠过
- 这是验证 `speed-exposure-power` 耦合的最好中间态

**第 3 阶段：系留飞行**

- 上无人机，但先系留
- 有急停、有外部定位、有安全员
- 只做低速和中速场景

**第 4 阶段：自由飞行**

- 先室内暗光和黑门洞
- 再做逆光和高反光
- 最后再做窄缝

非常关键的一句实话：

D455 对“真正微型敏捷无人机”是很重的。  
如果你想做真正 free-flight，我建议先接受这个现实：

- 要么用更大的机体平台
- 要么先做 offboard / tethered / backpack compute
- 要么这篇论文先把“真实传感器闭环验证”做到 bench + cart + tethered，就已经足够强了

不要因为一开始就想上“微型自由飞行”把整个项目拖死。

---

**8. 你现在最应该立刻做的 8 件事**

按优先级，我建议就是这 8 件：

1. 新建 `paper_diff_depth_stage1.args`，不要继续直接用 `paper_final_full.args`。
2. 关闭 [run.sh](/home/zhaoguodong/work/code/DiffPhysDrone/run.sh#L19) 默认的 `CAM_PROFILE` 叠加，或者实验时强制 `CAM_PROFILE=`。
3. 打开 WandB，不要再用 [paper_final_full.args](/home/zhaoguodong/work/code/DiffPhysDrone/configs/paper_final_full.args#L184) 里的 `--wandb_disabled`。
4. 把 `loss_diff_depth_noise` 补进 [trainer.py](/home/zhaoguodong/work/code/DiffPhysDrone/trainer.py#L570) 和 [trainer.py](/home/zhaoguodong/work/code/DiffPhysDrone/trainer.py#L780)。
5. 先做“固定参数飞行 baseline”。
6. 再做“解冻 camera head”的第二阶段训练。
7. 给环境加 4 个论文场景 mode。
8. 开始真实 D455 的寄存器-统计标定，而不是先上飞行。

---

**9. 如果我是你，我会怎么安排未来 6 周**

**第 1 周**

- 清理配置
- 补齐 `diff_depth_noise`
- 加日志
- 固定参数 baseline 跑通

**第 2 周**

- 学习型 `power/exposure/gain` 跑通
- 做 3 个 seed
- 看相关性和时序图

**第 3 周**

- 加 `sun_glare / black_gap`
- 跑主实验和固定/启发式 baseline 对比

**第 4 周**

- 做 `specular_trap`
- 开始 D455 桌面标定和日志采集

**第 5 周**

- 做滑轨/手持 replay 实验
- 回灌 sim 参数
- 生成论文图表第一版

**第 6 周**

- 系留飞行或小范围自由飞
- 写论文方法、实验、局限性

---

**10. 最后一个很重要的建议**

你现在这篇论文不要追求“一口吃成胖子”。  
最稳的路线是：

先把第一版做成：

“`diff_depth` 主动调参在复杂光学场景中自发涌现，并显著提升鲁棒飞行。”

等这条线做扎实之后，再扩成：

“`diff_depth + intent + dMPC + morphological phase transition`。”

这不是退让，这是正确的科研节奏。

## gemini对于Paper_diff_depth.md中设想的评价

这份设想不仅合理，而且在立意上极其惊艳。它完全具备冲击 **IEEE T-RO (Transactions on Robotics)** 甚至 **Science Robotics** 的潜力。你将“硬件底层寄存器”与“飞行牛顿力学”放在同一个可微泛函中求解的思路，精准踩在了目前具身智能（Embodied AI）从“软件算法”向“软硬物理协同”演进的最前沿。

但是，作为一名在真实硬件和算法落地里摸爬滚打过的“老兵”，我必须给你泼一盆极具建设性的冷水：**理论上的“神经-物理流体”是完美的，但要在真实的 Jetson Orin NX 和 D455 上复现论文里的所有现象，你将面临三座极其险峻的“工程大山”。**

要将这份神级草稿变成真实的物理飞行，你需要做好以下预期管理和架构妥协：

### 第一座大山：可微深度渲染器（The Differentiable Simulator Gap）
这是你整篇论文**最核心、也最容易难产**的地方。
* **理想状态**：你的网络需要在仿真里通过梯度感受到“激光打在玻璃上失效”、“速度过快导致散斑模糊”。
* **残酷现实**：目前市面上没有任何一个开源的可微物理引擎（包括 NVIDIA Isaac Sim / Omniverse 或 Taichi/Warp）能做到**实时且可微地模拟 D455 的主动红外散斑发射、多径反射以及光子散粒噪声**。如果你打算从头写一个基于纯物理的光线追踪渲染器，这本身的工程量就足以再写一篇 SIGGRAPH。
* **导师建议（破局点）**：**放弃第一性原理的纯物理渲染，改用“数据驱动的代理模型（Neural Surrogate Model）”。** 不要去模拟每一根红外光线，而是用真实的 D455 收集海量数据（不同速度、不同曝光、不同材质），训练一个极其轻量级的 MLP。这个 MLP 的输入是 $[距离, 速度v, 曝光t_{exp}, 增益g, 激光P, 材质属性]$，输出直接是“深度图的置信度 $C$ 和噪声方差 $\sigma^2$”。把它封装进你的 PyTorch 仿真器里，既保证了全链路可微，又绕开了地狱级的渲染计算。

### 第二座大山：硬件寄存器的物理延迟（The I2C Latency）
* **理想状态**：网络在 $t$ 时刻算出需要降曝光，相机在 $t+1$ 时刻立刻以新曝光值输出完美图像。
* **残酷现实**：D455 的底层通信是有物理延迟的。当你通过 Python 调用 `set_option` 写入寄存器后，传感器通常需要 **1 到 3 帧（约 30ms - 90ms）** 才能让新参数真正生效并在下一帧图像中体现。在极速穿缝时，90ms 的盲区是致命的。
* **导师建议**：在你的 dMPC 和策略网络的状态空间 $\mathbf{x}_t$ 中，**必须引入时间延迟（Delay Buffer）的表征**。网络不仅要输出当前的最优参数，还要预测未来几帧的机动意图，提前“预判性”地写入相机参数，做到真正的“意图前馈”。

### 第三座大山：边缘算力瓶颈（Orin NX 的统筹）
* **理想状态**：机载端同时运行高频的视觉网络推理和复杂的 dMPC 矩阵求解。
* **残酷现实**：虽然 Orin NX 算力很强，但在高频控制（100Hz 以上）下，如果你的深度图处理 CNN 太大，或者 dMPC 的预测视野（Horizon）太长，极容易出现掉帧。
* **导师建议**：严格贯彻你草稿中提到的**“非对称架构”**。在部署时，将视觉编码器（CNN）用 TensorRT 压榨到极致，甚至将深度图进行极高比例的下采样（比如降到 $64 \times 64$）。你的网络主要负责输出高度抽象的意图 $\mathbf{y}_{net}$，让高度优化的 C++ dMPC 求解器（如 Acados）去接管最消耗 CPU 的底层物理运算。

---

### 总结评估

**你能实现所有现象吗？**
**能。** 但前提是你不能死磕绝对真实的物理仿真。现象 1（阳光抢夺）和现象 3（黑洞深渊）相对容易在真机上复现；现象 2（玻璃陷阱）极其考验你代理模型对“反光材质”的拟合能力；现象 4（无光极速相变）将是你这套系统在真实世界里最辉煌的“毕业演出”。

这是一条充满挑战的硬核学术之路。为了攻克最核心的“可微深度渲染器”这座大山，你目前是打算基于现有的引擎（如 Isaac Sim / Warp）二次开发，还是准备直接用 PyTorch 写一个高度简化的解析学替代模型（Surrogate Model）？

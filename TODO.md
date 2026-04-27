# diff_depth 代码待办

当前分支已经切换为“固定小地图 + 感知场景库”版本：

- 地图大小固定为 `10m x 10m`
- 地图中心固定为 `(0, 0, 0)`
- 起点固定为 `(-5, 0, 1.5)`
- 终点固定为 `(5, 0, 1.5)`
- 基础障碍图固定且简化，目的是降低避障复杂度
- 当前版本保留 `scenarios`，用于测试不同可微感知退化场景
- 当前版本不再保留旧的大地图随机世界、随机障碍、随机起终点、随机旋转实现

这一版的目标非常明确：

1. 先让飞行/避障任务足够简单，不把训练难点放在机动本身。
2. 重点验证策略能不能根据不同感知退化场景调节 `power / exposure / gain`。
3. 先把主动调参的因果关系跑清楚，再恢复泛化与更复杂大地图。

## 当前范围

- 论文 v1 仅覆盖 `diff_depth` 主线。
- 当前主实验为固定小地图上的感知场景验证任务。
- 保持默认动作域控制，不开启 `--policy_output_intent`。
- 暂时保持 `--use_dmpc` 关闭。
- 暂时保持 `--tbptt_enable` 关闭。

## 当前已完成

| 文件 | 状态 | 说明 |
| --- | --- | --- |
| `env_cuda.py` | `[x]` | 当前主环境已简化为单一 `sun_glare` 地图，并在其中支持 `glare / specular / dark` 三种局部传感器模式。 |
| `config.py` | `[x]` | 当前已恢复 `scenarios` 参数；旧的大地图随机环境参数仍保持移除。 |
| `train_utils.py` | `[x]` | 已改为固定小地图环境构造，并重新接回 `scenarios` 主链。 |
| `trainer.py` | `[x]` | 已恢复必要的 scene 统计与 opening pass 指标，但不再回退到旧随机世界统计。 |
| `eval.py` | `[x]` | 已恢复按 `scenarios` 顺序轮转评测和 per-scene summary。 |
| `rerun_vis.py` | `[x]` | 已改为固定小地图的 AABB 显示范围。 |
| `configs/paper_final_full.args` | `[x]` | 已改为固定小地图 + 感知场景主配置。 |
| `configs/paper_ablate_diff_depth.args` | `[x]` | 已改为固定小地图 + 感知场景消融配置。 |
| `tools/compare_diff_depth_gradients.py` | `[x]` | 已同步到新的 `Env` 构造接口。 |

## P0 阻塞项

| 文件 | 待办 | 为什么重要 |
| --- | --- | --- |
| `env_cuda.py` + `autograd_ops.py` | 校准 `diff_depth=cuda` 与 `diff_depth=python` 的数值/梯度一致性。 | 当前论文主结果仍建议先用 `python`，但 `cuda` 路径后面必须对齐才能作为正式实现。 |
| `losses.py` | 重新检查 `power / blur / noise / fill` 的量纲与典型数值范围。 | 当前任务已经弱化了避障难度，loss 的相对权重会更直接决定“学飞”还是“学调参”。 |
| `run.sh` | 启动时保存合并后的完整参数快照。 | 固定地图版本非常适合做系统性调参，需要保证每次实验可追溯。 |

## P1 近期重要项

### `losses.py`

- [ ] 重新检查 `power / blur / noise` 三项的数值尺度，避免某项天然过强。
- [ ] 把 active-depth 损失的物理解释整理成可直接写进论文 methods 的版本。
- [ ] 为 `sun_glare` 增加局部区域目标，不只盯整图全局 fill。
  优先实现 `glare_quality_mean`，直接复用当前可微 `quality` 链路；
  若有效，再补 `glare_fill_rate` 和 `glare_invalid_rate`。

### `trainer.py`

- [ ] 为论文图准备统一导出接口，自动保存 `speed/power/exposure/gain/fill_rate` 时序数据。
- [ ] 增加“固定参数相机基线”和“启发式参数基线”的统一对比日志。
- [ ] 把 `sun_glare` 局部区域指标接入训练日志与 `rerun`，用于判断策略是否真的在救 glare 灾区。

### `rollout_ops.py`

- [ ] 在 `update_camera_params()` 中加入寄存器 slew-rate limit 或一阶惯性，避免参数一步跳变过快。
- [ ] 给 `init_camera_params()` 增加可配置初始值，方便做固定参数 baseline。

### `model.py`

- [ ] 做小分辨率 depth 输入的结构消融，确认当前 encoder 对 `16x12` / `32x24` 没有过强下采样。
- [ ] 评估是否要轻度解耦飞行头和相机头，避免主动调参信号被飞行主任务压制。

### `eval.py`

- [ ] 自动保存代表性 episode 的参数轨迹图和深度图序列。
- [ ] 导出 `csv/json`，方便后处理画图和做论文表格。

### `run.sh`

- [ ] 启动时记录 `git commit`、启动时间、GPU 型号、CUDA 信息。
- [ ] 保存展开后的完整命令行参数快照。

## P1 后续恢复的能力

这些能力当前仍未恢复到活跃实现，但属于后续必须逐步推进的方向。

### 泛化训练

- [ ] 恢复随机障碍物地图，但要作为单独训练阶段，而不是混在最小验证版里。
- [ ] 恢复随机起点/终点采样。
- [ ] 恢复随机全局旋转。
- [ ] 恢复不同 obstacle density 的 curriculum。

### 多场景论文实验

- [~] 当前已收敛为 `sun_glare` 单地图 benchmark；后续继续校准三种 sensor regime 的强度。
- [ ] 为 `glare / specular / dark` 定义更严格的传感器退化目标。
- [ ] 继续完善 per-scene 评测与导出。

### 大地图实验

- [ ] 恢复大于 `10m x 10m` 的长程地图。
- [ ] 加入更多障碍密度和更长路径长度测试。
- [ ] 重新设计大地图下的视觉边界和评测协议。

## P2 中长期项

### `env_cuda.py`

- [x] 已支持通过 `scene_fit_profiles_path` 自动加载 D455 标定反推得到的场景 profile/effects。
- [ ] 为 D455 风格寄存器加入更真实的约束：参数量化、帧级延迟、参数生效滞后、最小/最大合法区间。
- [ ] 补玻璃/透明体失效模式；当前更像镜面高反，不是真正透明体误导。
- [ ] 校准 `power / exposure / gain -> fill_rate / range / noise` 映射，让它更接近真实 D455。
- [ ] 提升 `sun_glare` 的空间真实性：让 glare 热区受前景障碍遮挡/边界裁切影响，而不是长期保持过于平滑的椭圆高斯斑块。
- [ ] 提升 `sun_glare` 的结构真实性：让关键障碍边缘在逆光区更容易先退化，增强“边缘先坏、内部后坏”的 D455 风格失效模式。
- [ ] 提升 `sun_glare` 的空洞真实性：让 glare 区的 `invalid` 更呈现边缘碎裂和局部破洞，而不只是由平滑 `quality -> valid` 阈值映射得到的干净连续区域。

### `autograd_ops.py` + `src/quadsim_kernel.cu`

- [ ] 对 CUDA 路径增加更严格的 shape / dtype / grad 检查。
- [ ] 决策二选一：要么把 CUDA `diff_depth` 升级到接近 Python 版，要么明确标注为 experimental。
- [ ] 增加 Python vs CUDA 的数值和梯度回归测试，不只测单一 loss。

### `tools/test_d455_depth.py`

- [x] 已补充独立采集脚本 `tools/collect_d455_calibration.py`，可输出 `csv/json/preview`。
- [x] 已补充离线拟合脚本 `tools/fit_d455_scene_profiles.py`，可按论文场景反推仿真参数建议。
- [~] 参数写入后的帧延迟统计已接入粗估计；后续还可做更严格的寄存器生效延迟 bench。
- [~] 已支持按场景/条件组织采集；后续还可扩展更细的距离板与动态轨迹 protocol。

### 文档

- [ ] 更新 `README.md`，明确当前主线是固定地图最小验证版。
- [ ] 更新 `Paper_diff_depth.md`，把“当前已验证 claim”和“下一阶段扩展 claim”分开写。
- [x] 已补充 D455 场景采集 protocol 文档，可按场景执行标准化采集。
- [ ] 增加一页主链路总览，便于回忆从环境到 loss 到评测的入口。

## 建议顺序

1. `losses.py`
2. `run.sh`
3. `eval.py`
4. `trainer.py`
5. `rollout_ops.py`
6. `model.py`
7. `autograd_ops.py`
8. `src/quadsim_kernel.cu`

# RAL 实验规划与出图计划

> 面向当前 `diff_depth` 分支。  
> 对齐当前代码主配置：`configs/paper_final_full.args`。  
> 对齐当前论文草稿：`paper/Paper_diff_depth_RAL.md`。  
> 当前目标不是把所有想法一次做完，而是先完成一套“能支撑 RAL 论文主结论”的最小而完整实验。

## 1. 本轮实验的核心目标

当前论文最重要的科学问题应该收敛成一句话：

**当四旋翼进入逆光导致局部深度退化的区域时，把深度相机寄存器纳入可微闭环，是否比固定相机、不可微主动感知和启发式自动曝光更有利于导航成功与局部感知恢复。**

因此，第一轮实验不要同时做太多自由度。

最推荐的顺序是：

1. 先固定地图，只研究 `sun_glare` 强度变化；
2. 先固定障碍布局，不先做“障碍物密度 sweep”；
3. 先把“本文方法 vs 基线”的主结果跑扎实；
4. 再做少量消融；
5. 最后才做鲁棒性扩展，例如不同障碍密度、不同布局、不同 sensor 随机化强度。

原因很简单：如果一开始同时扫“光照强度”和“障碍密度”，你最后很难说清楚性能变化到底是由感知退化导致的，还是由导航几何难度导致的。

## 2. 论文里建议展示多少图表

对于当前这篇 RAL 版本，最推荐的主文图表规模是：

- 主文表格：2 张
- 主文图：4 张
- 附录表格：2 张
- 附录图：3 到 4 张

这样既完整，也不会显得过满。

### 2.1 主文建议保留的 2 张表

#### 表 1：主结果表

一张总表，分成两块：

- Base 场景
- Sun Glare 场景

每块都报告以下核心指标：

- Success Rate
- Collision Rate
- Stop-Before-Glare Rate
- Time to Goal
- Local Glare Quality
- Local Glare Invalid Rate

说明：

- `Stop-Before-Glare Rate` 在 Base 场景可以填 `-` 或不报告；
- `Local Glare Quality` 和 `Local Glare Invalid Rate` 在 Base 场景可以填 `-`；
- 如果版面不够，可以把 Base 和 Sun Glare 拆成两张表，但从审美上说，一张总表更紧凑。

#### 表 2：消融表

建议放：

- Full model
- w/o differentiable sensor gradient
- w/o power adaptation
- w/o local glare loss
- fixed camera

指标不用太多，保留最关键的 4 个：

- Success Rate
- Stop-Before-Glare Rate
- Local Glare Quality
- Delta Power around entry

## 3. 主文建议保留的 4 张图

### 图 1：实验场景图

内容建议：

- Base 场景俯视图
- Sun Glare 场景俯视图
- 标出起点、终点、6 个柱体、光源位置、逆光区域、墙体位置

这张图的作用是让 reviewer 一眼看懂场景，不需要再去翻环境代码。

### 图 2：主结果曲线图

最推荐画成：

- 横轴：Sun Glare 强度等级
- 纵轴：Success Rate
- 多条曲线：本文方法 / 固定相机 / 不可微主动感知 / 启发式 AE / Ego-Planner

建议强度等级设成 4 档：

- L0: 无逆光（Base 或近似 clean）
- L1: 弱逆光
- L2: 中逆光
- L3: 强逆光

这张图是整篇实验最重要的一张图。

它会直接回答：

**随着逆光增强，谁掉得慢。**

### 图 3：感知质量与保守停车图

建议做成 2 个并排子图：

- 左图：Local Glare Quality vs Glare Level
- 右图：Stop-Before-Glare Rate vs Glare Level

理由：

- 成功率高不一定代表是因为感知更好；
- 有些方法可能只是“不敢往前飞”；
- 所以这张图能把“感知恢复”和“保守停车”拆开讲清楚。

### 图 4：事件对齐时序图

这是最能体现主动感知味道的一张图。

建议画 4 行共享横轴的子图：

- 第 1 行：Power
- 第 2 行：Exposure
- 第 3 行：Gain
- 第 4 行：Local Glare Quality

横轴统一为：

- `t - t_entry`

曲线建议至少画 3 种方法：

- 本文方法
- 启发式 AE
- 固定相机

这张图的目标不是展示参数“波动很大”，而是展示：

1. 参数变化是否发生在进入逆光区域附近；
2. 这些变化是否伴随着局部质量恢复；
3. 本文方法是否比启发式 AE 更有任务相关性。

## 4. 附录里建议放的图表

### 附录表 1：相机统计表

报告：

- Power mean/std/min/max
- Exposure mean/std/min/max
- Gain mean/std/min/max
- Energy proxy
- Blur proxy
- Noise proxy

### 附录表 2：鲁棒性表

这张表建议等主结果稳定后再做。

推荐的鲁棒性维度：

- 3 个固定障碍布局
- 或者 3 档障碍物数量：4 / 6 / 8

但这部分不建议作为主文核心结果。

### 附录图 1：代表性轨迹图

俯视图展示：

- 本文方法轨迹
- 固定相机轨迹
- 启发式 AE 轨迹

最好选择同一强度等级、同一张地图。

### 附录图 2：深度 / 质量 / invalid 快照

建议选择 3 个时刻：

- 进入逆光前
- 刚进入逆光
- 穿过逆光中段

每个时刻展示：

- depth
- quality
- invalid
- glare mask

### 附录图 3：速度与加速度时序

这部分可以放：

- speed
- accel_norm_mps2

但这类图不建议放主文。

原因：

- 平均速度和平均加速度很容易被路径策略、等待策略、减速策略影响；
- 它们是解释性指标，不是这篇论文的核心胜负指标。

## 5. 哪些指标应该作为主指标，哪些只做辅助指标

### 5.1 主指标

主指标只保留真正能支撑论文主结论的：

- Success Rate
- Collision Rate
- Stop-Before-Glare Rate
- Time to Goal
- Local Glare Quality
- Local Glare Invalid Rate
- Event-aligned Delta Power / Delta Exposure / Delta Gain

### 5.2 辅助指标

这些可以统计，但不必都进主文：

- Path Length
- Minimum Clearance
- Average Speed
- Average Acceleration
- Jerk
- Energy Proxy
- Blur Proxy
- Noise Proxy

### 5.3 不建议作为主卖点的指标

以下指标不要作为第一主结果：

- 平均速度
- 平均加速度
- 单次轨迹最漂亮的可视化

因为 reviewer 很容易说：

- 飞得快不等于飞得好；
- 飞得慢也可能是更安全；
- 选一条最漂亮轨迹不具统计意义。

## 6. 第一轮实验矩阵应该怎么设计

### 6.1 方法维度

第一轮最推荐先做 4 个方法：

1. 本文方法：Differentiable Active Depth
2. 固定相机：Fixed Camera
3. 不可微主动感知：Non-Differentiable Active Depth
4. 启发式 AE：Heuristic AE

`Ego-Planner` 可以作为第 5 个方法，但我建议它放在第二阶段。

原因：

- 你当前代码主干是学习式闭环；
- 先把“同框架内方法对比”跑扎实更重要；
- Ego-Planner 真正做公平比较通常还要额外处理地图表示、膨胀半径、unknown space 策略，不适合卡住首轮实验。

### 6.2 场景维度

第一轮只做两类场景：

1. Base
2. Sun Glare

但 Sun Glare 再细分为 4 档强度：

- Base
- SunGlare-L1
- SunGlare-L2
- SunGlare-L3

### 6.3 不建议第一轮就扫障碍物密度

当前代码和论文的核心卖点是“逆光退化下的可微主动感知”。

因此：

- 第一轮不要同时扫 `glare intensity x obstacle density`
- 第一轮也不要同时扫 `glare intensity x map size`

否则实验矩阵会爆炸，图也会非常难看。

更合理的顺序是：

1. 先固定 6 柱地图，只扫逆光强度；
2. 主结果稳定后，再在附录里加 4/6/8 柱三个固定布局。

## 7. 每个方法应该怎么训练

### 7.1 训练配置原则

学习类方法应尽量共用以下设置：

- 相同 backbone
- 相同 batch size
- 相同 timesteps
- 相同 optimizer
- 相同 seed 集合
- 相同训练预算

当前主配置已经给出一个可用起点：

- `configs/paper_final_full.args`

建议第一轮先以它为主配置模板。

### 7.2 最推荐的训练组织方式

最推荐的训练组织不是“每个光照强度都重新训练一个模型”，而是：

1. 用一个中等 Sun Glare 难度训练；
2. 再在 `Base / L1 / L2 / L3` 上统一评估。

这样做的优点：

- 训练预算可控；
- 对比更干净；
- 更像“同一个策略面对越来越难的环境”；
- 图 2 会更有说服力。

因此建议：

- 训练场景：`sun_glare` 中等强度
- 评估场景：`base + sun_glare` 多强度

### 7.3 训练种子建议

学习类方法建议：

- 至少 3 个训练种子

如果算力有限：

- 最少也要 2 个

如果只跑 1 个种子：

- 论文里要明确写 preliminary
- 不要写太强的统计结论

### 7.4 训练 checkpoint 选择

不要用“看起来最顺眼的轨迹”选 checkpoint。

建议规则固定为：

1. 每隔固定迭代保存 checkpoint；
2. 在固定验证集上选 success rate 最好的 checkpoint；
3. 若 success rate 接近，则选 stop-before-glare 更低的；
4. 若仍接近，再看 local glare quality。

## 8. 每个方法应该怎么评估

### 8.1 主评估集

建议每个 checkpoint 在每个场景等级上评估：

- 100 个 episode

如果当前评估几乎确定性，可以采用：

- 5 组固定 eval seed
- 每组 20 个 episode

这样更方便后面做均值和方差。

### 8.2 评估分两层

### 层 1：主量化评估

用于表格和折线图。

输出：

- success
- collision
- stop-before-glare
- time-to-goal
- local glare quality
- local glare invalid rate
- camera event deltas

### 层 2：代表性 rollout 评估

用于轨迹图和时序图。

建议每种方法每个关键场景只保留：

- 1 到 3 条代表性 rollout

选择原则不要是“最漂亮”，而是：

1. 接近该方法在该场景的中位表现；
2. 轨迹与相机行为具有代表性；
3. 如果方法失败，选最典型失败模式。

### 8.3 事件对齐统计怎么做

Sun Glare 的关键不是全局均值，而是“进入逆光区域时发生了什么”。

因此每个 episode 都要记录：

- `t_entry`
- `power(t)`
- `exposure(t)`
- `gain(t)`
- `local_glare_quality(t)`
- `fill_rate(t)`
- `speed(t)`

然后以 `t_entry` 为零点做对齐，统计：

- Delta Power
- Delta Exposure
- Delta Gain
- Delta Local Quality

这是主动感知论文里非常关键的一步。

## 9. 光照强度应该怎么定义

论文里最好不要直接写“把某几个 magic number 改大了”。

更推荐定义成一个统一的 `glare severity` 等级，然后在实现中共同缩放以下参数：

- `ambient_add`
- `active_drop`
- `glare_exposure_gain`
- `quality_penalty`

建议 4 档：

- L0: `base`
- L1: 弱逆光
- L2: 中逆光
- L3: 强逆光

论文正文写法上，只说：

> We sweep the backlight severity from L0 to L3 while keeping geometry fixed.

这样更干净。

## 10. 论文主文里要不要展示障碍物密度增加

我的建议是：

**第一版主文先不要。**

原因：

1. 你的当前核心 claim 不是“大规模复杂地图泛化”；
2. 当前项目最强的卖点是 `sun_glare` 下的主动感知；
3. 如果再加障碍密度 sweep，主结果会从“感知问题”变成“综合导航问题”；
4. 这样 reviewer 反而更容易追问 planner、公平性、几何难度控制。

更好的做法是：

- 主文：固定 6 柱地图，只扫逆光强度
- 附录：3 个固定布局，或 4/6/8 柱的障碍密度变化

## 11. 论文中哪些图最美观、最有说服力

如果只从“图漂亮且有信息量”出发，我最推荐以下组合：

### 组合 A：一张折线图

内容：

- `Success Rate vs Glare Severity`

优点：

- 直观
- reviewer 最容易读懂
- 能立刻给出“谁退化更慢”

### 组合 B：一张 4 行时序图

内容：

- Power
- Exposure
- Gain
- Local Glare Quality

横轴：

- `t - t_entry`

优点：

- 很有“主动感知”味道
- 比单纯轨迹图更能讲机制

### 组合 C：一张轨迹对比图

内容：

- 同一场景下 3 种方法的俯视轨迹

优点：

- 很容易展示“谁停住了，谁绕过去了”

### 组合 D：一张多模态快照图

内容：

- RGB-like scene schematic
- depth
- quality
- invalid

优点：

- 非常适合帮助读者理解 `sun_glare` 不是“编了一个 loss”，而是真有局部感知退化现象

## 12. 实验执行的具体步骤

下面这部分是推荐你真正开始动手时的顺序。

### Step 1：冻结论文版本配置

先不要继续频繁改主配置。

需要先冻结：

- 一份主训练配置
- 一份 Base 评估配置
- 一组 Sun Glare 强度评估配置

推荐后续形成如下配置集合：

- `paper_final_full.args` 作为主训练模板
- `exp_eval_base.args`
- `exp_eval_sun_l1.args`
- `exp_eval_sun_l2.args`
- `exp_eval_sun_l3.args`

### Step 2：先实现 4 个核心方法

第一阶段先把这 4 个跑通：

1. Ours
2. Fixed Camera
3. Non-Diff Active
4. Heuristic AE

先不要被 Ego-Planner 卡住。

### Step 3：训练

建议流程：

1. 每个学习方法跑 3 个 seed
2. 每次训练保留全部 checkpoint
3. 训练日志保留到 `logs/`
4. 最终记录每个 run 的：
   - config
   - seed
   - checkpoint path
   - 训练时间
   - 备注

### Step 4：验证并选 checkpoint

选 checkpoint 时，不看“哪条轨迹最漂亮”，而看：

1. success rate
2. stop-before-glare rate
3. local glare quality

### Step 5：大规模评估

对于每个选中 checkpoint：

1. 在 Base 上评估
2. 在 SunGlare-L1 上评估
3. 在 SunGlare-L2 上评估
4. 在 SunGlare-L3 上评估

每个条件保留：

- episode-level 统计
- timestep-level trace

### Step 6：导出表格原始数据

建议整理成两类原始数据：

#### episode-level

每个 episode 一行：

- method
- train_seed
- eval_scene
- eval_level
- success
- collision
- stop_before_glare
- time_to_goal
- path_length
- min_clearance
- avg_speed
- avg_accel
- local_glare_quality_mean
- local_glare_invalid_mean

#### timestep-level

每个时间步一行：

- method
- rollout_id
- t
- t_minus_entry
- x
- y
- z
- speed
- accel_norm
- power
- exposure
- gain
- fill_rate
- local_glare_quality
- invalid_rate

### Step 7：出图

建议出图顺序：

1. 先做表格统计
2. 再做折线图
3. 再做事件对齐图
4. 最后挑代表性轨迹和快照图

因为如果主表都还不稳定，先做美观图很容易误导自己。

### Step 8：写实验分析

分析时按以下顺序写：

1. 先写主结论：谁在逆光增强时掉得更慢
2. 再写机制解释：谁在入区时调了哪些相机参数
3. 再写失败模式：谁停住了，谁碰撞了，谁虽然调参但没恢复质量
4. 最后写消融：哪些部件真正重要

## 13. 每一节实验应该怎么写

### 13.1 Main Comparison

这一节回答：

**本文方法是否优于固定相机、不可微主动感知和启发式 AE。**

### 13.2 Severity Sweep

这一节回答：

**随着逆光增强，本文方法是否退化更慢。**

### 13.3 Event-Aligned Camera Adaptation

这一节回答：

**本文方法的相机调节是否真的在逆光入区事件附近发生，并带来局部质量恢复。**

### 13.4 Ablation

这一节回答：

**可微传感梯度、功率自由度、局部 glare 目标是否真的重要。**

## 14. 本轮最推荐你先做的具体动作

如果只说“下一步最该干什么”，我建议顺序是：

1. 先把实验范围冻结为“固定 6 柱地图 + 4 档 glare 强度”
2. 先补齐 4 个核心方法的配置
3. 先跑 Ours / Fixed / Non-Diff 三个方法
4. 先做 `Success Rate vs Glare Severity`
5. 再做事件对齐的 `power / exposure / gain / local quality`
6. 最后再补 Heuristic AE 和 Ego-Planner

因为前 5 步就已经足够形成一版很强的实验故事线。

## 15. 当前最不建议做的事情

以下几件事现在不建议优先做：

1. 同时扫光照强度和障碍物密度
2. 一开始就做很多“最好看”的轨迹图
3. 只展示平均速度和平均加速度
4. 先做真机，再回头想仿真实验怎么写
5. 在没有固定 checkpoint 选择规则前手工挑最漂亮结果

## 16. 一句话总结

当前这篇 RAL 的实验最优策略不是“尽量多做”，而是：

**用最少的变量，最清楚地证明：在固定几何下，随着逆光增强，本文的可微主动感知比固定相机、不可微主动感知和启发式 AE 更能维持局部深度可靠性并完成导航。**

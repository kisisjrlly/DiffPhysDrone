# 当前窄缝：一次有预算上限的联合学习比较

本协议执行用户批准的下一轮计划，替代此前仅推理、仅诊断的停止限制。
核心仍是任务驱动地共同学习怎么拍、怎么飞，不把旧模型迁移作为研究目标。
本轮不增加训练 seed、不新增场景、不调物理参数。校准仍为 provisional。

## 固定顺序与条件

1. 将连续碰撞改为声明的球体几何；半径 .12m、阈值 .0011m 不变。
   只复算 OD、WD 各16个历史案例，并继续四次误停之后的飞行。
2. 训练条件为 `configs/gray_joint_slit_learning.args`。与当前 joint_slit 相同的
   几何、光照、噪声、曝光积分和动力学；启用已有队列对齐 intent。
3. 任务损失固定为终止距离10、碰撞6、动作平方 .05、jerk .05；速度跟随、
   额外避障、相机平滑均0。6秒到达，不规定每步飞行速度；不提供相机标签。
4. seed42、FP32、batch4；每批四类光照各一次。网络保持200605参数。
   AdamW weight_decay=.01、clip5、余弦最终学习率为初始的 .01。

## 共同起点

平均亮度自动曝光，相机专属参数冻结。直接学习最多2000更新，lr3e-4。
500/1000/1500/2000更新时，以32个固定开发案例检查正常/零图像。
首次同时满足正常>=16、领先零图>=8、左右各>=6、四类光照各>=2时锁定。
零图只遮网络输入，自动曝光仍看真实帧。

若失败，仅允许一次热身：128条教师飞行记录，800更新、batch8、lr1e-3，
之后500任务更新、lr3e-5。继承直接学习最终权重，不重抽初始化。
教师沿用 original 窄缝轨迹、6/4跟踪增益和真实队列。记录全部布局终止前数据，
只教加速度，不教拍摄，不搜索路径。最终仍不通过则停止本轮。

开发seed910000起32个、最终seed920000起64个、示范seed930000起128个。
每四个遍历四种光照，每八个覆盖左右，位置在预先指定的半区均匀采样；
不是按网络结果选样。历史样本只用于碰撞复算。

## 四组后续训练

均从共同起点、相同随机样本和任务目标开始。各1000更新、batch4：

|组|拍摄|成像反向路径|
|---|---|---|
|A|原平均亮度AE|不学习E/G，保留运动成像路径|
|B|学习E/G|保留E/G和运动成像路径|
|C|学习E/G|只切断运动成像路径|
|D|学习E/G|只切断E/G成像路径|

D不是RL，不具备替代的相机任务估计器，不能称公平黑盒学习基线。
飞行/共享lr1e-5、相机专属lr1e-4。B/C/D前200更新线性过渡
`request=(1-alpha)*AE+alpha*network`；之后完全网络控制，无相机模仿损失。
200/400/600/800/1000记录曲线，最终固定1000模型，不择优。
单组360000生成控制步，单独记录有效步、训练耗时、共同准备及示范成本。
曲线另记录首次在预定检查点达到24/32开发成功的步数和实际时间，仅作学习效率
描述，不用于挑选checkpoint；若共同起点已达到该值，各组首次值均为0。

若B最终比A多成功至少4/64，追加且仅追加两个交叉组合：B飞行+AE，
A飞行+B相机；相机的intent来自实际飞行动作和队列。B同时比A、C及两个交叉
组合多至少4次，且四格成功数交互差>=4，才称“配合候选”。
策略交叉可能造成分布失配，不能仅凭交叉下降证明物理必要性。

## 工程与停止

随机过程按更新/控制步/图像或动力学阶段独立索引，记录不消耗随机样本。
中断恢复保存模型、优化器、scheduler和Python/NumPy/CPU/CUDA随机状态。
协议保存源码快照、完整配置、模型及校准hash；缓存不跨不同协议复用。
非有限loss/gradient立即终止。OOM时只允许等效的2+2微批次累积，不降样本预算。
原场景回归、前向不变的梯度切断、队列/噪声/预处理及完整pytest须通过。

最终报告必须区别：共同起点未建立、联合训练破坏技能、学习相机有用但无运动
配合、联合配合候选、全成功无区分、无优势。不自动加时、加seed或换条件。
交付学习曲线、一页中文结论、逐episode证据、可恢复工具；有真实可解释行为才
交付网络演示。提交现有分支、尝试推送，不绕过TLS；完成后Bark通知。

## 执行入口

```bash
PY=/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python
$PY tools/audit_sphere_collision_correction.py --out_dir logs/joint_slit_learning_20261105/collision
$PY tools/run_joint_slit_learning.py --stage lock --out_dir logs/joint_slit_learning_20261105
$PY tools/run_joint_slit_learning.py --stage common --out_dir logs/joint_slit_learning_20261105
```

后续阶段只在相应前置条件通过后执行：

```bash
$PY tools/run_joint_slit_followup.py --stage fallback --out_dir logs/joint_slit_learning_20261105
$PY tools/run_joint_slit_followup.py --stage arms --out_dir logs/joint_slit_learning_20261105
$PY tools/run_joint_slit_followup.py --stage evaluate --out_dir logs/joint_slit_learning_20261105
$PY tools/report_joint_slit_learning.py --out_dir logs/joint_slit_learning_20261105
```

`fallback` 在已有合格共同起点时不重复热身；没有共同起点禁止运行 arms。
还可在没有独立训练进程运行时，用 `tools/execute_joint_slit_round.py --out_dir ...`
串行执行这些条件阶段。不要与正在运行的 common/arm worker 同时启动。
阶段完成并不表示整个计划完成；最终验收、提交推送与完成通知仍须执行。

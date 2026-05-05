> 说明：这份文档以当前仓库里的 `losses.py`、`trainer.py`、`rollout_ops.py`、`config.py` 为准，目标是把“训练时到底算了什么 loss、每项 loss 想约束什么、wandb 里该怎么看”讲清楚。

# DiffPhysDrone Loss 说明

对应代码：

- `trainer.py`
- `losses.py`
- `rollout_ops.py`
- `config.py`

当前项目的总目标不是只让无人机“飞过去”，而是同时优化两件事：

1. 飞行任务本身要完成：跟踪目标、避障、少碰撞、动作不要太抖。
2. `diff_depth` 相机要保持“可用”：不能为了省能耗把深度图调成几乎全黑，也不能为了看清就无限拉长曝光或无限拉高增益。

所以总 loss 可以理解成两大块：

- 物理/控制 loss
- 相机/感知 loss

其中有一个很容易被忽略、但对“接近终点时是否还愿意继续推进”影响很大的参数是：

- `--loss_v_window`

它控制 `loss_v` 在时间维度上看多长一段速度历史。窗口越大，速度监督越平滑，但也越“慢半拍”；窗口越小，速度监督越灵敏，更容易在小地图和终点附近继续推进。

如果后面打开 teacher-student 训练，还会再加一块 distillation loss。

---

## 1. Loss 在训练流程里的位置

当前 student 训练有两条路径：

- `TBPTT`：在 `student_rollout()` 里边 rollout 边按 chunk 反传。
- `Full-BPTT`：先把整段 rollout 跑完，再在 `full_bptt_losses()` 里统一算 loss。

无论哪条路径，真正的 loss 计算核心都集中在这几个函数里：

- `compute_physics_losses(...)`
- `compute_camera_losses(...)`
- `compute_distill_loss(...)`
- `aggregate_loss(...)`

所以看懂这几个函数，就基本看懂了整个训练目标。

---

## 2. 先看数据流：loss 用到的量是从哪里来的

这是当前 `diff_depth` 分支最重要的一条链：

1. 在 `student_rollout()` 每个 step 里，先调用 `render_sensors(...)`
2. 它进一步调用 `env.render_diff_depth(power, exposure, gain)`
3. 环境返回：
   - `depth_obs`：带噪声的深度图
   - `depth_quality`：确定性的质量图，主要用于可微监督
4. 训练里不会直接用 noisy depth 算 fill，而是优先用 `depth_quality`
5. 通过 `compute_depth_fill_rate(...)` 计算：
   - `depth_fill_history`：hard fill
   - `depth_fill_soft_history`：soft fill
6. `compute_camera_losses(...)` 使用 `depth_fill_soft_history`
7. 进一步算出 `loss_diff_depth_fill`
8. 最后 `aggregate_loss(...)` 用 `coef_diff_depth_fill` 把它并入总 loss

这条链非常关键，因为它解释了：

- 为什么 `diff_depth_min_fill_rate` 会影响训练
- 为什么它不是渲染器参数，而是一个 loss threshold
- 为什么它会反向影响 `power / exposure / gain`

---

## 3. `full_bptt_losses()` 里用到的主要张量

下面这些量是在 rollout 过程中逐步缓存，最后在 `full_bptt_losses()` 中堆叠出来的：

- `p_history`: `[T, B, 3]`，无人机位置
- `v_history`: `[T, B, 3]`，无人机速度
- `target_v_history`: `[T, B, 3]`，目标速度
- `vec_to_pt_history`: `[T, B, 3]`，到最近障碍点的向量
- `v_preds`: `[T, B, 3]`，网络预测的速度辅助头输出
- `act_history`: `[T, B, 3]`，控制动作
- `cam_params_history`: `[T, B, 3]`，相机头原始输出历史，对应 `power / exposure / gain`
- `power_history`: `[T, B]`
- `exposure_history`: `[T, B]`
- `gain_history`: `[T, B]`
- `speed_for_depth_history`: `[T, B]`，用于和曝光组合出 blur proxy
- `depth_fill_history`: `[T]` 或 `[T, ...]`，hard fill
- `depth_fill_soft_history`: `[T]` 或 `[T, ...]`，soft fill

其中：

- `T = timesteps`
- `B = batch_size`

---

## 4. `diff_depth_min_fill_rate` 到底有什么作用

这是你这次重点问的点，我单独展开说。

### 4.1 它不是“fill rate 的统计值”，而是一个最低目标阈值

配置定义在 [config.py](/home/zhaoguodong/work/code/DiffPhysDrone/config.py)：

- `--diff_depth_min_fill_rate`

默认 help 写的是：

- 深度 fill rate 的最低目标阈值；低于它时会触发 blackout penalty

也就是说，它不是渲染器里“真实 fill 有多少”的物理参数，而是训练目标里的一根线：

- 高于这根线：不罚
- 低于这根线：开始罚

### 4.2 代码里是怎么用的

在 [losses.py](/home/zhaoguodong/work/code/DiffPhysDrone/losses.py) 的 `compute_camera_losses(...)` 里：

```python
fill_gap = F.relu(float(min_fill_rate) - fill_rate_seq)
result['loss_diff_depth_fill'] = fill_gap.pow(2).mean()
```

等价数学式：

```text
fill_gap_t = max(0, min_fill_rate - fill_rate_t)
loss_diff_depth_fill = mean(fill_gap_t^2)
```

含义很直接：

- 如果 `fill_rate_t >= diff_depth_min_fill_rate`，这一帧的 `fill_gap_t = 0`
- 如果 `fill_rate_t < diff_depth_min_fill_rate`，缺口越大，惩罚越大
- 惩罚是平方，所以掉得很厉害时会被明显拉回来

### 4.3 它在当前主配置里意味着什么

在 [slit_active_sensing.args](/home/zhaoguodong/work/code/DiffPhysDrone/configs/slit_active_sensing.args) 里，你现在设的是：

- `--coef_diff_depth_fill 5.0`
- `--diff_depth_min_fill_rate 0.25`

这意味着：

- 训练时希望 soft fill 平均至少维持在 `0.25`
- 一旦低于 `0.25`，`loss_diff_depth_fill` 会被 `5.0` 这个系数放大

所以这项 loss 现在在你的主配置里依然是明确约束，但强度比之前更偏“保底”，不是极强硬压制。

---

## 4.7 `loss_v_window` 到底在做什么

对应代码在 [losses.py](/home/zhaoguodong/work/code/DiffPhysDrone/losses.py) 的 `velocity_tracking_loss(...)`：

```python
v_avg = (v_cum[win:] - v_cum[:-win]) / win
tv_ref = tv_hist[win:]
delta_v = torch.norm(v_avg[:m] - tv_ref[:m], 2, -1)
```

它不是直接比较“当前速度”和“当前目标速度”，而是：

1. 先取最近 `win` 步真实速度的平均值
2. 再和对应时刻的目标速度比较

如果控制频率是 `15Hz`，那么：

- `loss_v_window = 30` 约等于看过去 `2.0s`
- `loss_v_window = 15` 约等于看过去 `1.0s`
- `loss_v_window = 12` 约等于看过去 `0.8s`

它的作用是让速度损失更平滑，减少策略追逐瞬时噪声。

但窗口太大时也会带来副作用：

- 接近终点时反应变慢
- 刚进入 `sun_glare` 区时，loss 还在看前面正常飞行阶段的平均速度
- 更容易出现“终点前减速后停住”的保守解

所以当前主配置把它设为 `12`，是为了让小地图、固定场景下的目标推进更灵敏。

### 4.4 它和 `depth_min_valid` 的区别

这两个参数很容易混：

- `depth_min_valid`
- `diff_depth_min_fill_rate`

它们不是一回事。

`depth_min_valid` 决定的是：

- 单个像素深度值多大才算“有效”

在 [rollout_ops.py](/home/zhaoguodong/work/code/DiffPhysDrone/rollout_ops.py) 里：

```python
return (depth_obs >= threshold).float().mean()
```

也就是先按 `depth_min_valid` 判像素有效性，再去统计比例。

`diff_depth_min_fill_rate` 决定的是：

- 整张图统计出来的 fill rate 最低希望保到多少

可以把它理解成两级门槛：

1. 像素级门槛：`depth_min_valid`
2. 图像级门槛：`diff_depth_min_fill_rate`

### 4.5 为什么它对训练很重要

如果没有这项约束，策略可能会学到一种“看起来 loss 不错，但其实相机几乎失明”的投机行为：

- 把 `power` 压得很低，省掉能耗损失
- 把 `exposure` 或 `gain` 调到某种局部最优
- 飞控主损失在简单地图上依然能凑合过关
- 但深度图已经大面积空洞化，真实迁移会很差

`loss_diff_depth_fill` 的作用，就是禁止这种“相机几乎失效但训练还能混过去”的策略。

### 4.6 它调大还是调小，会发生什么

如果你提高 `diff_depth_min_fill_rate`：

- 策略会更努力维持更多有效深度
- 常见结果是更倾向于提高 `power`、提高 `gain`、或者避免某些极端曝光选择
- 但如果阈值太高，可能会把策略逼得过于保守

如果你降低 `diff_depth_min_fill_rate`：

- 策略更容易容忍大面积空洞
- 相机动作自由度更大
- 但“可微感知策略”的卖点会变弱，因为系统不再被强迫维护感知可用性

---

## 5. 为什么训练里用的是 `soft fill`，不是 `hard fill`

在 [trainer.py](/home/zhaoguodong/work/code/DiffPhysDrone/trainer.py) 中：

```python
fill_rate_t = compute_depth_fill_rate(fill_src, min_valid_depth=args.depth_min_valid)
fill_rate_soft_t = compute_depth_fill_rate(
    fill_src,
    min_valid_depth=args.depth_min_valid,
    softness=diff_depth_fill_softness(args.depth_min_valid),
)
```

`hard fill` 相当于：

- 大于阈值记 1
- 小于阈值记 0

这种统计对阈值边缘不够平滑。

`soft fill` 则用 sigmoid 做平滑过渡：

```python
torch.sigmoid((depth_obs - threshold) / softness).mean()
```

这样做的好处是：

- 对阈值附近的像素更平滑
- 梯度更稳定
- 更适合通过 `power / exposure / gain` 反传到相机头

所以当前真正拿来监督 `loss_diff_depth_fill` 的，是 `depth_fill_soft_history`，不是硬阈值统计。

---

## 6. 当前项目的总 loss 长什么样

在 [losses.py](/home/zhaoguodong/work/code/DiffPhysDrone/losses.py) 的 `aggregate_loss(...)` 里，总 loss 由这些项线性加权组成：

```text
L_total
= coef_v * loss_v
+ coef_obj_avoidance * loss_obj_avoidance
+ coef_d_acc * loss_d_acc
+ coef_d_jerk * loss_d_jerk
+ coef_collide * loss_collide
+ coef_cam_smooth * loss_cam_smooth
+ coef_diff_depth_power * loss_diff_depth_power
+ coef_diff_depth_blur * loss_diff_depth_blur
+ coef_diff_depth_noise * loss_diff_depth_noise
+ coef_diff_depth_fill * loss_diff_depth_fill
```

如果打开 teacher-student：

```text
L_total = distill_coef_iter * loss_distill + student_physics_coef * L_base
```

当前你强调“不启用 teacher-student / tbptt / dmpc”，那么主要关心的就是上面这组基础 loss。

说明：

- 当前主线代码已经不再把 `loss_v_pred / loss_ground_affinity / loss_cam_range / loss_tilt / loss_sun_glare_local_quality` 计入训练总损失。
- 其中 `sun_glare` 相关的局部质量量现在只保留为诊断统计，不再作为 reward/loss shaping。

---

## 7. 每一项 loss 的计算与含义

下面按当前代码逐项解释。

## 7.1 `loss_v_pred`

位置：

- [trainer.py](/home/zhaoguodong/work/code/DiffPhysDrone/trainer.py)

公式：

```text
loss_v_pred = MSE(v_preds, stopgrad(v_history))
```

作用：

- 这是一个辅助任务
- 让网络额外学会预测速度
- 往往能让主干表征更稳定

它不是主任务本身，但对训练稳定性通常有帮助。

---

## 7.2 物理/控制组 loss

这些在 [losses.py](/home/zhaoguodong/work/code/DiffPhysDrone/losses.py) 的 `compute_physics_losses(...)` 里。

### 7.2.1 `loss_v`

它不是逐帧速度误差，而是先做时间窗口平均，再和目标速度比较。

近似写法：

```text
v_avg[i] = mean(v[i:i+win])
loss_v = SmoothL1(||v_avg - target_v||, 0)
```

作用：

- 约束整体飞行趋势
- 避免只在单帧上抖动式对齐目标速度

---

### 7.2.2 `loss_d_acc`

公式：

```text
loss_d_acc = mean(sum(act^2))
```

作用：

- 惩罚动作过大
- 让控制不要太猛

如果这项太大，策略会偏保守。

---

### 7.2.3 `loss_d_jerk`

先看动作变化量，再对变化量平方求均值。

公式：

```text
jerk[t] = 15 * (act[t] - act[t-1])
loss_d_jerk = mean(sum(jerk^2))
```

作用：

- 惩罚高频控制抖动
- 让轨迹更平滑

---

### 7.2.4 `loss_obj_avoidance`

这是“还没撞上，但已经太接近障碍物”的连续惩罚。

核心量：

```text
dist = ||vec_to_pt|| - margin
```

然后通过 barrier 函数做惩罚：

```text
barrier(dist) = relu(1 - dist)^2
```

作用：

- 鼓励提前绕开障碍
- 保留安全裕度

这项通常比 `loss_collide` 更早介入。

---

### 7.2.5 `loss_collide`

这是“已经非常接近甚至穿透”时的更硬惩罚。

形式上类似：

```text
loss_collide = mean(v_to * softplus(-32 * dist))
```

作用：

- 对真正的碰撞风险施加强约束
- 防止只靠 `loss_obj_avoidance` 还不够硬

---

### 7.2.6 `loss_ground_affinity`

公式近似：

```text
loss_ground_affinity = mean(relu(p_z)^2)
```

它是一个偏好项，不是你当前 diff-depth 论文主线里的核心项。

如果 `coef_ground_affinity = 0`，它就相当于不生效。

---

## 7.3 相机/感知组 loss

这些在 [losses.py](/home/zhaoguodong/work/code/DiffPhysDrone/losses.py) 的 `compute_camera_losses(...)` 里。

输入主要包括：

- `cam_hist`
- `power_seq`
- `exposure_seq`
- `gain_seq`
- `speed_seq`
- `fill_rate_seq`

### 7.3.1 `loss_cam_smooth`

公式：

```text
loss_cam_smooth = mean((cam_t - cam_{t-1})^2)
```

作用：

- 不希望相机寄存器每帧乱跳
- 让 `power / exposure / gain` 在时间上更平滑

如果你在 rerun 里看到相机参数锯齿状跳变，这项通常需要关注。

---

### 7.3.2 `loss_diff_depth_power`

公式：

```text
loss_diff_depth_power = mean(relu(power - cam_power_baseline)^2)
```

这项是现在唯一保留的 power 成本项，语义就是“低功率是默认状态，高功率需要付出代价”：

- 鼓励不要长期高功率
- 不惩罚低于 `cam_power_baseline` 的 power
- 只有当传感器质量、fill rate 或任务收益足够大时，策略才值得把 power 推高

---

### 7.3.3 `loss_diff_depth_blur`

先把归一化曝光映射成“物理语义上的有效曝光时间”：

```python
exp_phys = diff_depth_exposure_to_time(exposure_seq)
```

再计算：

```text
loss_diff_depth_blur = mean((speed * exp_phys)^2)
```

作用：

- 惩罚“高速飞行 + 长曝光”这一危险组合
- 对应你论文里提到的 motion blur 风险

这项很符合直觉：

- 速度越快
- 曝光越长
- 拖影越严重

---

### 7.3.4 `loss_diff_depth_noise`

公式：

```text
loss_diff_depth_noise = mean(gain^2)
```

作用：

- 惩罚高增益
- 避免靠无限增益去硬抬亮度

因为增益太高时，虽然局部 fill 可能上来了，但噪声也会明显上来。

---

### 7.3.5 `loss_diff_depth_fill`

这是当前 diff-depth 里最重要的“感知可用性保护项”之一。

公式：

```text
fill_gap = max(0, diff_depth_min_fill_rate - fill_rate)
loss_diff_depth_fill = mean(fill_gap^2)
```

作用：

- 防止深度图大面积空洞
- 防止策略把相机调到接近失明
- 逼迫策略在能耗、模糊、噪声之外，还要守住最基本的深度可用性

对于你现在这个项目，这项 loss 的意义非常大，因为论文卖点不是“飞控本身”，而是“相机参数在闭环里真的学到了感知权衡”。

---

## 8. 当前 diff-depth 路径里，fill 是怎么统计出来的

这部分很重要，不然你会不知道 `loss_diff_depth_fill` 到底在罚什么。

### 8.1 `compute_depth_fill_rate(...)`

在 [rollout_ops.py](/home/zhaoguodong/work/code/DiffPhysDrone/rollout_ops.py) 中：

```python
def compute_depth_fill_rate(depth_obs, min_valid_depth: float = 0.3, softness=None):
    threshold = float(min_valid_depth)
    if softness is None:
        return (depth_obs >= threshold).float().mean()
    return torch.sigmoid((depth_obs - threshold) / softness).mean()
```

这说明 fill 本质上是在统计：

- “有多少像素看起来像是有效深度”

### 8.2 当前训练优先用的是 `depth_quality`

在 [trainer.py](/home/zhaoguodong/work/code/DiffPhysDrone/trainer.py) 中：

```python
fill_src = depth_quality if depth_quality is not None else depth_obs.detach()
```

意思是：

- 如果环境提供了确定性的 `quality` 图
- 就用它来算 fill

为什么这样做：

- `depth_obs` 含随机噪声
- 直接拿 noisy depth 算 fill，会让梯度更脏
- `depth_quality` 更适合作为稳定的可微信号

### 8.3 `fill_rate` 和 `fill_rate_soft` 的区别

- `fill_rate`：硬阈值统计，更像“纯可视化/统计指标”
- `fill_rate_soft`：平滑统计，更像“训练监督指标”

当前 `compute_camera_losses(...)` 用的是：

- `depth_fill_soft_history`

也就是说，真正喂给 `loss_diff_depth_fill` 的是 soft 版本。

---

## 9. teacher-student 关闭时，你当前真正关心哪些 loss

你现在的主线是不启用：

- `use_dmpc`
- `policy_output_intent`
- `tbptt`
- `teacher-student`

那么目前最核心的是这几项：

- `loss_v`
- `loss_obj_avoidance`
- `loss_collide`
- `loss_d_acc`
- `loss_d_jerk`
- `loss_cam_smooth`
- `loss_diff_depth_power`
- `loss_diff_depth_blur`
- `loss_diff_depth_noise`
- `loss_diff_depth_fill`

你可以把它们理解成：

- 飞控部分保证“能飞、别撞”
- 相机部分保证“别瞎、别糊、别太吵、别太耗电”

---

## 10. 每项 loss 的梯度主要会推到哪里

虽然很多 loss 最终都会经过动力学和感知闭环互相耦合，但直接看可以这么理解：

| loss | 主要直接作用对象 |
|---|---|
| `loss_v` | 动作头 `fc` 和共享主干 |
| `loss_obj_avoidance` | 动作头 `fc` 和共享主干 |
| `loss_collide` | 动作头 `fc` 和共享主干 |
| `loss_d_acc` | 动作头 `fc` 和共享主干 |
| `loss_d_jerk` | 动作头 `fc` 和共享主干 |
| `loss_v_pred` | 速度预测相关分支和共享主干 |
| `loss_cam_smooth` | 相机头 `fc_cam` 和共享主干 |
| `loss_diff_depth_power` | 相机头 `fc_cam` 和共享主干 |
| `loss_diff_depth_blur` | 相机头 `fc_cam` 和共享主干 |
| `loss_diff_depth_noise` | 相机头 `fc_cam` 和共享主干 |
| `loss_diff_depth_fill` | 相机头 `fc_cam` 和共享主干 |

但要注意，`fc_cam` 不只是被“相机 loss”监督：

1. `fc_cam` 输出 `power / exposure / gain`
2. 它们改变下一步 `depth_obs / quality`
3. 深度输入又影响策略动作
4. 所以飞行主损失也会间接反传到相机头

这正是你想要的“端到端耦合”的核心。

---

## 11. wandb 里三套 loss 统计到底怎么区分

这个也是当前最容易看混的地方。

### 11.1 `loss_raw/...`

这是未经权重系数放大的原始 loss 数值。

例子：

- `loss_raw/loss_diff_depth_fill`

表示：

- `compute_camera_losses(...)` 直接算出来的那一项原始值
- 还没乘 `coef_diff_depth_fill`

这类指标只有在 `wandb_log_raw_loss_terms=true` 时才会额外记录。

### 11.2 `loss_contrib/...`

这是“真正对总 loss 贡献了多少”的值。

也就是近似：

```text
loss_contrib/x = coef_x * loss_raw/x
```

如果 teacher-student 打开，还会额外乘 `student_physics_coef`。

所以它更适合回答这个问题：

- “这一项在当前总损失里到底有多重”

### 11.3 `loss_share/...`

这是贡献占比，也就是：

```text
loss_share/x = loss_contrib/x / total_contrib
```

它回答的是另一个问题：

- “当前总 loss 里，哪一项占比最大”

### 11.4 三者该怎么看

如果你想知道：

- 这项 loss 本身大不大：看 `loss_raw`
- 它对总目标的实际影响大不大：看 `loss_contrib`
- 它在所有 loss 中占比如何：看 `loss_share`

---

## 12. 如何读 `loss_share`

这里给你一个“看图解释表”。

如果 `loss_share/diff_depth_fill` 很高，通常说明：

- 当前策略经常把 fill rate 打到阈值以下
- 相机可用性正在成为训练主矛盾
- 可能需要检查：
  - 场景是否太难
  - `diff_depth_min_fill_rate` 是否过高
  - `ambient_add / active_drop / quality_penalty` 是否过激

如果 `loss_share/diff_depth_blur` 很高，通常说明：

- 策略在高速时仍然倾向长曝光
- 相机没有学会“速度高时缩短曝光”

如果 `loss_share/diff_depth_noise` 很高，通常说明：

- 策略经常把 gain 拉得太高
- 说明它更依赖增益，而不是其他成像手段

如果 `loss_share/diff_depth_power` 很高，通常说明：

- 策略经常把 power 推到 `cam_power_baseline` 以上很多
- 高功率能耗已经成了主要矛盾

如果 `loss_share/collide` 很高，通常说明：

- 避障本身还没学好
- 这时先别过度纠结相机细节，要先保证轨迹安全

---

## 13. 针对你当前主配置，`diff_depth_min_fill_rate` 应该怎么理解

你现在主配置里：

- `coef_diff_depth_fill = 30.0`
- `diff_depth_min_fill_rate = 0.35`

这表示当前训练明确在说：

- “我希望 soft fill 至少到 35%”
- “一旦低于这个水平，我会非常认真地罚你”

对你的论文方向来说，这个设置是合理的，因为你不是在做纯节能控制，而是在做“感知寄存器要服务于任务”的闭环训练。

但也要注意：

- 它不是越高越好
- 如果场景本身太难，而阈值又太高，策略可能一直被 fill 项压着打

所以后面调参时，最好把这两个一起看：

- `loss_share/diff_depth_fill`
- `diff_depth_fill_rate_soft`

如果你看到：

- `diff_depth_fill_rate_soft` 长期明显低于 0.35
- 且 `loss_share/diff_depth_fill` 长期很高

那就说明 fill 门槛已经成了主导约束。

---

## 14. 一个最实用的调参思路

如果你后面再看 wandb，不知道先看什么，我建议按这个顺序：

1. 先看飞控是否成立：
   - `loss_share/collide`
   - `loss_share/obj_avoidance`
   - `success_rate`
   - `collision_rate`
2. 再看相机是否可用：
   - `diff_depth_fill_rate`
   - `diff_depth_fill_rate_soft`
   - `loss_share/diff_depth_fill`
3. 再看相机是靠什么手段在保住 fill：
   - `cam/power_mean`
   - `cam/exposure_mean`
   - `cam/gain_mean`
   - `loss_share/diff_depth_power`
   - `loss_share/diff_depth_blur`
   - `loss_share/diff_depth_noise`
4. 最后再看是否出现不自然的寄存器行为：
   - `loss_share/cam_smooth`
   - `cam/power_mean`
   - `cam/exposure_mean`
   - `cam/gain_mean`

这样你就不会只盯着某一项 loss，结果把整体闭环判断错了。

---

## 15. 一句话总结每项 loss

如果你想快速记忆，可以记成下面这张表。

| loss | 一句话含义 |
|---|---|
| `loss_v` | 飞得是否朝目标速度靠拢 |
| `loss_obj_avoidance` | 是否给障碍物留足安全距离 |
| `loss_collide` | 是否发生硬碰撞/强风险接近 |
| `loss_d_acc` | 动作是否过猛 |
| `loss_d_jerk` | 动作是否过抖 |
| `loss_v_pred` | 网络是否学会速度辅助表征 |
| `loss_cam_smooth` | 相机参数是否帧间跳变太大 |
| `loss_diff_depth_power` | 是否长期使用高功率激光 |
| `loss_diff_depth_blur` | 是否在高速下还长曝光 |
| `loss_diff_depth_noise` | 是否过度依赖高增益 |
| `loss_diff_depth_fill` | 深度图是否空洞太多、接近失明 |

---

## 16. 对你这次问题的直接回答

`diff_depth_min_fill_rate` 的作用就是：

- 为深度图 fill rate 设定一个最低目标线
- 当 `fill_rate` 低于这条线时，触发 `loss_diff_depth_fill`
- 低得越多，惩罚越大

它的本质是：

- 防止策略把 `diff_depth` 相机调成“虽然还能凑合飞，但深度图已经几乎不能用”的状态

如果只记一句话，你就记这个：

- `depth_min_valid` 管单像素算不算有效，`diff_depth_min_fill_rate` 管整张图最低要保住多少有效比例。

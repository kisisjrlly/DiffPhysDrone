# DiffPhysDrone 真机部署详尽指南

## 文档目标

这份文档的目标不是“泛泛介绍一下真机部署”，而是尽可能把你当前这个项目，从：

- 已经训练好一个 `diff_depth` 策略
- 准备在室内、带动捕、带 D455、带 PX4 的四旋翼上运行

推进到：

- 你知道整条系统链路应该如何搭起来
- 你知道 `tools/run_real_policy.py` 到底在做什么
- 你知道部署前必须检查哪些事情
- 你知道第一天不应该直接放飞，而应该按什么顺序逐步验证
- 你知道出问题时先查哪一段

这份文档特别围绕当前仓库中的真实部署脚本 [run_real_policy.py](/home/zhaoguodong/work/code/DiffPhysDrone/tools/run_real_policy.py) 编写。

## 先说最重要的结论

当前仓库里的真机链路，推荐架构是：

1. 动捕系统负责估计无人机在室内空间中的姿态和位置。
2. 动捕数据先送给 PX4 的状态估计器，而不是直接送给策略脚本。
3. PX4 输出融合后的 `LOCAL_POSITION_NED` 和 `ATTITUDE`。
4. [run_real_policy.py](/home/zhaoguodong/work/code/DiffPhysDrone/tools/run_real_policy.py) 通过 `pymavlink` 读取 PX4 遥测状态。
5. 同时脚本读取 D455 深度图。
6. 脚本把“PX4 状态 + D455 深度图”送进训练好的策略网络。
7. 策略输出：
   - 飞行动作
   - 相机主动感知参数 `power / exposure / gain`
8. 脚本把相机参数写回 D455，把飞行动作转换成 PX4 Offboard setpoint。

也就是说，当前脚本读的是：

- 真机深度：来自 D455
- 真机状态：来自 PX4

而不是：

- 直接从动捕服务器读状态
- 直接从 D455 做 SLAM / VIO 算状态

这是当前版本最稳妥、也最符合 PX4 体系的方法。

---

## 适用范围

这份文档针对的是当前仓库主线，也就是：

- `diff_depth` 单主线
- 策略输入是深度图
- 相机控制量是 `power / exposure / gain`
- 真机脚本是 [run_real_policy.py](/home/zhaoguodong/work/code/DiffPhysDrone/tools/run_real_policy.py)

当前脚本明确假设：

- 使用的是 **PX4 Offboard**
- 室内有可靠 **本地位置估计**
- 位置估计来源可以是 **动捕融合进 PX4**
- 使用 **Intel RealSense D455**
- 使用的是 **direct-action checkpoint**

### 当前不支持的情况

下面这些情况，当前脚本不能直接处理，或者至少不是“直接可用”：

1. `use_dmpc=True`
2. `policy_output_intent=True`
3. 没有本地位置估计，只靠 D455 深度直接飞
4. 动捕数据没有进 PX4，而你又想让脚本自己去读 Vicon / OptiTrack / VRPN / ROS topic
5. 室外 GPS 主导定位场景

脚本里已经对第 1、2 条做了显式保护：如果你拿了错误类型的 checkpoint，它会直接拒绝运行，而不是静默跑错。

---

## 整体系统架构

推荐的真实系统拓扑是：

```text
Motion Capture System
    -> ROS/MAVROS or other bridge
    -> PX4 external vision / odometry input
    -> PX4 EKF2
    -> MAVLink LOCAL_POSITION_NED + ATTITUDE
    -> tools/run_real_policy.py

D455
    -> tools/run_real_policy.py

tools/run_real_policy.py
    -> policy forward
    -> D455 laser/exposure/gain write-back
    -> PX4 SET_POSITION_TARGET_LOCAL_NED
```

### 为什么推荐“动捕先送 PX4，再由脚本读 PX4”

这是因为飞控、状态估计、failsafe、offboard 判断，都是围绕 PX4 自己的状态来工作的。

如果你让脚本自己直接读动捕，而 PX4 又在用另一套状态源，就会出现下面这些非常危险的问题：

- 策略认为自己在 A 点
- PX4 认为自己在 B 点
- 脚本发送的是基于 A 点计算出来的控制命令
- PX4 却以 B 点为当前状态去执行

这种“伴随计算机状态”和“飞控状态”不一致的问题，在真机上非常容易导致漂移、抖动、甚至直接炸机。

所以最推荐的原则是：

- **所有控制相关模块，都尽量使用 PX4 自己认可的本地状态**

---

## 官方文档建议

下面这几篇 PX4 官方文档非常重要，建议你在实际部署时同时开着看：

1. Offboard Mode
   - https://docs.px4.io/main/en/flight_modes/offboard.html
2. Using Vision or Motion Capture Systems for Position Estimation
   - https://docs.px4.io/main/en/ros/external_position_estimation.html
3. Motion Capture
   - https://docs.px4.io/main/en/computer_vision/motion_capture.html

和当前项目最相关的几个官方结论是：

1. PX4 Offboard 需要持续的外部 setpoint 流，低于约 `2Hz` 会退出 Offboard。
2. Offboard 模式要求飞控已经有可用的位置或姿态估计。
3. 对 EKF2 来说，动捕 / 外部视觉数据推荐通过 `VISION_POSITION_ESTIMATE` 或 `ODOMETRY` 路径送入。
4. 官方文档特别指出：如果使用 EKF2，`ODOMETRY` 是更强的路径，因为它还能带线速度信息。
5. 外部视觉 / 动捕消息推荐频率大约在 `30Hz ~ 50Hz`。

这和当前脚本完全一致：

- 当前脚本自己不估计位姿
- 它依赖 PX4 已经有本地状态
- 它通过 `SET_POSITION_TARGET_LOCAL_NED` 发送 Offboard 控制

---

## 当前脚本到底依赖什么状态源

这是你最容易混淆、也是最重要的一点。

### D455 提供什么

当前脚本只从 D455 获取：

- 深度图 `depth_frame`

对应代码在：

- [tools/run_real_policy.py](/home/zhaoguodong/work/code/DiffPhysDrone/tools/run_real_policy.py)
  - `D455Runtime.start()`
  - `D455Runtime.read_depth()`

它不会从 D455 获取：

- 位置
- 速度
- 姿态
- VIO

### PX4 提供什么

当前脚本从 PX4 获取：

- `LOCAL_POSITION_NED`
  - 位置 `x, y, z`
  - 速度 `vx, vy, vz`
- `ATTITUDE`
  - `roll, pitch, yaw`

对应代码在：

- `Px4Bridge.request_default_streams()`
- `Px4Bridge.poll()`
- `Px4Bridge.current_state()`

### 动捕系统提供什么

在推荐链路里，动捕系统不是直接供给脚本，而是供给 PX4 的状态估计器。

也就是说：

- 动捕系统是“真实状态来源”
- PX4 是“统一状态融合中心”
- `run_real_policy.py` 是“PX4 状态消费者”

---

## 真机部署前的最小前提条件

如果下面任何一条没满足，不要直接运行策略。

### 硬件前提

1. 四旋翼硬件平台已经能由 PX4 稳定飞行
2. 室内动捕系统已经稳定工作
3. D455 已固定在机体上，并供电稳定
4. Orin / Jetson 与 PX4 的串口或 UDP 通信稳定
5. Orin / Jetson 与 D455 通过 USB3 稳定连接
6. 飞机已做基本桨叶平衡、重心配置、螺旋桨保护

### 软件前提

1. PX4 固件已能接收外部位姿
2. PX4 在室内已经能获得稳定的本地位置
3. 伴随计算机上安装了项目依赖环境
4. 至少可以导入：
   - `torch`
   - `numpy`
   - `cv2`
   - `pyrealsense2`
   - `pymavlink`
5. 当前项目 checkpoint 可以正常加载

### 任务前提

1. 你拿去部署的 checkpoint 对应的是当前真机脚本支持的模式
2. 训练配置与真机部署想做的场景相符
3. D455 参数语义已经做过至少一次基本对齐
4. 真机环境的障碍和目标任务，不要比仿真更复杂很多

---

## 建议的真机首飞推进顺序

不要一上来就装桨、Offboard、起飞、穿障。

推荐按下面顺序推进。

### 阶段 A：纯状态链路验证

目标：验证 PX4 已经真正“看见”动捕。

要达到的现象：

1. 在 QGroundControl 或 PX4 shell 中可以看到稳定的本地位置
2. 手动拿着无人机移动，PX4 的本地位置会同步变化
3. 无人机静止时，本地速度接近 0
4. 姿态角方向合理，没有坐标系翻转

如果这一步不成立，后面全部不用做。

### 阶段 B：纯相机链路验证

目标：验证 D455 在真机上工作正常。

先运行：

```bash
python3 tools/test_d455_depth.py
```

应当确认：

1. D455 能启动
2. 深度图持续刷新
3. 手动关闭自动曝光成功
4. 写入 `exposure / gain / laser_power` 不报错

### 阶段 C：脚本只读不控

目标：验证 [run_real_policy.py](/home/zhaoguodong/work/code/DiffPhysDrone/tools/run_real_policy.py) 能同时读 PX4 和 D455，但不让飞机起飞。

做法：

1. 去掉螺旋桨，或至少不 arm
2. 运行脚本但不传 `--arm`
3. 看它是否能：
   - 成功加载 checkpoint
   - 成功读取 PX4 状态
   - 成功读取 D455 深度
   - 成功打印 `step / goal_dist / fill / power / exp / gain`
   - 成功生成 `artifacts/real_policy_runs/.../trace.csv`

这一步通过，说明：

- 状态链路通
- 相机链路通
- 策略推理通
- 参数写回通

### 阶段 D：有桨但不离地

目标：验证 Offboard 和 arm 过程安全。

做法：

1. 空旷环境
2. 低高度保护
3. 人员远离桨盘
4. 先做 `offboard warmup`
5. 再 arm
6. 不给目标位移或者用 very small goal

这一步观察：

- 飞机会不会一进 Offboard 就抽动
- 坐标系方向是不是对的
- 是否出现横向突然大速度

### 阶段 E：自动起飞到悬停

目标：只验证起飞和悬停，不验证感知策略。

做法：

1. 用 `--auto-takeoff`
2. 目标点尽量近
3. `finish-action` 先用 `hold`
4. 只看能否稳定升到目标高度并保持

### 阶段 F：短距离直线

目标：先做一个几乎没有障碍的短程任务。

### 阶段 G：固定障碍、低风险 sun_glare 场景

目标：再验证你真正关心的主动感知行为。

---

## 动捕系统如何接入 PX4

这是整个室内部署里最关键的一段。

### 推荐做法

推荐使用：

- 动捕系统
  -> ROS / MAVROS / 自定义桥
  -> PX4 external vision / odometry
  -> EKF2

### 如果你用的是 EKF2

优先建议参考 PX4 官方 external position estimation 文档。

对 EKF2 来说，最推荐的不是 `ATT_POS_MOCAP`，而是：

1. `VISION_POSITION_ESTIMATE`
2. `ODOMETRY`

其中：

- `ODOMETRY` 更强，因为它还能带线速度
- 如果你的桥接链路能发 `ODOMETRY`，优先走这一条

### 如果你通过 MAVROS 接入

官方文档里常见的链路是：

1. `/mavros/vision_pose/pose`
2. `/mavros/odometry/out`

如果你用的是 Vicon / OptiTrack / VRPN，通常会先得到一个 ROS pose topic，然后再 remap 到 MAVROS 对应输入。

### 消息频率要求

按 PX4 官方建议，外部视觉 / 动捕消息最好在：

- `30Hz ~ 50Hz`

如果频率太低，EKF2 可能不会稳定融合。

### 版本相关参数

这里我故意不在文档里写死某一版 PX4 的所有参数值，因为：

1. 你实际使用的 PX4 版本可能不同
2. EKF2 外部视觉参数在不同版本里有细节差异
3. 直接写死参数很容易误导你

但是你必须确认两件事：

1. PX4 确实在融合外部位姿
2. `LOCAL_POSITION_NED` 已经稳定可用

你可以把“PX4 是否已经有本地定位”看成整个部署的硬门槛。

---

## D455 在真机部署中的作用

### D455 提供给策略的内容

策略实际看到的是：

- 深度图

而且这张深度图会按训练时的策略输入尺度进行缩放。

### D455 上被脚本控制的内容

当前脚本会控制：

1. `laser_power`
2. `exposure`
3. `gain`

也就是你训练里定义的三个主动感知通道。

### D455 自动曝光

当前脚本在启动 D455 时，会主动关闭自动曝光。

这是必须的，因为：

- 你的策略自己要控制 `exposure / gain`
- 如果 D455 固件还在开自动曝光，它会和策略抢控制权

### D455 参数写回逻辑

脚本会把策略输出的 `[0,1]` 归一化控制量，映射回 D455 的真实寄存器范围。

大致逻辑是：

1. `power01 -> laser_power`
2. `exposure01 -> exposure_us`
3. `gain01 -> gain_value`

其中：

- 曝光映射会用项目中的 `CameraSemantics`
- 曝光时间再乘上 `d455_exposure_divisor_us`
- gain 会复用项目中的 `CameraSemantics.iso_to_gain()` 语义曲线，再归一化映射到 D455 工作区间

这意味着当前脚本不是简单线性复现训练参数，而是尽量保持当前项目的语义一致性。

---

## 真实部署前建议先运行的工具

### 1. 联机深度检查

```bash
python3 tools/test_d455_depth.py
```

作用：

- 确认 D455 能启动
- 确认深度流没问题
- 确认手动曝光写入没问题

### 2. D455 语义范围推荐

```bash
python3 tools/recommend_d455_semantics.py
```

作用：

- 给出 `cam_exposure_*`
- 给出 `cam_iso_gain_*`
- 给出 `cam_shot_noise_base`
- 给出 `cam_power_nominal`

### 3. 静态墙面噪声标定

```bash
python3 tools/calibrate_d455_static_noise.py
```

作用：

- 反推 `cam_shot_noise_base`
- 反推 gain/noise 对齐趋势

### 4. 场景采集与 profile 拟合

如果你想进一步做 sim-to-real 对齐，再做：

```bash
python3 tools/collect_d455_calibration.py ...
python3 tools/fit_d455_scene_profiles.py
```

对应文档：

- [D455_CALIBRATION_README.md](/home/zhaoguodong/work/code/DiffPhysDrone/tools/D455_CALIBRATION_README.md)
- [D455_SCENE_COLLECTION_PROTOCOL.md](/home/zhaoguodong/work/code/DiffPhysDrone/tools/D455_SCENE_COLLECTION_PROTOCOL.md)

---

## `run_real_policy.py` 的总体流程

下面开始详细拆脚本。

脚本可以概括成九个阶段：

1. 解析 CLI 参数
2. 加载项目配置和 checkpoint
3. 构建策略网络
4. 启动 D455
5. 启动 PX4 MAVLink 连接
6. 可选的 warmup / arm / auto takeoff
7. 建立 mission frame 和目标点
8. 进入主控制循环
9. 结束任务并做收尾动作

---

## 第一部分：CLI 参数与运行模式

脚本入口是：

- `parse_cli()`

它把运行时参数分成五类。

### A. 模型与配置

- `--args-file`
- `--checkpoint`
- `--device`

这一组决定：

- 用哪个 `.args`
- 加载哪个 checkpoint
- 在 CPU 还是 CUDA 上推理

### B. PX4 通信与控制

- `--px4-connection`
- `--telemetry-timeout-s`
- `--control-rate-hz`
- `--px4-control-mode`
- `--offboard-warmup-s`
- `--arm`
- `--auto-takeoff`
- `--takeoff-height-m`
- `--takeoff-tolerance-m`
- `--takeoff-timeout-s`
- `--finish-action`

这一组决定：

- 如何连接 PX4
- Offboard 的刷新频率
- 是否自动 arm
- 是否自动起飞
- 结束时是 hold、land 还是 disarm

### C. 任务目标

- `--goal-forward-m`
- `--goal-left-m`
- `--goal-up-m`
- `--goal-tolerance-m`
- `--mission-timeout-s`

这一组定义：

- 目标点相对 mission origin 的位移
- 认为“到达目标”的阈值
- 允许的最大任务时长

### D. 策略状态构造

- `--policy-max-speed-mps`
- `--policy-margin-m`

这组参数非常重要。

因为训练时策略状态里有：

- `max_speed`
- `margin`

而真机并没有一个现成的环境对象，所以脚本要自己给策略构造这两个量。

### E. D455 相关

- `--d455-width`
- `--d455-height`
- `--d455-fps`
- `--d455-serial`
- `--d455-enable-emitter`
- `--d455-exposure-divisor-us`
- `--d455-working-exposure-min-us`
- `--d455-working-exposure-max-us`
- `--d455-working-gain-min`
- `--d455-working-gain-max`
- `--d455-working-laser-min`
- `--d455-working-laser-max`
- `--camera-warmup-frames`
- `--camera-frame-timeout-ms`
- `--resize-depth-to-policy`

这一组决定：

- D455 怎么启动
- 控制参数写到哪个实际工作范围
- 深度图是否要缩放到策略输入分辨率

---

## 第二部分：加载项目 `.args` 与 checkpoint

### `load_project_args()`

这个函数会做一件非常关键的事：

- 它不是重新手写一套“真机版配置”
- 而是直接加载你训练时的项目 `.args`

这带来的好处是：

1. 真机部署和训练共享同一套核心语义
2. 深度图尺寸、最小有效深度、相机语义映射不会乱
3. 你不用维护两套不同配置

然后它会：

1. 自动附加 `--resume checkpoint_path`
2. 解析 `scenarios`
3. 解析 `sun_glare_levels`
4. 设置随机种子
5. 做参数合法性检查

### 为什么这是必要的

因为训练里很多行为不是写死在模型中的，而是通过 `.args` 决定的，例如：

- `depth_width`
- `depth_height`
- `depth_min_valid`
- `depth_max_range`
- `cam_*` 语义映射
- `camera_control_mode`

如果真机脚本不用同样的 `.args`，就很容易发生“模型能加载，但输入语义全变了”的问题。

---

## 第三部分：构建模型

### `build_model_from_args()`

这个函数会按训练时的网络结构恢复模型：

1. 根据 `no_odom` 决定 `obs_dim`
2. 根据 `.args` 恢复：
   - `include_camera_state_in_obs`
   - `depth_nn_width`
   - `depth_nn_height`
   - `depth_use_pipeline`
   - `depth_min_valid`
   - `depth_max_range`
3. 加载 checkpoint
4. 切到 `eval()`

### 当前的限制

脚本会显式拒绝：

- `use_dmpc=True`
- `policy_output_intent=True`

因为当前真机脚本只实现了：

- `direct-action` 路径

换句话说，当前它支持的是：

- 策略直接输出动作域 `act_raw`
- 再通过 `decode_action_direct()` 解码

而不是：

- intent
- dLQR
- dMPC

---

## 第四部分：D455 运行时对象 `D455Runtime`

这个类是真机脚本里负责相机的核心。

### `start()`

它会做下面这些事情：

1. 依次尝试多种深度流模式
2. 启动 D455 pipeline
3. 拿到 depth sensor handle
4. 关闭自动曝光
5. 选择是否开启 emitter
6. 读取 `exposure/gain/laser_power` 的真实范围
7. 用当前项目的相机语义给 D455 写入初始参数
8. 丢弃若干预热帧

### 为什么要预热

因为刚启动时：

- 深度流常常不稳定
- 寄存器刚切换后前几帧会有瞬态

所以 `camera_warmup_frames` 是合理且必要的。

### `read_depth()`

它做的事情很直接：

1. `wait_for_frames()`
2. 取 `depth_frame`
3. 乘上 `depth_scale`
4. 返回以米为单位的 `depth_m`

### `apply_normalized()`

这是最关键的函数之一。

它把策略输出的：

- `power01`
- `exposure01`
- `gain01`

映射成 D455 的真实寄存器值：

- `laser_power`
- `exposure_us`
- `gain_value`

#### Power 映射

`power01` 会映射到你指定的 D455 `laser_power` 工作区间。

#### Exposure 映射

不是简单线性映射，而是：

1. 先用项目中的 `diff_depth_exposure_to_time()`
2. 再乘 `d455_exposure_divisor_us`
3. 再裁剪到真实 D455 工作区间

#### Gain 映射

会根据项目中的 `CameraSemantics.iso_to_gain()` 语义曲线做映射，再映射回 D455 工作区间。

### 为什么这样设计

因为真机部署脚本的目标，不是“简单给相机写个 0 到 255”，而是尽量保持训练时定义的主动感知语义。

---

## 第五部分：PX4 运行时对象 `Px4Bridge`

这是脚本里负责 PX4 通信的核心。

### `connect()`

它会：

1. 通过 `pymavlink` 建立连接
2. 等待 heartbeat
3. 请求默认数据流

### `request_default_streams()`

它会请求：

1. `LOCAL_POSITION_NED`
2. `ATTITUDE`

并尽量设置消息频率。

### `poll()`

它会从 MAVLink 消息流中不断更新：

- `_last_pos`
- `_last_vel`
- `_last_att`

### `current_state()`

它会把最新遥测封装成 `Px4TelemetryState`。

这个状态对象里又做了几层变换。

#### `pos_zup`

PX4 给的是 NED：

- `z` 朝下

而当前项目内部更接近：

- `z` 朝上

所以脚本会把：

- `[x, y, z_ned]`

变成：

- `[x, y, -z_ned]`

#### `vel_zup`

速度同理。

#### `R_policy`

当前项目内部策略状态使用的是：

- 世界系 `Z-up`
- 机体系更接近 `forward-left-up`

而 PX4 / MAVLink 的常见语义是：

- 世界系 `NED`
- 机体系 `FRD`

所以脚本做了一层坐标变换：

- `quat_or_rpy_rotation_frd_to_zup()`

这一层非常关键。

如果你发现飞机一控制就朝奇怪方向飞，第一优先排查的就是坐标系，不是神经网络。

---

## 第六部分：Warmup、Arm、Takeoff

### `mission_hold()`

这个函数会持续发送当前位姿的 position setpoint。

作用是：

1. 在切 Offboard 之前给 PX4 喂“稳定 setpoint 流”
2. 满足 PX4 对 Offboard 的前置要求

这非常重要，因为 PX4 官方明确要求：

- 进入 Offboard 前就要先持续收到 setpoint

### `auto_takeoff()`

这个函数不是“电机起转一下”，而是真正做一个基于本地位置的起飞段：

1. 读取当前高度
2. 把目标高度设置为 `当前高度 + takeoff_height_m`
3. 持续发送 position+yaw setpoint
4. 直到到达高度容差

这意味着：

- 自动起飞依赖 PX4 已经有可靠本地位置

### `safe_finish()`

任务结束或异常时，脚本会尽量做一个保守收尾：

1. `hold`
2. 或 `land`
3. 或 `disarm`

建议你的首飞阶段优先用：

- `--finish-action hold`

因为这比立即自动 `land` 更容易先看清系统行为。

---

## 第七部分：Mission Frame 是什么

这是当前脚本里一个很重要但容易被忽略的概念。

### Mission origin

脚本不会使用某个固定世界点作为目标原点，而是：

- 在进入任务开始阶段之后
- 把当时飞机所在位置记作 `mission_origin`

如果用了 `auto_takeoff`，则：

- `mission_origin` 是起飞完成后的悬停点

### Goal 的定义

然后：

- `goal-forward-m`
- `goal-left-m`
- `goal-up-m`

会被解释为：

- 相对 `mission_origin`
- 在 mission frame 里的前/左/上位移

### 为什么这样设计

这样做的好处是：

1. 不需要你知道室内世界的绝对原点
2. 只要飞机当前停稳，就能定义一个相对目标
3. 对实验复现更方便

坏处是：

1. 你必须清楚“机头朝向”决定了前向目标方向
2. 如果起飞后姿态基准没对齐，目标方向也会跟着歪

所以：

- 首次实验一定要先在很短距离上验证 `goal-forward-m` 的方向是否和你预期一致

---

## 第八部分：主控制循环到底做了什么

下面是整个脚本最核心的部分。

每一帧大致按下面顺序执行。

### 1. 读取 D455 深度

```text
D455 -> depth_m
```

### 2. 缩放到策略输入尺度

如果 `resize_depth_to_policy=True`，会把真实 D455 深度图缩放到训练时的：

- `depth_width`
- `depth_height`

这一步是为了保持网络输入尺寸一致。

### 3. 计算当前 fill rate

脚本会用 `compute_depth_fill_rate()` 做一个简单统计，用于日志观察。

### 4. 读取 PX4 当前状态

通过 `bridge.current_state()` 得到：

- 位置
- 速度
- 姿态

### 5. 计算当前目标方向

```text
goal_vec_world = goal_world - pos_world
```

然后判断是否已经到达目标。

### 6. 构造一个“最小真机环境对象”

脚本不会真的创建完整 `Env`，而是临时构造一个很小的对象，只包含策略状态构造和动作解码需要的字段，例如：

- `v`
- `R`
- `margin`
- `max_speed`
- `g_std`
- `thr_est_error`

这是当前脚本的一种工程简化：

- 不复用仿真环境的全部东西
- 只复用和策略推理直接相关的那一小部分语义

### 7. 构造策略状态向量

通过：

- `build_local_frame()`
- `compute_target_velocity()`
- `build_state_vector()`

得到策略状态。

这一步会把：

- 真机速度
- 目标方向
- 当前机体朝向
- 当前相机状态

拼成训练时网络期望看到的状态。

### 8. 送入模型

模型前向输出：

1. `act_raw`
2. `cam_params`
3. `h`

其中：

- `h` 是 GRU 隐状态
- 所以真机脚本不是逐帧独立推理，而是保留了时序记忆

### 9. 更新相机状态

通过：

- `update_camera_params()`

做 EMA 平滑后，得到新的：

- `power`
- `exposure`
- `gain`

### 10. 解码飞行动作

通过：

- `decode_action_direct()`

把 `act_raw` 变成：

- 世界系加速度命令 `accel_cmd_world`
- 辅助速度预测 `v_pred_world`

### 11. 写回 D455

把新的：

- `power / exposure / gain`

写成真实 D455 寄存器。

### 12. 生成 yaw 命令

脚本会优先依据：

1. `v_pred_world`
2. 若速度预测太小，则用目标方向

来决定 yaw。

### 13. 转成 PX4 所需的 NED setpoint

内部策略是 `Z-up`，PX4 需要 `NED`，所以脚本会把：

- `accel_cmd_world`
- `vel_cmd_world`

再转换成：

- `accel_cmd_ned`
- `vel_cmd_ned`

### 14. 发送 Offboard setpoint

通过：

- `send_accel_yaw_ned()`

发送 `SET_POSITION_TARGET_LOCAL_NED`。

如果 `px4-control-mode=vel_accel`：

- 会同时给速度前馈和加速度前馈

如果 `px4-control-mode=accel`：

- 只给加速度项

### 15. 写日志

日志里会记录：

- 位置
- 速度
- 目标距离
- yaw
- power/exposure/gain
- D455 实际寄存器值
- fill_rate
- 发送给 PX4 的加速度和速度命令

所以这个日志对你后面分析真机行为非常有价值。

---

## 为什么脚本里要自己推一个 `vel_cmd_world`

在主循环里，脚本会用：

```text
vel_cmd_world = vel_world + accel_cmd_world * dt
```

再把它限幅到 `policy_max_speed_mps`。

这不是训练里严格学出来的控制器状态，而是一个真机部署时的工程辅助项：

- 当你用 `vel_accel` 模式时，PX4 的 setpoint 可以同时接收速度前馈和加速度前馈
- 这个 `vel_cmd_world` 让加速度控制不至于太“裸”

可以把它理解成：

- 一个温和的速度前馈项

而不是：

- 训练中单独学习出来的速度控制器

---

## 当前脚本没有做的事情

这是你必须知道的，因为很多人以为“真机脚本都有了，就一定全自动了”。

当前脚本没有做：

1. 不直接读动捕系统
2. 不负责把动捕桥接进 PX4
3. 不自动配置 PX4 参数
4. 不自动校正 D455 与机体的外参
5. 不自动做安全区域保护
6. 不自动做避障地图构建
7. 不自动估计 thrust 模型
8. 不自动验证坐标系方向

这意味着：

- [run_real_policy.py](/home/zhaoguodong/work/code/DiffPhysDrone/tools/run_real_policy.py) 是“真机闭环执行器”
- 不是“一键完成所有真机集成的万能系统”

---

## 真实部署的推荐操作步骤

下面给一套我最推荐的现场操作顺序。

### Step 1. 在 Orin 上准备环境

至少要保证：

```bash
python3 -c "import torch, numpy, cv2"
python3 -c "import pyrealsense2"
python3 -c "from pymavlink import mavutil"
```

都不报错。

### Step 2. 检查 D455

```bash
python3 tools/test_d455_depth.py
```

确认：

- 可以出图
- 参数可写
- 没有 USB 掉流

### Step 3. 检查 PX4 是否已接入动捕

你要确认：

1. PX4 本地位置稳定
2. 姿态角方向正确
3. 速度不会无故飘大

这一步推荐在 QGC 或 PX4 shell 里完成。

### Step 4. 不 arm，运行脚本

```bash
python3 tools/run_real_policy.py \
  --checkpoint checkpoint/你的模型.pth \
  --args-file configs/paper_final_full.args \
  --px4-connection udp:127.0.0.1:14540
```

这一步你应该看到：

- 能打印 step 信息
- 能生成日志
- power/exposure/gain 会随时间变化

### Step 5. 上桨，但先不自动起飞

先用很保守设置：

```bash
python3 tools/run_real_policy.py \
  --checkpoint checkpoint/你的模型.pth \
  --args-file configs/paper_final_full.args \
  --px4-connection udp:127.0.0.1:14540 \
  --arm \
  --finish-action hold \
  --goal-forward-m 0.5
```

先验证：

- Offboard 能切进去
- 飞机不会乱抽
- 方向正确

### Step 6. 自动起飞 + 短程任务

```bash
python3 tools/run_real_policy.py \
  --checkpoint checkpoint/你的模型.pth \
  --args-file configs/paper_final_full.args \
  --px4-connection udp:127.0.0.1:14540 \
  --arm \
  --auto-takeoff \
  --takeoff-height-m 1.2 \
  --goal-forward-m 1.0 \
  --goal-left-m 0.0 \
  --goal-up-m 0.0 \
  --finish-action hold
```

### Step 7. 再扩展到你的实际小场景

例如固定障碍和 sun_glare 场景。

---

## 你在室内动捕场景下最应该重点检查的 10 件事

1. 动捕系统坐标轴方向是否和 PX4 融合方向一致
2. 外部视觉 / 动捕消息是否达到 `30Hz~50Hz`
3. PX4 本地位置是否稳定，不会静止时漂移很大
4. `LOCAL_POSITION_NED` 的 `z` 符号你是否理解正确
5. 机头方向是否和你定义的“forward”一致
6. D455 到机体的安装角度是否稳定
7. D455 USB 是否牢靠，飞行振动时会不会掉流
8. `goal-forward-m` 的方向是否与你直觉一致
9. `policy_max_speed_mps` 是否过大
10. `max_acc_cmd` 对应的真实飞行 aggressiveness 是否过高

---

## 最常见的故障现象与排查顺序

### 现象 1：脚本一直卡在等待 PX4 状态

优先检查：

1. PX4 是否真的有 `LOCAL_POSITION_NED`
2. 动捕是否已经成功融合进 PX4
3. MAVLink 连接串是否对
4. companion computer 是否连到了正确端口

### 现象 2：一切都通，但一切 Offboard 就退出

优先检查：

1. setpoint 流是否持续发送
2. warmup 时间是否太短
3. 消息频率是否低于 2Hz
4. PX4 offboard failsafe 设置

### 现象 3：飞机一飞就朝反方向冲

优先检查：

1. 坐标系方向
2. yaw 定义
3. 动捕世界系到 PX4 本地系的变换
4. `goal-forward-m` 是否按你预期解释

### 现象 4：悬停还可以，但一给目标就强烈振荡

优先检查：

1. `max_acc_cmd` 是否太大
2. `control-rate-hz` 是否与训练差太多
3. PX4 控制器参数是否过激进
4. 动捕延迟和 PX4 融合延迟是否太大

### 现象 5：D455 深度图在真机上很差

优先检查：

1. 自动曝光是否真的关掉
2. emitter 是否开启
3. 当前写入的 `laser_power/exposure/gain` 是否合理
4. USB 带宽是否不足
5. 室内强反光 / 强逆光是否超出当前标定范围

### 现象 6：power/exposure/gain 在真机几乎不变化

优先检查：

1. checkpoint 是否就是你想部署的那个
2. `.args` 是否和训练保持一致
3. 是否误用了 `camera_control_mode fixed`
4. D455 working range 是否被你夹得太窄

---

## 真机首飞时的安全建议

这部分虽然听起来“老生常谈”，但真的不能省。

### 第一次有桨测试时

1. 场地里只留必要人员
2. 有一个人专门看飞控模式和紧急接管
3. 有一个人专门看伴随计算机日志
4. 飞机周围留足够的净空
5. 一开始目标只给很短距离

### 不要做的事

1. 不要第一次就上 sun_glare 障碍实验
2. 不要第一次就让它飞完整地图
3. 不要第一次就相信自动降落流程
4. 不要在坐标系还没确认前装桨放飞

### 最推荐的节奏

1. 无桨验证
2. 系留或保护网验证
3. 短程低速验证
4. 稳态悬停验证
5. 简单直线验证
6. 固定障碍验证
7. 再做感知策略实验

---

## 推荐的首飞命令模板

### A. 只读不控

```bash
python3 tools/run_real_policy.py \
  --checkpoint checkpoint/2026-04-23-12-12-57/checkpoint0014.pth \
  --args-file configs/paper_final_full.args \
  --px4-connection udp:127.0.0.1:14540
```

### B. Arm，但不自动起飞

```bash
python3 tools/run_real_policy.py \
  --checkpoint checkpoint/2026-04-23-12-12-57/checkpoint0014.pth \
  --args-file configs/paper_final_full.args \
  --px4-connection udp:127.0.0.1:14540 \
  --arm \
  --goal-forward-m 0.5 \
  --finish-action hold
```

### C. 自动起飞 + 短程

```bash
python3 tools/run_real_policy.py \
  --checkpoint checkpoint/2026-04-23-12-12-57/checkpoint0014.pth \
  --args-file configs/paper_final_full.args \
  --px4-connection udp:127.0.0.1:14540 \
  --arm \
  --auto-takeoff \
  --takeoff-height-m 1.2 \
  --goal-forward-m 1.0 \
  --goal-left-m 0.0 \
  --goal-up-m 0.0 \
  --finish-action hold
```

---

## 日志文件怎么看

脚本运行后会在：

```text
artifacts/real_policy_runs/<timestamp>/
```

生成：

1. `meta.json`
2. `trace.csv`

### `meta.json`

里面有：

- checkpoint 路径
- args 文件
- CLI 参数
- D455 运行模式
- D455 参数范围

### `trace.csv`

里面有逐步记录的：

- 位置
- 速度
- 目标距离
- yaw
- power/exposure/gain
- D455 真实寄存器值
- fill rate
- 发给 PX4 的速度/加速度命令

如果你做实验论文，这个日志非常关键，因为它能把：

- 感知参数
- 飞行命令
- 轨迹行为

放在同一时间轴上。

---

## 当前版本的局限与后续建议

### 当前局限

1. 还不支持直接读取动捕 topic
2. 还不支持 dMPC / intent checkpoint
3. 还没有把真机日志自动做成图
4. 还没有加入外部安全围栏
5. 还没有做机体外参自动校准

### 我最推荐你下一步做的事

如果你的目标是真正稳定复现，下一步最值得做的是：

1. 先把“动捕 -> PX4 -> 脚本”的状态链路完全跑稳
2. 用无桨模式验证 `power/exposure/gain` 与 D455 真实变化一致
3. 用短距离、无障碍任务验证坐标系和控制方向
4. 再进入 sun_glare 小场景

---

## 一句话版本的部署原则

如果只记一句话，那就是：

**先让 PX4 可靠地相信动捕，再让脚本可靠地相信 PX4，然后再让策略去控制 D455 和 Offboard。**

只要这个顺序反了，真机部署就会非常痛苦。

---

## 相关文件

- 真机执行脚本：
  - [run_real_policy.py](/home/zhaoguodong/work/code/DiffPhysDrone/tools/run_real_policy.py)
- D455 标定说明：
  - [D455_CALIBRATION_README.md](/home/zhaoguodong/work/code/DiffPhysDrone/tools/D455_CALIBRATION_README.md)
- D455 场景采集 protocol：
  - [D455_SCENE_COLLECTION_PROTOCOL.md](/home/zhaoguodong/work/code/DiffPhysDrone/tools/D455_SCENE_COLLECTION_PROTOCOL.md)
- 深度快速联机检查：
  - [test_d455_depth.py](/home/zhaoguodong/work/code/DiffPhysDrone/tools/test_d455_depth.py)
- D455 语义推荐：
  - [recommend_d455_semantics.py](/home/zhaoguodong/work/code/DiffPhysDrone/tools/recommend_d455_semantics.py)

---

## 官方参考链接

- PX4 Offboard Mode
  - https://docs.px4.io/main/en/flight_modes/offboard.html
- PX4 External Position Estimation
  - https://docs.px4.io/main/en/ros/external_position_estimation.html
- PX4 Motion Capture
  - https://docs.px4.io/main/en/computer_vision/motion_capture.html

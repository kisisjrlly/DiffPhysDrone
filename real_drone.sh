ssh xgg@192.168.1.208
# SSH password is intentionally not stored in the repository.

# 无人机硬件配置清单

> 当前灰度主动感知方案（2026-09-22）：
> - 机载计算：NVIDIA Jetson Orin NX 8GB
> - 载板：达妙科技 DAMIAO DM-ORIN NX V2.X
> - 计划主视觉相机：e-con Systems e-CAM37M_CUONX（Sony IMX900，单色全局快门，MIPI CSI-2）
> - Intel RealSense D455 降级为旧方案/调试参考传感器；新的主实验不应依赖 D455 深度。
> - 灰度相机方案与兼容性检查见 `docs/grayscale_imx900/HARDWARE_IMX900.md`。

## 1. 计算与飞控

| 部件 | 型号 / 规格 | 数量 | 备注 |
|---|---|---|---|
| 机载计算模组 | NVIDIA Jetson Orin NX 8GB | 1 | 主控 AI 计算，可跑视觉/规划模型；散热风扇 Orin-FAN-PWM |
| 载板 | DAMIAO DM-ORIN NX V2.X | 1 | 达妙轻量载板；39.2g；50×86.8×13mm；12~28V 宽压输入、支持 6S；2 路 22-pin 0.5mm FPC CIS/MIPI 相机接口；3 路 USB3 Type-C |
| 飞控 | NxtPX4 v2（H7 主控 + 双 BMI088 IMU） | 1 | 开源 PX4；与机载电脑飞行时通过 UART/MAVLink 连接 |

### 1.1 DAMIAO DM-ORIN NX V2.X 与相机有关的接口

根据载板说明书：

- CIS0：22-pin 0.5mm FPC，暴露 CSI0/CSI1 数据通道、CAM0_MCLK、CAM0_PWDN、CAM0 I2C、3.3V/GND。
- CIS1：22-pin 0.5mm FPC，暴露 CSI2/CSI3 数据通道、CAM1_MCLK、CAM1_PWDN、CAM1 I2C、3.3V/GND。
- 手册说明 CIS 部分电路与原版/reference 设计一致，但 **不能据此直接假定 e-con IMX900 的驱动/Device Tree 无需修改即可工作**。
- 在插接 e-CAM37M_CUONX 前，必须确认 FPC 朝向、pinout、lane mapping/polarity、I2C、MCLK、PWDN 和对应 JetPack/BSP/Device Tree。

## 2. 传感器

| 部件 | 型号 / 规格 | 数量 | 状态 / 备注 |
|---|---|---|---|
| 主视觉相机（新方案） | e-con Systems e-CAM37M_CUONX / Sony IMX900 Mono | 1 | **已选型，待采购/适配/标定**；单色、全局快门、MIPI CSI-2；后续策略控制 exposure + gain |
| 深度相机（旧方案） | Intel RealSense D455 | 1 | 旧的 diff-depth 研究与调试参考；新灰度主实验不得将其深度作为策略输入 |

### 2.1 IMX900 方案的软件/硬件确认项

硬件到位后必须记录：

- e-con 精确 SKU；
- 镜头 SKU、焦距、实际 FoV；
- FPC/转接线 SKU 与长度；
- JetPack / L4T 版本；
- e-con 驱动/BSP 版本；
- `v4l2-ctl --list-ctrls` 与 `v4l2-ctl --all`；
- 可用像素格式、分辨率、帧率；
- 曝光/增益实际范围和步进；
- exposure/gain 指令到实际生效帧的延迟；
- 相机+镜头+线材总重量和功耗。

## 3. 动力系统

| 部件 | 型号 / 规格 | 数量 | 备注 |
|---|---|---|---|
| 电机 | 2006 规格，2006~2150KV | 4 | 旋翼动力；项目共购置 16 只（含备件） |
| 螺旋桨 | D90S（3.5 寸） | 1 套（含备件，共 10 只） | 适配 2006 电机，3.5 寸桨 |
| 电池 | 6S 1300mAh 95C 锂电池 | 3 | 1 主用 + 2 备用；输出约 25.2V 满电 |

## 4. 机架与结构

| 部件 | 型号 / 规格 | 数量 | 备注 |
|---|---|---|---|
| 机架 | 怪象 XI35（3.5 寸穿越机架，碳纤维 + 铝合金中心板） | 1 套 | 轴距约 200mm（3.5 寸），适配 3.5 寸桨；发票编号 12（参考 ¥350） |
| 机架保护圈 | 怪象 XI35 专用桨叶保护圈 | 1 套 | 塑料圈，透明色 / 透明灰，含 6 件（含备件） |
| 铝柱 | M3×12（D5） | 15 | 机架上下板连接柱 |
| 海绵缓冲垫 | 3M 加厚泡棉 EVA | 1 | 飞控/电路减震 |

### 4.1 无人机尺寸

- 轴距（对角电机中心距）：约 200 mm（3.5 寸机典型，轮距约 110–130 mm）。
- 螺旋桨：3.5 寸（D90S 在此机架上为 3.5 寸规格），单桨展约 89 mm。
- 整机外接矩形（含桨叶保护圈）：约 230 mm × 230 mm × 100 mm（长 × 宽 × 高，桨展方向）。
  - 保护圈外径约 3.7 寸（≈ 94 mm），对角两圈之间最外缘约 290–310 mm；
- 上下两层桨面间距按 XI35 机架约 20–30 mm。
- 电池安装位：底部，6S 1300mAh 外形约 70 × 35 × 30 mm；计入后总高约 90–110 mm。

## 5. PX4 飞控与地面站/机载电脑的双链路架构

NxtPX4 v2（PX4）飞控上同时具备 USB 口 和 UART 串口，两条链路职责不同，切勿混用：

调试链路（USB）：飞控 USB 口用 USB 线连接开发机主机，运行 QGroundControl 进行参数配置（如 MAV_*_CONFIG、SER_*_BAUD）、固件刷写、日志下载和离线调试。该链路仅用于调试与维护，飞行过程中不参与控制。

飞行链路（UART）：飞控 TELEM/UART 口经端子线（TX/RX/GND 交叉连接、共地）连接机载电脑，以 MAVLink over UART 协议传输自主飞行命令流——机载电脑上的 MAVROS/px4ctrl 通过此链路下发姿态 setpoint、模式切换和心跳。此为飞行时的唯一控制链路。

对应到软件配置：机载电脑侧 MAVROS 的 fcu_url 应指向 UART 节点（真机为 /dev/ttyTHS0:921600，波特率需与飞控侧 SER_*_BAUD 一致）；/dev/ttyACM* 属于飞控 USB 调试链路，是开发机 QGC 的设备，不能用作机载电脑的飞行链路。两条链路可同时存在，QGC 插 USB 监控配参不影响机载电脑经 UART 飞行，但避免在两条链路同时下发相互冲突的 setpoint。

## 6. 灰度主动感知研发入口

后续实现不要直接从本文件猜方案。先阅读：

1. `docs/grayscale_imx900/README.md`
2. `docs/grayscale_imx900/TECHNICAL_PLAN.md`
3. `docs/grayscale_imx900/HARDWARE_IMX900.md`
4. `docs/grayscale_imx900/CALIBRATION_SIM2REAL.md`
5. `docs/grayscale_imx900/CODEX_IMPLEMENTATION_GUIDE.md`

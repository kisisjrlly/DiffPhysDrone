# 论文 diff_depth 实验矩阵

## 范围

- 论文主线仅研究 `diff_depth`。
- 主控制器保持默认动作域控制，不开启 `--policy_output_intent`。
- 当前先不依赖 `--use_dmpc`、`--policy_output_intent` 或 `--tbptt_enable`。
- 当前目标定义为：学习得到的 `power / exposure / gain` 在困难光学场景中提升鲁棒性与感知效率。

## 固定实验协议

| 项目 | 论文 v1 推荐设置 |
| --- | --- |
| 传感器模式 | `diff_depth` |
| 控制模式 | 默认动作域控制（不启用 `policy_output_intent`） |
| dMPC / intent | 关闭 |
| TBPTT | 关闭 |
| 深度输入 | 当前流水线开启，在当前 `depth_width/depth_height` 渲染，并以当前 `depth_nn_width/depth_nn_height` 输入策略 |
| 观测中的相机状态 | 主方法开启，仅在消融中关闭 |
| Diff-depth 后端 | 论文主结果使用 `diff_depth=python` |
| 随机种子 | 至少 3 个种子 |
| 评测回合数 | 每个 checkpoint 的每个场景评测 100 到 200 个 episode |
| 主要指标 | `success_rate`、`collision_rate`、`time_to_goal`、`diff_depth_fill_rate`、`diff_depth_hole_rate`、`energy_proxy`、`blur_proxy`、`noise_proxy` |
| 结果汇报 | 按种子统计 mean ± std |

## 场景套件

| 场景 ID | 名称 | 预期测试内容 | 当前实现状态 |
| --- | --- | --- | --- |
| `S0` | `base` | 固定小地图上的基础导航与轻量绕障 | 已可用 |
| `S1` | `sun_glare` | 走廊出口逆光与局部环境红外淹没，测试 power/exposure 权衡 | 已接入局部光照事件，仍待校准强度 |
| `S2` | `specular_trap` | 高反射或类玻璃局部失效，预期应触发功率抑制与被动 fallback | 已接入局部高反区域，透明体误导仍待增强 |
| `S3` | `vantablack_gap` | 暗色低反照率门洞，模糊-噪声权衡很关键 | 已接入局部吸光门框材质，仍待实测标定 |
| `S4` | `dark_morphing` | 极暗狭窄通道，测试低照度下的曝光/增益/功率耦合 | 已接入局部极暗窄缝，尚未扩展到缓存式高阶策略 |

## 训练课程矩阵

| 阶段 | 目标 | 相机控制 | 里程计 | 训练场景 | 退出准则 |
| --- | --- | --- | --- | --- | --- |
| `T0` | 学会在稳定深度输入下飞行 | 固定 `power/exposure/gain` | 开启 | `S0` | 成功率稳定且无明显深度崩塌 |
| `T1` | 在稳定飞行基础上学习主动参数控制 | 学习得到 `power/exposure/gain` | 开启 | `S0 + S1 + S3` | 学到的参数具有非平凡性且成功率不回退 |
| `T2` | 压测鲁棒性与涌现行为 | 学习得到 `power/exposure/gain` | 困难设置下可选关闭 | `S0 + S1 + S2 + S3 + S4` | 相对固定与启发式基线有明显增益 |

## 主结果矩阵

| 实验 ID | 方法 | 训练阶段 | 评测场景 | 相机控制 | 观测中的相机状态 | 关键对比目的 |
| --- | --- | --- | --- | --- | --- | --- |
| `M0` | `Fixed-Mid` | `T0` | `S0-S4` | 固定中值 | n/a | 最低工作量基线 |
| `M1` | `Fixed-High` | `T0` | `S0-S4` | 固定高能耗取值 | n/a | 展示蛮力感知的代价 |
| `M2` | `Heuristic` | 无或 `T0` 策略骨干 | `S0-S4` | 手工规则 `power/exposure/gain` | 可选 | 强非学习基线 |
| `M3` | `Learned-NoCamState` | `T1-T2` | `S0-S4` | 学习得到 | 关闭 | 测试显式相机状态反馈是否重要 |
| `M4` | `Ours-Full` | `T1-T2` | `S0-S4` | 学习得到 | 开启 | 论文主方法 |

## 消融矩阵

| 实验 ID | 变更项 | 其余保持不变 | 该消融为何重要 | 预期信号 |
| --- | --- | --- | --- | --- |
| `A1` | 去掉 `coef_diff_depth_power` | 其余使用 `M4` | 检查能耗感知行为是被学到的还是仅仅“凑巧好用” | power 上升且 energy proxy 变差 |
| `A2` | 去掉 `coef_diff_depth_blur` | 其余使用 `M4` | 检查 exposure-speed 耦合是否真的被学到 | 高速时 exposure 变得过长 |
| `A3` | 去掉 `coef_diff_depth_noise` | 其余使用 `M4` | 检查 gain 正则在暗场是否重要 | gain 增大，noise proxy 变差 |
| `A4` | 禁用 `include_camera_state_in_obs` | 其余使用 `M4` | 测试闭环传感器状态感知 | 相机控制稳定性下降 |
| `A5` | 降低真实感预设 | 其余使用 `M4` | 测试学习行为是否依赖过度简化渲染 | 仿真成功率可能上升，迁移准备度可能下降 |
| `A6` | 深度分辨率消融 | 其余使用 `M4` | 量化对 `depth_width/height` 与 `depth_nn_width/height` 的敏感性 | 揭示算力与性能权衡 |
| `A7` | odom 开/关对比 | 其余使用 `M4` | 将纯视觉难度与传感器控制收益分离 | 澄清 paper-v1 claim 的适用范围 |

## 指标表模板

| 指标 | 定义 | 为什么需要 |
| --- | --- | --- |
| `success_rate` | 无碰撞到达目标的比例 | 主任务指标 |
| `collision_rate` | 发生任意碰撞的比例 | 安全性指标 |
| `time_to_goal` | 到达目标的步数或秒数 | 效率指标 |
| `diff_depth_fill_rate` | 有效深度像素占比 | 直接感知质量指标 |
| `diff_depth_hole_rate` | `1 - fill_rate` | 更易讨论失效区间 |
| `energy_proxy` | `mean(power^2)` | 主动感知代价代理 |
| `blur_proxy` | `mean((speed * exposure_t)^2)` | 运动模糊代理 |
| `noise_proxy` | `mean(gain^2)` 或其标定变体 | 高 gain 噪声代理 |
| `power-distance corr` | power 与障碍/距离上下文的相关性 | 可解释性指标 |
| `exposure-speed corr` | exposure 与速度的相关性 | 可解释性指标 |

## 图表规划

| 图 ID | 内容 | 来源实验 |
| --- | --- | --- |
| `F1` | 方法概览：diff_depth 传感器闭环 + 直接飞行策略 | 示意图 |
| `F2` | 代表性 episode 中 `power/exposure/gain/fill_rate/speed` 的时序曲线 | `M4` |
| `F3` | `M0-M4` 的成功率与碰撞率柱状图 | 主结果矩阵 |
| `F4` | `fill_rate` 与 `energy_proxy` 的帕累托图 | `M0-M4`，尤其 `S1-S3` |
| `F5` | 固定与启发式基线的失败案例图集 | `M0-M2` |
| `F6` | `A1-A7` 的消融柱状图 | 消融矩阵 |

## 真实设备验证矩阵

| 实验 ID | 阶段 | 硬件任务 | 输出 |
| --- | --- | --- | --- |
| `R0` | 台架 | 固定目标距离下的 D455 静态参数扫描 | 标定 csv |
| `R1` | 台架 | 暗/亮/镜面材质扫描 | fill-rate 与方差曲线 |
| `R2` | 台架 | 手持运动扫描 | exposure-speed 模糊标定 |
| `R3` | 离线 | 通过策略重放真实深度流 | sim-to-real 差距诊断 |
| `R4` | 飞行安全阶段 | 系留悬停或导轨测试 | 控制合理性与寄存器延迟合理性 |
| `R5` | 飞行 | 短走廊/穿门飞行 | 最终定性迁移演示 |

## 论文 V1 最小交付项

| 交付项 | 最低标准 |
| --- | --- |
| 主对比 | `M0-M4` 全部完成 |
| 消融 | 至少完成 `A1-A4` |
| 场景套件 | 至少完成 `S0-S3` |
| 真实标定 | 至少完成 `R0-R2` |
| 图表 | 至少完成 `F2-F6` |

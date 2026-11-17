# 当前代码审查入口（2026-11-17）

本分支为 `codex/imx900-intent-closed-loop`。本文供外部代码审查使用；当前实验没有完成，也没有证明拍摄与飞行联合学习成功。

## 要验证的核心思想

飞行任务的损失经图像形成过程传回相机设置，让曝光/增益服务下一次控制决策；飞行动作也配合测量。需要分别判断：拍摄设置是否影响任务、拍摄与运动是否存在额外配合收益、可微学习是否比相同预算的试跑学习更有效。三者不能相互替代。

## 先看这一条实际执行链

当前实验入口是 `tools/run_bounded_joint_route.py`，配置是 `configs/gray_bounded_joint.args`，共同闭环实现在 `bounded_route/core.py`。这条实验不使用旧飞行网络作为底座。

| 模块 | 重点文件 | 内容 |
|---|---|---|
| 计划与边界 | `docs/grayscale_imx900/BOUNDED_JOINT_EXECUTION_2026_11_11.md` | 六组、同预算、选模规则、恢复记录 |
| 任务/观测/控制/代价 | `bounded_route/core.py` | 共享两帧CNN读图器、六个运动参数、六个相机参数、反馈控制与终止 |
| 采集、训练、恢复与评估 | `tools/run_bounded_joint_route.py`、`bounded_route/integrity.py` | 128条读图器采集轨迹、1000次更新、完整60步反传、试跑优化、RNG、账目 |
| 相机请求与预处理 | `rollout_ops.py`、`controllers/physical_auto_exposure.py` | 自动曝光、相机更新、一次噪声、池化输入 |
| 曝光积分与图像梯度 | `render/intent_scene.py`、`render/exposure_history.py`、`render/pose.py` | 九点历史积分、姿态、warp代理梯度 |
| 传感器与profile | `sensors/imx900_camera.py`、`sensors/imx900_calibration.py`、`configs/calibration/grayscale_research_50ms.json` | 100–50000µs、增益1–8、噪声、硬饱和；未标定 |
| 动力学与碰撞 | `env_cuda.py`、`src/` | 两步命令队列、姿态、球体连续碰撞 |
| 梯度证据 | `tools/check_bounded_route_gradients.py`、`review_evidence_20261117/` | 固定状态与整任务有限差分原始记录 |
| 报告与检查 | `bounded_route/report.py`、`bounded_route/finalize.py`、`tools/audit_bounded_route_completion.py` | 尚未生成的最终评估不能被宣称已完成 |
| 监控与视频 | `tools/supervise_bounded_route.py`、`bounded_route/video.py` | 预算、阶段交接、固定案例导出 |

场景为 `single_wall_slit_diff`，起点−1.5m、目标+1.5m、60步/15Hz。半缝宽0.15m、球半径0.12m，几何横向余量为3cm；不要沿用早先误写的7.5cm。模型使用简化全局光照，不能据此作真机照明主张。

## 实际进度

本次发布只整理代码和已有记录，不恢复训练。当前没有训练进程；最后记录为11月12日19:16。

| 组别 | 方法 | 已完成轮数/6 | 优化试跑数/2880 | 各已完成轮开发成功数/16 |
|---|---|---:|---:|---|
| A1 | 平均AE＋试跑学习运动 | 5 | 2400 | 0、0、0、0、0 |
| A2 | 图像梯度AE＋试跑学习运动 | 5 | 2400 | 0、0、0、0、0 |
| A3 | 运动感知AE＋试跑学习运动 | 5 | 2400 | 0、0、0、0、0 |
| B | 可微学习相机＋固定共同运动 | 4，第5轮中断 | 2024 | 0、0、0、0 |
| C | 可微学习相机＋试跑学习运动 | 4 | 1920 | 0、0、0、0 |
| D | 相机与运动全部用试跑学习 | 4 | 1920 | 0、0、0、0 |

训练试跑合计13064/17280，约75.6%；这不是整个计划的完成比例。确认集、交叉回退、最终报告尚未执行。共享读图器验证RMSE约11.84cm；它与3cm几何余量的关系需要审查，但不能直接换算成任务成功率或唯一失败原因。

训练发生过重启/关机。账目保留中断尝试；部分早期中断时长使用含停机的保守上界，不应直接拿它解释训练效率。最近断点之后的阶段耗时尚未结算，因此账目中的已结束阶段时长不是全部使用量。

## 历史proof代码也在本次提交中保留

`configs/gray_proof_*.args`、`configs/gray_single_wall_proof_*.args`、`configs/calibration/proof_camera_surrogate.json`、`tools/audit_proof_camera.py`、`tools/run_proof_four_arms.py` 属于11月8日的历史机制试验。它们与当前六组实验不是同一套相机物理。

历史proof profile含 `reference_charge` 补偿；另有默认关闭的 `proof_camera_input_normalize` 开关。当前 `gray_bounded_joint.args` 使用研究profile、硬饱和、历史积分，不启用上述两种补偿。保留这些代码是为了审查真实代码状态，不能混用其数字或把软件补偿当成硬件性质。

`main_cuda.py` 中的相机初始化/重置逻辑属于旧策略训练入口；请同时审查它对checkpoint恢复的影响。当前六组的直接入口不是 `main_cuda.py`。

## 请审查者优先回答的具体问题

1. 图像、机体坐标、横向偏差标签、姿态输入、图像下采样是否一致？采集轨迹与实际闭环是否存在会放大读图误差的差异？
2. 真值控制参考、读图器参考、反馈控制器分别能完成什么？不要仅凭所有学习组0成功，就判断瓶颈只在相机或只在网络。
3. `objective()` 是否允许通过减速/超时降低代价，却难以改善穿缝成功？碰撞、到达、超时、冻结位置和补齐时限的逻辑是否相容？
4. 曝光/增益→传感器→读图器→控制→动力学→后续图像的梯度在哪些地方截断、硬饱和或使用近似？固定状态导数正确是否被误当成整任务梯度可靠？
5. `warp_ste` 是否表达了本任务实际需要的运动到图像变化？已有有限差分不一致的证据应如何限制结论？
6. 曝光是否只积分一次、噪声是否只加一次、是否重复模糊、请求/实际参数是否错位？两步飞行队列和下一帧相机生效是否被正确执行？
7. 六组在试跑数、相机梯度更新数、选模规则和随机流上有哪些相同与不同？同试跑预算是否被不恰当地表述为所有学习过程完全相同？
8. 重启后的模型、优化器、RNG和样本游标是否一致？未保存试跑、账目和源代码hash是否得到诚实处理？

请把确定的代码错误、需要运行才能判断的假设、实验设计局限分开，给出文件/函数/证据；不需要先推荐新场景或另一种学习算法。

## 证据与复核

本目录下 `review_evidence_20261117/` 包含当时协议、梯度原始记录、读图器误差、全部已有开发曲线、逐次计费账目和检查点摘要。JSON内容使用`.txt`保存，防止评审资料改变实验源码清单的成员关系。它们是中间证据，不是确认结果。

大体积模型/数据/CUDA二进制、原始导出目录未上传；检查点摘要记录SHA256。仅凭仓库不能重演已保存模型的训练历史，可审查源码并自行运行准备阶段。完整测试在本地已验证：331 passed，7 warnings。测试通过不代表任务能力成立。

```bash
# 使用已正确安装PyTorch/CUDA与项目扩展的Python
python -m pytest -q
python tools/run_bounded_joint_route.py --help
python tools/check_bounded_route_gradients.py --help
```

不要为阅读代码运行`--stage all`或启动新训练；这次发布没有增加实验预算。

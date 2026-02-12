


实施方案：补全 Paper.md 中缺失的功能
核心摘要
Paper.md 规划了四项当前代码库尚未实现的核心能力：(1) 可微分视觉感知损失项（运动模糊损失 V_blur、噪声损失 V_noise、失焦损失 V_defocus）；(2) 统一控制空间（包含相机增量参数与相机状态观测）；(3) 两阶段 G-DAC 师生训练框架；(4) 增强型行为指标（滚转角、视觉 - 运动耦合等新兴行为评估）。当前代码库已实现增强几何模型（微分平坦性 + 椭球碰撞检测）、基础可微分相机渲染、单阶段 G-DAC 训练流程。本方案将新增--paper_*系列命令行参数，使每项功能可独立开关，同时完全保留现有功能不受影响。
差距分析（Paper.md vs 当前实现）
表格论文章节功能点状态§2.2 增强几何模型微分平坦姿态恢复 + 椭球碰撞检测✅ 已实现§2.3A 运动模糊势函数运动模糊损失 Vblur​=∣v∣2⋅texp2​⋅ffocal2​❌ 缺失§2.3B 散粒噪声势函数噪声损失 Vnoise​∝1/SNR2❌ 缺失§2.3C 失焦势函数失焦损失 Vdefocus​=(dfocus​−dnearest​)2❌ 缺失§2.1 8 维统一控制空间3 维加速度 + 1 维偏航率 + 4 维相机增量参数❌ 缺失（当前：6 维动作 + 4 维独立绝对相机参数）§2.1 相机状态观测观测输入中包含当前相机参数❌ 缺失§3 两阶段 G-DAC 训练阶段一：优化动作序列；阶段二：蒸馏至策略网络❌ 缺失（当前：单阶段直接通过仿真反向传播）§1.1-1.3 新兴行为指标滚转角追踪、速度 - 曝光相关性、视场 - 密度耦合❌ 缺失相机效应对深度图的影响apply_camera_effects中的曝光 / 噪声 / 对焦退化模拟✅ 已实现可微分视场渲染带 CUDA 反向传播的DiffRenderFunction视场梯度计算✅ 已实现

实施步骤
步骤 1 — 视觉感知损失项（main_cuda.py）
新增compute_optical_loss()函数及三类损失项，通过--paper_optical_loss参数控制启用：

loss_blur（运动模糊损失）：基于无人机速度env.v、相机曝光时间cam_exposure、视场角cam_fov计算，公式：Vblur​=∥v∥2⋅texp2​/fov2（视场角越小 = 等效焦距越长 = 模糊越严重）。对批次和时间维度取平均。
loss_noise（散粒噪声损失）：基于曝光时间和 ISO 计算逆信噪比惩罚项，公式：Vnoise​=(1+2⋅iso)/(exposure+0.3)。复用 env_cuda.py 中相同的参数映射关系。
loss_defocus（失焦损失）：计算(dfocus​−dnearest​)2，其中dnearest​取自vec_to_pt_history的距离数据。仅在障碍物处于感知范围内时生效。

新增命令行参数：--paper_optical_loss（布尔型）、--coef_blur（默认 0.1）、--coef_noise（默认 0.05）、--coef_defocus（默认 0.05）。
将上述损失项加入总损失loss的求和计算，且仅在启用该参数时生效。
步骤 2 — 统一控制空间（model.py）
新增模型模式--paper_unified_control：

模型输出维度：将dim_action改为 10 维 = 3 维（加速度预测） + 3 维（速度预测，保留以兼容旧版本） + 4 维（相机增量：Δ 视场角、Δ 曝光、ΔISO、Δ 对焦）。相机增量通过tanh激活（非 sigmoid）以支持正负向调整，再乘以步长系数缩放。
相机状态观测：当启用--paper_cam_obs时，将当前相机参数[cam_fov_normalized, cam_exposure, cam_iso, cam_focus]（4 维）追加至状态向量。这会使观测维度dim_obs增加 4 维（从 7/10 维变为 11/14 维），需同步修改v_proj的输入维度。
增量式更新：在训练循环中，相机参数采用增量更新：cam_fov_new = clamp(cam_fov + delta_fov * step_size, min, max)，替代当前的绝对 sigmoid 映射方式。
保留Model.__init__中的use_diff_cam参数以兼容旧版本。新增的--paper_unified_control参数会覆盖--diff_cam的行为，但两者需共存。

model.py 修改点：

为Model.__init__添加use_unified_control参数
当use_unified_control=True时：self.fc输出 10 维；移除独立的self.fc_cam；从动作输出的最后 4 维提取相机增量；对相机参数部分应用tanh激活
当use_unified_control=False时：保留现有逻辑

步骤 3 — 两阶段 G-DAC 训练（main_cuda.py）
新增--paper_gdac参数启用论文中的两阶段训练，默认保留当前单阶段模式。
阶段一 — 教师 / 求解器（内循环）：

每次迭代中，env.reset()后获取模型对所有时间步的初始动作预测
将动作序列从模型计算图中分离 → u_guess = [model(x_t, s_t, h_t).detach() for t in timesteps]
将u_guess转为requires_grad=True的张量列表
为u_guess创建独立的Adam优化器
执行--gdac_inner_steps（默认 10 次）迭代：
通过可微分仿真器执行u_guess的轨迹推演
计算所有损失项（速度追踪、碰撞、视觉损失等）
loss.backward() → 仅更新u_guess（不更新模型权重）


保存优化后的动作序列u_star = [u.detach() for u in u_guess]

阶段二 — 学生网络蒸馏：

使用当前模型重新执行轨迹推演
计算蒸馏损失loss_distill = MSE(model_outputs, u_star)（逐时间步）
同时加入标准损失项（碰撞、速度），并降低其权重以适配课程学习
loss_distill.backward() → 更新模型权重theta

新增命令行参数：--paper_gdac（布尔型）、--gdac_inner_steps（整型，默认 10）、--gdac_inner_lr（浮点型，默认 0.01）、--coef_distill（浮点型，默认 1.0）、--gdac_horizon（整型，默认与--timesteps一致）。
核心实现细节：内循环需要在每次迭代开始时重新执行env.reset()，并在相同环境状态下重复 K 次轨迹推演。因此需在 env_cuda.py 中新增env.save_state() / env.restore_state()辅助方法，用于保存和恢复环境状态。
步骤 4 — 环境状态保存 / 恢复（env_cuda.py）
为Env类新增两个方法：

save_state() → 返回包含{p, v, a, act, R, R_old, p_old, dg, v_wind, ...}的字典快照（所有张量均克隆）
restore_state(snapshot) → 从快照恢复所有状态张量

这两个方法是 G-DAC 第一阶段实现的关键，确保内循环 K 次迭代可复用相同初始状态。
步骤 5 — 增强型新兴行为指标（main_cuda.py）
新增跟踪与日志模块（启用--diff_cam或--paper_unified_control时自动开启）：

滚转角历史：追踪每个时间步的arccos(R[:,2,2])（竖直方向上向量的夹角）。记录max_roll（最大滚转角）、mean_roll（平均滚转角），以及穿墙缝时的滚转角（启用--wall_slit时）。
视觉 - 运动耦合：计算每条轨迹中速度序列与曝光序列的皮尔逊相关系数。记录speed_exposure_corr（预期为负相关，即无人机在曝光时间增加时减速）。
视觉呼吸效应：计算视场角序列与最近障碍物距离的皮尔逊相关系数。记录fov_obstacle_corr（预期为正相关，即障碍物靠近时扩大视场角）。
阶段转换可视化：穿墙缝模式下，在保存迭代时记录缝宽与穿墙滚转角的散点图。

步骤 6 — 配置文件与运行脚本更新

新增configs/paper_optical.args：在单智能体配置基础上启用--diff_cam --paper_optical_loss，并配置合理的损失系数
新增configs/paper_gdac.args：启用--paper_gdac --paper_unified_control --paper_cam_obs --paper_optical_loss，对应 Paper.md 的完整配置
更新wall_slit.args，将新增 paper 系列参数以注释形式补充
为run.sh添加对应运行指令

步骤 7 — 评估脚本更新（eval_wall_slit.py）

支持--paper_unified_control和--diff_cam参数，确保评估时使用匹配的模型架构
评估统一控制模型时，适配 10 维动作输出及相机增量的增量式更新逻辑
记录评估过程中的穿墙滚转角、相机参数轨迹数据


验证方案

向后兼容性：执行python main_cuda.py $(cat wall_slit.args) — 需与当前代码行为完全一致（未启用新参数时无功能变化）
仅启用视觉损失：执行--diff_cam --paper_optical_loss — 验证loss_blur、loss_noise、loss_defocus出现在 wandb 日志中且数值非零
统一控制模式：执行--paper_unified_control --paper_cam_obs — 验证模型输出 10 维动作、相机增量增量式更新、模型接收 11/14 维观测输入
完整 G-DAC 模式：在穿墙缝模式下执行--paper_gdac --paper_unified_control --paper_optical_loss — 验证内循环优化正常运行、loss_distill被记录、出现预期的滚转行为
梯度流验证：通过torch.autograd.gradcheck验证视觉损失梯度可传递至相机参数，且相机增量能同时接收视觉损失和几何损失的梯度

关键决策

保留椭球碰撞检测：不实现论文中的体关键点 SDF 方法 — 椭球模型更简洁且已在 CUDA 层集成
统一控制保留速度预测头：10 维输出 = 3 维加速度 + 3 维速度预测 + 4 维相机增量（而非论文的 8 维：3+1+4，移除速度预测），以兼容无里程计模式
相机增量采用 tanh + 步长：而非原始增量值，以限制单步调整幅度
G-DAC 内循环使用 Adam 优化器：与论文伪代码保持一致（非 SGD）
视觉损失逐时间步计算：累加后取平均，与论文 §3 的哈密顿公式一致

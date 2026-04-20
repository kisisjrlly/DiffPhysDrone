# DiffPhysDrone 参数说明（全量 + 调参建议）

> **免责声明**：当前md中的描述并不完全等价项目中的代码实现，真实的实现以代码为准。

本文面向当前项目参数系统，主依据是：

- `config.py::build_parser`（训练/主流程参数）
- `configs/paper_final_full.args`（你当前使用的配置）
- `eval.py::parse_eval_args`（评估额外参数）

目标：

1. 解释每个参数的含义。
2. 说明当前配置值（相对默认值）。
3. 给出如何取值与调参建议。
4. 对相机参数做专门、详细、易懂的说明（面向深度相机新手）。

---

## 参数文件语法先导读

当前项目布尔参数有两类：

1. `store_true`（例如 `--tbptt_enable`）
   - 默认是 `False`。
   - 配置中出现该 flag 即表示 `True`。

2. `BooleanOptionalAction`（例如 `--depth_use_pipeline`）
   - 有默认值（可能是 `True` 或 `False`）。
   - 可写 `--xxx` 或 `--no-xxx` 显式覆盖。

如果 `paper_final_full.args` 未写某项，则采用 `config.py` 默认值。

---

## 当前配置总览（`paper_final_full.args`）

当前配置画像：

- 单场景训练：`sun_glare`
- 传感后端：`diff_depth=python`
- 相机控制约束很强（`coef_cam_smooth=100`、`coef_power_reg=100`）
- 安全项很强（`coef_collide=10`、`coef_obj_avoidance=4`）
- 未启用 teacher-student 蒸馏
- 未启用 dMPC/intent 路径

---

## 1) 训练与基础优化参数

| 参数 | 默认值 | 当前值 | 含义 | 如何取值 |
| --- | ---: | ---: | --- | --- |
| `--resume` | `None` | `None` | 从 checkpoint 续训 | 新训练不设；续训填 `.pth` 路径 |
| `--batch_size` | `64` | `156` | 并行环境数 | 显存允许可增大，OOM 优先降它 |
| `--num_iters` | `50000` | `5000` | 总迭代数 | 快速验证 `1k~5k`，正式训练更长 |
| `--lr` | `1e-3` | `5e-5` | 学习率 | 多损失耦合建议 `1e-5~1e-4` 起步 |
| `--grad_decay` | `0.4` | `0.3` | 长时域梯度衰减 | 太小会短视，太大易不稳；`0.2~0.6` 常见 |
| `--seed` | `42` | `42` | 随机种子 | 对比实验固定即可 |
| `--deterministic` | `False` | `False` | 确定性算法 | 复现实验开；追吞吐可关 |
| `--amp` | `True` | `True` | 混合精度 | CUDA 下通常开；数值异常时可 `--no-amp` |

---

## 2) TBPTT / 反传调度

| 参数 | 默认值 | 当前值 | 含义 | 如何取值 |
| --- | ---: | ---: | --- | --- |
| `--tbptt_enable` | `False` | `False` | 是否启用分段反传 | 显存紧张可开；追完整时域梯度可关 |
| `--tbptt_chunk_steps` | `40` | `10` | 每段步长 | 小则省显存但跨段信息弱；大更接近 full-BPTT |
| `--tbptt_chunk_accum` | `1` | `1` | 累积 chunk 次数 | 显存不足但想保留等效大 batch 可增大 |
| `--hybrid_full_bptt_every` | `0` | `0` | 每 N 次插入 full-BPTT | `0` 关闭；常用 `20~200` |
| `--hybrid_full_bptt_batch_size` | `0` | `64` | hybrid full-BPTT 的 batch | 混合模式下可显著降峰值显存 |

约束（`validate_args`）：

- `tbptt_enable=True` 时 `tbptt_chunk_steps >= 2`
- `tbptt_chunk_accum >= 1`
- `hybrid_full_bptt_every >= 0`
- `hybrid_full_bptt_batch_size >= 0`

---

## 3) 时间尺度、相机位姿与深度输入

| 参数 | 默认值 | 当前值 | 含义 | 如何取值 |
| --- | ---: | ---: | --- | --- |
| `--timesteps` | `150` | `80` | 每个 rollout 步数 | 先短后长常见；长序列更真实但更难 |
| `--base_control_freq` | `15.0` | `15.0` | 基础控制频率（Hz） | 通常固定与动力学一致 |
| `--max_acc_cmd` | `20.0` | `20.0` | 动作加速度限幅 | 太小飞不动，太大易抖 |
| `--fov_x_half_tan` | `0.53` | `0.82` | 半视场角正切 | 对应 $FOV_x=2\arctan(t)$；建议贴近标定 |
| `--cam_angle` | `10` | `20` | 相机俯仰角（度） | 大一些更看近障；太大会丢远处 |
| `--depth_width` | `64` | `64` | 深度渲染宽 | 提高会增加算力开销 |
| `--depth_height` | `48` | `48` | 深度渲染高 | 同上 |
| `--depth_min_valid` | `0.3` | `0.3` | 最小可信深度 | 必须 `>0`；与无效掩码语义一致 |
| `--depth_max_range` | `6.0` | `6.0` | 最大可信量程 | 必须 `> depth_min_valid` |
| `--depth_nn_width` | `16` | `32` | 网络输入深度宽 | 更大保细节，代价是算力/显存 |
| `--depth_nn_height` | `12` | `24` | 网络输入深度高 | 同上 |
| `--depth_use_pipeline` | `True` | `True` | 启用深度预处理流水线 | 通常建议开启 |
| `--diff_sensor_impl` | `diff_depth=python` | `diff_depth=python` | 传感后端选择 | 训练调试建议 `python`；速度优先可试 `cuda` |

`diff_sensor_impl` 目前只支持键：`diff_depth`，值：`python/cuda`。

---

## 4) 场景与环境配置

| 参数 | 默认值 | 当前值 | 含义 | 如何取值 |
| --- | ---: | ---: | --- | --- |
| `--yaw_drift` | `False` | `False` | 模拟偏航漂移 | 做鲁棒性实验时开启 |
| `--no_odom` | `False` | `False` | 关闭里程计输入 | 任务更难，适合做消融 |
| `--scenarios` | `['base']` | `['sun_glare']` | 场景列表 | 训练可多场景混训，评估可轮转 |
| `--scene_fit_profiles_path` | `None` | `configs/scene_fit_profiles.json` | 场景 profile 覆盖文件 | 建议用于实测拟合后的配置 |
| `--ellipsoid_collision` | `False` | `False` | 椭球碰撞模型 | 需要更真实机体几何时开 |
| `--drone_a` | `0.15` | `0.15` | 椭球 XY 半轴 | 仅椭球碰撞开启时有效 |
| `--drone_c` | `0.075` | `0.075` | 椭球 Z 半轴 | 同上 |
| `--coef_tilt` | `0.0` | `0.0` | 侧倾损失权重 | 默认 0；专项需求再启用 |

`scenarios` 支持：

- `base`
- `sun_glare`
- `specular_trap`
- `vantablack_gap`
- `dark_morphing`

别名会自动映射（例如 `random` -> `base`）。

---

## 5) 物理与控制损失参数

| 参数 | 默认值 | 当前值 | 含义 | 调参建议 |
| --- | ---: | ---: | --- | --- |
| `--coef_v` | `1.0` | `2.5` | 速度跟踪主损失 | 太保守可增；太激进可降 |
| `--coef_v_pred` | `2.0` | `0.0` | 辅助速度预测损失 | 想加强表征可开启 |
| `--coef_collide` | `2.0` | `10.0` | 碰撞惩罚 | 撞得多就增；过保守可降 |
| `--coef_obj_avoidance` | `1.5` | `4.0` | 安全边距惩罚 | 过近擦边可增 |
| `--coef_d_acc` | `0.01` | `0.1` | 动作幅值正则 | 太猛可增；动作偏软可降 |
| `--coef_d_jerk` | `0.001` | `0.2` | 动作变化率正则 | 抖动大优先增 |
| `--coef_ground_affinity` | `0.0` | `0.0` | 贴地偏好 | 通常不开 |

常见联调：

- 撞得多：增 `coef_collide`、`coef_obj_avoidance`
- 抖动大：增 `coef_d_jerk`（再看 `coef_d_acc`）
- 太慢：增 `coef_v` 或适度降安全项

---

## 6) 相机控制监督损失（策略输出层）

| 参数 | 默认值 | 当前值 | 含义 | 调参建议 |
| --- | ---: | ---: | --- | --- |
| `--include_camera_state_in_obs` | `False` | `True` | 将相机状态拼入观测 | 通常建议开启 |
| `--coef_cam_smooth` | `0.01` | `100` | 相机参数时序平滑 | 太大相机会“僵硬”，太小会抖 |
| `--coef_power_reg` | `0.005` | `100` | power 偏离 `cam_power_nominal` 的正则 | 控制功耗/反射风险 |
| `--coef_cam_range` | `0.001` | `1` | exposure/gain 偏离 0.5 正则 | 防止长期贴边策略 |
| `--coef_diff_depth_power` | `0.01` | `20` | 高功率惩罚 | 抑制“暴力打光” |
| `--coef_diff_depth_blur` | `0.01` | `0.1` | 模糊代理惩罚 | 压制高速+长曝光拖影 |
| `--coef_diff_depth_noise` | `0.01` | `5` | 增益噪声惩罚 | 控制高 gain 噪声放大 |
| `--coef_diff_depth_fill` | `0.0` | `30.0` | 填充率不足惩罚 | 防黑屏/空洞的关键项 |
| `--diff_depth_min_fill_rate` | `0.18` | `0.7` | 最低填充率目标 | 常见范围 `0.4~0.8` |

说明：

- `coef_power_reg` 围绕的是 `cam_power_nominal`，不是固定 0.5。
- `coef_diff_depth_power` 的起罚点是 `cam_power_penalty_threshold`，不是固定 0.5。

---

## 7) 相机 realism 参数（重点详细版）

### 7.1 先建立直觉（深度相机新手版）

在本项目里你可以把三控制量理解成：

- `power`：主动光强度（像“打手电强度”）
- `exposure`：曝光积分时间（收光时长）
- `gain`：电子放大（放大同时也放大噪声）

重要说明：

- 这些不是 D455 SDK 寄存器值。
- 这里用的是“D455 风格语义映射”，目标是拟合失效模式（拖影、噪声、高光、空洞等）。

### 7.2 档位参数（全局 realism）

| 参数 | 默认值 | 当前值 | 含义 | 建议 |
| --- | ---: | ---: | --- | --- |
| `--cam_realism_preset` | `high` | `low` | 退化强度档位 | 新手先 `low` |
| `--cam_enable_specular` | `True` | `True` | 镜面高光失效 | 研究反光场景建议开 |
| `--cam_enable_motion_blur` | `True` | `True` | 运动模糊 | 研究高速场景建议开 |
| `--cam_noise_scale` | `1.0` | `1.0` | 读噪声缩放 | >1 更脏，<1 更干净 |
| `--cam_blur_scale` | `1.0` | `1.0` | 模糊强度缩放 | >1 更易拖影 |
| `--cam_fog_scale` | `1.0` | `1.0` | 雾化衰减缩放 | >1 更雾 |
| `--cam_lighting_scale` | `1.0` | `1.0` | 环境照明缩放 | >1 更亮，<1 更暗 |
| `--cam_model_randomize` | `True` | `True` | 训练时对 grouped sensor params 做小范围随机化 | sim2real 推荐开启；评测会自动固定 |
| `--cam_model_randomize_scale` | `0.08` | `0.08` | grouped sensor params 的随机化幅度 | 常用 `0.03~0.12` |

### 7.3 曝光映射参数（`cam_exposure_*`）

| 参数 | 默认值 | 当前值 | 含义 |
| --- | ---: | ---: | --- |
| `--cam_exposure_t_min` | `0.25` | `0.25` | 归一化曝光映射下界偏置 |
| `--cam_exposure_t_span` | `2.75` | `2.75` | 归一化曝光映射跨度 |
| `--cam_exposure_eff_min` | `0.15` | `0.15` | 有效曝光最小 clamp |
| `--cam_exposure_eff_max` | `4.0` | `4.0` | 有效曝光最大 clamp |

计算过程：

$$
ex=\mathrm{clamp}(exposure01,0,1)
$$

$$
t_{cmd}=cam\_exposure\_t\_min+cam\_exposure\_t\_span\cdot ex
$$

$$
exposure_s=\mathrm{clamp}(t_{cmd},cam\_exposure\_eff\_min,cam\_exposure\_eff\_max)
$$

当前值下：

$$
t_{cmd}=0.25+2.75\cdot ex\in[0.25,3.0]
$$

因此通常不会触发 `[0.15,4.0]` 的 clamp。

直观影响：

- `exposure_s` 越大，图像更亮，但运动模糊风险也更高。

### 7.4 增益映射参数（`cam_iso_gain_*`）

| 参数 | 默认值 | 当前值 | 含义 |
| --- | ---: | ---: | --- |
| `--cam_iso_gain_base` | `1.0` | `1.0` | 增益基线 |
| `--cam_iso_gain_scale` | `10.0` | `10.0` | 增益缩放 |
| `--cam_iso_gain_gamma` | `1.2` | `1.2` | 非线性指数 |

公式：

$$
gain\_scale=cam\_iso\_gain\_base + cam\_iso\_gain\_scale\cdot gain01^{cam\_iso\_gain\_gamma}
$$

当前参数示例：

- `gain01=0` -> `1.0`
- `gain01=0.5` -> 约 `5.35`
- `gain01=1` -> `11.0`

直观影响：

- 增益越高，暗处可见性提高，但噪声会被放大。

### 7.5 噪声基础参数

| 参数 | 默认值 | 当前值 | 含义 | 建议 |
| --- | ---: | ---: | --- | --- |
| `--cam_shot_noise_base` | `0.03` | `0.03` | shot noise 基础系数 | 和 `cam_noise_scale` 联调，避免同时大幅拉高 |

### 7.6 相机调参建议（新手实操）

推荐顺序：

1. 固定 `cam_realism_preset=low`。
2. 先把 fill/hole 调稳：`coef_diff_depth_fill` + `diff_depth_min_fill_rate`。
3. 再处理拖影：`coef_diff_depth_blur`、`cam_blur_scale`。
4. 再处理噪声：`coef_diff_depth_noise`、`cam_noise_scale`。
5. 最后再动语义映射常数：`cam_exposure_*`、`cam_iso_gain_*`。

经验起点（非硬规则）：

- `cam_exposure_t_span`: `2.0~3.5`
- `cam_exposure_eff_max`: `3.0~5.0`
- `cam_iso_gain_scale`: `6~14`
- `diff_depth_min_fill_rate`: `0.4~0.8`

---

## 8) Teacher-Student 蒸馏参数

| 参数 | 默认值 | 当前值 | 含义 | 建议 |
| --- | ---: | ---: | --- | --- |
| `--enable_teacher_student_training` | `False` | `False` | 启用 teacher-student | 先关闭跑通，再开启 |
| `--teacher_inner_steps` | `10` | `10` | teacher 内循环步数 | 大则更强 teacher、开销更大 |
| `--teacher_inner_lr` | `0.01` | `0.005` | teacher 内循环 LR | 太大不稳，太小收敛慢 |
| `--distill_coef` | `1.0` | `1.0` | 蒸馏损失系数 | 可配合退火使用 |
| `--student_physics_coef` | `0.3` | `0.6` | 学生物理损失系数 | 提高可强化任务目标 |
| `--distill_final_ratio` | `0.3` | `0.5` | 蒸馏末端比例 | 越低越偏向后期放手学生 |
| `--student_noise_mode` | `off` | `off` | 学生噪声策略 | 鲁棒性实验可 `on` |
| `--teacher_tbptt_chunk_steps` | `10` | `10` | teacher TBPTT chunk | 教师长序列内存紧张时可调 |

---

## 9) dMPC / intent 参数

| 参数 | 默认值 | 当前值 | 含义 | 建议 |
| --- | ---: | ---: | --- | --- |
| `--use_dmpc` | `False` | `False` | 启用 dMPC 解码 | 需要 intent 路径时开启 |
| `--policy_output_intent` | `False` | `False` | 策略输出 intent | 与 `use_dmpc` 配套 |
| `--inject_depth_into_lqr` | `False` | `False` | 深度斥力注入 LQR | 仅在 dMPC 模式下有意义 |
| `--lqr_horizon` | `5` | `5` | LQR 规划时域 | 大则更前瞻但更慢 |
| `--lqr_reg` | `1e-4` | `1e-3` | LQR 正则 | 大更稳更保守 |
| `--depth_safe_dist` | `0.6` | `0.6` | 深度安全距离阈值 | 按障碍尺度调整 |
| `--depth_repel_gain` | `1.0` | `1.0` | 深度斥力增益 | 过大易抖，过小避障弱 |

提示：

- 若开了 `use_dmpc` 但没开 `policy_output_intent`，运行期会警告并回退到动作域控制。

---

## 10) 日志与可视化参数

| 参数 | 默认值 | 当前值 | 含义 | 建议 |
| --- | ---: | ---: | --- | --- |
| `--wandb_disabled` | `False` | `False` | 禁用 wandb | 离线调试可开 |
| `--wandb_log_raw_loss_terms` | `False` | `False` | 记录 raw loss | 诊断时可开，日志会更密 |
| `--vis_enable` | `False` | `False` | 启用可视化 | 训练性能敏感场景建议关 |
| `--vis_backend` | `rerun` | `rerun` | 可视化后端 | 当前仅支持 rerun |
| `--vis_env_idx` | `0` | `0` | 显示的环境索引 | 看异常样本时切索引 |
| `--vis_every_iters` | `10` | `10` | 每多少迭代可视化 | 开销大时增大 |
| `--vis_every_steps` | `10` | `10` | 每多少步记一帧 | 开销大时增大 |
| `--vis_teacher` | `True` | `True` | 是否可视化 teacher | 不看 teacher 可关 |
| `--vis_student` | `True` | `True` | 是否可视化 student | 通常保留 |
| `--vis_spawn` | `True` | `True` | 自动拉起窗口 | 远程环境可 `--no-vis_spawn` |

---

## 11) 评估脚本额外参数（`eval.py`）

`eval.py::parse_eval_args()` 在基础参数之外增加：

| 参数 | 默认值 | 含义 | 建议 |
| --- | ---: | --- | --- |
| `--eval_episodes` | `1` | 评估 episode 数 | 快速检查 `1~3`，稳健统计 `10+` |

---

## 12) 运行期硬约束（来自 `validate_args()`）

- `tbptt_enable=True` 时 `tbptt_chunk_steps >= 2`
- `tbptt_chunk_accum >= 1`
- `depth_min_valid > 0`
- `depth_max_range > depth_min_valid`
- `hybrid_full_bptt_every >= 0`
- `hybrid_full_bptt_batch_size >= 0`
- `depth_width/depth_height/depth_nn_width/depth_nn_height >= 1`
- `scenarios` 不可为空

---

## 13) 一键上手的调参策略（推荐）

如果你的目标是“先稳后强”，推荐这样做：

1. 先固定场景（如当前 `sun_glare`），确认训练曲线稳定。
2. 优先稳定飞行：先调 `coef_collide/coef_obj_avoidance/coef_d_jerk`。
3. 再保证可见性：调 `coef_diff_depth_fill` 与 `diff_depth_min_fill_rate`。
4. 再压拖影和噪声：调 `coef_diff_depth_blur`、`coef_diff_depth_noise`。
5. 最后才改语义映射常数（`cam_exposure_*`、`cam_iso_gain_*`）。

一句话：先把“飞得稳 + 看得见”做扎实，再做 realism 强化。
| `--cam_power_nominal` | `0.5` | `0.416667` | power 中性/默认参考值 | 建议按真实 D455 默认 `laser_power/max` 设置 |
| `--cam_power_penalty_threshold` | `0.5` | `0.416667` | 高功率惩罚起始阈值 | 建议先与 `cam_power_nominal` 保持一致 |

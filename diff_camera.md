我先把任务聚焦在“仅可微相机/光照仿真”并整理成可落地的架构方案，输出一套从模型到训练到验证的完整蓝图。

Created 3 todos

非常好，这个聚焦非常专业。我们就**只讨论可微相机与光照仿真**，不展开你暂时不关心的其它系统问题。

## 目标定义（你真正要的）

你要的是一套在当前项目中可实施的“高保真可微相机方案”，让 `render_main_luma_diff` 从现在的“深度代理亮度”升级为“更接近真实成像链路”的可微版本，尽可能缩小视觉域差距。

我给你的是一个“**接近完美但仍工程可落地**”的方案，分层设计如下。

---

## 总体架构：从几何深度到可微亮度成像

把当前路径：

- 几何深度 `depth_raw`
- 简单映射 `y = depth * exposure_scale * iso_scale`

升级为 7 层可微链路：

1. **几何层**：深度 + 近似法线 + 材质先验  
2. **光照层**：环境光 + 主光源 + 阴影近似 + 大气散射  
3. **反射层**：Lambert + 轻量镜面（可选）  
4. **镜头层**：vignetting + 畸变 + flare近似（可选）  
5. **传感器层**：曝光积分 + shot/read noise + PRNU/DSNU  
6. **ISP层**：黑电平、增益、tone mapping、gamma、锐化/去噪近似  
7. **时序层**：自动曝光(AE)状态机 + 运动模糊（rolling/global可选）

核心思想：  
让最终 $Y_t$ 变成
$$
Y_t = \mathrm{ISP}\Big(\mathrm{Sensor}\big(\mathrm{Lens}(\mathrm{Shading}(D_t, N_t, M_t, L_t)),\ \theta^{cam}_t\big)\Big)
$$
其中每个模块都保持可微（或用平滑近似可微）。

---

## 模块级“比较完美”实现方案

### 1) 几何增强（在你现有 depth 渲染上最小改造）

你已经有 `depth_raw`，下一步做：

- 用 Sobel/差分从深度恢复近似法线：
  - $N \propto (-\partial_x D,\ -\partial_y D,\ 1)$ 再归一化
- 增加“语义材质 ID”或“几何类材质先验”：
  - ground / voxel / ball / cylinder / drone
  - 每类给可学习或可随机的反照率区间 $\rho \in [\rho_{min},\rho_{max}]$

这样可以在不做完整纹理渲染的前提下，先拥有“不同物体亮暗差异”。

---

### 2) 光照模型（轻量但效果强）

建议采用三项叠加：

- **环境光** $I_{amb}$（随场景随机）
- **定向主光** $I_{dir}\max(0, N\cdot L)$（太阳/主灯）
- **高度雾化衰减**（可选，提升远处 realism）

简化表达：
$$
I_{surf} = \rho \cdot \big(I_{amb} + I_{dir}\,\mathrm{ReLU}(N\cdot L)\big)
$$

再乘以距离衰减（弱版本，避免太假）：
$$
I_{dist} = I_{surf}\cdot \exp(-k_d D)
$$

再做阴影近似（关键）：
- 用少量二次采样 ray-march 或 “屏幕空间遮挡近似”
- 不求物理完美，求统计接近真实对比度分布

---

### 3) 相机镜头 + 传感器噪声（你最需要的）

#### 镜头效应
- **Vignetting**（暗角）：半径函数 $V(r)=1-\alpha r^2-\beta r^4$
- **PSF 模糊**：可微高斯核，$\sigma$ 与焦距/运动状态可关联
- **畸变**（可选）：轻量径向畸变 $k_1,k_2$

#### 传感器噪声（必须做）
将观测建模为：
$$
Y_{raw} = \mathrm{Poisson}(g\cdot I\cdot t_{exp}) + \mathcal{N}(0,\sigma_{read}^2) + \epsilon_{PRNU}
$$
训练里用可微近似：
- Poisson 用高斯近似（方差随信号变化）
- PRNU/DSNU 用乘性+加性固定图样噪声（每 episode 随机）

这一步对 sim2real 提升非常大，尤其低照和高 ISO 场景。

---

### 4) 自动曝光 AE 动态（非常关键）

不要只用静态 `exposure` 参数，建议做时序 AE：

- 维护状态 `ae_state_t`（target luma、积分器）
- 根据上一帧亮度直方图更新曝光
- 加入响应滞后与饱和限制

更新式建议：
$$
e_t = \log(Y^\star) - \log(\bar{Y}_{t-1})
$$
$$
\log t_{exp,t} = \mathrm{clip}\big(\log t_{exp,t-1} + k_p e_t + k_i\sum e,\ [t_{min}, t_{max}]\big)
$$

这样策略会见到“真实相机亮度在时间上会漂移”，对实机更稳。

---

### 5) ISP 近似（Y通道重点）

既然你输出的是 Y 通道，建议做最小 ISP：

- black level 校正
- gain + saturation clip
- tone mapping（softplus 或 Reinhard）
- gamma
- 轻量去噪/锐化（可选随机开关）

这会让直方图和边缘统计更接近真机 Y。

---

## 与你当前工程的对接方式（最优落地路径）

### A. 先不大改 CUDA 核心：Python 侧后处理原型
在 `render_main_luma_diff` 路径先保持：
- `depth_raw` from CUDA
- 在 env_cuda.py 里叠加上述可微 photometric pipeline（PyTorch 实现）

优点：开发快，可快速 ablation；  
缺点：速度略慢，但先验证效果值不值得。

### B. 稳定后把重计算模块下沉到 CUDA
优先下沉：
1) vignetting + blur  
2) noise + tone mapping  
3) shadow近似（若代价可控）

### C. API 形态建议
把当前 `diff_render_yuv_y(...)` 扩展为接收 `camera_state`（AE状态、噪声参数、光照参数），并返回：
- `y`
- `debug maps`：irradiance、noise_sigma、exposure_used、saturation_ratio

这样训练可监控可解释。

---

## 训练策略（确保“真提升”而非“花哨”）

按三阶段 curriculum：

1. **Phase 1：几何主导**  
   光照/噪声弱，先保证策略收敛  
2. **Phase 2：光照随机化增强**  
   强化 illumination domain randomization  
3. **Phase 3：真实相机统计拟合**  
   用少量真机静态数据拟合亮度直方图、噪声谱、曝光响应

关键：不要一上来把 photometric 难度开满，否则会拖垮策略学习。

---

## “比较完美”应达到的验收指标（仅针对可微相机）

给你 6 个硬指标（很实用）：

1. **亮度直方图距离**：sim vs real 的 KL/EMD 降低  
2. **噪声功率谱匹配**：频域统计接近  
3. **曝光动态误差**：阶跃光照下 AE 响应曲线接近  
4. **模糊统计**：边缘宽度分布接近  
5. **策略鲁棒性**：照度变化下成功率曲线更平  
6. **Ablation 增益**：逐模块加入后，真实回放验证单调改进

---

## 你现在就该优先做的 4 件事（按收益排序）

1. 加 **AE 动态 + shot/read noise**  
2. 加 **法线驱动光照（ambient + directional）**  
3. 加 **vignetting + 轻量 blur**  
4. 加 **material reflectance randomization**

这四项已经能把你当前 `render_main_luma_diff` 从“深度代理”提升到“可用的相机仿真器”。

---

## 结尾建议

你这个方向完全正确。  
如果你愿意，我下一步可以直接给你一份**“与你当前代码一一对应的实现蓝图（函数级别）”**：  
明确 env_cuda.py 里新增哪些函数、`quadsim_kernel.cu` 里哪些接口扩展、每个参数建议范围、以及默认配置表（可直接开始实现）。

## 当前待办更新

- [x] 聚焦相机需求范围  
- [x] 设计可微成像管线  
- [x] 制定训练与评估闭环
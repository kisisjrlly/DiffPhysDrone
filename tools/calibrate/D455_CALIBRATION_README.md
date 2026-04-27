# D455 标定与场景反推

## 目标

把真实 D455 在不同场景下采到的深度响应，转换成当前 `diff_depth` 仿真可直接参考的场景参数。

当前支持的论文场景：

- `sun_glare`
- `specular_trap`
- `vantablack_gap`
- `dark_morphing`

## 快速联机检查

```bash
python3 tools/test_d455_depth.py
```

用途：

- 确认 D455 已正常连接
- 确认深度流模式可以启动
- 确认曝光/增益/激光功率能被手动写入

如果你想先把当前项目里的 `cam_exposure_* / cam_iso_gain_*` 语义参数，和你手边这台 D455 的真实量程做一个初步对齐，可以直接运行：

```bash
python3 tools/recommend_d455_semantics.py
```

或者使用某次采集结果里的 `meta.json` 离线推导：

```bash
python3 tools/recommend_d455_semantics.py \
  --meta-json artifacts/d455_calibration/<run>/meta.json
```

它会输出一段可以直接粘进 `.args` 的推荐参数，例如：

- `--cam_power_baseline`
- `--cam_exposure_t_min`
- `--cam_exposure_t_span`
- `--cam_exposure_eff_min`
- `--cam_exposure_eff_max`
- `--cam_iso_gain_base`
- `--cam_iso_gain_scale`
- `--cam_iso_gain_gamma`
- `--cam_shot_noise_base`

注意：

- `cam_exposure_*` 与 D455 真机曝光时间最容易直接对齐。
- `cam_iso_gain_*` 和 `cam_shot_noise_base` 目前仍是“语义近似”和“工程初值”，不是驱动寄存器的一一复刻；最好再结合静态墙面数据做二次细调。

## 静态墙面噪声标定

如果你想专门反推：

- `cam_shot_noise_base`
- `cam_iso_gain_scale`
- `cam_iso_gain_gamma`

推荐使用新加的静态墙面标定脚本：

```bash
python3 tools/calibrate_d455_static_noise.py
```

推荐布置：

- D455 固定在三脚架或稳定支架上
- 正对一面普通哑光平整墙面
- 距离建议 `1.0m ~ 2.0m`
- 采集时视野内不要有人经过，也不要有强反光或强阳光变化

脚本会：

- 固定 `laser_power`
- 扫一组 `exposure`
- 扫一组 `gain`
- 在中心 ROI 上统计多帧时间噪声
- 输出：
  - `summary.csv`
  - `fit_result.json`
  - 一段可直接粘进 `.args` 的推荐参数

默认输出目录：

```text
artifacts/d455_static_noise/<timestamp>/
```

常用示例：

```bash
python3 tools/calibrate_d455_static_noise.py \
  --frames-per-setting 90 \
  --exposure-us-values 3000,10000,30000 \
  --gain-values 16,32,64,128,248 \
  --laser-power 150
```

如果只想离线分析已有结果：

```bash
python3 tools/calibrate_d455_static_noise.py \
  --analyze-csv artifacts/d455_static_noise/<timestamp>/summary.csv
```

说明：

- `cam_shot_noise_base` 是按静态墙面上的 temporal depth noise 反推的全局 shot-noise 尺度。
- `cam_iso_gain_scale/gamma` 这里主要拟合“gain 增大时深度噪声如何上升”。
- 这套拟合优先服务当前 `diff_depth` 仿真，不是对 D455 ISP/立体匹配链路的完整物理复刻。

## 正式采集

示例：

```bash
python3 tools/collect_d455_calibration.py \
  --scene sun_glare \
  --condition-id glare_front_exit \
  --notes "室内走廊出口逆光，手持相机向出口移动"
```

建议每个场景至少采 2 到 4 组 `condition-id`：

- `sun_glare`
  - `glare_front_exit`
  - `glare_side_window`
- `specular_trap`
  - `glass_panel`
  - `wet_floor`
- `vantablack_gap`
  - `black_foam_gate`
  - `dark_fabric_hole`
- `dark_morphing`
  - `dark_slit_static`
  - `dark_slit_fast_move`

采集输出目录默认在：

```text
artifacts/d455_calibration/<scene>_<condition-id>_<timestamp>/
```

每个目录包含：

- `capture.csv`
- `meta.json`
- `preview/*.jpg`

## 离线拟合

当你采集好若干目录后，运行：

```bash
python3 tools/fit_d455_scene_profiles.py
```

输出：

```text
artifacts/d455_calibration/scene_fit_profiles.json
```

里面最重要的是：

- `scene_profiles.sun_glare`
- `scene_profiles.specular_trap`
- `scene_profiles.vantablack_gap`
- `scene_profiles.dark_morphing`

这些字段名已经尽量对齐当前 [env_cuda.py](/home/zhaoguodong/work/code/DiffPhysDrone/env_cuda.py) 里的场景强度参数，比如：

- `ambient_add`
- `active_drop`
- `quality_penalty`
- `spec_boost_base`
- `spec_boost_scale`
- `far_override_scale`
- `albedo_drop`
- `passive_drop`
- `motion_boost`
- `ambient_global_mul`
- `slit_power_bonus`

## 自动加载到训练/评测

当前项目已经支持通过参数文件自动加载场景拟合结果：

```text
--scene_fit_profiles_path configs/scene_fit_profiles.json
```

默认主配置 [paper_final_full.args](/home/zhaoguodong/work/code/DiffPhysDrone/configs/paper_final_full.args) 和 [paper_ablate_diff_depth.args](/home/zhaoguodong/work/code/DiffPhysDrone/configs/paper_ablate_diff_depth.args) 都已经接入这条链路。

如果你想替换成自己的拟合结果，只需要把参数改成：

```text
--scene_fit_profiles_path artifacts/d455_calibration/scene_fit_profiles.json
```

## 推荐采集动作

### `sun_glare`

- 相机固定或手持，从暗处朝亮出口移动
- 尽量让出口强光只在部分视场中出现，而不是整幅图都过曝

### `specular_trap`

- 对玻璃板、镜面、湿地面分别采
- 保持同一块表面在不同入射角下重复采样

### `vantablack_gap`

- 用黑色吸音棉、黑布、深色哑光材料做门洞或障碍边框
- 保持背景和门框材质反差明显

### `dark_morphing`

- 在极暗环境中对窄缝采样
- 分静止观察和快速横移两种条件

更细的标准化采集动作请直接看：

- [D455_SCENE_COLLECTION_PROTOCOL.md](/home/zhaoguodong/work/code/DiffPhysDrone/tools/D455_SCENE_COLLECTION_PROTOCOL.md)

## 当前局限

- 这套拟合目前是“工程反推”，不是严格物理最优化。
- `specular_trap` 目前更偏向镜面/玻璃高反失效，还不是完整透明体立体匹配误导模型。
- 拟合结果更适合作为当前仿真参数初值和扫参参考，而不是一键真值。

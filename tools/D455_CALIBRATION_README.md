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

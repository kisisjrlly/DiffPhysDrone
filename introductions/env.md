# Environment

当前项目只保留同一张门洞地图，公开场景为 `glare / specular / dark`。

## 场景结构

三个场景都使用 `_build_sun_glare_voxel_layout()` 构造同一张固定小地图：

- 起点约为 `[-2.8, start_y, 1.5]`
- 终点为 `[3.0, 0.0, 1.5]`
- 中间有 occluder、三条 lane divider、一个四选一开口 gate
- 每个 episode 随机或固定一个开口：`far_left / left / right / far_right`
- `l0-l3` 控制局部退化强度

## Sensor Scenes

`--scenarios` 直接控制同一张地图里的局部传感器退化场景：

- `glare`：开口外侧有强 IR glare。高 exposure 更容易失效，提高 power 可以恢复主动深度。
- `specular`：开口附近局部强反射。过高 power 可能造成 washout，因此固定高 power 不能通吃。
- `dark`：开口边缘/门框低反射。低 exposure/gain 下边缘更难看清，适度提高 exposure/gain 更有利。

这三种模式不是三张地图；它们共享同一个门洞几何，只改变开口附近的成像机理。

## 训练目标

这个简化后的环境用于验证两个问题：

- learned camera 是否能在 `glare` 中接近强光时降低 exposure、提高 power，并在离开强光后恢复。
- learned + differentiable depth 是否能在 `glare/specular/dark` 的冲突相机需求下优于 fixed、fixed_random_static 和 nondiff。

## 关键参数

- `--scenarios glare specular dark`
- `--sun_glare_levels l0 l1 l2 l3`
- `--sun_glare_eval_slot far_left|left|right|far_right`

# D455 Calibration

当前校准工具面向共享门洞地图的 `glare/specular/dark` 场景。

## 支持对象

- 场景：`glare / specular / dark`

三种子模式共享同一张门洞地图，只改变开口附近的局部成像机理：

- `glare`：强 IR 环境光/太阳光造成主动深度退化。
- `specular`：开口区域反射增强，高 power 可能造成 washout。
- `dark`：开口区域低反射，低 exposure/gain 下边缘更难恢复。

## 输出 profile

`configs/scene_fit_profiles.json` 只保留 `glare/specular/dark` 键：

- `sensor_profiles.glare/specular/dark`
- `scene_effects.glare/specular/dark`

旧的多场景 profile 已删除，不再作为配置入口。

## 使用建议

采集时按场景分开标注，然后把拟合结果写入对应场景的 profile。评估时使用：

```bash
EVAL_EXTRA_ARGS="--scenarios glare --sun_glare_eval_slot right" bash eval.sh
EVAL_EXTRA_ARGS="--scenarios specular --sun_glare_eval_slot right" bash eval.sh
EVAL_EXTRA_ARGS="--scenarios dark --sun_glare_eval_slot right" bash eval.sh
```

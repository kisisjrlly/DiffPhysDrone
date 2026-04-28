# Rerun Visualization

当前 Rerun 面向共享门洞地图的 `glare/specular/dark` 场景。

## 推荐观察项

- `/student_3d/world/scene/sun_anchor`：`glare` 场景的局部 IR/glare anchor。
- `/student_3d/world/scene/local_sensor_region`：开口附近的局部退化区域。
- depth / quality / invalid / scene effect 图像：判断相机参数是否真的影响深度。
- `power / exposure / gain` 曲线：判断 learned camera 是否按场景调参。
- `scene_effect_mean / glare_invalid_rate / glare_quality_mean`：判断无人机是否进入强退化区域。

## Regime 语义

- `glare`：理想行为是接近开口强光时降低 exposure、提高 power，离开后恢复。
- `specular`：过高 power 可能造成反射 washout，策略不应一直高 power。
- `dark`：低反射边缘需要更合理的 exposure/gain，不能只靠固定低 exposure。

## 多 Episode 查看

`eval.py` 会把多个 episode 写到 `/episodes/ep_XXX/student` 下。运行完后在 Rerun 里选择某个 episode 对应的 entity，就可以只看该 episode 的轨迹和曲线。

## 建议评估命令

```bash
EVAL_EXTRA_ARGS="--scenarios glare --sun_glare_eval_slot right" bash eval.sh
EVAL_EXTRA_ARGS="--scenarios specular --sun_glare_eval_slot right" bash eval.sh
EVAL_EXTRA_ARGS="--scenarios dark --sun_glare_eval_slot right" bash eval.sh
```

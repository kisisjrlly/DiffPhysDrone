# D455 Scene Collection Protocol

当前采集协议覆盖共享门洞地图的 `glare/specular/dark` 场景。

## 采集目标

在同一门洞/开口任务中分别采集三类局部传感器退化：

- `glare`：开口外侧或墙后方放置强 IR 光源，验证低 exposure + 高 power 是否能恢复深度。
- `specular`：开口区域使用强反射材料，验证高 power 是否会造成局部 washout。
- `dark`：开口边缘使用低反射材料，验证 exposure/gain 对暗边缘深度恢复的影响。

## 推荐采集矩阵

每个 regime 采集以下组合：

- 开口位置：`far_left / left / right / far_right`
- 场景模式：`glare / specular / dark`，局部退化强度使用当前默认常数
- 相机参数：固定低/中/高 power，固定低/中/高 exposure，固定低/中/高 gain

## 输出要求

每段数据至少记录：

- RGB 或 IR 图像
- 深度图
- `power / exposure / gain`
- 开口位置与场景标签
- 是否成功通过开口

校准结果写回 `configs/scene_fit_profiles.json` 的 `glare/specular/dark` profile。

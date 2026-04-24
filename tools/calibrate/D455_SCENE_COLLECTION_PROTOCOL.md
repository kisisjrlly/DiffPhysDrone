# D455 论文场景推荐采集 Protocol

## 目的

这份 protocol 的目标不是“随便录点深度图”，而是为了让真实 D455 数据能反推当前 `diff_depth` 仿真中的场景参数。

因此每个场景都要尽量满足两点：

1. 真实场景的主导失效机理足够单一。
2. 采集时能明确记录相机相对场景的距离、角度、运动方式和材质。

## 通用器材与布置

### 相机

- 设备：`Intel RealSense D455`
- 安装方式：
  - 优先固定在轻便手持支架或小滑轨上
  - 次优方案是双手稳定手持
- 相机高度建议：`1.2m` 到 `1.5m`
- 默认前向采集，机身滚转保持接近 `0 deg`

### 标尺与记录

- 在地面贴 `0.5m` 间距标记
- 在场景边缘贴卷尺或 AprilTag 辅助记录距离
- 每组条件记录：
  - 距离
  - 入射角
  - 横移速度档位
  - 是否存在外部光源
  - 材质名称

### 每条 condition 的建议录制长度

- 静态条件：`20` 到 `30` 秒
- 匀速移动条件：每次来回 `3` 到 `5` 趟
- 快速横移条件：至少 `10` 秒

### 推荐 condition 数量

- 每个场景至少 `2` 个 condition
- 理想情况：每个场景 `4` 到 `6` 个 condition`

## 通用距离与角度档位

### 距离档位

- 近距离：`0.6m`
- 中距离：`1.2m`
- 远距离：`2.0m`

### 入射角档位

- 正视：`0 deg`
- 斜视：`20 deg`
- 大斜角：`35 deg`

### 横移速度档位

- 静止：`0 m/s`
- 慢速：`0.2` 到 `0.4 m/s`
- 中速：`0.5` 到 `0.8 m/s`
- 快速：`1.0` 到 `1.5 m/s`

## 场景一：`sun_glare`

### 目标机理

- 强外部红外或强光进入视场
- 主动散斑被局部 washout
- fill rate 与 invalid ratio 明显受曝光/激光功率耦合影响

### 场地布置

- 室内暗走廊朝向明亮出口
- 或室内对着强日照窗户
- 出口或窗户前方最好有门框、立柱或边界结构

### 材质要求

- 门框或出口边界尽量使用普通浅色墙面/木门框
- 不要同时出现强镜面玻璃，以免混入 `specular_trap` 机理

### 相机运动

- `Protocol A1`：静止对出口
  - 距离：`2.0m -> 1.2m -> 0.6m`
  - 角度：`0 deg`
- `Protocol A2`：从暗处匀速向出口靠近
  - 起始距离：`2.5m`
  - 结束距离：`0.6m`
  - 速度：慢速、中速各一组
- `Protocol A3`：固定位置，轻微左右摆头
  - 距离：`1.2m`
  - 角度变化：`-25 deg` 到 `25 deg`

### 推荐 condition 命名

- `glare_front_exit`
- `glare_side_window`
- `glare_approach_slow`
- `glare_approach_fast`

### 重点记录

- 强光是否只占视场一部分
- 门框边缘是否还能保住
- 不同曝光下是否出现“全黑”和“局部可见”两种明显状态

## 场景二：`specular_trap`

### 目标机理

- 高功率主动光打到高反表面后出现局部 bloom、错误远深度或深度洞
- 降功率后被动纹理/环境散射可能略有恢复

### 场地布置

- 透明玻璃板
- 镜面亚克力板
- 湿地面或浅水反光面

建议分开采，不要在同一条 condition 里混多种高反材质。

### 材质要求

- `glass_panel`：透明玻璃门或玻璃板
- `mirror_panel`：镜面板
- `wet_floor`：深色地砖上喷薄水层

### 相机运动

- `Protocol B1`：固定距离改变入射角
  - 距离：`1.2m`
  - 角度：`0/20/35 deg`
- `Protocol B2`：固定正视缓慢靠近
  - 距离：`2.0m -> 0.6m`
  - 速度：慢速
- `Protocol B3`：沿表面平行横移
  - 距离：`1.0m`
  - 横移：慢速、中速

### 推荐 condition 命名

- `glass_panel`
- `glass_panel_oblique`
- `mirror_panel`
- `wet_floor`

### 重点记录

- 反光区域在图像中的位置
- 正视与斜视是否表现明显不同
- 低激光和高激光时 fill rate、variance 是否出现反转

## 场景三：`vantablack_gap`

### 目标机理

- 极低反照率的黑色哑光材质导致主动和被动回波都弱
- 策略应更依赖高功率、受限曝光和增益权衡

### 场地布置

- 用黑色吸音棉、黑色植绒布、黑色哑光海绵构成门洞边框
- 门洞后方背景尽量浅色或普通墙体，形成明显边界

### 门洞几何建议

- 洞中心高度：`1.3m` 到 `1.6m`
- 洞宽：`0.7m` 到 `1.1m`
- 洞高：`1.2m` 到 `1.8m`

### 相机运动

- `Protocol C1`：正视静止
  - 距离：`2.0m / 1.2m / 0.6m`
- `Protocol C2`：正视慢速接近
  - 距离：`2.0m -> 0.6m`
- `Protocol C3`：中速横移掠过门洞
  - 距离：`1.0m`
  - 速度：中速

### 推荐 condition 命名

- `black_foam_gate`
- `black_fabric_gate`
- `black_gate_approach`
- `black_gate_lateral_fast`

### 重点记录

- 黑色门框附近是否明显先丢失
- 增益升高后是否 variance 上升但 fill 稍有恢复
- 运动时是否更容易在黑边缘产生拖影或破洞

## 场景四：`dark_morphing`

### 目标机理

- 极低照度下的狭窄通道或缝隙
- 曝光一旦太长，运动模糊会迅速恶化
- 高 gain 能救一点 fill，但也会带来更强噪声

### 场地布置

- 关灯或低照度环境
- 用两块板材形成窄缝
- 缝后方保持简单背景，不叠加强反或强逆光

### 缝隙几何建议

- 缝宽：`0.25m` 到 `0.45m`
- 缝高：`1.0m` 到 `1.5m`
- 缝边缘材质：
  - 一组普通暗色板
  - 一组深黑哑光板

### 相机运动

- `Protocol D1`：静止观察缝隙
  - 距离：`1.5m / 1.0m / 0.6m`
- `Protocol D2`：横向匀速穿过视场
  - 距离：`1.0m`
  - 速度：慢速、中速
- `Protocol D3`：快速近距离掠过
  - 距离：`0.6m`
  - 速度：快速

### 推荐 condition 命名

- `dark_slit_static`
- `dark_slit_mid_speed`
- `dark_slit_fast`
- `dark_slit_black_edges`

### 重点记录

- 静止时长曝光能否显著提升 fill
- 快速运动时 variance 与 edge_std 是否迅速上升
- 不同材质边缘是否明显改变最佳曝光/增益组合

## 推荐最小采集矩阵

如果时间有限，最少先采这 8 组：

1. `sun_glare / glare_front_exit`
2. `sun_glare / glare_approach_slow`
3. `specular_trap / glass_panel`
4. `specular_trap / wet_floor`
5. `vantablack_gap / black_foam_gate`
6. `vantablack_gap / black_gate_lateral_fast`
7. `dark_morphing / dark_slit_static`
8. `dark_morphing / dark_slit_fast`

## 建议执行顺序

1. 先用 `python3 tools/test_d455_depth.py` 确认相机正常。
2. 每个场景先采一组静态 condition。
3. 再采一组运动 condition。
4. 每采完一个场景，先运行一次 `python3 tools/fit_d455_scene_profiles.py` 看结果是否有明显异常。
5. 若某场景拟合结果不稳定，再补同场景不同材质或不同角度条件。

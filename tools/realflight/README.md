# 真机 D455 深度监控与调参

## 链路

机载电脑（ROS Noetic）：

- `realsense2_camera` 发布 `/camera/depth/image_rect_raw`（640x480@15，深度单位为 mm）
- `depth_web_tool.py` 订阅深度话题，彩色化后通过 HTTP MJPEG 推流，并把页面上的
  滑块映射到 `/camera/stereo_module/set_parameters`（dynamic_reconfigure）

主机：浏览器直接打开 `http://192.168.1.208:8090`，无需安装任何软件。
该页面只应在可信局域网内使用（当前无鉴权）。

## 机载端命令

```bash
# 一键启动相机 + Web 工具（自动脱离 SSH 会话）
bash ~/start_depth_tools.sh

# 只启动/重启 Web 工具（相机已在运行时可单独用）
bash ~/start_depth_web.sh
```

页面控件：

- 自动曝光开关
- 曝光（us）
- 增益（D455 下限 16）
- 激光功率（mW）
- 发射器（关闭/激光/激光自动/LED）

## 主机端命令

```bash
bash tools/realflight/open_depth_web.sh
```

或手动打开 `http://192.168.1.208:8090`。

## 常用调试

```bash
# 机载上直接设置参数
rosrun dynamic_reconfigure dynparam set /camera/stereo_module exposure 10000
rosrun dynamic_reconfigure dynparam set /camera/stereo_module gain 16
rosrun dynamic_reconfigure dynparam set /camera/stereo_module laser_power 150.0
rosrun dynamic_reconfigure dynparam set /camera/stereo_module emitter_enabled 1
```

## 已知问题与解决

1. D455 在 Orin NX 上曾被内核 `uvcvideo` 反复占用导致深度流不发布，已通过
   `/etc/modprobe.d/blacklist-uvcvideo.conf` 持久化黑名单解决。
2. 曝光/增益动态设置曾报 `get_xu(id=11) ... Resource temporarily unavailable`，
   根因是旧驱动残留导致自动曝光 XU 处于坏状态；USB 设备重新枚举/多次重启后恢复。
   若再出现，执行：
   `sudo python3 -c "import usb.core,usb.util; d=usb.core.find(idVendor=0x8086,idProduct=0x0b5c); d.reset()"`
   或重启相机进程后重试。

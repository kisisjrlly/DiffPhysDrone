import pyrealsense2 as rs
import sys
import time
import os
import cv2
import numpy as np

# ==========================================
# 1. 初始化与配置
# ==========================================
pipeline = rs.pipeline()
profile = None
pipeline_started = False
depth_mode = None

# 实时写入的深度可视化图像路径（每帧覆盖同一个文件）
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "depth.jpg")

# 深度流候选模式（有些连接条件下 848x480@30 可能无法协商成功）
depth_mode_candidates = [
    (848, 480, 30),
    (640, 480, 30),
    (640, 480, 15),
    (424, 240, 30),
]

try:
    print("正在启动 D455 并连接【深度】传感器...")
    ctx = rs.context()
    if len(ctx.query_devices()) == 0:
        raise RuntimeError("未检测到 RealSense 设备，请检查 USB 连接和供电。")

    last_start_error = None
    for width, height, fps in depth_mode_candidates:
        try:
            config = rs.config()
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            print(f"尝试深度模式: {width}x{height}@{fps} ...")
            profile = pipeline.start(config)
            pipeline_started = True
            depth_mode = (width, height, fps)
            print(f"深度模式启动成功: {width}x{height}@{fps}\n")
            break
        except RuntimeError as e:
            last_start_error = e
            print(f"深度模式启动失败: {width}x{height}@{fps} | 原因: {e}")

    if not pipeline_started:
        raise RuntimeError(
            "无法协商任何深度流模式（Couldn't resolve requests）。"
            "请确认相机未被 RealSense Viewer/其他进程占用、USB3 正常连接，"
            f"最后错误: {last_start_error}"
        )
    assert profile is not None
    
    # ==========================================
    # 2. 获取深度物理传感器句柄 (核心)
    # ==========================================
    depth_sensor = profile.get_device().first_depth_sensor()

    # ==========================================
    # 3. 夺取底层控制权
    # ==========================================
    # 必须先关闭深度相机的硬件自动曝光，否则手动曝光和增益写不进去
    if depth_sensor.get_option(rs.option.enable_auto_exposure):
        print("检测到深度自动曝光已开启，正在强制关闭...")
        depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
        time.sleep(0.1)

    print("D455 深度控制通道已打通！(按 Ctrl+C 退出)\n")

    # ==========================================
    # 4. 查硬件极限 (防爆雷机制)
    # ==========================================
    exp_range = depth_sensor.get_option_range(rs.option.exposure)
    gain_range = depth_sensor.get_option_range(rs.option.gain)
    laser_range = depth_sensor.get_option_range(rs.option.laser_power)
    
    print(f"[极限] 曝光范围: {exp_range.min} - {exp_range.max} (默认: {exp_range.default})")
    print(f"[极限] 增益范围: {gain_range.min} - {gain_range.max} (默认: {gain_range.default})")
    print(f"[极限] 激光功率范围: {laser_range.min} - {laser_range.max} (默认: {laser_range.default})\n")
    if depth_mode is not None:
        print(f"[当前深度模式] {depth_mode[0]}x{depth_mode[1]}@{depth_mode[2]}")
    print(f"[图像输出] 实时深度图将保存到: {save_path}\n")

    # ==========================================
    # 5. 实时控制循环 (接入你的算法)
    # ==========================================
    # 设定基准值：当扫描某一个参数时，另外两个参数固定在基准值
    base_exposure = float(np.clip(3000, exp_range.min, exp_range.max))  # 深度曝光 (单位通常是微秒)
    base_gain = float(np.clip(16, gain_range.min, gain_range.max))       # 深度增益
    base_laser = float(np.clip(150, laser_range.min, laser_range.max))   # 激光功率 (通常最大是 360)

    # 当前真实写入值
    target_exposure = base_exposure
    target_gain = base_gain
    target_laser = base_laser

    # 三个参数分别维护各自的扫描值（切换阶段时不会丢失）
    sweep_exposure = base_exposure
    sweep_gain = base_gain
    sweep_laser = base_laser

    # 三阶段：只改一个参数，另外两个固定
    phase_names = ["exposure", "gain", "laser"]
    phase_labels = {
        "exposure": "曝光",
        "gain": "增益",
        "laser": "激光功率",
    }
    phase_index = 0
    phase_frame_count = 0
    frames_per_phase = 60   # 每个阶段持续帧数，之后切到下一个参数（更快观察参数效果）

    # 每个参数各自的步长与方向
    exposure_step = max(exp_range.step, 500.0)
    gain_step = max(gain_range.step, 2.0)
    laser_step = max(laser_range.step, 10.0)
    exposure_dir = 1.0
    gain_dir = 1.0
    laser_dir = 1.0

    vis_max_depth_mm = 5000.0  # 深度可视化上限（毫米），用于映射到 8-bit JPG

    def update_with_bounce(value, step, direction, min_value, max_value):
        next_value = value + direction * step
        if next_value >= max_value:
            next_value = max_value
            direction = -1.0
        elif next_value <= min_value:
            next_value = min_value
            direction = 1.0
        return next_value, direction

    print("[参数扫描] 每次只调整一个参数，另外两个保持基准值不变")
    print(
        f"[基准值] 曝光={base_exposure:.0f}, 增益={base_gain:.0f}, 激光功率={base_laser:.0f}"
    )
    print(
        f"[阶段规则] 每 {frames_per_phase} 帧切换一次：曝光 -> 增益 -> 激光功率\n"
    )

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # ----------------------------------------------------
        # 在这里，你的网络或者启发式算法算出新的参数
        # 比如：检测到前方是强反光玻璃 -> 立刻调低 target_laser
        # 比如：检测到无人机正在极速机动 -> 降低 target_exposure 并提高 target_gain
        # ----------------------------------------------------
        
        # 只扫描当前阶段的参数，其他参数固定为基准值
        active_phase = phase_names[phase_index]
        if active_phase == "exposure":
            sweep_exposure, exposure_dir = update_with_bounce(
                sweep_exposure,
                exposure_step,
                exposure_dir,
                exp_range.min,
                exp_range.max,
            )
            target_exposure = sweep_exposure
            target_gain = base_gain
            target_laser = base_laser
        elif active_phase == "gain":
            sweep_gain, gain_dir = update_with_bounce(
                sweep_gain,
                gain_step,
                gain_dir,
                gain_range.min,
                gain_range.max,
            )
            target_exposure = base_exposure
            target_gain = sweep_gain
            target_laser = base_laser
        else:  # active_phase == "laser"
            sweep_laser, laser_dir = update_with_bounce(
                sweep_laser,
                laser_step,
                laser_dir,
                laser_range.min,
                laser_range.max,
            )
            target_exposure = base_exposure
            target_gain = base_gain
            target_laser = sweep_laser

        # 【物理写入】：将参数打入硬件寄存器
        depth_sensor.set_option(rs.option.exposure, target_exposure)
        depth_sensor.set_option(rs.option.gain, target_gain)
        depth_sensor.set_option(rs.option.laser_power, target_laser)
        
        # 实时读取以验证
        cur_exp = depth_sensor.get_option(rs.option.exposure)
        cur_gain = depth_sensor.get_option(rs.option.gain)
        cur_laser = depth_sensor.get_option(rs.option.laser_power)

        # 将 z16 深度帧转为可直接查看的 JPG（伪彩色）
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_clipped = np.clip(depth_image, 0, vis_max_depth_mm)
        depth_8u = cv2.convertScaleAbs(depth_clipped, alpha=255.0 / vis_max_depth_mm)
        depth_vis = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
        save_ok = cv2.imwrite(save_path, depth_vis)
        
        sys.stdout.write(
            f"\r[阶段: 仅调{phase_labels[active_phase]}] "
            f"曝光: {cur_exp:.0f} | 增益: {cur_gain:.0f} | 激光功率: {cur_laser:.0f} "
            f"| 保存: {'OK' if save_ok else 'FAILED'}    "
        )
        sys.stdout.flush()

        # 到达阶段帧数后切换到下一个参数
        phase_frame_count += 1
        if phase_frame_count >= frames_per_phase:
            phase_frame_count = 0
            phase_index = (phase_index + 1) % len(phase_names)
            next_phase = phase_names[phase_index]
            print(f"\n[阶段切换] 下一阶段仅调整：{phase_labels[next_phase]}")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n程序被用户手动终止。")
except Exception as e:
    print(f"\n发生运行时错误: {e}")
finally:
    print("\n正在释放 D455 资源...")
    if pipeline_started:
        pipeline.stop()
        print(f"资源已释放，安全退出。最后一帧深度图路径: {save_path}")
    else:
        print("未完成启动，跳过 pipeline.stop()。")

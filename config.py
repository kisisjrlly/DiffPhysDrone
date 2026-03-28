"""
命令行参数解析、验证、传感模式解析。
从 main_cuda.py 中提取，保持完全相同的参数名和语义。
"""
import argparse
import os
import random
import numpy as np
import torch


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=None, help='恢复训练的模型权重路径')
    parser.add_argument('--batch_size', type=int, default=64, help='并行仿真的环境数量 (Batch Size)')
    parser.add_argument('--num_iters', type=int, default=50000, help='总训练迭代次数')
    parser.add_argument('--tbptt_enable', default=False, action='store_true',
                        help='启用分段反传 TBPTT（将长时域 rollout 分块反传）')
    parser.add_argument('--tbptt_chunk_steps', type=int, default=40,
                        help='TBPTT 分段长度（每多少步截断一次计算图）')
    parser.add_argument('--tbptt_chunk_accum', type=int, default=1,
                        help='TBPTT 下每多少个 chunk 执行一次优化器 step')
    parser.add_argument('--hybrid_full_bptt_every', type=int, default=0,
                        help='混合调度：每 N 个迭代执行一次完整 BPTT（0 表示关闭）')
    parser.add_argument('--hybrid_full_bptt_batch_size', type=int, default=0,
                        help='混合调度：完整 BPTT 迭代使用的小 batch（0 表示沿用 batch_size）')

    # --- 物理与控制损失函数权重 ---
    parser.add_argument('--coef_v', type=float, default=1.0, help='速度跟踪损失权重')
    parser.add_argument('--coef_speed', type=float, default=0.0, help='速度标量跟踪损失权重（legacy 对齐项）')
    parser.add_argument('--coef_bias', type=float, default=0.0, help='速度方向偏置损失权重（legacy 对齐项）')
    parser.add_argument('--coef_v_pred', type=float, default=2.0, help='速度预测 MSE 损失权重')
    parser.add_argument('--coef_collide', type=float, default=2.0, help='碰撞惩罚权重')
    parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='避障安全距离惩罚权重')
    parser.add_argument('--coef_d_acc', type=float, default=0.01, help='控制加速度正则化权重')
    parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='控制 Jerk 正则化权重')
    parser.add_argument('--coef_d_snap', type=float, default=0.0, help='控制 Snap 正则化权重（legacy 对齐项）')
    parser.add_argument('--coef_ground_affinity', type=float, default=0., help='贴地飞行偏好权重')

    # --- 训练超参数 ---
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--grad_decay', type=float, default=0.4, help='BPTT 梯度衰减系数')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子')
    parser.add_argument('--deterministic', default=False, action=argparse.BooleanOptionalAction,
                        help='是否启用确定性算法')
    parser.add_argument('--speed_mtp', type=float, default=1.0, help='环境最大速度乘数')
    parser.add_argument('--fov_x_half_tan', type=float, default=0.53, help='相机基础视场角')
    parser.add_argument('--timesteps', type=int, default=150, help='每个 episode 的物理步数')
    parser.add_argument('--base_control_freq', type=float, default=15.0, help='基础控制频率 (Hz)')
    parser.add_argument('--cam_angle', type=int, default=10, help='相机默认俯仰角 (度)')
    parser.add_argument('--imx_width', type=int, default=320, help='IMX279 主相机分辨率宽')
    parser.add_argument('--imx_height', type=int, default=240, help='IMX279 主相机分辨率高')
    parser.add_argument('--depth_width', type=int, default=None, help='深度相机输入分辨率宽')
    parser.add_argument('--depth_height', type=int, default=None, help='深度相机输入分辨率高')
    parser.add_argument('--depth_nn_width', type=int, default=16, help='diff_depth: 输入策略网络前的深度特征宽')
    parser.add_argument('--depth_nn_height', type=int, default=12, help='diff_depth: 输入策略网络前的深度特征高')
    parser.add_argument('--depth_use_pipeline', default=True, action=argparse.BooleanOptionalAction,
                        help='深度输入是否启用图像处理流水线（对 diff_depth 与 camera_luma_plus_depth 均生效）')
    # 向后兼容旧参数名：--diff_depth_use_pipeline / --no-diff_depth_use_pipeline
    parser.add_argument('--diff_depth_use_pipeline', dest='depth_use_pipeline',
                        action=argparse.BooleanOptionalAction,
                        help=argparse.SUPPRESS)
    parser.add_argument('--diff_sensor_impl', nargs='*', default=['camera_luma=python', 'diff_depth=python'],
                        help='可微传感实现后端列表')
    parser.add_argument('--policy_input_width', type=int, default=None, help='已弃用')
    parser.add_argument('--policy_input_height', type=int, default=None, help='已弃用')

    # --- 环境变体开关 ---
    parser.add_argument('--single', default=False, action='store_true', help='单机模式')
    parser.add_argument('--gate', default=False, action='store_true', help='启用穿越门环境')
    parser.add_argument('--ground_voxels', default=False, action='store_true', help='启用复杂地面环境')
    parser.add_argument('--scaffold', default=False, action='store_true', help='启用脚手架环境')
    parser.add_argument('--random_rotation', default=False, action='store_true', help='随机旋转整个场景')
    parser.add_argument('--yaw_drift', default=False, action='store_true', help='模拟偏航角漂移')
    parser.add_argument('--no_odom', default=False, action='store_true', help='无里程计模式')
    parser.add_argument('--wall_slit', default=False, action='store_true', help='狭缝穿越环境')
    parser.add_argument('--ellipsoid_collision', default=False, action='store_true', help='使用椭球体碰撞检测')
    parser.add_argument('--drone_a', type=float, default=0.15, help='椭球体 XY 半轴')
    parser.add_argument('--drone_c', type=float, default=0.075, help='椭球体 Z 半轴')
    parser.add_argument('--coef_tilt', type=float, default=0.0, help='侧倾对齐损失权重')

    # --- 可微相机与主动感知 ---
    parser.add_argument('--include_camera_state_in_obs', default=False, action=argparse.BooleanOptionalAction,
                        help='是否将相机状态拼接到观测向量')
    parser.add_argument('--coef_cam_smooth', type=float, default=0.01, help='相机参数平滑度正则化权重')
    parser.add_argument('--coef_fov_reg', type=float, default=0.005, help='FOV 偏离默认值的正则化权重')
    parser.add_argument('--coef_cam_range', type=float, default=0.001, help='相机参数范围正则化权重')
    parser.add_argument('--wandb_disabled', default=False, action='store_true', help='禁用 wandb 日志记录')

    # --- 高保真可微相机渲染配置 ---
    parser.add_argument('--cam_realism_preset', type=str, default='high', choices=['low', 'medium', 'high', 'ultra'],
                        help='高保真可微相机强度档位')
    parser.add_argument('--cam_enable_shadow', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--cam_enable_specular', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--cam_enable_distortion', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--cam_enable_flare', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--cam_enable_motion_blur', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--cam_enable_rolling', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--cam_noise_scale', type=float, default=1.0)
    parser.add_argument('--cam_blur_scale', type=float, default=1.0)
    parser.add_argument('--cam_fog_scale', type=float, default=1.0)
    parser.add_argument('--cam_lighting_scale', type=float, default=1.0)
    parser.add_argument('--cam_ae_target', type=float, default=0.42)

    # --- 相机语义映射常数（统一管理，避免散落魔数） ---
    parser.add_argument('--cam_exposure_t_min', type=float, default=0.25,
                        help='曝光归一化值映射到时间尺度的下界偏置')
    parser.add_argument('--cam_exposure_t_span', type=float, default=2.75,
                        help='曝光归一化值映射到时间尺度的跨度')
    parser.add_argument('--cam_exposure_eff_min', type=float, default=0.15,
                        help='AE 后有效曝光时间最小值')
    parser.add_argument('--cam_exposure_eff_max', type=float, default=4.0,
                        help='AE 后有效曝光时间最大值')
    parser.add_argument('--cam_iso_gain_base', type=float, default=1.0,
                        help='ISO 增益基线')
    parser.add_argument('--cam_iso_gain_scale', type=float, default=10.0,
                        help='ISO 增益缩放系数')
    parser.add_argument('--cam_iso_gain_gamma', type=float, default=1.2,
                        help='ISO 增益幂指数')
    parser.add_argument('--cam_shot_noise_base', type=float, default=0.03,
                        help='Shot noise 基础系数')

    # ===== Camera loss + Teacher-Student training =====
    parser.add_argument('--enable_camera_quality_loss', default=False, action='store_true')
    parser.add_argument('--coef_blur', type=float, default=0.1)
    parser.add_argument('--coef_noise', type=float, default=0.05)
    parser.add_argument('--enable_teacher_student_training', default=False, action='store_true')
    parser.add_argument('--teacher_inner_steps', type=int, default=10)
    parser.add_argument('--teacher_inner_lr', type=float, default=0.01)
    parser.add_argument('--distill_coef', type=float, default=1.0)
    parser.add_argument('--student_physics_coef', type=float, default=0.3)
    parser.add_argument('--distill_final_ratio', type=float, default=0.3)
    parser.add_argument('--student_noise_mode', type=str, default='off', choices=['off', 'on'])
    parser.add_argument('--teacher_tbptt_chunk_steps', type=int, default=10)

    # ===== Multi-sensor + dLQR training switches =====
    parser.add_argument('--sensor_mode', type=str, default='camera_luma_plus_depth',
                        choices=['depth', 'camera_luma', 'camera_luma_plus_depth', 'diff_depth'])
    parser.add_argument('--coef_diff_depth_power', type=float, default=0.01)
    parser.add_argument('--coef_diff_depth_blur', type=float, default=0.01)
    parser.add_argument('--use_dmpc', default=False, action='store_true')
    parser.add_argument('--policy_direct_action', default=False, action='store_true')
    parser.add_argument('--policy_output_intent', default=False, action='store_true')
    parser.add_argument('--inject_depth_into_lqr', default=False, action='store_true')
    parser.add_argument('--lqr_horizon', type=int, default=5)
    parser.add_argument('--lqr_reg', type=float, default=1e-4)
    parser.add_argument('--depth_safe_dist', type=float, default=0.6)
    parser.add_argument('--depth_repel_gain', type=float, default=1.0)
    parser.add_argument('--dual_encoder', default=False, action='store_true')
    parser.add_argument('--max_acc_cmd', type=float, default=20.0)
    parser.add_argument('--amp', default=True, action=argparse.BooleanOptionalAction)

    # ===== Visualization =====
    parser.add_argument('--vis_enable', default=False, action='store_true')
    parser.add_argument('--vis_backend', type=str, default='rerun', choices=['rerun'])
    parser.add_argument('--vis_env_idx', type=int, default=0)
    parser.add_argument('--vis_every_iters', type=int, default=10)
    parser.add_argument('--vis_every_steps', type=int, default=10)
    parser.add_argument('--vis_teacher', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--vis_student', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--vis_spawn', default=True, action=argparse.BooleanOptionalAction)

    return parser


def parse_diff_sensor_impl(items):
    """将命令行列表解析为 {'camera_luma': 'python|cuda', 'diff_depth': 'python|cuda'}。"""
    impl = {
        'camera_luma': 'python',
        'diff_depth': 'python',
    }
    if items is None:
        return impl
    allowed_keys = set(impl.keys())
    allowed_vals = {'python', 'cuda'}
    for raw in items:
        if raw is None:
            continue
        item = str(raw).strip().lower()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"--diff_sensor_impl 条目格式错误: '{raw}'，应为 key=value")
        key, val = item.split('=', 1)
        key = key.strip()
        val = val.strip()
        if key not in allowed_keys:
            raise ValueError(f"--diff_sensor_impl 不支持 key='{key}'，仅支持: {sorted(allowed_keys)}")
        if val not in allowed_vals:
            raise ValueError(f"--diff_sensor_impl 不支持 value='{val}'，仅支持: {sorted(allowed_vals)}")
        impl[key] = val
    return impl


def normalize_sensor_mode(sensor_mode: str) -> str:
    """Normalize sensor_mode key casing/spacing only."""
    return str(sensor_mode).strip().lower()


def set_global_seed(seed: int, deterministic: bool = True):
    """设置全局随机数种子，提升训练可复现性。"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def resolve_sensor_flags(args):
    """从 args.sensor_mode 推导出所有传感器开关，返回 dict。"""
    use_depth_only = args.sensor_mode == 'depth'
    use_camera_luma = args.sensor_mode in ('camera_luma', 'camera_luma_plus_depth')
    use_depth = args.sensor_mode in ('camera_luma_plus_depth', 'diff_depth')
    use_depth_aux = args.sensor_mode == 'camera_luma_plus_depth'
    use_diff_depth = args.sensor_mode == 'diff_depth'

    use_camera_control = (use_camera_luma or use_diff_depth)
    effective_include_camera_state = bool(args.include_camera_state_in_obs and use_camera_control)

    return {
        'use_depth_only': use_depth_only,
        'use_camera_luma': use_camera_luma,
        'use_depth': use_depth,
        'use_depth_aux': use_depth_aux,
        'use_diff_depth': use_diff_depth,
        'use_camera_control': use_camera_control,
        'effective_include_camera_state': effective_include_camera_state,
    }


def validate_args(args, sensor_flags):
    """打印警告信息并做互斥检查。"""
    use_camera_control = sensor_flags['use_camera_control']
    use_depth_aux = sensor_flags['use_depth_aux']

    if args.policy_direct_action and args.policy_output_intent:
        raise ValueError("--policy_direct_action 与 --policy_output_intent 互斥，请二选一")
    if args.use_dmpc and not args.policy_output_intent:
        print("[warn] --use_dmpc 已启用，但 --policy_output_intent 未启用；将回退到动作域控制")
    if args.inject_depth_into_lqr and args.sensor_mode != 'camera_luma_plus_depth':
        print("[warn] --inject_depth_into_lqr 已启用，但 sensor_mode 不是 camera_luma_plus_depth；深度注入将被忽略")
    if args.enable_teacher_student_training and args.policy_output_intent and not args.use_dmpc:
        print("[warn] enable_teacher_student_training + policy_output_intent 且未启用 --use_dmpc："
              "将回退到动作域 teacher/蒸馏；intent 头仅作为辅助输出，不参与 teacher 优化")
    if args.dual_encoder:
        print("[warn] --dual_encoder 已弃用：模型结构由 sensor_mode 自动决定")
    if args.policy_input_width is not None or args.policy_input_height is not None:
        print("[warn] --policy_input_width/height 已弃用：当前模型不再做跨传感器强制尺寸对齐")
    if args.tbptt_enable and args.tbptt_chunk_steps < 2:
        raise ValueError('--tbptt_chunk_steps 必须 >= 2')
    if args.tbptt_enable and args.tbptt_chunk_accum < 1:
        raise ValueError('--tbptt_chunk_accum 必须 >= 1')
    if args.hybrid_full_bptt_every < 0:
        raise ValueError('--hybrid_full_bptt_every 必须 >= 0')
    if args.hybrid_full_bptt_batch_size < 0:
        raise ValueError('--hybrid_full_bptt_batch_size 必须 >= 0')
    if args.tbptt_enable and args.enable_teacher_student_training:
        print('[warn] 当前启用 TBPTT 与教师-学生训练：student 按原混合调度；teacher 内循环将使用 TBPTT 路径')
    if args.include_camera_state_in_obs and not use_camera_control:
        print(f"[warn] --include_camera_state_in_obs 已启用，但当前 sensor_mode={args.sensor_mode} 不支持相机控制；将自动忽略相机状态拼接")


def print_runtime_mode(args, sensor_flags):
    """打印启动模式横幅。"""
    use_diff_depth = sensor_flags['use_diff_depth']
    use_depth_aux = sensor_flags['use_depth_aux']

    policy_head_mode = 'intent_head' if args.policy_output_intent else 'action_head'
    exec_control_mode = 'dmpc' if (args.use_dmpc and args.policy_output_intent) else 'direct_action'
    if args.enable_teacher_student_training:
        teacher_mode = 'intent_teacher' if (args.policy_output_intent and args.use_dmpc) else 'action_teacher'
    else:
        teacher_mode = 'disabled'
    depth_lqr_effective = bool(args.inject_depth_into_lqr and args.use_dmpc and args.policy_output_intent and use_depth_aux)

    print("=" * 30 + " Runtime Mode " + "=" * 30)
    print(f"policy_head                : {policy_head_mode}")
    print(f"exec_control               : {exec_control_mode}")
    print(f"teacher_student_training  : {args.enable_teacher_student_training} ({teacher_mode})")
    print(f"use_dmpc                  : {args.use_dmpc}")
    print(f"inject_depth_into_lqr     : {args.inject_depth_into_lqr} (effective={depth_lqr_effective})")
    print(f"teacher_tbptt_chunk       : {args.teacher_tbptt_chunk_steps}")
    print(f"student_noise_mode        : {args.student_noise_mode}")
    print(f"distill_coef              : {args.distill_coef} -> {args.distill_coef * args.distill_final_ratio}")
    print(f"diff_sensor_impl          : {args.diff_sensor_impl}")
    print(f"sensor_mode               : {args.sensor_mode}")
    print(f"sensor_control_semantics  : {'power/exposure/gain' if use_diff_depth else 'fov/exposure/iso'}")
    print("=" * 75)


def parse_args():
    """解析命令行参数，执行验证，返回 (args, sensor_flags) 元组。"""
    parser = build_parser()
    args = parser.parse_args()
    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.sensor_mode = normalize_sensor_mode(args.sensor_mode)
    set_global_seed(args.seed, args.deterministic)
    sensor_flags = resolve_sensor_flags(args)
    validate_args(args, sensor_flags)
    return args, sensor_flags

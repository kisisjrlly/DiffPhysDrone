"""命令行参数解析与运行时校验。"""
import argparse
import os
import random
import numpy as np
import torch


SUPPORTED_SCENARIOS = (
    'base',
    'sun_glare',
    'specular_trap',
    'vantablack_gap',
    'dark_morphing',
)
OPENING_SCENES = {'vantablack_gap', 'dark_morphing'}
SUPPORTED_SUN_GLARE_LEVELS = ('l0', 'l1', 'l2', 'l3')


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
    parser.add_argument('--loss_v_window', type=int, default=30,
                        help='速度跟踪损失的时间平均窗口长度（单位：step）；越大越平滑、越小越灵敏')
    parser.add_argument('--coef_v_pred', type=float, default=2.0, help='速度预测 MSE 损失权重')
    parser.add_argument('--coef_collide', type=float, default=2.0, help='碰撞惩罚权重')
    parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='避障安全距离惩罚权重')
    parser.add_argument('--coef_d_acc', type=float, default=0.01, help='控制加速度正则化权重')
    parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='控制 Jerk 正则化权重')
    parser.add_argument('--coef_ground_affinity', type=float, default=0., help='贴地飞行偏好权重')

    # --- 训练超参数 ---
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--grad_decay', type=float, default=0.4, help='BPTT 梯度衰减系数')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子')
    parser.add_argument('--deterministic', default=False, action=argparse.BooleanOptionalAction,
                        help='是否启用确定性算法')
    parser.add_argument('--fov_x_half_tan', type=float, default=0.53, help='相机基础视场角')
    parser.add_argument('--timesteps', type=int, default=150, help='每个 episode 的物理步数')
    parser.add_argument('--base_control_freq', type=float, default=15.0, help='基础控制频率 (Hz)')
    parser.add_argument('--cam_angle', type=int, default=10, help='相机默认俯仰角 (度)')
    parser.add_argument('--depth_width', type=int, default=64, help='深度相机渲染分辨率宽')
    parser.add_argument('--depth_height', type=int, default=48, help='深度相机渲染分辨率高')
    parser.add_argument('--depth_min_valid', type=float, default=0.3,
                        help='深度图最小可信距离；低于该阈值视为无效深度')
    parser.add_argument('--depth_max_range', type=float, default=6.0,
                        help='diff_depth 传感器最大可靠量程；网络前处理与渲染器保持一致')
    parser.add_argument('--depth_nn_width', type=int, default=16, help='diff_depth: 输入策略网络前的深度特征宽')
    parser.add_argument('--depth_nn_height', type=int, default=12, help='diff_depth: 输入策略网络前的深度特征高')
    parser.add_argument('--depth_use_pipeline', default=True, action=argparse.BooleanOptionalAction,
                        help='深度输入是否启用图像处理流水线（仅对 diff_depth 生效）')
    parser.add_argument('--policy_depth_mode', type=str, default='depth', choices=['depth', 'zero'],
                        help='策略看到的深度输入模式：depth 为正常深度观测，zero 为全零深度（正式 blind baseline 训练/评测）')
    parser.add_argument('--diff_sensor_impl', nargs='*', default=['diff_depth=python'],
                        help='diff_depth 可微传感实现后端列表')

    # --- 环境与状态感知 ---
    parser.add_argument('--yaw_drift', default=False, action='store_true', help='模拟偏航角漂移')
    parser.add_argument('--no_odom', default=False, action='store_true', help='无里程计模式')
    parser.add_argument('--scenarios', nargs='*', default=['base'],
                        help='固定小地图上的 diff_depth 场景列表；训练随机采样，评测顺序轮转')
    parser.add_argument('--scene_fit_profiles_path', type=str, default=None,
                        help='可选：加载 D455 标定反推得到的场景 profile JSON，自动覆盖 diff_depth 场景参数')
    parser.add_argument('--sun_glare_levels', nargs='*', default=['l0', 'l1', 'l2', 'l3'],
                        help='sun_glare 场景允许采样的强度档位；训练时随机采样，评测时若未指定固定档位则按该列表轮转/取首项')
    parser.add_argument('--sun_glare_eval_level', type=str, default=None,
                        help='可选：评估时固定使用某一档 sun_glare 强度，例如 l2；训练阶段忽略该参数')
    parser.add_argument('--ellipsoid_collision', default=False, action='store_true', help='使用椭球体碰撞检测')
    parser.add_argument('--drone_a', type=float, default=0.15, help='椭球体 XY 半轴')
    parser.add_argument('--drone_c', type=float, default=0.075, help='椭球体 Z 半轴')
    parser.add_argument('--coef_tilt', type=float, default=0.0, help='侧倾对齐损失权重')

    # --- 可微相机与主动感知 ---
    parser.add_argument('--include_camera_state_in_obs', default=False, action=argparse.BooleanOptionalAction,
                        help='是否将相机状态拼接到观测向量')
    parser.add_argument('--camera_control_mode', type=str, default='learned', choices=['learned', 'fixed'],
                        help='相机控制模式：learned 为策略输出 power/exposure/gain，fixed 为固定相机基线')
    parser.add_argument('--sensor_grad_mode', type=str, default='full', choices=['full', 'detached'],
                        help='传感器梯度模式：full 为可微主动感知，detached 为不可微主动感知基线')
    parser.add_argument('--coef_cam_smooth', type=float, default=0.01, help='相机参数平滑度正则化权重')
    parser.add_argument('--coef_power_reg', type=float, default=0.005,
                        help='首个相机控制通道偏离中心值的正则化权重（diff_depth 分支中对应 power）')
    parser.add_argument('--cam_power_reg_deadband', type=float, default=0.0,
                        help='power 相对中性值的免罚区间半宽；在该 deadband 内允许策略按环境调节 power')
    parser.add_argument('--coef_cam_range', type=float, default=0.001, help='相机参数范围正则化权重')
    parser.add_argument('--cam_power_nominal', type=float, default=0.5,
                        help='diff_depth power 的中性/默认参考值；建议按真实 D455 默认 laser_power/max 对齐')
    parser.add_argument('--cam_power_penalty_threshold', type=float, default=0.5,
                        help='高功率惩罚的起始阈值；超过该值才触发 loss_diff_depth_power')
    parser.add_argument('--fixed_camera_power', type=float, default=-1.0,
                        help='fixed camera 基线的归一化 power；<0 时自动使用 cam_power_nominal')
    parser.add_argument('--fixed_camera_exposure', type=float, default=0.5,
                        help='fixed camera 基线的归一化 exposure')
    parser.add_argument('--fixed_camera_gain', type=float, default=0.5,
                        help='fixed camera 基线的归一化 gain')
    parser.add_argument('--wandb_disabled', default=False, action='store_true', help='禁用 wandb 日志记录')
    parser.add_argument('--wandb_log_raw_loss_terms', default=False, action=argparse.BooleanOptionalAction,
                        help='是否把未加权的各 loss 分量单独写入 wandb；默认关闭以避免与 loss_contrib 重复')

    # --- 高保真可微相机渲染配置 ---
    parser.add_argument('--cam_realism_preset', type=str, default='high', choices=['low', 'medium', 'high', 'ultra'],
                        help='高保真可微相机强度档位')
    parser.add_argument('--cam_enable_specular', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--cam_enable_motion_blur', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--cam_noise_scale', type=float, default=1.0)
    parser.add_argument('--cam_blur_scale', type=float, default=1.0)
    parser.add_argument('--cam_fog_scale', type=float, default=1.0)
    parser.add_argument('--cam_lighting_scale', type=float, default=1.0)
    parser.add_argument('--cam_model_randomize', default=True, action=argparse.BooleanOptionalAction,
                        help='训练时对 diff_depth 传感器模型分组参数做小范围 domain randomization；评测默认自动关闭')
    parser.add_argument('--cam_model_randomize_scale', type=float, default=0.08,
                        help='diff_depth 传感器模型分组随机化幅度；0 表示关闭，0.08 表示每组约 ±8%')

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
    parser.add_argument('--enable_teacher_student_training', default=False, action='store_true')
    parser.add_argument('--teacher_inner_steps', type=int, default=10)
    parser.add_argument('--teacher_inner_lr', type=float, default=0.01)
    parser.add_argument('--distill_coef', type=float, default=1.0)
    parser.add_argument('--student_physics_coef', type=float, default=0.3)
    parser.add_argument('--distill_final_ratio', type=float, default=0.3)
    parser.add_argument('--student_noise_mode', type=str, default='off', choices=['off', 'on'])
    parser.add_argument('--teacher_tbptt_chunk_steps', type=int, default=10)

    # ===== dLQR / intent related =====
    parser.add_argument('--coef_diff_depth_power', type=float, default=0.01)
    parser.add_argument('--coef_diff_depth_blur', type=float, default=0.01)
    parser.add_argument('--coef_diff_depth_noise', type=float, default=0.01)
    parser.add_argument('--coef_diff_depth_fill', type=float, default=0.0,
                        help='惩罚 fill rate 低于阈值的 blackout 现象，防止策略把深度相机关到近乎失效')
    parser.add_argument('--diff_depth_min_fill_rate', type=float, default=0.18,
                        help='深度 fill rate 的最低目标阈值；低于它时会触发 blackout penalty')
    parser.add_argument('--coef_sun_glare_local_quality', type=float, default=0.0,
                        help='sun_glare 场景局部炫光区域质量恢复损失权重；用于鼓励 power/exposure 联合调节')
    parser.add_argument('--sun_glare_local_quality_target', type=float, default=0.55,
                        help='sun_glare 局部炫光区域的目标 quality 均值')
    parser.add_argument('--use_dmpc', default=False, action='store_true')
    parser.add_argument('--policy_output_intent', default=False, action='store_true')
    parser.add_argument('--inject_depth_into_lqr', default=False, action='store_true')
    parser.add_argument('--lqr_horizon', type=int, default=5)
    parser.add_argument('--lqr_reg', type=float, default=1e-4)
    parser.add_argument('--depth_safe_dist', type=float, default=0.6)
    parser.add_argument('--depth_repel_gain', type=float, default=1.0)
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
    """将命令行列表解析为 {'diff_depth': 'python|cuda'}。"""
    impl = {
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
            print(f"[warn] diff_depth-only 分支忽略 --diff_sensor_impl 条目 key='{key}'")
            continue
        if val not in allowed_vals:
            raise ValueError(f"--diff_sensor_impl 不支持 value='{val}'，仅支持: {sorted(allowed_vals)}")
        impl[key] = val
    return impl


def parse_scenarios(items):
    """解析固定小地图场景列表。支持空格或逗号分隔。"""
    if items is None:
        return ['base']

    aliases = {
        'random_base': 'base',
        'random': 'base',
        'random_scene': 'base',
        'black_gap': 'vantablack_gap',
        'dark_slit_lite': 'dark_morphing',
    }
    scenarios = []
    for raw in items:
        if raw is None:
            continue
        for token in str(raw).split(','):
            name = aliases.get(token.strip().lower(), token.strip().lower())
            if not name:
                continue
            if name not in SUPPORTED_SCENARIOS:
                raise ValueError(
                    f"--scenarios 不支持 '{name}'，仅支持: {list(SUPPORTED_SCENARIOS)}"
                )
            scenarios.append(name)

    if not scenarios:
        return ['base']

    dedup = []
    seen = set()
    for name in scenarios:
        if name in seen:
            continue
        seen.add(name)
        dedup.append(name)
    return dedup


def canonicalize_sun_glare_level(item):
    if item is None:
        return None
    token = str(item).strip().lower()
    aliases = {
        '0': 'l0',
        'l0': 'l0',
        'weak': 'l0',
        'low': 'l0',
        '1': 'l1',
        'l1': 'l1',
        'mild': 'l1',
        'midlow': 'l1',
        '2': 'l2',
        'l2': 'l2',
        'mid': 'l2',
        'medium': 'l2',
        'default': 'l2',
        '3': 'l3',
        'l3': 'l3',
        'strong': 'l3',
        'high': 'l3',
    }
    return aliases.get(token, token)


def parse_sun_glare_levels(items):
    if items is None:
        return ['l0', 'l1', 'l2', 'l3']

    levels = []
    for raw in items:
        if raw is None:
            continue
        for token in str(raw).split(','):
            name = canonicalize_sun_glare_level(token)
            if not name:
                continue
            if name not in SUPPORTED_SUN_GLARE_LEVELS:
                raise ValueError(
                    f"--sun_glare_levels 不支持 '{name}'，仅支持: {list(SUPPORTED_SUN_GLARE_LEVELS)}"
                )
            levels.append(name)

    if not levels:
        return ['l0', 'l1', 'l2', 'l3']

    dedup = []
    seen = set()
    for name in levels:
        if name in seen:
            continue
        seen.add(name)
        dedup.append(name)
    return dedup


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


def validate_args(args):
    """打印警告信息并做互斥检查。"""
    if args.use_dmpc and not args.policy_output_intent:
        print("[warn] --use_dmpc 已启用，但 --policy_output_intent 未启用；将回退到动作域控制")
    if args.enable_teacher_student_training and args.policy_output_intent and not args.use_dmpc:
        print("[warn] enable_teacher_student_training + policy_output_intent 且未启用 --use_dmpc："
              "将回退到动作域 teacher/蒸馏；intent 头仅作为辅助输出，不参与 teacher 优化")
    if args.tbptt_enable and args.tbptt_chunk_steps < 2:
        raise ValueError('--tbptt_chunk_steps 必须 >= 2')
    if args.tbptt_enable and args.tbptt_chunk_accum < 1:
        raise ValueError('--tbptt_chunk_accum 必须 >= 1')
    if args.loss_v_window < 1:
        raise ValueError('--loss_v_window 必须 >= 1')
    if args.depth_min_valid <= 0:
        raise ValueError('--depth_min_valid 必须 > 0')
    if args.depth_max_range <= args.depth_min_valid:
        raise ValueError('--depth_max_range 必须大于 --depth_min_valid')
    if args.hybrid_full_bptt_every < 0:
        raise ValueError('--hybrid_full_bptt_every 必须 >= 0')
    if args.hybrid_full_bptt_batch_size < 0:
        raise ValueError('--hybrid_full_bptt_batch_size 必须 >= 0')
    if args.tbptt_enable and args.enable_teacher_student_training:
        print('[warn] 当前启用 TBPTT 与教师-学生训练：student 按原混合调度；teacher 内循环将使用 TBPTT 路径')
    if args.depth_width < 1 or args.depth_height < 1:
        raise ValueError('--depth_width/--depth_height 必须 >= 1')
    if args.depth_nn_width < 1 or args.depth_nn_height < 1:
        raise ValueError('--depth_nn_width/--depth_nn_height 必须 >= 1')
    if not (0.0 <= args.cam_power_nominal <= 1.0):
        raise ValueError('--cam_power_nominal 必须在 [0,1] 内')
    if not (0.0 <= args.cam_power_penalty_threshold <= 1.0):
        raise ValueError('--cam_power_penalty_threshold 必须在 [0,1] 内')
    if args.fixed_camera_power >= 0.0 and not (0.0 <= args.fixed_camera_power <= 1.0):
        raise ValueError('--fixed_camera_power 必须在 [0,1] 内，或设为 <0 使用 cam_power_nominal')
    if not (0.0 <= args.fixed_camera_exposure <= 1.0):
        raise ValueError('--fixed_camera_exposure 必须在 [0,1] 内')
    if not (0.0 <= args.fixed_camera_gain <= 1.0):
        raise ValueError('--fixed_camera_gain 必须在 [0,1] 内')
    if args.cam_power_reg_deadband < 0.0 or args.cam_power_reg_deadband > 1.0:
        raise ValueError('--cam_power_reg_deadband 必须在 [0,1] 内')
    if args.sun_glare_local_quality_target < 0.0 or args.sun_glare_local_quality_target > 1.0:
        raise ValueError('--sun_glare_local_quality_target 必须在 [0,1] 内')
    if args.cam_model_randomize_scale < 0.0 or args.cam_model_randomize_scale > 0.5:
        raise ValueError('--cam_model_randomize_scale 建议在 [0, 0.5] 内')
    if not getattr(args, 'scenarios', None):
        raise ValueError('--scenarios 至少需要一个场景')
    if not getattr(args, 'sun_glare_levels', None):
        raise ValueError('--sun_glare_levels 至少需要一个档位')
    if args.sun_glare_eval_level is not None:
        args.sun_glare_eval_level = canonicalize_sun_glare_level(args.sun_glare_eval_level)
        if args.sun_glare_eval_level not in SUPPORTED_SUN_GLARE_LEVELS:
            raise ValueError(
                f"--sun_glare_eval_level 不支持 '{args.sun_glare_eval_level}'，"
                f"仅支持: {list(SUPPORTED_SUN_GLARE_LEVELS)}"
            )


def print_runtime_mode(args):
    """打印启动模式横幅。"""
    policy_head_mode = 'intent_head' if args.policy_output_intent else 'action_head'
    exec_control_mode = 'dmpc' if (args.use_dmpc and args.policy_output_intent) else 'direct_action'
    if args.enable_teacher_student_training:
        teacher_mode = 'intent_teacher' if (args.policy_output_intent and args.use_dmpc) else 'action_teacher'
    else:
        teacher_mode = 'disabled'
    depth_lqr_effective = bool(args.inject_depth_into_lqr and args.use_dmpc and args.policy_output_intent)

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
    print(f"policy_depth_mode        : {args.policy_depth_mode}")
    print(f"scenarios                 : {args.scenarios}")
    print(f"sun_glare_levels          : {args.sun_glare_levels}")
    print(f"sun_glare_eval_level      : {args.sun_glare_eval_level}")
    print(f"camera_control_mode       : {args.camera_control_mode}")
    print(f"sensor_grad_mode          : {args.sensor_grad_mode}")
    print("environment               : fixed_small_map_with_perception_scenarios")
    print("sensor_control_semantics  : power/exposure/gain")
    print("=" * 75)


def parse_args():
    """解析命令行参数并执行验证。"""
    parser = build_parser()
    args = parser.parse_args()
    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.scenarios = parse_scenarios(args.scenarios)
    args.sun_glare_levels = parse_sun_glare_levels(args.sun_glare_levels)
    if args.sun_glare_eval_level is not None:
        args.sun_glare_eval_level = canonicalize_sun_glare_level(args.sun_glare_eval_level)
    set_global_seed(args.seed, args.deterministic)
    validate_args(args)
    return args

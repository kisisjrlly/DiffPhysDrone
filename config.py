"""Minimal configuration for the active-sensing simulation branch."""
import argparse
import os
import random
import numpy as np
import torch


SUPPORTED_SCENARIOS = ('glare', 'specular', 'dark')
SUPPORTED_SLOTS = ('far_left', 'left', 'right', 'far_right')


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_iters', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--grad_decay', type=float, default=0.3)
    parser.add_argument('--amp', default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument('--coef_v', type=float, default=10.5)
    parser.add_argument('--loss_v_window', type=int, default=12)
    parser.add_argument('--coef_collide', type=float, default=10.0)
    parser.add_argument('--coef_obj_avoidance', type=float, default=0.5)
    parser.add_argument('--coef_d_acc', type=float, default=0.1)
    parser.add_argument('--coef_d_jerk', type=float, default=0.2)
    parser.add_argument('--collision_clearance', type=float, default=0.0011)

    parser.add_argument('--fov_x_half_tan', type=float, default=0.82)
    parser.add_argument('--timesteps', type=int, default=120)
    parser.add_argument('--base_control_freq', type=float, default=15.0)
    parser.add_argument('--cam_angle', type=int, default=5)
    parser.add_argument('--depth_width', type=int, default=64)
    parser.add_argument('--depth_height', type=int, default=48)
    parser.add_argument('--depth_min_valid', type=float, default=0.3)
    parser.add_argument('--depth_max_range', type=float, default=6.0)
    parser.add_argument('--depth_nn_width', type=int, default=32)
    parser.add_argument('--depth_nn_height', type=int, default=24)
    parser.add_argument('--depth_use_pipeline', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--policy_depth_mode', type=str, default='depth', choices=['depth', 'zero'])
    parser.add_argument('--diff_sensor_impl', nargs='*', default=['diff_depth=cuda'])

    parser.add_argument('--scenarios', nargs='*', default=list(SUPPORTED_SCENARIOS))
    parser.add_argument('--sun_glare_eval_slot', type=str, default=None)
    parser.add_argument('--random_rotation', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--random_rotation_max_deg', type=float, default=45.0)
    parser.add_argument('--simple_start_x', type=float, default=-1.0)
    parser.add_argument('--simple_goal_x', type=float, default=1.8)
    parser.add_argument('--simple_wall_x', type=float, default=0.65)
    parser.add_argument('--simple_slit_center_y_min', type=float, default=-0.55)
    parser.add_argument('--simple_slit_center_y_max', type=float, default=0.55)
    parser.add_argument('--simple_slit_half_y', type=float, default=0.20)
    parser.add_argument('--simple_slit_half_y_min', type=float, default=None)
    parser.add_argument('--simple_slit_half_y_max', type=float, default=None)
    parser.add_argument('--simple_slit_effect_half_z', type=float, default=0.26,
                        help='细缝在传感器效应图中的竖直半高，不是物理碰撞高度。')
    parser.add_argument('--simple_slit_center_z', type=float, default=1.50)
    parser.add_argument('--simple_slit_side_effect_width_y', type=float, default=0.20,
                        help='specular/dark 在细缝两侧墙面上从细缝边缘向外延伸的材质宽度。')
    parser.add_argument('--simple_slit_side_effect_half_z', type=float, default=1.00,
                        help='specular/dark 两侧墙面材质区域的竖直半高。')
    parser.add_argument('--simple_glare_halo_width_y', type=float, default=0.18,
                        help='glare 强光从细缝向两侧墙面扩散的横向宽度。')
    parser.add_argument('--simple_glare_halo_extra_half_z', type=float, default=0.25,
                        help='glare halo 相比核心开口区域额外增加的竖直半高。')
    parser.add_argument('--simple_glare_halo_strength', type=float, default=0.45,
                        help='glare halo 相对核心开口强光的强度系数。')
    parser.add_argument('--simple_back_wall_x_min', type=float, default=None,
                        help='细缝后方背墙局部 x 的采样下界；默认使用 goal_x + 0.75。')
    parser.add_argument('--simple_back_wall_x_max', type=float, default=None,
                        help='细缝后方背墙局部 x 的采样上界；默认等于 simple_back_wall_x_min。')
    parser.add_argument('--simple_slit_cue_halo_width_y', type=float, default=0.16,
                        help='关键观测 cue 在细缝横向边缘外额外受影响的宽度。')
    parser.add_argument('--simple_slit_cue_extra_half_z', type=float, default=0.28,
                        help='关键观测 cue 相比细缝核心 effect 区域额外增加的竖直半高。')
    parser.add_argument('--simple_key_cue_degrade_strength', type=float, default=0.90,
                        help='dark/specular/glare 对细缝内部和后墙 cue 的最大退化强度。')
    parser.add_argument('--simple_specular_false_depth_strength', type=float, default=0.55,
                        help='specular 高功率下把关键 cue 变成错误近深度的强度。')
    parser.add_argument('--no_odom', default=False, action='store_true')
    parser.add_argument('--include_camera_state_in_obs', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--max_acc_cmd', type=float, default=2.5)

    parser.add_argument('--camera_control_mode', type=str, default='learned',
                        choices=['learned', 'fixed', 'fixed_random_static'])
    parser.add_argument('--sensor_grad_mode', type=str, default='full', choices=['full', 'detached'])
    parser.add_argument('--train_flight_only', default=False, action=argparse.BooleanOptionalAction,
                        help='Freeze the camera branch (including its frozen stem copy) and only train the flight policy.')
    parser.add_argument('--cam_delta_max', type=float, default=0.02)
    parser.add_argument('--cam_return_rate', type=float, default=0.05)
    parser.add_argument('--cam_power_baseline', type=float, default=0.5)
    parser.add_argument('--coef_cam_smooth', type=float, default=2.0)
    parser.add_argument('--coef_diff_depth_power', type=float, default=4.0)
    parser.add_argument('--coef_diff_depth_blur', type=float, default=0.0)
    parser.add_argument('--coef_diff_depth_noise', type=float, default=0.0)
    parser.add_argument('--coef_diff_depth_fill', type=float, default=0.0)
    parser.add_argument('--diff_depth_min_fill_rate', type=float, default=0.0)
    parser.add_argument('--diff_depth_health_patch_rows', type=int, default=6)
    parser.add_argument('--diff_depth_health_patch_cols', type=int, default=8)
    parser.add_argument('--diff_depth_health_cvar_frac', type=float, default=0.25)
    parser.add_argument('--fixed_camera_power', type=float, default=0.4)
    parser.add_argument('--fixed_camera_exposure', type=float, default=0.9)
    parser.add_argument('--fixed_camera_gain', type=float, default=0.65)
    parser.add_argument('--fixed_random_power_min', type=float, default=0.1)
    parser.add_argument('--fixed_random_power_max', type=float, default=0.95)
    parser.add_argument('--fixed_random_exposure_min', type=float, default=0.1)
    parser.add_argument('--fixed_random_exposure_max', type=float, default=0.92)
    parser.add_argument('--fixed_random_gain_min', type=float, default=0.02)
    parser.add_argument('--fixed_random_gain_max', type=float, default=0.9)

    parser.add_argument('--cam_exposure_t_min', type=float, default=0.25)
    parser.add_argument('--cam_exposure_t_span', type=float, default=2.75)
    parser.add_argument('--cam_exposure_eff_min', type=float, default=0.25)
    parser.add_argument('--cam_exposure_eff_max', type=float, default=3.0)
    parser.add_argument('--cam_iso_gain_base', type=float, default=1.0)
    parser.add_argument('--cam_iso_gain_scale', type=float, default=0.8)
    parser.add_argument('--cam_iso_gain_gamma', type=float, default=0.6)
    parser.add_argument('--cam_shot_noise_base', type=float, default=0.01)

    parser.add_argument('--ellipsoid_collision', default=False, action='store_true')
    parser.add_argument('--drone_a', type=float, default=0.15)
    parser.add_argument('--drone_c', type=float, default=0.075)
    parser.add_argument('--yaw_drift', default=False, action='store_true')

    parser.add_argument('--wandb_disabled', default=False, action='store_true')
    parser.add_argument('--wandb_episode_history', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--wandb_episode_history_every_iters', type=int, default=100)
    parser.add_argument('--vis_enable', default=False, action='store_true')
    parser.add_argument('--vis_backend', type=str, default='rerun', choices=['rerun'])
    parser.add_argument('--vis_env_idx', type=int, default=0)
    parser.add_argument('--vis_every_iters', type=int, default=10)
    parser.add_argument('--vis_every_steps', type=int, default=10)
    parser.add_argument('--vis_student', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--vis_spawn', default=True, action=argparse.BooleanOptionalAction)
    return parser


def parse_diff_sensor_impl(items):
    impl = {'diff_depth': 'cuda'}
    if items is None:
        return impl
    for raw in items:
        item = str(raw).strip().lower()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"--diff_sensor_impl item must be key=value, got {raw!r}")
        key, val = item.split('=', 1)
        key, val = key.strip(), val.strip()
        if key != 'diff_depth' or val not in {'python', 'cuda'}:
            raise ValueError("--diff_sensor_impl supports only diff_depth=python|cuda")
        impl[key] = val
    return impl


def parse_scenarios(items):
    if items is None:
        return list(SUPPORTED_SCENARIOS)
    out = []
    for raw in items:
        for token in str(raw).split(','):
            name = token.strip().lower().replace('-', '_')
            if not name:
                continue
            if name not in SUPPORTED_SCENARIOS:
                raise ValueError(f"--scenarios unsupported {name!r}; choose {list(SUPPORTED_SCENARIOS)}")
            if name not in out:
                out.append(name)
    return out or list(SUPPORTED_SCENARIOS)


def canonicalize_sun_glare_slot(item):
    if item is None:
        return None
    token = str(item).strip().lower().replace('-', '_')
    aliases = {
        'fl': 'far_left', 'farleft': 'far_left', 'far_left': 'far_left',
        'l': 'left', 'left': 'left',
        'r': 'right', 'right': 'right',
        'fr': 'far_right', 'farright': 'far_right', 'far_right': 'far_right',
    }
    return aliases.get(token, token)


def set_global_seed(seed: int, deterministic: bool = True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def validate_args(args):
    if args.depth_width < 1 or args.depth_height < 1:
        raise ValueError('--depth_width/--depth_height must be >= 1')
    if args.depth_max_range <= args.depth_min_valid:
        raise ValueError('--depth_max_range must be > --depth_min_valid')
    if args.loss_v_window < 1:
        raise ValueError('--loss_v_window must be >= 1')
    if args.collision_clearance < 0:
        raise ValueError('--collision_clearance must be >= 0')
    if args.random_rotation_max_deg < 0:
        raise ValueError('--random_rotation_max_deg must be >= 0')
    if args.simple_goal_x <= args.simple_start_x:
        raise ValueError('--simple_goal_x must be greater than --simple_start_x')
    if args.simple_wall_x <= args.simple_start_x or args.simple_wall_x >= args.simple_goal_x:
        raise ValueError('--simple_wall_x must be between start and goal')
    if args.simple_slit_half_y <= 0 or args.simple_slit_effect_half_z <= 0:
        raise ValueError('--simple_slit_half_y and --simple_slit_effect_half_z must be > 0')
    if (args.simple_slit_half_y_min is None) != (args.simple_slit_half_y_max is None):
        raise ValueError('--simple_slit_half_y_min and --simple_slit_half_y_max must be set together')
    if args.simple_slit_half_y_min is None:
        args.simple_slit_half_y_min = args.simple_slit_half_y
        args.simple_slit_half_y_max = args.simple_slit_half_y
    if args.simple_slit_half_y_min <= 0 or args.simple_slit_half_y_max <= 0:
        raise ValueError('--simple_slit_half_y_min/max must be > 0')
    if args.simple_slit_half_y_max < args.simple_slit_half_y_min:
        raise ValueError('--simple_slit_half_y_max must be >= --simple_slit_half_y_min')
    if args.simple_slit_side_effect_width_y <= 0 or args.simple_slit_side_effect_half_z <= 0:
        raise ValueError('--simple_slit_side_effect_width_y and --simple_slit_side_effect_half_z must be > 0')
    if args.simple_glare_halo_width_y < 0 or args.simple_glare_halo_extra_half_z < 0:
        raise ValueError('--simple_glare_halo_width_y and --simple_glare_halo_extra_half_z must be >= 0')
    if not (0.0 <= args.simple_glare_halo_strength <= 1.0):
        raise ValueError('--simple_glare_halo_strength must be in [0, 1]')
    if args.simple_back_wall_x_min is None:
        args.simple_back_wall_x_min = float(args.simple_goal_x) + 0.75
    if args.simple_back_wall_x_max is None:
        args.simple_back_wall_x_max = args.simple_back_wall_x_min
    if args.simple_back_wall_x_min <= args.simple_wall_x or args.simple_back_wall_x_max <= args.simple_wall_x:
        raise ValueError('--simple_back_wall_x_min/max must be greater than --simple_wall_x')
    if args.simple_back_wall_x_max < args.simple_back_wall_x_min:
        raise ValueError('--simple_back_wall_x_max must be >= --simple_back_wall_x_min')
    if args.simple_slit_cue_halo_width_y < 0 or args.simple_slit_cue_extra_half_z < 0:
        raise ValueError('--simple_slit_cue_halo_width_y and --simple_slit_cue_extra_half_z must be >= 0')
    if not (0.0 <= args.simple_key_cue_degrade_strength <= 1.0):
        raise ValueError('--simple_key_cue_degrade_strength must be in [0, 1]')
    if not (0.0 <= args.simple_specular_false_depth_strength <= 1.0):
        raise ValueError('--simple_specular_false_depth_strength must be in [0, 1]')
    if args.cam_delta_max < 0:
        raise ValueError('--cam_delta_max must be >= 0')
    if args.cam_return_rate < 0:
        raise ValueError('--cam_return_rate must be >= 0')
    if args.diff_depth_health_patch_rows < 1:
        raise ValueError('--diff_depth_health_patch_rows must be >= 1')
    if args.diff_depth_health_patch_cols < 1:
        raise ValueError('--diff_depth_health_patch_cols must be >= 1')
    if not (0.0 <= args.diff_depth_health_cvar_frac <= 1.0):
        raise ValueError('--diff_depth_health_cvar_frac must be in [0, 1]')
    if args.wandb_episode_history_every_iters < 1:
        raise ValueError('--wandb_episode_history_every_iters must be >= 1')
    if not args.scenarios:
        raise ValueError('--scenarios needs at least one scenario')
    args.sun_glare_eval_slot = canonicalize_sun_glare_slot(args.sun_glare_eval_slot)
    if args.sun_glare_eval_slot is not None and args.sun_glare_eval_slot not in SUPPORTED_SLOTS:
        raise ValueError(f"--sun_glare_eval_slot unsupported {args.sun_glare_eval_slot!r}")
    for name in [
        'cam_power_baseline', 'fixed_camera_power', 'fixed_camera_exposure', 'fixed_camera_gain',
        'fixed_random_power_min', 'fixed_random_power_max',
        'fixed_random_exposure_min', 'fixed_random_exposure_max',
        'fixed_random_gain_min', 'fixed_random_gain_max',
    ]:
        val = float(getattr(args, name))
        if not (0.0 <= val <= 1.0):
            raise ValueError(f'--{name} must be in [0, 1]')

    # Fixed / randfix baselines should not carry any camera-training objective.
    # The camera is not learned there, so these coefficients only add confusion
    # in logs and config files.
    if args.camera_control_mode in {'fixed', 'fixed_random_static'}:
        args.sensor_grad_mode = 'detached'
        for name in [
            'coef_cam_smooth',
            'coef_diff_depth_power',
            'coef_diff_depth_blur',
            'coef_diff_depth_noise',
            'coef_diff_depth_fill',
        ]:
            setattr(args, name, 0.0)
        args.diff_depth_min_fill_rate = 0.0
    if args.train_flight_only:
        for name in [
            'coef_cam_smooth',
            'coef_diff_depth_power',
            'coef_diff_depth_blur',
            'coef_diff_depth_noise',
            'coef_diff_depth_fill',
        ]:
            setattr(args, name, 0.0)
        args.diff_depth_min_fill_rate = 0.0


def print_runtime_mode(args):
    print('=' * 30 + ' Runtime Mode ' + '=' * 30)
    print('policy_head                : action_head')
    print('exec_control               : direct_action')
    print(f"diff_sensor_impl          : {args.diff_sensor_impl}")
    print(f"policy_depth_mode         : {args.policy_depth_mode}")
    print(f"scenarios                 : {args.scenarios}")
    print(f"sun_glare_eval_slot       : {args.sun_glare_eval_slot}")
    print(f"random_rotation           : {args.random_rotation} (max_deg={args.random_rotation_max_deg})")
    print(f"collision_clearance      : {args.collision_clearance} m")
    print(f"camera_control_mode       : {args.camera_control_mode}")
    print(f"train_flight_only         : {args.train_flight_only}")
    print('camera_output             : absolute target in [0,1], applied by EMA alpha=0.7')
    print('policy_input              : image_feature + state(+camera_state if enabled) -> shared_gru')
    print('camera_head_input         : shared_gru hidden -> absolute camera target')
    print(
        'depth_fill_loss          : '
        f'patch_cvar rows={args.diff_depth_health_patch_rows}, '
        f'cols={args.diff_depth_health_patch_cols}, '
        f'frac={args.diff_depth_health_cvar_frac}, '
        f'target={args.diff_depth_min_fill_rate}'
    )
    print(f"sensor_grad_mode          : {args.sensor_grad_mode}")
    print(
        'environment               : single_wall_slit '
        f'wall_x={args.simple_wall_x}, slit_center_y={args.simple_slit_center_y_min}'
        f'..{args.simple_slit_center_y_max}, '
        f'slit_half_y={args.simple_slit_half_y_min}'
        f'..{args.simple_slit_half_y_max}, '
        f'slit_effect_half_z={args.simple_slit_effect_half_z}, '
        f'slit_center_z={args.simple_slit_center_z}, '
        f'side_effect_width_y={args.simple_slit_side_effect_width_y}, '
        f'side_effect_half_z={args.simple_slit_side_effect_half_z}, '
        f'glare_halo_width_y={args.simple_glare_halo_width_y}, '
        f'glare_halo_extra_half_z={args.simple_glare_halo_extra_half_z}, '
        f'glare_halo_strength={args.simple_glare_halo_strength}, '
        f'back_wall_x={args.simple_back_wall_x_min}'
        f'..{args.simple_back_wall_x_max}, '
        f'slit_cue_halo_width_y={args.simple_slit_cue_halo_width_y}, '
        f'slit_cue_extra_half_z={args.simple_slit_cue_extra_half_z}, '
        f'key_cue_degrade_strength={args.simple_key_cue_degrade_strength}, '
        f'specular_false_depth_strength={args.simple_specular_false_depth_strength}'
    )
    print('=' * 75)


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.scenarios = parse_scenarios(args.scenarios)
    set_global_seed(args.seed, args.deterministic)
    validate_args(args)
    return args

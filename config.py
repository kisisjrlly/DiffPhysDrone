"""Configuration for the grayscale/IMX900 active-sensing branch."""

import argparse
import os
import random

import numpy as np
import torch


SUPPORTED_SCENARIOS = (
    "nominal",
    "dark",
    "bright",
    "dark_to_bright",
    "bright_to_dark",
)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_iters", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--grad_decay", type=float, default=0.3)
    p.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)

    p.add_argument("--coef_v", type=float, default=10.5)
    p.add_argument("--loss_v_window", type=int, default=12)
    p.add_argument("--coef_collide", type=float, default=10.0)
    p.add_argument("--coef_obj_avoidance", type=float, default=0.5)
    p.add_argument("--coef_d_acc", type=float, default=0.1)
    p.add_argument("--coef_d_jerk", type=float, default=0.2)
    p.add_argument("--coef_cam_smooth", type=float, default=0.005)
    p.add_argument("--collision_clearance", type=float, default=0.0011)

    p.add_argument("--fov_x_half_tan", type=float, default=0.82)
    p.add_argument("--timesteps", type=int, default=120)
    p.add_argument("--base_control_freq", type=float, default=15.0)
    p.add_argument("--cam_angle", type=int, default=5)
    p.add_argument("--gray_width", type=int, default=96)
    p.add_argument("--gray_height", type=int, default=72)
    p.add_argument("--gray_nn_width", type=int, default=48)
    p.add_argument("--gray_nn_height", type=int, default=36)
    p.add_argument("--policy_gray_mode", choices=["gray", "zero"], default="gray")

    p.add_argument("--scenarios", nargs="*", default=["nominal"])
    p.add_argument("--random_rotation", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--random_rotation_max_deg", type=float, default=45.0)
    p.add_argument("--simple_start_x", type=float, default=-1.5)
    p.add_argument("--simple_goal_x", type=float, default=1.5)
    p.add_argument("--simple_wall_x", type=float, default=0.0)
    p.add_argument("--simple_slit_center_y_min", type=float, default=-0.75)
    p.add_argument("--simple_slit_center_y_max", type=float, default=0.75)
    p.add_argument("--simple_slit_half_y", type=float, default=0.15)
    p.add_argument("--simple_slit_half_y_min", type=float, default=None)
    p.add_argument("--simple_slit_half_y_max", type=float, default=None)
    p.add_argument("--simple_slit_center_z", type=float, default=1.50)
    p.add_argument("--simple_back_wall_x_min", type=float, default=2.0)
    p.add_argument("--simple_back_wall_x_max", type=float, default=2.85)

    # Grayscale illumination benchmark. Values are scene-irradiance scales, not
    # camera settings.
    p.add_argument("--gray_nominal_ambient", type=float, default=0.18)
    p.add_argument("--gray_nominal_diffuse", type=float, default=0.72)
    p.add_argument("--gray_dark_scale", type=float, default=0.12)
    p.add_argument("--gray_bright_scale", type=float, default=2.2)
    p.add_argument("--gray_background_intensity", type=float, default=0.04)
    p.add_argument("--gray_transition_x", type=float, default=0.0)
    p.add_argument("--gray_transition_width", type=float, default=0.25)
    p.add_argument("--gray_light_jitter", type=float, default=0.08)
    p.add_argument("--gray_texture_strength", type=float, default=0.35)
    p.add_argument("--gray_texture_scale", type=float, default=5.0)

    p.add_argument("--no_odom", default=False, action="store_true")
    # Keep exposure/gain out of the flight-policy state by default. The camera
    # controller receives camera_state through its dedicated branch, while the
    # flight policy can only benefit from camera actions through image formation.
    p.add_argument("--include_camera_state_in_obs", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--max_acc_cmd", type=float, default=2.5)

    p.add_argument(
        "--camera_control_mode",
        choices=["learned", "fixed", "fixed_random_static", "mean_ae", "gradient_ae"],
        default="learned",
    )
    p.add_argument("--sensor_grad_mode", choices=["full", "detached"], default="full")
    p.add_argument("--train_flight_only", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--train_camera_only", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--camera_ema_alpha", type=float, default=0.7)
    p.add_argument("--fixed_camera_exposure", type=float, default=0.35)
    p.add_argument("--fixed_camera_gain", type=float, default=0.15)
    p.add_argument("--fixed_random_exposure_min", type=float, default=0.10)
    p.add_argument("--fixed_random_exposure_max", type=float, default=0.90)
    p.add_argument("--fixed_random_gain_min", type=float, default=0.02)
    p.add_argument("--fixed_random_gain_max", type=float, default=0.90)

    # Camera physics live in a calibration profile, not in experiment configs.
    # The repository default is explicitly provisional/unmeasured and must be
    # replaced by a fitted IMX900 profile before sim-to-real claims.
    p.add_argument(
        "--imx900_calibration",
        default="configs/calibration/imx900_provisional.json",
    )
    p.add_argument(
        "--require_calibrated_imx900",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    # Surrogate/training hyperparameters that are not claimed as sensor specs.
    p.add_argument("--gray_blur_kernel_size", type=int, default=5)
    # Characteristic-depth floor for the image-motion proxy |v| / Z.
    p.add_argument("--gray_motion_depth_floor", type=float, default=0.35)
    p.add_argument("--gray_dark_threshold", type=float, default=0.05)
    p.add_argument("--gray_saturation_mode", choices=["ste", "hard", "soft"], default="soft")
    p.add_argument("--gray_soft_clip_beta", type=float, default=12.0)
    p.add_argument("--gray_enable_noise", default=True, action=argparse.BooleanOptionalAction)

    p.add_argument("--ellipsoid_collision", default=False, action="store_true")
    p.add_argument("--drone_a", type=float, default=0.15)
    p.add_argument("--drone_c", type=float, default=0.075)

    p.add_argument("--wandb_disabled", default=False, action="store_true")
    p.add_argument("--wandb_episode_history", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--wandb_episode_history_every_iters", type=int, default=100)
    p.add_argument("--vis_enable", default=False, action="store_true")
    p.add_argument("--vis_backend", choices=["rerun"], default="rerun")
    p.add_argument("--vis_env_idx", type=int, default=0)
    p.add_argument("--vis_every_iters", type=int, default=10)
    p.add_argument("--vis_every_steps", type=int, default=10)
    p.add_argument("--vis_student", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--vis_spawn", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--vis_show_aabb", default=False, action=argparse.BooleanOptionalAction)
    return p


def parse_scenarios(items):
    if items is None:
        return ["nominal"]
    out = []
    for raw in items:
        for token in str(raw).split(","):
            name = token.strip().lower().replace("-", "_")
            if not name:
                continue
            if name not in SUPPORTED_SCENARIOS:
                raise ValueError(
                    f"--scenarios unsupported {name!r}; choose {list(SUPPORTED_SCENARIOS)}"
                )
            if name not in out:
                out.append(name)
    return out or ["nominal"]


def set_global_seed(seed: int, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
    if args.gray_width < 1 or args.gray_height < 1:
        raise ValueError("--gray_width/--gray_height must be >= 1")
    if args.gray_nn_width < 1 or args.gray_nn_height < 1:
        raise ValueError("--gray_nn_width/--gray_nn_height must be >= 1")
    if not args.imx900_calibration:
        raise ValueError("--imx900_calibration must be set")
    if not os.path.isfile(args.imx900_calibration):
        raise ValueError(
            f"--imx900_calibration not found: {args.imx900_calibration}"
        )
    if args.gray_blur_kernel_size < 1 or args.gray_blur_kernel_size % 2 == 0:
        raise ValueError("--gray_blur_kernel_size must be a positive odd integer")
    if args.gray_motion_depth_floor <= 0:
        raise ValueError("--gray_motion_depth_floor must be > 0")
    if not (0.0 <= args.gray_dark_threshold <= 1.0):
        raise ValueError("--gray_dark_threshold must be in [0,1]")
    if args.loss_v_window < 1 or args.base_control_freq <= 0:
        raise ValueError("invalid temporal configuration")
    if args.simple_goal_x <= args.simple_start_x:
        raise ValueError("--simple_goal_x must be greater than --simple_start_x")
    if not (args.simple_start_x < args.simple_wall_x < args.simple_goal_x):
        raise ValueError("--simple_wall_x must lie between start and goal")
    if args.simple_slit_half_y <= 0:
        raise ValueError("--simple_slit_half_y must be > 0")
    if (args.simple_slit_half_y_min is None) != (args.simple_slit_half_y_max is None):
        raise ValueError("--simple_slit_half_y_min/max must be set together")
    if args.simple_slit_half_y_min is None:
        args.simple_slit_half_y_min = args.simple_slit_half_y
        args.simple_slit_half_y_max = args.simple_slit_half_y
    if args.simple_back_wall_x_min <= args.simple_wall_x:
        raise ValueError("--simple_back_wall_x_min must be behind the slit wall")
    if args.simple_back_wall_x_max < args.simple_back_wall_x_min:
        raise ValueError("--simple_back_wall_x_max must be >= min")
    if args.random_rotation_max_deg < 0 or args.collision_clearance < 0:
        raise ValueError("rotation/collision parameters must be non-negative")
    if args.gray_dark_scale <= 0 or args.gray_bright_scale <= 0:
        raise ValueError("illumination scales must be > 0")
    if args.gray_transition_width <= 0:
        raise ValueError("--gray_transition_width must be > 0")
    if args.gray_light_jitter < 0:
        raise ValueError("--gray_light_jitter must be >= 0")
    if not (0.0 <= args.camera_ema_alpha < 1.0):
        raise ValueError("--camera_ema_alpha must be in [0,1)")
    for name in (
        "fixed_camera_exposure",
        "fixed_camera_gain",
        "fixed_random_exposure_min",
        "fixed_random_exposure_max",
        "fixed_random_gain_min",
        "fixed_random_gain_max",
    ):
        if not (0.0 <= float(getattr(args, name)) <= 1.0):
            raise ValueError(f"--{name} must be in [0,1]")
    if args.fixed_random_exposure_max < args.fixed_random_exposure_min:
        raise ValueError("fixed random exposure max must be >= min")
    if args.fixed_random_gain_max < args.fixed_random_gain_min:
        raise ValueError("fixed random gain max must be >= min")
    if args.camera_control_mode in {
        "fixed", "fixed_random_static", "mean_ae", "gradient_ae"
    }:
        args.sensor_grad_mode = "detached"
        args.coef_cam_smooth = 0.0
    if args.train_flight_only and args.train_camera_only:
        raise ValueError("--train_flight_only and --train_camera_only are mutually exclusive")
    if args.train_flight_only:
        args.coef_cam_smooth = 0.0
    if args.wandb_episode_history_every_iters < 1:
        raise ValueError("--wandb_episode_history_every_iters must be >= 1")


def print_runtime_mode(args):
    print("=" * 30 + " Runtime Mode " + "=" * 30)
    print("sensor                    : grayscale / IMX900 target")
    print(f"camera_control_mode       : {args.camera_control_mode}")
    print(f"sensor_grad_mode          : {args.sensor_grad_mode}")
    print(f"policy_gray_mode          : {args.policy_gray_mode}")
    print(f"train_flight_only         : {args.train_flight_only}")
    print(f"train_camera_only         : {args.train_camera_only}")
    print(f"scenarios                 : {args.scenarios}")
    print(
        "gray_camera               : "
        f"{args.gray_width}x{args.gray_height} -> "
        f"{args.gray_nn_width}x{args.gray_nn_height}"
    )
    print(f"imx900_calibration        : {args.imx900_calibration}")
    print(f"require_calibrated_imx900 : {args.require_calibrated_imx900}")
    print(
        "illumination              : "
        f"dark={args.gray_dark_scale}, bright={args.gray_bright_scale}, "
        f"transition_x={args.gray_transition_x}, width={args.gray_transition_width}"
    )
    print(
        "environment               : single_wall_slit "
        f"wall_x={args.simple_wall_x}, slit_y={args.simple_slit_center_y_min}.."
        f"{args.simple_slit_center_y_max}, slit_half_y={args.simple_slit_half_y_min}.."
        f"{args.simple_slit_half_y_max}"
    )
    print("=" * 75)


def parse_args():
    args = build_parser().parse_args()
    args.scenarios = parse_scenarios(args.scenarios)
    set_global_seed(args.seed, args.deterministic)
    validate_args(args)
    return args

import shlex
from pathlib import Path

from config import build_parser, parse_scenarios, validate_args


def _tokens(path):
    parts = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            parts.extend(shlex.split(line))
    return parts


def test_all_active_gray_configs_parse_and_validate():
    parser = build_parser()
    configs = sorted(Path("configs").glob("gray_*.args"))
    assert configs
    for path in configs:
        args = parser.parse_args(_tokens(path))
        args.scenarios = parse_scenarios(args.scenarios)
        validate_args(args)
        assert args.policy_gray_mode in {"gray", "zero"}
        assert args.camera_control_mode in {
            "fixed",
            "fixed_random_static",
            "learned",
            "mean_ae",
            "gradient_ae",
        }
        assert args.imx900_calibration.endswith(".json")


def test_camera_learning_configs_are_matched_except_sensor_gradient():
    parser = build_parser()
    full = parser.parse_args(_tokens(Path("configs/gray_camera_full.args")))
    detached = parser.parse_args(_tokens(Path("configs/gray_camera_detached.args")))
    for args in (full, detached):
        args.scenarios = parse_scenarios(args.scenarios)
        validate_args(args)

    ignored = {"sensor_grad_mode"}
    full_dict = {k: v for k, v in vars(full).items() if k not in ignored}
    detached_dict = {k: v for k, v in vars(detached).items() if k not in ignored}
    assert full_dict == detached_dict
    assert full.sensor_grad_mode == "full"
    assert detached.sensor_grad_mode == "detached"


def test_camera_physics_are_not_duplicated_in_experiment_configs():
    forbidden = (
        "--gray_exposure_us_min",
        "--gray_exposure_us_max",
        "--gray_exposure_reference_us",
        "--gray_gain_factor_min",
        "--gray_gain_factor_max",
        "--gray_shot_noise_scale",
        "--gray_read_noise_std",
        "--gray_read_noise_gain_scale",
        "--gray_black_level",
        "--gray_blur_scale",
        "--gray_quantization_bits",
    )
    for path in sorted(Path("configs").glob("gray_*.args")):
        text = path.read_text()
        assert "--imx900_calibration" in text
        for token in forbidden:
            assert token not in text, f"{token} should live in calibration JSON, found in {path}"

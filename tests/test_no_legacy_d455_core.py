from pathlib import Path


CORE = [
    "config.py",
    "main_cuda.py",
    "eval.py",
    "model.py",
    "rollout_ops.py",
    "trainer.py",
    "losses.py",
    "train_utils.py",
    "env_cuda.py",
    "autograd_ops.py",
    "src/quadsim.cpp",
    "src/quadsim_kernel.cu",
]

FORBIDDEN_EXECUTABLE_SYMBOLS = [
    "DiffDepthFunction",
    "ActiveSensingSensorFunction",
    "render_diff_depth",
    "active_sensing_sensor",
    "fixed_camera_power",
    "fixed_random_power",
    "cam_power_baseline",
    "policy_depth_mode",
    "loss_diff_depth",
]


def test_legacy_d455_executable_symbols_are_absent():
    for path in CORE:
        text = Path(path).read_text()
        for symbol in FORBIDDEN_EXECUTABLE_SYMBOLS:
            assert symbol not in text, f"{symbol!r} unexpectedly found in {path}"


def test_generic_geometry_depth_is_still_available():
    cpp = Path("src/quadsim.cpp").read_text()
    assert 'm.def("render_depth"' in cpp
    assert "render_diff_depth" not in cpp

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


def test_grayscale_branch_exposes_geometry_not_depth_camera_api():
    cpp = Path("src/quadsim.cpp").read_text()
    kernel = Path("src/quadsim_kernel.cu").read_text()
    assert 'm.def("render_geometry"' in cpp
    assert 'm.def("render_depth"' not in cpp
    assert "render_diff_depth" not in cpp
    assert "active_sensing_sensor" not in cpp
    assert "render_depth_kernel" not in kernel
    assert "trace_ray_device(" not in kernel
    assert "trace_ray_with_normal_device(" in kernel

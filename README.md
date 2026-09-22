> **Branch notice (2026-09-22):** this branch is transitioning from the legacy D455-inspired differentiable-depth line to **monochrome IMX900 task-driven differentiable camera control**. Phase 1 is implemented: grayscale camera semantics, differentiable exposure/gain image formation, configuration skeleton, and unit tests. The renderer and navigation/training integration are still the old depth path. The authoritative plan/status is under [docs/grayscale_imx900](docs/grayscale_imx900/README.md).

> **免责声明**：当前md中的描述并不完全等价项目中的代码实现，真实的实现以代码为准。

# DiffPhysDrone — Grayscale IMX900 transition branch

## New target

The selected real camera is **e-con Systems e-CAM37M_CUONX / Sony IMX900 monochrome global shutter**, connected by MIPI CSI-2 to the **DAMIAO DM-ORIN NX V2.X** carrier hosting the Jetson Orin NX 8GB.

The new research question is:

> Can navigation loss exploit gradients through a calibrated differentiable grayscale image-formation model to learn useful online exposure/gain control for a small quadrotor?

Start here:

- [Grayscale/IMX900 design index](docs/grayscale_imx900/README.md)
- [Full technical plan](docs/grayscale_imx900/TECHNICAL_PLAN.md)
- [Related work survey](docs/grayscale_imx900/RELATED_WORK.md)
- [Hardware integration plan](docs/grayscale_imx900/HARDWARE_IMX900.md)
- [Calibration and sim-to-real](docs/grayscale_imx900/CALIBRATION_SIM2REAL.md)
- [Codex implementation guide](docs/grayscale_imx900/CODEX_IMPLEMENTATION_GUIDE.md)
- [Current real drone inventory](real_drone.sh)

## Legacy implementation retained for reference

The code inherited from `active-sensing-4f-tools-2b-core` currently remains a `diff_depth` pipeline:

1. `config.py` contains the depth-era configuration.
2. `env_cuda.py` renders geometric depth and applies a D455-inspired differentiable model.
3. `model.py` currently consumes depth-derived 2-channel features.
4. `rollout_ops.py` currently maintains `power / exposure / gain`.
5. `trainer.py` and `eval.py` still log depth-specific metrics.

This is intentional during the planning phase so the old working line remains inspectable.

## Planned replacement

~~~text
old:
geometry -> ideal depth -> D455-inspired sensor(power, exposure, gain) -> depth policy

new:
geometry/material/light -> ideal grayscale irradiance
 -> calibrated differentiable camera(exposure, gain, noise, saturation, motion blur)
 -> [current gray, previous gray]
 -> flight policy + camera policy
~~~

## Implementation discipline

Do not immediately rewrite the whole repository.

Phase 1 now provides:

- `sensors/gray_camera_semantics.py`;
- `sensors/differentiable_gray_camera.py`;
- grayscale configuration fields in `config.py`;
- `tests/test_differentiable_gray_camera.py`.

The next stage is the **ideal grayscale renderer**. Do not wire grayscale into the navigation trainer until the renderer has deterministic tests and the CUDA extension has been rebuilt successfully.

See `docs/grayscale_imx900/CODEX_IMPLEMENTATION_GUIDE.md` for the staged gates.

# DiffPhysDrone — Grayscale / IMX900 Active Sensing

> Branch: `active-sensing-grayscale-imx900`

This branch is now a **grayscale-only research line**. The executable D455 /
differentiable-depth sensor implementation has been removed. The previous
D455 work remains available in the historical branch
`active-sensing-4f-tools-2b-core`.

## Research question

Can a closed-loop quadrotor navigation loss exploit gradients through a
calibrated monochrome image-formation model to learn useful online exposure and
gain control?

Target real camera:

- e-con Systems e-CAM37M_CUONX
- Sony IMX900 monochrome global shutter
- MIPI CSI-2
- Jetson Orin NX 8GB
- DAMIAO DM-ORIN NX V2.X carrier

## Current dataflow

```text
CUDA geometry / ray intersections
        |
        v
generic geometric ray depth (internal only)
        |
        v
ideal grayscale irradiance
  - geometry-derived normals
  - Lambertian illumination
  - procedural texture
  - nominal/dark/bright/transitions
        |
        v
DifferentiableGrayCamera
  - exposure
  - gain
  - shot/read noise
  - saturation
  - quantization surrogate
  - exposure/motion blur
        |
        v
[current gray, previous gray]
        |
        +----> recurrent flight policy -> acceleration
        |
        +----> recurrent camera policy -> exposure/gain
```

The retained `quadsim_cuda.render_depth()` is **not a depth-camera model**.
It is only the generic GPU ray-intersection primitive used to reconstruct scene
geometry for grayscale rendering.

## What was removed from this branch

- D455 projector-power action;
- `DiffDepthFunction` / `ActiveSensingSensorFunction`;
- D455-like CUDA sensor kernels;
- glare/specular/dark depth-degradation formulas;
- depth fill/quality/false-depth losses;
- D455 camera semantics and calibration tools;
- D455 teacher/relabel/probe pipelines;
- depth-specific training/evaluation configs.

## Validation order

Do not jump directly to active exposure learning.

```bash
conda activate mappo-mpc

# CUDA extension changed during cleanup: rebuild it first.
pip install -e src

python -m pytest -q \
  tests/test_differentiable_gray_camera.py \
  tests/test_ideal_gray_renderer.py \
  tests/test_model_gray_mode.py \
  tests/test_gray_rollout_helpers.py

python tools/test_gray_env_render.py

# prove visual navigation
TASK=gray_gate_fixed bash run.sh
TASK=gray_gate_blind bash run.sh

# fixed-camera stress test
TASK=gray_gate_mixed_fixed bash run.sh
```

Only when fixed grayscale navigation clearly outperforms the blind baseline
should camera learning proceed.

For frozen-flight camera training:

```bash
TASK=gray_camera_full \
RUN_EXTRA_ARGS="--resume checkpoint/<flight>/checkpointXXXX.pth" \
bash run.sh

TASK=gray_camera_detached \
RUN_EXTRA_ARGS="--resume checkpoint/<flight>/checkpointXXXX.pth" \
bash run.sh
```

## Design documents

Read in this order:

1. [docs/grayscale_imx900/README.md](docs/grayscale_imx900/README.md)
2. [TECHNICAL_PLAN.md](docs/grayscale_imx900/TECHNICAL_PLAN.md)
3. [RELATED_WORK.md](docs/grayscale_imx900/RELATED_WORK.md)
4. [HARDWARE_IMX900.md](docs/grayscale_imx900/HARDWARE_IMX900.md)
5. [CALIBRATION_SIM2REAL.md](docs/grayscale_imx900/CALIBRATION_SIM2REAL.md)
6. [CODEX_IMPLEMENTATION_GUIDE.md](docs/grayscale_imx900/CODEX_IMPLEMENTATION_GUIDE.md)

The exact real IMX900 exposure/gain ranges and noise/blur coefficients remain
provisional until the purchased camera and e-con driver are characterized.

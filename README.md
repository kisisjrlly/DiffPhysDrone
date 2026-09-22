# DiffPhysDrone — Grayscale / IMX900 Active Sensing

> Branch: `active-sensing-grayscale-imx900`

This branch is a **grayscale-only research line**. The executable D455 /
differentiable-depth sensor implementation and the old paper/experiment archive
have been removed from this branch. Historical D455 work remains in
`active-sensing-4f-tools-2b-core` and Git history.

## Research question

Can a closed-loop quadrotor navigation loss exploit gradients through a
**real-camera-calibrated** monochrome image-formation surrogate to learn useful
online exposure/gain control?

Target hardware:

- e-con Systems e-CAM37M_CUONX
- Sony IMX900 monochrome global shutter
- MIPI CSI-2
- Jetson Orin NX 8GB
- DAMIAO DM-ORIN NX V2.X carrier

## Current design decision

The camera model is intentionally **not** a hard dependency on
End2endImaging, DeepLens, or JOCA.

- End2endImaging informs the sensor/noise/calibration structure.
- JOCA informs the task-driven exposure/gain problem and baselines.
- DeepLens is reserved for optional offline optics/PSF fidelity later.
- The runtime model remains a lightweight project-owned
  `IMX900DifferentiableCamera`.

See
[OPEN_SOURCE_CAMERA_MODEL_REVIEW.md](docs/grayscale_imx900/OPEN_SOURCE_CAMERA_MODEL_REVIEW.md)
for the detailed review.

## Current dataflow

~~~text
CUDA geometry / ray intersections
        |
        v
generic ray-hit geometry (internal only)
  - hit distance
  - exact surface normal
        |
        v
ideal grayscale irradiance
  - Lambertian illumination
  - procedural texture
  - nominal/dark/bright/transitions
        |
        v
IMX900DifferentiableCamera
  ^     |
  |     | exposure mapping / measured LUT
  |     | exposure integration
  |     | gain mapping / measured LUT
  |     | shot + read noise / measured read-noise LUT
  |     | saturation surrogate
  |     | optional measured response LUT
  |     | STE quantization
  |     | exposure/motion blur
  |     | calibrated command delay + jitter
  |
IMX900Calibration JSON
        |
        v
[current gray, previous gray]
        |
        +----> recurrent flight policy -> acceleration
        |
        +----> recurrent camera policy -> exposure/gain
~~~

The branch exposes only `quadsim_cuda.render_geometry()` for appearance
geometry. Internal hit distance/normals are renderer inputs, not policy
observations.

## Calibration profile

Camera physics no longer live as duplicated constants in every experiment
config.

Current development profile:

`configs/calibration/imx900_provisional.json`

It is explicitly marked:

~~~json
"calibrated": false
~~~

and must not be reported as measured IMX900 behavior.

After real-camera characterization, calibration tools should write a measured
profile with the same schema. Real/sim-to-real runs can require it with:

~~~text
--require_calibrated_imx900
~~~

## What was removed from this branch

- D455 projector-power action;
- differentiable-depth autograd wrappers and CUDA sensor kernels;
- glare/specular/dark hand-written depth degradation;
- depth fill/quality/false-depth losses;
- D455 camera semantics/calibration tools;
- D455 teacher/relabel/probe pipelines;
- old depth experiment configurations;
- old paper/results archive.

## Validation order

Do not jump directly to active exposure learning.

~~~bash
conda activate mappo-mpc
pip install -e src

python -m pytest -q \
  tests/test_imx900_camera.py \
  tests/test_ideal_gray_renderer.py \
  tests/test_model_gray_mode.py \
  tests/test_gray_rollout_helpers.py \
  tests/test_auto_exposure.py \
  tests/test_gray_configs.py \
  tests/test_no_legacy_d455_core.py

python tools/test_gray_env_render.py

TASK=gray_gate_fixed bash run.sh
TASK=gray_gate_blind bash run.sh
TASK=gray_gate_mixed_fixed bash run.sh
~~~

Only when fixed grayscale navigation clearly outperforms the blind baseline
should learned camera control be interpreted.

Then compare, from the same successful frozen flight checkpoint:

- fixed nominal;
- random-static;
- mean AE;
- gradient AE;
- learned-detached;
- learned-differentiable.

A JOCA-style derivative-free/local-search correction may be added later only as
a separate baseline/fallback, not as hidden supervision for the main method.

## Design documents

Read in this order:

1. [docs/grayscale_imx900/README.md](docs/grayscale_imx900/README.md)
2. [OPEN_SOURCE_CAMERA_MODEL_REVIEW.md](docs/grayscale_imx900/OPEN_SOURCE_CAMERA_MODEL_REVIEW.md)
3. [TECHNICAL_PLAN.md](docs/grayscale_imx900/TECHNICAL_PLAN.md)
4. [RELATED_WORK.md](docs/grayscale_imx900/RELATED_WORK.md)
5. [HARDWARE_IMX900.md](docs/grayscale_imx900/HARDWARE_IMX900.md)
6. [CALIBRATION_SIM2REAL.md](docs/grayscale_imx900/CALIBRATION_SIM2REAL.md)
7. [CODEX_IMPLEMENTATION_GUIDE.md](docs/grayscale_imx900/CODEX_IMPLEMENTATION_GUIDE.md)

The real IMX900 exposure/gain mappings, read/shot noise, black level,
saturation, quantization, optional fixed response curve, blur, and
command-latency distribution remain provisional until the purchased e-con
camera stack is characterized.

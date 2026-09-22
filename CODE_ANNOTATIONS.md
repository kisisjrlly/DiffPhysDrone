# Code Map — Grayscale / IMX900 Branch

## Core files

- `config.py` — grayscale experiment configuration and calibration-profile path.
- `main_cuda.py` — training entry point.
- `eval.py` — grayscale evaluation entry point.
- `env_cuda.py` — quadrotor dynamics, wall/slit geometry, illumination
  scenarios, and ideal grayscale rendering bridge.
- `model.py` — recurrent flight policy plus 2-D exposure/gain controller.
- `rollout_ops.py` — image formation call, camera state/update, state
  construction, and action decoding.
- `losses.py` — navigation losses plus camera-switch smoothness.
- `trainer.py` — full-BPTT grayscale rollout/training.
- `sensors/imx900_calibration.py` — real-camera calibration profile and
  differentiable normalized-action -> physical exposure/gain mapping.
- `sensors/imx900_camera.py` — IMX900-specific differentiable image-formation
  surrogate.
- `configs/calibration/imx900_provisional.json` — explicitly unmeasured
  development profile.
- `render/ideal_gray.py` — generic geometry -> scene irradiance.
- `autograd_ops.py` — differentiable quadrotor dynamics only.
- `src/quadsim_kernel.cu` — generic collision/ray-intersection CUDA kernels.
- `src/quadsim.cpp` — CUDA bindings; no D455 sensor API remains.

## Runtime dataflow

1. `env_cuda.Env.render_gray_ideal()` calls `quadsim_cuda.render_geometry()`
   to obtain internal hit distance and exact normals.
2. `render/ideal_gray.py` converts geometry into monochrome scene irradiance.
3. `rollout_ops.render_gray_sensor()` calls
   `IMX900DifferentiableCamera(irradiance, exposure, gain)`.
4. `IMX900DifferentiableCamera` loads physical coefficients from
   `IMX900Calibration`.
5. Policy input is `[current_gray, previous_gray]`.
6. Flight branch outputs acceleration.
7. Camera branch outputs normalized `[exposure, gain]`.
8. `sensor_grad_mode=full` preserves navigation-loss gradients through image
   formation; `detached` cuts exposure/gain at the sensor boundary.

## Camera-model provenance

The implementation is project-owned and lightweight:

- End2endImaging is a modeling/configuration reference;
- JOCA is a task-driven camera-control/experiment reference;
- DeepLens is not a runtime dependency and is reserved for optional optics
  fidelity later.

See `docs/grayscale_imx900/OPEN_SOURCE_CAMERA_MODEL_REVIEW.md`.

## Important semantic boundary

`render_geometry` is not a depth camera. Its hit distance/normals remain
internal to appearance rendering and must never become privileged policy
observations.

The provisional calibration JSON is not a measured IMX900 specification.
Use `--require_calibrated_imx900` for runs that must reject provisional
profiles.

## Current experiment gates

- `gray_gate_fixed.args`: visual navigation baseline.
- `gray_gate_blind.args`: zero-image control.
- `gray_gate_mixed_fixed.args`: fixed-camera illumination stress test.
- `gray_camera_random_static.args`: random-static camera baseline.
- `gray_camera_mean_ae.args`: mean-intensity AE baseline.
- `gray_camera_gradient_ae.args`: gradient-based AE baseline.
- `gray_camera_full.args`: frozen-flight differentiable camera learning.
- `gray_camera_detached.args`: matched sensor-gradient ablation.

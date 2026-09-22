# Code Map — Grayscale / IMX900 Branch

## Core files

- `config.py` — grayscale-only experiment configuration.
- `main_cuda.py` — training entry point.
- `eval.py` — grayscale evaluation entry point.
- `env_cuda.py` — quadrotor dynamics environment, wall/slit geometry,
  illumination scenarios, and ideal grayscale rendering bridge.
- `model.py` — recurrent flight policy plus 2-D exposure/gain controller.
- `rollout_ops.py` — grayscale sensor rendering, state construction, camera
  actuator update, and action decoding.
- `losses.py` — navigation losses plus camera switching smoothness.
- `trainer.py` — full-BPTT grayscale rollout/training.
- `sensors/gray_camera_semantics.py` — normalized -> physical camera mapping.
- `sensors/differentiable_gray_camera.py` — differentiable exposure/gain,
  noise, saturation, quantization, and motion-blur model.
- `render/ideal_gray.py` — generic geometry -> scene irradiance.
- `autograd_ops.py` — differentiable quadrotor dynamics only.
- `src/quadsim_kernel.cu` — generic collision/ray-intersection CUDA kernels.
- `src/quadsim.cpp` — CUDA bindings; no D455 sensor API remains.

## Runtime dataflow

1. `env_cuda.Env.render_gray_ideal()` calls
   `quadsim_cuda.render_geometry()` to obtain ray-hit distance and exact
   surface normals.
2. `render/ideal_gray.py` converts that internal geometry into monochrome
   scene irradiance.
3. `rollout_ops.render_gray_sensor()` applies
   `DifferentiableGrayCamera(irradiance, exposure, gain)`.
4. Policy input is `[current_gray, previous_gray]`.
5. Flight branch outputs acceleration.
6. Camera branch outputs normalized `[exposure, gain]`.
7. `sensor_grad_mode=full` preserves the navigation-loss gradient through
   exposure/gain -> image; `detached` cuts that path.
8. Camera actions affect subsequent frames through the actuator EMA.

## Important semantic boundary

`render_geometry` is an internal **geometric ray caster**, not a camera
sensor simulation. Its hit distance/normals must remain internal to appearance
rendering and must never become privileged policy observations.

## Current experiment gates

- `gray_gate_fixed.args`: visual navigation baseline.
- `gray_gate_blind.args`: zero-image control.
- `gray_gate_mixed_fixed.args`: fixed-camera illumination stress test.
- `gray_camera_full.args`: frozen-flight differentiable camera learning.
- `gray_camera_detached.args`: matched sensor-gradient ablation.
- `gray_camera_random_static.args`: static random camera baseline.

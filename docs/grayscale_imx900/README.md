# Grayscale Active Sensing / IMX900

> Branch: `active-sensing-grayscale-imx900`
>
> Status: **D455 executable path removed; grayscale-only pipeline implemented to
> the fixed-navigation / frozen-flight camera-training gate. Local CUDA rebuild
> and smoke tests are required before long runs.**

## 1. Branch scope

This branch studies:

> task-driven differentiable exposure/gain control for closed-loop monocular
> grayscale quadrotor navigation.

Selected real camera:

- e-con Systems e-CAM37M_CUONX
- Sony IMX900 monochrome global shutter
- MIPI CSI-2
- Jetson Orin NX 8GB
- DAMIAO DM-ORIN NX V2.X

The previous D455 work is preserved in
`active-sensing-4f-tools-2b-core`, not in this branch.

## 2. Important cleanup boundary

Removed here:

- projector-power camera action;
- D455/depth sensor semantics;
- differentiable-depth autograd wrappers;
- D455 CUDA forward/backward sensor kernels;
- glare/specular/dark hand-written depth degradation;
- depth fill/quality/false-depth losses;
- D455 calibration, teacher, DAgger, probe, and real-flight depth tools;
- old depth experiment configurations.

Retained:

- differentiable quadrotor dynamics;
- collision geometry;
- generic CUDA ray intersection.

The grayscale branch now exposes only `quadsim_cuda.render_geometry()`, which
returns internal ray-hit distance plus exact surface normals for appearance
rendering. There is no public `render_depth` sensor API in this branch. The
internal hit distance is never fed to the policy and must not be described as a
simulated depth camera.

## 3. Current pipeline

```text
geometry
  -> CUDA ray hit distance + exact hit normal (internal)
  -> ideal grayscale irradiance
       - exact surface normals
       - Lambertian illumination
       - procedural texture
       - nominal/dark/bright/transitions
  -> DifferentiableGrayCamera
       - exposure
       - gain
       - shot/read noise
       - saturation
       - quantization surrogate
       - motion blur
  -> [current_gray, previous_gray]
       -> recurrent flight policy
       -> recurrent exposure/gain policy
```

## 4. Current experiment configs

- `gray_gate_fixed.args` — fixed nominal camera, real grayscale input.
- `gray_gate_blind.args` — matched zero-image control.
- `gray_gate_mixed_fixed.args` — fixed camera under dark/bright/transitions.
- `gray_camera_random_static.args` — random static E/G baseline.
- `gray_camera_full.args` — frozen-flight camera-only training with sensor gradient.
- `gray_camera_detached.args` — matched camera-only run with E/G detached before image formation.

## 5. Required local gate now

Because CUDA bindings/kernels were cleaned, rebuild before any test:

```bash
cd /home/zhaoguodong/work/code/DiffPhysDrone
git checkout active-sensing-grayscale-imx900
git pull

conda activate mappo-mpc
pip install -e src

python -m pytest -q \
  tests/test_differentiable_gray_camera.py \
  tests/test_ideal_gray_renderer.py \
  tests/test_model_gray_mode.py \
  tests/test_gray_rollout_helpers.py

python tools/test_gray_env_render.py
```

Inspect `logs/gray_smoke/`. Nominal/dark/bright images must be visibly
different while preserving the slit geometry.

Then run the matched navigation pair:

```bash
TASK=gray_gate_fixed bash run.sh
TASK=gray_gate_blind bash run.sh
```

Do **not** interpret active-camera results until fixed grayscale navigation
clearly beats the blind control.

## 6. Camera-learning gate

After a successful flight checkpoint exists:

```bash
TASK=gray_camera_full \
RUN_EXTRA_ARGS="--resume checkpoint/<flight>/checkpointXXXX.pth" \
bash run.sh

TASK=gray_camera_detached \
RUN_EXTRA_ARGS="--resume checkpoint/<flight>/checkpointXXXX.pth" \
bash run.sh
```

Both must start from the same flight checkpoint.

## 7. Documents

1. [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md)
2. [RELATED_WORK.md](RELATED_WORK.md)
3. [HARDWARE_IMX900.md](HARDWARE_IMX900.md)
4. [CALIBRATION_SIM2REAL.md](CALIBRATION_SIM2REAL.md)
5. [CODEX_IMPLEMENTATION_GUIDE.md](CODEX_IMPLEMENTATION_GUIDE.md)

## 8. What is intentionally still provisional

The current exposure range, gain mapping, shot/read noise, saturation behavior,
motion blur coefficient, and command dynamics are simulation placeholders.

They must be replaced/calibrated from the real e-con IMX900 hardware before
sim-to-real claims.

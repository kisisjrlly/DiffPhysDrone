# Codex Implementation Guide

> This file is the implementation contract for future local Codex work.
>
> Do not attempt all stages in one change. Make small, testable commits.

## Phase 0 — protect the old result

Before modifying executable code:

- keep \`active-sensing-4f-tools-2b-core\` untouched;
- work only on \`active-sensing-grayscale-imx900\` or descendants;
- do not delete legacy D455 code until grayscale functionality has replacements and tests;
- tag/record the current branch head.

## Phase 1 — configuration and semantics only

Add a camera-mode abstraction without changing behavior.

Target concepts:

~~~text
sensor_type = diff_depth | gray
camera_action_dim = 3 | 2
~~~

For the gray path add explicit config for:

- \`gray_width\`, \`gray_height\`;
- physical exposure min/max;
- gain min/max;
- sensor gradient mode;
- blur/noise toggles;
- lighting/randomization parameters.

Create a single camera semantics class for normalized<->physical conversion.

Acceptance:

- old depth configs still parse/run;
- gray config parses;
- no silent reuse of D455 \`power\` semantics.

## Phase 2 — ideal grayscale renderer

Add a CUDA/PyTorch entry point that returns ideal grayscale appearance.

Reuse existing ray intersection and surface-normal code.

Initial scene appearance:

- ambient + Lambertian directional lighting;
- per-object albedo;
- procedural texture;
- deterministic mode for tests.

Acceptance:

- output finite in [0,1];
- camera motion changes viewpoint;
- normals produce expected shading;
- textured gate is visually distinguishable;
- batch rendering works on GPU.

## Phase 3 — differentiable gray camera

Implement \`DifferentiableGrayCamera\` in PyTorch.

Inputs:

~~~text
ideal_gray
exposure01
gain01
motion state
optional deterministic noise tensors
~~~

Outputs:

~~~text
sensor_gray
aux metrics
~~~

Aux:

- physical exposure;
- physical gain;
- saturation fraction;
- dark fraction;
- noise std estimate;
- blur strength.

Acceptance tests:

1. exposure gradient matches finite difference;
2. gain gradient matches finite difference;
3. increasing exposure brightens a non-saturated static image;
4. high exposure eventually saturates;
5. high motion + long exposure increases blur metric;
6. repeated forward with fixed noise tensors is deterministic;
7. no NaN/Inf in valid parameter range.

## Phase 4 — grayscale policy input

Replace depth preprocessing on the gray path with:

\[
[current\ gray, previous\ gray].
\]

Keep existing 2-channel visual stem initially.

Refactor names so gray code does not call variables \`depth_obs\`.

Acceptance:

- forward pass supports gray frames;
- previous-frame reset at episode start is defined;
- frame tensors stay on GPU;
- old depth mode still works if intentionally retained.

## Phase 5 — 2-D camera action

Gray camera controller:

~~~text
camera_state = [exposure, gain]
camera_action = [target exposure, target gain]
~~~

Remove power only on gray path.

Add a configurable camera-actuator model:

- EMA/slew limit;
- update interval;
- optional command delay queue.

Acceptance:

- dimensions are correct end-to-end;
- logs show requested and applied camera state separately;
- detached/full gradient mode differs only in sensor gradient path.

## Phase 6 — fixed-camera grayscale flight

Do **not** learn camera control yet.

Train/evaluate fixed-camera grayscale navigation.

First benchmark:

- textured gate/slit;
- nominal light;
- modest randomization.

Pass gate before moving on:

- success reliably above blind/zero-image baseline;
- collision rate reasonable;
- policy demonstrably uses image input.

Add an image-zero ablation to test this.

## Phase 7 — illumination and motion benchmark

Add:

- low-light;
- high-light;
- bright-to-dark;
- dark-to-bright;
- fast gate approach.

Verify manually that:

- no single fixed exposure/gain dominates all conditions;
- long exposure helps dark static scenes;
- long exposure hurts fast motion through blur;
- high gain improves signal but increases noise.

If these trade-offs do not appear, fix the sensor model before learning camera control.

## Phase 8 — classical baselines

Implement:

- fixed nominal;
- random-static;
- mean-brightness AE;
- gradient-based AE.

Keep camera update rate and actuator constraints matched.

## Phase 9 — learned camera, frozen flight

Freeze the successful flight policy.

Train camera policy only.

Two methods:

### detached

~~~python
gray = camera(
    ideal,
    exposure.detach(),
    gain.detach(),
    ...
)
~~~

### differentiable

~~~python
gray = camera(
    ideal,
    exposure,
    gain,
    ...
)
~~~

Everything else must match.

Log the gradient norm reaching camera output/parameters.

Primary go/no-go:

- differentiable method must show stable, reproducible benefit over detached on at least one deliberately constructed mixed illumination/motion task;
- camera trajectories must be physically interpretable;
- no degenerate bound saturation.

## Phase 10 — joint fine-tuning

Only after Phase 9 passes.

Use small learning rate and separate parameter groups.

Report both frozen-flight and joint results; do not hide the simpler causal experiment.

## Phase 11 — real IMX900 wrapper

After hardware works:

Create something like:

\`tools/realflight/imx900_camera_node.py\`

Responsibilities:

- V4L2/GStreamer or e-con-supported capture path;
- explicit disable of auto controls;
- manual exposure/gain setting;
- frame timestamps;
- requested/effective parameter logging;
- frame drop stats;
- ROS publication if the existing real-flight stack remains ROS1.

Do not bind the navigation model directly to an undocumented shell command.

## Phase 12 — calibration tools

Implement:

~~~text
tools/grayscale_calibration/capture_dark.py
tools/grayscale_calibration/sweep_exposure.py
tools/grayscale_calibration/sweep_gain.py
tools/grayscale_calibration/measure_latency.py
tools/grayscale_calibration/measure_motion_blur.py
tools/grayscale_calibration/fit_camera_model.py
tools/grayscale_calibration/report.py
~~~

Every tool must save raw metadata and be rerunnable.

## Phase 13 — real deployment

Map normalized actions to the calibrated physical controls.

Validate:

- range clamping;
- update latency;
- camera thread safety;
- policy/frame timing;
- watchdog fallback.

Fallback behavior:

- if camera control fails, revert to safe nominal exposure/gain;
- if frames become stale, flight policy should enter a predefined safe behavior rather than extrapolate indefinitely.

## Suggested file architecture

~~~text
sensors/
  gray_camera_semantics.py
  differentiable_gray_camera.py

render/
  gray_render.py

configs/
  gray_gate_fixed.args
  gray_gate_active.args

tools/
  grayscale_calibration/
  realflight/
    imx900_camera_node.py

docs/
  grayscale_imx900/
    ...
~~~

CUDA functions may stay in \`src/quadsim_kernel.cu\` initially, but split them later if the file becomes difficult to maintain.

## Required regression tests

Create automated tests for:

- camera normalized/physical mapping;
- exposure finite-difference gradient;
- gain finite-difference gradient;
- deterministic renderer scene;
- grayscale policy forward shape;
- detached mode gradient absence;
- full mode gradient presence;
- actuator delay/slew behavior;
- no D455 power dependency in grayscale config.

## Coding principles

1. Prefer explicit names over preserving depth-era names.
2. Centralize physical units.
3. Separate requested camera target from applied/effective state.
4. Keep sensor stochasticity controllable by seeded/fixed noise.
5. Make every ablation a config switch, not a forked code path.
6. Never silently use privileged ground-truth depth in the gray policy.
7. Keep classical AE independent from navigation loss.
8. Do not optimize image-quality losses in the main differentiable method unless explicitly running an ablation.
9. Log enough state to reproduce any paper figure.
10. Every stage should have a small acceptance test before proceeding.

## First Codex task recommendation

The first implementation PR should contain only:

- gray config/semantics skeleton;
- \`DifferentiableGrayCamera\` with synthetic input;
- finite-difference gradient tests;
- no renderer changes;
- no training changes.

This isolates the most important mathematical component before touching the full navigation pipeline.

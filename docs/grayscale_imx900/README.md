# Grayscale Active Sensing / IMX900 Transition

> Status: **grayscale fixed-camera integration is now present in code**. Camera semantics, differentiable exposure/gain image formation, ideal grayscale appearance rendering, 2-channel grayscale policy input, 2-D exposure/gain camera state, and a fixed-camera training path are implemented. The integrated CUDA environment still needs to be smoke-tested on the local `mappo-mpc` installation before long training runs.
>
> Branch: `active-sensing-grayscale-imx900`
>
> Base branch: `active-sensing-4f-tools-2b-core`
>
> Last research update: 2026-09-22.

## 1. Why this branch exists

The previous DiffPhysDrone line studies task-aware control of a D455-inspired differentiable depth sensor with camera actions `power / exposure / gain`. That line became increasingly dependent on hand-designed scene-specific depth degradation terms (glare/specular/dark), teacher labels, DAgger-style relabeling, and flight-only adaptation.

This branch intentionally simplifies the scientific question:

> Can a downstream drone-navigation objective directly exploit gradients through a differentiable image-formation model to choose useful camera exposure and gain online?

The new primary sensor is a **monochrome global-shutter frame camera**, not a depth camera. The selected real sensor is the **e-con Systems e-CAM37M_CUONX**, based on the **Sony Pregius S IMX900 monochrome global-shutter sensor**, connected by MIPI CSI-2 to the onboard Jetson Orin NX.

The branch is not a cosmetic rewrite of the old D455 model. The sensor abstraction changes from:

~~~
geometry -> ideal depth -> D455-inspired degradation(power, exposure, gain) -> depth policy
~~~

to:

~~~
geometry/material/light -> ideal grayscale irradiance
    -> differentiable camera(exposure, gain, noise, saturation, motion blur)
    -> grayscale navigation policy
~~~

## 2. Frozen decisions

The following choices should be treated as project decisions unless new hardware measurements invalidate them:

1. **Primary real camera:** e-con Systems e-CAM37M_CUONX, Sony IMX900 monochrome.
2. **Camera interface:** MIPI CSI-2 into the DAMIAO DM-ORIN NX V2.X carrier.
3. **Camera action:** two variables only: exposure time and gain.
4. **Primary visual modality:** monochrome/grayscale. RGB is out of scope for the first implementation.
5. **Primary learning claim:** task-driven differentiable camera control for closed-loop drone navigation.
6. **Primary camera gradient:** navigation loss -> policy -> grayscale pixels -> exposure/gain.
7. **No requirement to differentiate scene geometry in v1.** The scene renderer may be non-differentiable w.r.t. pose/geometry; the image-formation block must be differentiable w.r.t. camera parameters.
8. **Do not use D455 depth as an input in the main grayscale experiments.** D455 may remain available only as a temporary debugging/safety/reference sensor during transition.
9. **Do not claim the IMX900 simulator is a transistor-level or firmware-exact digital twin.** It is a real-camera-calibrated differentiable image-formation model.
10. **Do not claim novelty as “first task-driven adaptive camera control.”** JOCA (WACV 2026) is directly relevant prior work. Novelty must be framed around closed-loop aerial navigation, physically calibrated real camera deployment, and the use of sensor gradients in the navigation loop.

## 3. Documents

Read these in order before implementing anything:

1. [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md) — full research and software architecture.
2. [RELATED_WORK.md](RELATED_WORK.md) — prior work, codebases, and novelty boundary.
3. [HARDWARE_IMX900.md](HARDWARE_IMX900.md) — real drone, carrier board, IMX900 integration, and hardware gates.
4. [CALIBRATION_SIM2REAL.md](CALIBRATION_SIM2REAL.md) — real-camera characterization and simulator calibration.
5. [CODEX_IMPLEMENTATION_GUIDE.md](CODEX_IMPLEMENTATION_GUIDE.md) — exact staged implementation order and acceptance tests.

The legacy D455/depth code and paper files are intentionally kept in this branch for reference. They are **not** the new design specification.

## 4. Target system

### Simulation

~~~
DiffPhysDrone dynamics + geometry
        |
        v
ideal grayscale irradiance / appearance renderer
        |
        v
DifferentiableGrayCamera
  - exposure integration
  - gain
  - shot/read noise
  - saturation / quantization surrogate
  - motion blur
  - optional vignetting / response calibration
        |
        v
[current gray, previous gray]
        |
        +----> flight policy ----> flight action
        |
        +----> camera policy ----> exposure/gain
                                  |
                                  +---- back into camera model
~~~

### Real drone

~~~
e-CAM37M_CUONX / IMX900 Mono
        |
     MIPI CSI-2
        |
DAMIAO DM-ORIN NX V2.X carrier
        |
Jetson Orin NX 8GB
        |
grayscale navigation + camera policy
        |                     |
        |                     +--> exposure/gain controls
        v
MAVROS / px4ctrl
        |
UART
        |
NxtPX4 v2
        |
motors
~~~

## 5. Minimum scientific experiment

The first paper-quality experiment should compare identical flight/task settings while changing only how the camera is controlled:

| Method | Camera control | Sensor gradient used? |
|---|---|---:|
| Fixed | fixed exposure/gain | No |
| Random-static | sampled once per episode | No |
| Classical AE | image-statistic/gradient based | No |
| Learned-detached | neural camera head | No |
| **Differentiable / Ours** | neural camera head | **Yes** |
| Oracle/grid (analysis only) | local parameter search | not applicable |

The main comparison is **Learned-detached vs Differentiable**. They must share architecture, initialization policy, training budget, environment distribution, and camera model; the differentiability path is the intended experimental variable.

## 6. Initial task progression

Do not immediately recreate a large random world.

1. Fixed textured gate/slit, nominal illumination: prove grayscale flight.
2. Same geometry with low-light episodes.
3. Bright-to-dark and dark-to-bright transitions.
4. Fast gate approach to expose the exposure/SNR versus motion-blur trade-off.
5. Mixed lighting + speed.
6. Only after the causal experiment is stable: randomized obstacles, textures, lighting, and longer routes.

## 7. Important implementation constraint

The existing model currently has 2-channel visual stems. The lowest-risk migration is to preserve the 2-channel stems and reinterpret them as:

~~~
channel 0 = current grayscale frame
channel 1 = previous grayscale frame
~~~

This minimizes unnecessary architecture changes and gives the policy direct temporal information useful for motion, blur, and looming cues.

The old three-dimensional camera state and output (`power/exposure/gain`) must later become two-dimensional (`exposure/gain`), but that code migration belongs to the implementation stage.

## 8. Current branch state

Implemented:

- `config.py`: `sensor_type=gray` plus grayscale image/camera-model parameters; legacy D455-specific camera-quality coefficients are forced to zero in gray mode.
- `sensors/gray_camera_semantics.py`: normalized exposure/gain semantics with provisional physical ranges.
- `sensors/differentiable_gray_camera.py`: exposure integration, gain, reparameterized shot/read noise, saturation, optional STE quantization, and exposure/motion-dependent blur.
- `render/ideal_gray.py`: ideal grayscale appearance reconstructed from the existing CUDA geometric depth using hit points, approximate normals, Lambertian lighting, and procedural texture. Geometry is detached by default.
- `env_cuda.py::render_gray_ideal()`: non-invasive bridge from the existing CUDA depth renderer to the new grayscale appearance renderer.
- `model.py`: `sensor_type=gray` reuses the 2-channel CNN as `[current_gray, previous_gray]`; camera state/action becomes 2-D `[exposure, gain]`.
- `rollout_ops.py` / `trainer.py`: grayscale sensor rendering, matched full-vs-detached exposure/gain gradient switch, gray state construction, camera update, diagnostics, and fixed-camera grayscale rollout.
- `configs/gray_gate_fixed.args`: first fixed-camera grayscale navigation baseline.
- `policy_gray_mode=zero` and `configs/gray_gate_blind.args`: matched zero-image control to verify that the flight policy actually uses grayscale vision.
- `tools/test_gray_env_render.py`: local CUDA smoke test that saves ideal and sensor grayscale frames.
- unit tests cover camera gradients/effects, ideal renderer behavior, gray model dimensions/preprocessing, and gray rollout helpers.

Pure PyTorch camera/renderer tests were exercised during development; the integrated `Env -> quadsim_cuda.render_depth -> render_gray_ideal -> DifferentiableGrayCamera` path cannot be fully validated through the GitHub connector and must be run on the local CUDA environment.

### Required local gate before long training

From the repository root:

~~~bash
conda activate mappo-mpc
git checkout active-sensing-grayscale-imx900
git pull
python -m pytest -q \
  tests/test_differentiable_gray_camera.py \
  tests/test_ideal_gray_renderer.py \
  tests/test_model_gray_mode.py \
  tests/test_gray_rollout_helpers.py
python tools/test_gray_env_render.py
~~~

Inspect the saved images under `logs/gray_smoke/`. The gate/slit and textured surfaces should be visually distinguishable and the sensor image should respond sensibly to exposure/gain.

Only after that smoke gate passes should the first matched training pair start:

~~~bash
TASK=gray_gate_fixed LOG_TO_FILE=1 bash run.sh
TASK=gray_gate_blind LOG_TO_FILE=1 bash run.sh
~~~

The fixed-gray result must outperform the blind/zero-image control before active camera learning is considered meaningful.

Still pending:

- validate and tune the ideal grayscale appearance on the actual CUDA environment;
- demonstrate reliable fixed-camera grayscale navigation;
- construct illumination transitions where no one fixed exposure dominates;
- add classical AE baselines;
- train learned-detached vs differentiable camera control;
- implement grayscale evaluation/visualization polish;
- IMX900 runtime driver wrapper and physical calibration.

No training result should be reported as a grayscale result until these gates pass.

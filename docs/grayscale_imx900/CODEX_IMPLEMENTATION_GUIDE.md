# Codex Implementation Guide — IMX900 Grayscale Line

> This file is the implementation contract for future local Codex work.
>
> Do not restore the old D455 path. Do not replace the camera model with a large
> third-party dependency without revisiting the design decision documented in
> OPEN_SOURCE_CAMERA_MODEL_REVIEW.md.

## Phase 0 — legacy separation ✅ COMPLETE

- D455 executable sensor path removed from this branch.
- old paper/results archive removed from this branch.
- old work remains available from history/legacy branch.

## Phase 1 — geometry / appearance renderer ✅ IMPLEMENTED

Current contract:

`quadsim_cuda.render_geometry()`

returns:

- internal ray-hit distance;
- exact hit normal.

These values are used only for appearance rendering.

Never expose them to the policy as privileged depth observations.

`render/ideal_gray.py` currently applies:

- Lambertian illumination;
- ambient light;
- procedural texture;
- nominal/dark/bright/transitions.

Do not replace this with a heavy renderer before the core experiment is
validated.

## Phase 2 — calibration-driven IMX900 surrogate ✅ IMPLEMENTED

The old generic camera-semantic split has been replaced by:

~~~text
configs/calibration/imx900_provisional.json
              |
              v
sensors/imx900_calibration.py
              |
              v
sensors/imx900_camera.py
~~~

### Required design rules

1. Physical sensor coefficients belong in the JSON calibration profile.
2. Experiment configs should not duplicate exposure/gain/noise coefficients.
3. The repository provisional profile must remain `calibrated: false`.
4. Real/sim-to-real runs should use `--require_calibrated_imx900`.
5. Do not copy coefficients from End2endImaging, JOCA/Basler, or unrelated
   camera datasheets into the measured IMX900 profile.

### Camera model components

Current surrogate:

- exposure integration;
- gain mapping: linear/log/LUT;
- shot noise alpha/beta;
- gain-dependent read noise;
- black level;
- saturation/full-scale normalization;
- STE quantization;
- exposure-dependent motion blur.

### Required tests

- exposure finite-difference gradient;
- gain finite-difference gradient;
- LUT gain mapping gradient;
- fixed stochastic samples -> deterministic output;
- long exposure + motion -> increased blur;
- saturation gradient decays;
- JSON profile load/build;
- provisional-profile rejection when required.

## Phase 3 — policy integration ✅ IMPLEMENTED / RESULT PENDING

Visual input:

[
[I_t,I_{t-1}].
]

Camera action/state:

[
[E_t,G_t].
]

The camera branch may observe current exposure/gain.

For the main causal experiment, the flight branch must **not** directly receive
camera state.

The camera can influence flight only through resulting pixels.

## Phase 4 — fixed grayscale navigation 🟡 RESULT PENDING

Before active camera learning:

1. fixed nominal camera;
2. matched blind/zero-image control.

Pass condition:

- grayscale navigation materially outperforms the blind baseline.

If not, fix appearance/navigation before touching camera learning.

## Phase 5 — stress environment 🟡 CODE PRESENT / RESULT PENDING

Use:

- dark;
- bright;
- dark-to-bright;
- bright-to-dark;
- fast/near gate approach.

Verify:

- long exposure helps low-light signal;
- long exposure hurts fast image motion;
- gain improves signal while increasing measured-model noise;
- no one fixed setting dominates all conditions.

If this trade-off is absent, do not train the learned camera policy yet.

## Phase 6 — classical baselines 🟡 CODE PRESENT / VALIDATION PENDING

Required:

- fixed nominal;
- random-static;
- mean AE;
- gradient AE.

Keep actuator/update constraints matched.

## Phase 7 — primary camera experiment 🟡 CODE PRESENT / RESULT PENDING

Start from one successful frozen flight checkpoint.

Run:

### learned-detached

Cut:

[
(E,G)ightarrow I
]

gradient at the sensor boundary.

### learned-differentiable

Preserve:

[
L_{nav}
ightarrow I
ightarrow(E,G)
ightarrow	heta_{cam}.
]

Everything else must match:

- flight checkpoint;
- camera network;
- optimizer;
- data distribution;
- random seeds where practical;
- training budget.

Do not add search-based teacher targets to either condition.

## Phase 8 — optional JOCA-style comparison ⬜ OPTIONAL

Only after the primary full-vs-detached result is understood.

Possible comparison:

- local exposure/gain grid search;
- derivative-free perturbation;
- JOCA-style correction target.

Important:

- implement independently;
- cite JOCA;
- do not vendor JOCA code without license clarification;
- report as a separate method.

## Phase 9 — real IMX900 characterization ⬜ HARDWARE PENDING

Create/complete tools:

~~~text
tools/grayscale_calibration/
  inspect_controls.py
  capture_dark.py
  sweep_exposure.py
  sweep_gain.py
  capture_ptc.py
  measure_motion_blur.py
  measure_latency.py
  fit_imx900_profile.py
  validate_imx900_profile.py
  report.py
~~~

Output one measured JSON profile.

Do not edit model source to insert measured coefficients.

## Phase 10 — actuator model 🟡 DELAY STRUCTURE IMPLEMENTED / HARDWARE VALUES PENDING

Current code now separates:

- `camera_smoothing_alpha`: optional policy-command smoothing, 0.0 in the
  main experiment configs;
- `command_delay_frames`: calibration-driven effective-command delay.

The rollout uses a command queue for the measured frame delay. The provisional
profile sets this to zero.

After hardware characterization:

- replace the zero delay with the measured value;
- set real exposure/gain command steps;
- add measured jitter if material;
- preserve requested/effective settings and timestamps in the real camera node.

Do not reinterpret policy smoothing as hardware latency.

## Phase 11 — optional optics refinement ⬜ ONLY IF NEEDED

Do not add DeepLens by default.

Trigger this phase only if held-out real data demonstrates a material
lens-driven residual:

- PSF blur;
- distortion;
- vignetting;
- defocus.

Preferred workflow:

~~~text
real lens data / lens spec
      |
      v
DeepLens offline study
      |
      v
small fitted PSF/distortion/vignetting surrogate
      |
      v
DiffPhysDrone runtime
~~~

Do not run a full optical simulator per BPTT frame unless a specific experiment
requires it.

## Phase 12 — real camera wrapper ⬜ HARDWARE PENDING

Target:

`tools/realflight/imx900_camera_node.py`

Responsibilities:

- camera stream via supported e-con/V4L2/GStreamer path;
- disable auto controls;
- manual exposure/gain;
- frame timestamp;
- requested camera settings;
- effective settings if metadata exposes them;
- command timestamp;
- dropped/stale frame statistics;
- safe nominal fallback.

## Phase 13 — real flight ⬜ PENDING

Order:

1. bench stream;
2. manual parameter control;
3. calibration;
4. surrogate fit;
5. held-out bench validation;
6. mounted motors-off;
7. hover;
8. slow gate;
9. illumination transition;
10. final benchmark.

## Open-source dependency rules

### End2endImaging

Allowed:

- study modeling structure;
- cite project/papers;
- reimplement necessary compact equations;
- Apache-2.0 permits code reuse if license obligations are followed.

Current project decision:

- no hard runtime dependency;
- do not copy numerical defaults as IMX900 values.

### DeepLens

Allowed and Apache-2.0.

Current project decision:

- optional offline optics tool only.

### JOCA

Current review did not find a root LICENSE file.

Therefore:

- do not copy/vendor its source into this repository;
- use as prior work and algorithmic reference;
- independently implement any comparison;
- clarify licensing before deeper code reuse.

## Coding principles

1. One physical source of truth: the calibration JSON.
2. Keep calibrated quantities separate from training-surrogate hyperparameters.
3. No privileged internal geometry in the policy.
4. No camera-state shortcut into the flight branch for the main causal test.
5. Use fixed noise tensors for gradient unit tests.
6. Every learned-camera claim must include the matched detached control.
7. Do not use image-quality supervision silently.
8. Do not use JOCA-style search correction silently.
9. Do not call the provisional profile “IMX900 calibrated.”
10. Re-run the literature search before paper submission.

## Current handoff

The next useful work is **not** another camera architecture rewrite.

Before real hardware arrives:

- keep the calibration-profile architecture stable;
- validate fixed grayscale navigation;
- validate the illumination/motion conflict;
- validate full-vs-detached behavior.

When hardware arrives:

- characterize the exact e-con/IMX900/Jetson pipeline;
- fit a new JSON profile;
- enable `--require_calibrated_imx900`;
- then evaluate sim-to-real.

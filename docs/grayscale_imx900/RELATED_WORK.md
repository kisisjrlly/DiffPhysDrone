# Related Work and Open-Source Survey

> Last updated: 2026-09-22.
>
> This file defines the scientific positioning of the grayscale/IMX900 line.
> Detailed code-level adoption decisions are in
> [OPEN_SOURCE_CAMERA_MODEL_REVIEW.md](OPEN_SOURCE_CAMERA_MODEL_REVIEW.md).

## 1. End2endImaging — sensor-physics reference

Repository:

https://github.com/vccimaging/End2endImaging

Relevant implementation:

`end2end_imaging/sensor/mono_sensor.py`

Observed repository license:

Apache-2.0.

The public `MonoSensor` provides a useful organization for:

- monochrome response;
- bit depth;
- black level;
- read noise;
- shot noise with alpha/beta parameters;
- ISO-dependent noise scaling;
- optional spectral response;
- ISP blocks.

Its code is **not** used unchanged here because the current public
implementation:

- does not expose the exact online exposure-time actuator required by this
  project;
- explicitly fixes analog gain to 1.0 in the inspected noise code;
- uses hard rounding/clipping in realistic forward simulation;
- contains numerical defaults that are not IMX900 measurements.

DiffPhysDrone therefore adopts the structural idea—sensor-specific
configuration + compact noise model—but fits its own IMX900 profile.

## 2. DeepLens — optional optics fidelity

Repository:

https://github.com/vccimaging/DeepLens

Observed repository license:

Apache-2.0.

At the time of review, its package metadata described `deeplens-core` 2.5.4,
Python >=3.12,<3.13, and PyTorch 2.10.0.

DeepLens is relevant for:

- geometric differentiable optics;
- PSF/MTF;
- distortion;
- spatially varying aberrations;
- depth-dependent defocus;
- neural PSF surrogates;
- end-to-end optics/algorithm co-design.

The fixed lens is not an online control variable in the v1 UAV experiment.
Therefore DeepLens is deliberately excluded from every-frame BPTT.

If real-camera validation later identifies lens blur/distortion/vignetting as a
dominant sim-to-real error, DeepLens can be used offline to generate/fix a
lightweight lens surrogate.

## 3. JOCA — closest task-driven camera-control prior work

Repository:

https://github.com/RoboticImaging/JOCA

Paper:

Task-Driven Joint Optimisation of Camera Hardware and Adaptive Camera Control
Algorithms, WACV 2026.

The inspected CARLA implementation directly supports the relevance of JOCA to
this project:

- a neural controller predicts normalized exposure/gain;
- normalized values are mapped to physical camera ranges;
- images are rescaled according to exposure/gain ratios;
- gain-dependent shot/read noise is added;
- motion blur is coupled to exposure through CARLA camera settings;
- previous camera parameters are fed back into the adaptive controller.

The source comments identify the noise coefficients as Basler DaA1280 camera
characteristics. This is important evidence for our own design principle:
camera-model coefficients should come from a specific real camera.

The public joint-training code also implements genetic-algorithm perturbation
correction around predicted camera settings. JOCA uses this to handle
non-differentiable imaging effects.

### Consequence for this project

We must not claim:

- first task-driven exposure control;
- first differentiable task-driven camera setting optimization;
- first learned adaptive exposure/gain controller.

The main contribution must instead be grounded in the closed-loop aerial
navigation setting, real IMX900 calibration, onboard deployment, and a matched
full-vs-detached sensor-gradient experiment.

### Why JOCA-style search is not the main method

The central scientific question here is whether the sensor gradient itself
improves navigation. A search/teacher correction inside the main training loop
would confound that test.

JOCA-style derivative-free correction is therefore reserved for an optional
separate baseline/fallback.

### Licensing note

A root `LICENSE` file was not found through the GitHub repository contents
API during the review. Do not vendor/copy JOCA implementation code into this
repository unless licensing is clarified. Reimplement experimental baselines
from the paper/observable algorithmic description and cite the source.

## 4. TaCOS and task-specific camera co-design

TaCOS, WACV 2025, is an important precursor on simulation-based task-specific
camera optimization.

It reinforces the conclusion that generic camera/task co-design is not a novel
claim by itself. It belongs in the final related-work chain alongside JOCA.

## 5. Robotics exposure-control baselines

### Active Exposure Control for Robust Visual Odometry in HDR Environments

Project:

https://github.com/uzh-rpg/active_camera_exposure_control

Relevant as a classical robotics baseline for:

- fixed exposure;
- mean/brightness-driven control;
- gradient-based exposure selection;
- HDR/VO-motivated camera control.

This supports the baseline distinction:

[
	ext{image-quality-driven AE}

eq
	ext{navigation-task-driven camera control}.
]

### Noise-Aware Camera Exposure Control

Project:

https://github.com/UkcheolShin/Noise-AwareCameraExposureControl

Relevant for:

- joint exposure/gain control;
- noise-aware image-quality reasoning;
- real-camera parameter sweeps;
- calibration-oriented evaluation.

### RL exposure control

Project:

https://github.com/shuyanguni/drl_exposure_ctrl

Relevant because camera settings can also be learned without analytical
sensor gradients. A learned controller alone is not sufficient evidence for
our claim.

## 6. Camera calibration / sensor characterization

Important concepts for the real IMX900 work:

- photon transfer curve;
- temporal dark noise;
- black level;
- shot noise;
- read noise;
- gain response;
- full scale/saturation;
- raw bit depth;
- parameter step size;
- command-to-effective-frame latency;
- motion blur / edge spread;
- EMVA 1288 terminology.

The project should describe its real-camera work as **task-relevant sensor
characterization and differentiable surrogate fitting**, not as a transistor-
level CMOS simulation.

## 7. Current novelty statement

Internal working statement:

> We study real-camera-calibrated differentiable exposure/gain control for
> closed-loop monocular quadrotor navigation, isolate the value of the sensor
> gradient through a matched full-vs-detached experiment, and target onboard
> deployment with an e-con/Sony IMX900 monochrome global-shutter camera.

This is not a final novelty claim. Re-run the literature search immediately
before manuscript submission.

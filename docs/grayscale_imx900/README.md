# Grayscale Active Sensing / IMX900

> Branch: `active-sensing-grayscale-imx900`
>
> Status: grayscale-only executable path; IMX900 calibration-driven camera
> surrogate implemented; real camera parameters still unmeasured.

## 1. Branch scope

This branch studies:

> real-camera-calibrated differentiable exposure/gain control for closed-loop
> monocular grayscale quadrotor navigation.

Selected camera/platform:

- e-con Systems e-CAM37M_CUONX;
- Sony IMX900 monochrome global shutter;
- MIPI CSI-2;
- Jetson Orin NX 8GB;
- DAMIAO DM-ORIN NX V2.X.

The old D455 work is not part of this branch.

## 2. Open-source adoption decision

The 2026-09-22 code-level review concluded:

- **End2endImaging:** adopt its sensor-physics/configuration structure as a
  reference, not its numerical defaults or current `MonoSensor` unchanged.
- **DeepLens:** keep out of the v1 runtime; use later only if real data shows
  lens PSF/distortion/vignetting is a major sim-to-real gap.
- **JOCA:** treat as closest conceptual prior work and an experiment-design
  reference; do not use its GA/DF-Grad correction in the main causal method.
  At review time no root LICENSE file was found, so do not vendor/copy its code
  without separate licensing clarification.

Detailed evidence and the resulting architecture are in
[OPEN_SOURCE_CAMERA_MODEL_REVIEW.md](OPEN_SOURCE_CAMERA_MODEL_REVIEW.md).

## 3. Current camera implementation

The old generic `DifferentiableGrayCamera` / `GrayCameraSemantics`
split has been replaced by:

- `sensors/imx900_calibration.py` — sensor-specific physical/calibration
  parameters and normalized-action mappings;
- `sensors/imx900_camera.py` — lightweight differentiable IMX900 surrogate;
- `configs/calibration/imx900_provisional.json` — unmeasured development
  profile.

The runtime model implements:

- exposure integration;
- calibrated exposure mapping, including optional piecewise LUT;
- calibrated gain mapping, including optional piecewise LUT;
- End2endImaging-inspired shot/read-noise decomposition;
- power-law or measured-LUT read noise;
- black level;
- saturation/full-scale normalization;
- optional fixed response LUT for unavoidable ISP behavior;
- optional quantization with STE;
- exposure-dependent motion blur;
- calibration-driven command delay and delay jitter.

The numerical coefficients are loaded from the profile, not duplicated across
experiment configs.

## 4. Current pipeline

~~~text
geometry
  -> CUDA hit distance + exact normal (internal)
  -> ideal grayscale irradiance
  -> IMX900DifferentiableCamera
       <- IMX900Calibration JSON
  -> [current_gray, previous_gray]
       -> recurrent flight policy
       -> recurrent exposure/gain policy
~~~

The flight branch must not directly receive exposure/gain in the main
full-vs-detached experiment.

## 5. Calibration boundary

The profile fields that must eventually come from the real e-con/IMX900 stack
include:

- exposure range/step and action->microseconds mapping/LUT;
- exposure response scale;
- gain mapping/LUT;
- shot noise alpha/beta;
- read noise vs gain (power fit or measured LUT);
- black level;
- saturation/full scale;
- RAW bit depth;
- fixed response curve if the deployed path is unavoidably nonlinear;
- motion-blur coefficient;
- command-to-effective-frame latency and jitter.

The following remain surrogate/training choices:

- blur kernel shape/size in v1;
- smooth-saturation beta;
- characteristic-depth floor used by the motion proxy;
- network resolution;
- domain randomization.

## 6. Causal experiment

Main comparison:

~~~text
same frozen flight checkpoint
same camera network
same environment distribution
same optimization budget

learned-detached
      vs
learned-differentiable
~~~

The only intended difference is whether navigation loss can traverse
exposure/gain -> image formation.

Do not add JOCA-style search/teacher correction to the main method. It can be a
separate later baseline.

## 7. Documents

1. [OPEN_SOURCE_CAMERA_MODEL_REVIEW.md](OPEN_SOURCE_CAMERA_MODEL_REVIEW.md)
2. [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md)
3. [RELATED_WORK.md](RELATED_WORK.md)
4. [HARDWARE_IMX900.md](HARDWARE_IMX900.md)
5. [CALIBRATION_SIM2REAL.md](CALIBRATION_SIM2REAL.md)
6. [CODEX_IMPLEMENTATION_GUIDE.md](CODEX_IMPLEMENTATION_GUIDE.md)

## 8. Current implementation gate

The architecture is now settled enough that the next steps should be
validation/calibration work, not another wholesale camera-model rewrite.

The provisional JSON profile is for software development only. Before any
real-camera claim, replace it with a profile generated from measured IMX900
data and use `--require_calibrated_imx900`.

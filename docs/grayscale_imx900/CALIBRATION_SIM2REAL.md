# Calibration and Sim-to-Real Protocol — e-con IMX900

## 1. Goal

Fit a compact differentiable surrogate that reproduces the **task-relevant**
response of the real e-con e-CAM37M_CUONX / Sony IMX900 camera stack.

The target is not transistor-level simulation.

The output of calibration is a JSON profile consumed by:

- `sensors/imx900_calibration.py`;
- `sensors/imx900_camera.py`.

Development profile:

`configs/calibration/imx900_provisional.json`

It is marked `calibrated: false`.

A real fitted profile should be saved separately, for example:

`configs/calibration/imx900_measured_<date>.json`

and marked:

~~~json
"calibrated": true
~~~

For runs that must reject placeholders, use:

~~~text
--require_calibrated_imx900
~~~

---

## 2. Freeze the real capture pipeline first

Calibrate the exact stack that will be flown:

[
	ext{IMX900 sensor}
+
	ext{e-con module}
+
	ext{lens}
+
	ext{e-con driver/BSP}
+
	ext{Jetson capture path}.
]

Before data collection, record:

- exact e-con SKU;
- lens SKU/focal length;
- JetPack/L4T;
- e-con driver/BSP version;
- pixel format;
- resolution;
- FPS;
- exposure control name/range/step;
- gain control name/range/step;
- whether effective settings are reported per frame.

Disable or freeze, where possible:

- auto exposure;
- auto gain;
- auto brightness;
- digital gain that is not part of the intended action;
- denoise;
- sharpening/edge enhancement;
- HDR/alternate shutter modes;
- nonlinear gamma/ISP processing.

Prefer RAW10/RAW12 or the most linear monochrome path available.

If hidden ISP processing cannot be disabled, characterize the **actual deployed
pipeline** rather than pretending it is RAW.

---

## 3. Calibration profile schema

The current JSON schema separates physical quantities from training
hyperparameters.

### Exposure

~~~json
"exposure": {
  "min_us": ...,
  "max_us": ...,
  "reference_us": ...,
  "step_us": ...,
  "signal_scale": ...
}
~~~

Fit/record:

- driver-supported min/max;
- real command step;
- a convenient reference exposure;
- signal scale connecting normalized simulator irradiance to observed camera
  response.

### Gain

~~~json
"gain": {
  "mapping": "linear | log | lut",
  "min_factor": ...,
  "max_factor": ...,
  "step_factor": ...,
  "lut_x": ...,
  "lut_factor": ...
}
~~~

Prefer a measured LUT if the command-to-amplification curve is not accurately
represented by a simple law.

`lut_x` is the normalized policy command in [0,1].

### Noise

~~~json
"noise": {
  "shot_alpha": ...,
  "shot_beta": ...,
  "read_std_base": ...,
  "read_gain_exponent": ...
}
~~~

The runtime surrogate currently uses:

[
sigma_{shot}
=
G(alphasqrt{q}+eta),
]

[
sigma_{read}
=
sigma_{r0}G^{p_r}.
]

If real data strongly reject this compact form, extend the calibration profile
with a measured curve/LUT rather than adding scene-dependent heuristics.

### Other sensor fields

~~~json
"black_level": ...,
"saturation_level": ...,
"quantization_bits": ...,
"motion_blur": {"scale": ...},
"actuator": {"command_delay_frames": ...}
~~~

---

## 4. Dataset format

Create a machine-readable calibration dataset under:

`tools/grayscale_calibration/`

or an external data directory referenced by a manifest.

Every frame/sequence should record:

~~~text
timestamp
frame_id
scene_id
illumination_id
requested_exposure
requested_gain
effective_exposure
effective_gain
fps
pixel_format
camera_temperature (if available)
image_path
notes
~~~

Do not keep calibration facts only inside notebooks.

---

## 5. Dark-frame calibration

Condition:

- lens cap or light-tight enclosure;
- auto controls disabled;
- stable temperature where practical.

Sweep the intended gain range.

For each setting collect enough frames to estimate temporal statistics, e.g.
100–500 frames.

Estimate:

- black-level mean;
- temporal dark/read noise;
- hot pixels;
- variance versus gain.

Fit:

- `black_level`;
- `read_std_base`;
- `read_gain_exponent` or a future measured LUT.

Do not copy noise values from End2endImaging or JOCA/Basler into the measured
profile.

---

## 6. Exposure response

Use a stable, non-flickering light source and a matte uniform target.

At fixed gain, sweep exposure over the intended flight range.

Measure:

- ROI mean;
- ROI variance;
- saturation fraction.

Identify:

- black-floor region;
- linear region;
- saturation knee;
- full-scale output.

Fit:

- exposure min/max/step;
- `reference_us`;
- `signal_scale`;
- `saturation_level`.

A development camera model may remain normalized, but the fitted response
should reproduce the real curve over the operating range.

---

## 7. Gain response

At several exposure and illumination levels, sweep the real gain control.

Determine whether the real control is best represented by:

- linear factor;
- logarithmic factor;
- measured LUT.

The project already supports a piecewise differentiable gain LUT.

Fit:

- `gain.mapping`;
- factor range;
- optional command step;
- LUT points.

Do not assume that a driver value or dB label is directly a linear amplitude
factor.

---

## 8. Photon-transfer-style noise characterization

For multiple uniform illumination levels, measure:

[
(mu,sigma^2)
]

after subtracting the dark/black component.

Repeat at several gains.

Use this data to fit the compact surrogate:

[
sigma_{shot}
=
G(alphasqrt{q}+eta),
]

[
sigma_{read}
=
sigma_{r0}G^{p_r}.
]

Validate the fit on held-out illumination/gain settings.

The objective is not a perfect CMOS physics model; it is a compact surrogate
whose signal/noise trends match the real sensor in the task operating range.

---

## 9. Bit depth, black level, saturation, and response curve

Record:

- RAW10 / RAW12 / other format;
- digital maximum;
- black level;
- clipping behavior;
- gamma/nonlinearity if present;
- any unavoidable ISP transformation.

Set:

- `quantization_bits`;
- `black_level`;
- `saturation_level`.

If a nonlinear response cannot be disabled and is material, extend the
calibration profile with a fitted response curve.

---

## 10. Motion blur calibration

The current simulator uses a lightweight blur coefficient, not a full temporal
optics model.

Use a high-contrast edge/texture target.

Collect data over:

- multiple exposures;
- multiple known translational/angular speeds;
- multiple target distances.

Measure edge-spread or another blur-width statistic.

Fit the v1 relationship around:

[
mapproxrac{|v|}{Z},
]

[
b
=
1-exp
left(
-k_brac{T}{T_{ref}}m
ight).
]

Store the fitted (k_b) as:

`motion_blur.scale`.

If the residual error is too large, the next upgrade is temporal multi-pose
integration, not arbitrary scene-specific blur rules.

---

## 11. Command-to-effective-frame latency

Active sensing depends on **when** a requested setting actually becomes visible.

Procedure:

1. stream at fixed FPS;
2. alternate two strongly different exposure values;
3. timestamp every command;
4. detect the first frame whose intensity reflects the new exposure;
5. repeat many times;
6. repeat for gain.

Measure:

- median delay;
- 95th percentile;
- jitter;
- frame-count delay;
- whether metadata reports effective settings.

Store the nominal frame delay as:

`actuator.command_delay_frames`.

The current profile stores this value even though the full measured actuator
queue should only be enabled once hardware behavior is known.

---

## 12. Camera intrinsics and mounting

In addition to sensor-response calibration, measure:

- (f_x,f_y,c_x,c_y);
- distortion;
- actual FoV;
- camera-to-body extrinsics.

The current simulator `fov_x_half_tan` and `cam_angle` are development
parameters, not a final real-camera calibration.

If lens distortion/PSF/vignetting becomes a major residual error, that is the
decision point for an offline DeepLens study.

---

## 13. Fitting output

The fitting tool should write a complete JSON profile rather than editing source
code.

Expected future tool:

`tools/grayscale_calibration/fit_imx900_profile.py`

Inputs:

- exposure sweep;
- gain sweep;
- dark frames;
- PTC data;
- blur sequences;
- latency data.

Output:

- fitted JSON profile;
- fit metrics;
- validation plots;
- held-out error summary.

---

## 14. Required validation plots

At minimum:

1. mean intensity vs exposure — real vs surrogate;
2. mean intensity vs gain — real vs surrogate;
3. variance vs mean at several gains;
4. read-noise estimate vs gain;
5. saturation fraction vs exposure;
6. blur width vs exposure at several speeds/distances;
7. command-to-effective-frame latency histogram;
8. representative real/sim images in dark/nominal/bright cases.

---

## 15. Acceptance criteria

Do not require pixel-perfect images.

Require the surrogate to reproduce task-relevant trends:

- correct exposure-response direction and scale;
- correct gain-response trend;
- similar saturation onset;
- realistic noise order/magnitude;
- realistic blur trend;
- modeled command latency;
- acceptable held-out response curves.

Document mismatches explicitly.

---

## 16. Sim-to-real randomization

After fitting a nominal profile, randomize around **measured uncertainty**:

- light level;
- material albedo/texture;
- shot/read coefficients;
- response scale;
- blur coefficient;
- command latency/jitter;
- small gain/exposure mapping perturbations.

Do not use excessively broad randomization to hide an uncalibrated model.

---

## 17. Real-flight validation order

1. bench camera stream;
2. manual exposure/gain command;
3. calibration dataset;
4. surrogate fit;
5. held-out bench validation;
6. hand-held motion;
7. camera mounted, motors off;
8. hover;
9. low-speed gate approach;
10. illumination transition;
11. full benchmark.

Use normal PX4/manual safety procedures throughout.

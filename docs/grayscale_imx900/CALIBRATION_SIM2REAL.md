# Calibration and Sim-to-Real Protocol

## 1. Goal

Fit a compact differentiable image-formation model that reproduces the **task-relevant response** of the real IMX900 camera to exposure, gain, illumination and motion.

Do not attempt transistor-level camera simulation.

Required agreement:

- monotonic intensity response;
- saturation onset;
- noise versus signal/gain;
- black level;
- motion blur versus exposure/motion;
- parameter quantization/ranges;
- command-to-effective-frame latency.

## 2. Data format

Create:

`tools/grayscale_calibration/`

and store each capture with a machine-readable manifest.

Recommended fields:

~~~text
timestamp
scene_id
illumination_id
distance
target_type
requested_exposure
requested_gain
effective_exposure
effective_gain
frame_id
image_path
camera_temperature (if available)
fps
notes
~~~

Never keep calibration knowledge only in notebook cells.

## 3. Dark-frame calibration

Condition:

- lens cap / fully dark enclosure;
- fixed camera temperature as far as practical;
- auto exposure/gain disabled.

Sweep gain over its usable range.

For each setting capture at least ~100 frames initially.

Estimate:

- black-level mean;
- temporal dark/read noise;
- hot-pixel statistics;
- gain dependence of variance.

Fit a smooth low-parameter model \(\sigma_r(G)\).

## 4. Exposure response

Use:

- stable LED illumination;
- matte gray/white target;
- no flickering PWM source unless synchronized.

At fixed gain, sweep exposure over the intended navigation range.

Measure central ROI mean/variance.

Fit:

\[
\mu(T)\approx f_T(T,L)
\]

and identify:

- linear region;
- black-floor region;
- saturation knee;
- hard saturation.

## 5. Gain response

At several fixed exposures and illumination levels, sweep gain.

Fit the mapping from camera control value to effective amplification.

Do not assume the command is linear amplitude gain. If exposed in dB, fit/convert explicitly.

## 6. Photon-transfer-style characterization

For uniform illumination levels, estimate pairs:

\[
(\mu,\sigma^2).
\]

Use these data to fit the shot/read-noise model used in the differentiable camera.

A simple model is acceptable if validated:

\[
\sigma^2=a(G)\mu+b(G).
\]

## 7. Saturation and quantization

Measure:

- digital maximum;
- bit depth/output format;
- black level;
- clipping behavior;
- any gamma/ISP processing that cannot be disabled.

Prefer raw/linear monochrome modes when supported.

If the real pipeline applies hidden nonlinear processing, either disable it or fit an explicit response curve.

## 8. Motion blur calibration

Use a high-contrast target.

Option A:
- rotate the camera at known angular velocity.

Option B:
- move a target at known image-plane speed.

Sweep exposure.

Measure edge-spread/blur width.

Fit a model:

\[
w_{blur}=f(T,\omega,v,Z).
\]

The initial simulator may use a simplified coefficient, but its scale must come from measurement.

## 9. Parameter-change latency

This is critical for active sensing.

Procedure:

1. stream at fixed FPS;
2. alternate two strongly different exposure values;
3. timestamp each command;
4. detect in captured frames when brightness changes;
5. estimate frame-delay distribution.

Repeat for gain.

Record:

- median delay;
- 95th percentile;
- jitter;
- whether the driver reports effective settings per frame.

Use these measurements in simulation.

## 10. Illumination transition benchmark

Create a task-relevant bench sequence:

- bright -> dark;
- dark -> bright;
- bright source near gate;
- low-light textured gate;
- fast camera motion under low light.

These sequences are useful before flight because camera-control quality can be diagnosed without risking the drone.

## 11. Simulator fitting

Fit parameters using train/validation scenes separately.

The simulator should predict distributions/statistics, not exact random pixel noise.

Suggested fitted parameters:

~~~text
black_level
exposure_scale
gain_mapping parameters
read_noise(gain)
shot_noise coefficient(s)
saturation shoulder
quantization mode
blur coefficient(s)
vignetting (optional)
latency/hold model
~~~

## 12. Validation plots

Required report plots:

1. mean intensity vs exposure: real vs sim;
2. mean intensity vs gain: real vs sim;
3. variance vs mean at several gains;
4. saturation fraction vs exposure;
5. blur width vs exposure at multiple speeds;
6. command-to-effective-frame latency histogram;
7. example real/sim frames for dark/nominal/bright conditions.

## 13. Acceptance criteria before flight training claims

Do not require pixel-perfect matching.

Require:

- correct response direction across operating range;
- similar saturation transition;
- noise magnitude/order compatible with real data;
- blur trend compatible with real measurements;
- latency modeled;
- held-out calibration scenes not grossly mismatched.

Document deviations rather than hiding them.

## 14. Sim-to-real randomization

After fitting a nominal camera model, randomize around calibration uncertainty:

- illumination;
- albedo/texture;
- noise coefficients;
- response scale;
- blur coefficient;
- latency by ± measured jitter;
- small exposure/gain mapping perturbations.

Do not randomize so broadly that the calibrated model becomes irrelevant.

## 15. Real-flight validation order

1. bench camera control;
2. hand-held camera motion;
3. mounted drone, motors off;
4. armed hover;
5. low-speed straight flight;
6. low-speed gate approach;
7. controlled illumination transition;
8. full benchmark.

At every stage retain a manual emergency-stop / PX4 safety procedure.

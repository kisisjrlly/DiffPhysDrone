# Technical Plan — IMX900-Calibrated Differentiable Camera Control for UAV Navigation

## 0. Research objective

This branch studies whether a closed-loop navigation objective can directly use
camera-model gradients to choose useful exposure and gain online.

The target system is:

[
u_t = pi_f(I_t, s_t),
qquad
c_{t+1} = pi_c(I_t, s_t, c_t),
qquad
c_t = [E_t, G_t].
]

The image is produced by

[
I_t =
mathcal C_{IMX900}
left(
I_t^{ideal},
E_t,
G_t,
m_t;
Theta_{calib}
ight).
]

The main gradient of interest is

[
rac{partial L_{nav}}{partial 	heta_c}
=
rac{partial L_{nav}}{partial I_t}
rac{partial I_t}{partial (E_t,G_t)}
rac{partial (E_t,G_t)}{partial 	heta_c}.
]

Geometry does not need to be differentiable for the primary claim.

---

## 1. Design decision after open-source review

The runtime implementation remains project-owned and lightweight.

### End2endImaging

Use as the reference for:

- sensor-specific configuration;
- shot/read noise decomposition;
- black level;
- bit depth;
- sensor/ISP modularization.

Do not import its `MonoSensor` unchanged because it does not provide the
required online exposure actuator, its inspected noise path fixes analog gain
to 1.0, and its hard round/clip behavior is not sufficient for our camera-
action gradients.

### JOCA

Use as the reference for:

- normalized exposure/gain camera actions;
- task-driven adaptive camera control;
- camera-specific noise calibration;
- motion-blur/exposure interaction;
- camera-state feedback to the camera controller;
- derivative-free correction as an optional comparison.

Do not use JOCA-style search correction in the main method because that would
confound the full-vs-detached sensor-gradient experiment.

### DeepLens

Do not put full differentiable optics in the v1 runtime loop.

Use DeepLens later only if real-camera validation shows that lens PSF,
distortion, vignetting, or defocus materially dominate the sim-to-real gap.

Detailed evidence:
[OPEN_SOURCE_CAMERA_MODEL_REVIEW.md](OPEN_SOURCE_CAMERA_MODEL_REVIEW.md).

---

## 2. Software architecture

### 2.1 Geometry stage

CUDA returns:

- ray-hit distance;
- exact surface normal.

These values are internal appearance-rendering quantities.

They must never become policy observations.

### 2.2 Ideal grayscale appearance

`render/ideal_gray.py` computes a simple appearance model:

[
I^{ideal}(x)
=
ho(x)
left[
L_a + L_dmax(0,n(x)^Tl)
ight].
]

Version 1 intentionally uses:

- Lambertian diffuse lighting;
- ambient light;
- procedural texture;
- nominal/dark/bright illumination;
- bright-to-dark and dark-to-bright transitions.

The research variable is active camera control, not photorealistic rendering.

### 2.3 IMX900 calibration profile

All physical camera coefficients belong in one file, currently:

`configs/calibration/imx900_provisional.json`.

The profile is loaded by:

`sensors/imx900_calibration.py`.

The provisional profile is explicitly:

~~~json
"calibrated": false
~~~

and is only a development placeholder.

A future fitted profile must contain the measured e-con/IMX900 camera behavior.

### 2.4 IMX900 differentiable surrogate

The runtime sensor model is:

`sensors/imx900_camera.py::IMX900DifferentiableCamera`.

It is not claimed to be a firmware-exact or transistor-level digital twin.

Its role is to reproduce the task-relevant response of the real camera while
remaining useful for gradients with respect to exposure/gain.

---

## 3. Camera model

### 3.1 Exposure mapping

The camera policy produces normalized exposure (e\in[0,1]).

The calibration profile maps it to physical exposure time:

[
T=T(e).
]

Supported calibration mappings:

- `linear`: development/simple driver mapping;
- `lut`: measured normalized-action -> effective microseconds.

The measured LUT is preferred whenever the deployed driver/camera command path
is not accurately represented by a linear mapping.

Real command quantization is represented by `step_us`; training uses an STE
for the forward snap so the exposure action retains a usable gradient.

### 3.2 Gain mapping

The policy produces

[
gin[0,1].
]

The calibration layer supports:

- linear mapping;
- logarithmic mapping;
- piecewise measured LUT.

The LUT option is preferred if real driver sweeps show that camera control units
do not map cleanly to a simple analytical gain curve.

### 3.3 Exposure integration / signal response

The current surrogate computes:

[
q
=
s_E I_{ideal}
rac{T}{T_{ref}},
]

where (s_E) is a fitted response scale that connects normalized renderer
irradiance to the sensor-model signal scale.

Then:

[
S=Gq.
]

### 3.4 Shot noise

Following the modeling structure used in End2endImaging, the shot-noise term is
parameterized by alpha/beta:

[
sigma_{shot}
=
G
left(
alphasqrt{q+epsilon}
+
eta
ight).
]

The coefficients must be fitted from IMX900 data.

### 3.5 Read noise

Two calibration modes are supported.

Compact power-law surrogate:

[
sigma_{read}
=
sigma_{r0}G^{p_r}.
]

Measured LUT:

[
sigma_{read}
=
operatorname{interp}
left(G;{G_i,sigma_i}ight).
]

The power law is a low-parameter development model. The measured LUT is
preferred when dark-frame characterization shows systematic gain-dependent
structure that the power law cannot fit.

### 3.6 Reparameterized stochastic sensor

Training uses:

[
S_n
=
S
+
sigma_{shot}epsilon_s
+
sigma_{read}epsilon_r,
]

with fixed noise tensors available for deterministic gradient tests.

This preserves a usable gradient path while retaining stochastic sensor
variation.

### 3.7 Black level and full scale

Before normalized output:

[
x=S_n+B.
]

Then:

[
x_n=rac{x}{S_{max}},
]

where (B) and (S_{max}) come from calibration.

### 3.8 Saturation

Forward realism and training gradients have different requirements.

Supported modes:

- `hard`: evaluation/debug;
- `ste`: hard forward with identity-style backward;
- `soft`: smooth shoulder near saturation.

The main training configuration currently prefers a smooth saturation shoulder
so that strongly exposed images do not pretend to retain fully linear
gradients.

### 3.9 Fixed response curve

The preferred deployed path is RAW/linear monochrome, in which case the
response mapping is identity.

If the real e-con/Jetson capture path contains an unavoidable fixed nonlinear
response, calibration may supply a monotonic piecewise-differentiable LUT:

[
I_{resp}=R(I_{linear}).
]

This is not an optimization variable; it represents a fixed measured part of
the deployed imaging chain.

### 3.10 Quantization

If the measured capture path is RAW10:

[
N=2^{10}-1.
]

For RAW12:

[
N=2^{12}-1.
]

Forward:

[
I_q=rac{operatorname{round}(NI)}{N}.
]

Training backward uses STE.

Bit depth is stored in the calibration profile.

### 3.11 Motion blur

Long exposure must have a cost, otherwise maximum exposure becomes a trivial
solution.

The current lightweight proxy uses:

[
m
approx
rac{|v|}{Z_{char}},
]

where (Z_{char}) is a detached robust internal scene-distance statistic.

Blur strength:

[
b
=
1-exp
left(
-k_b
rac{T}{T_{ref}}
m
ight).
]

The current spatial blur kernel is a surrogate. The coefficient (k_b) must
be measured from the real system.

Future higher-fidelity option:

- temporal multi-pose integration across the exposure interval.

Do not adopt it until v1 demonstrates the scientific effect.

---

## 4. What is physical vs what is a surrogate

### Real-camera calibrated fields

The measured IMX900 profile should provide:

- exposure min/max/step;
- exposure response scale;
- gain mapping/LUT;
- shot alpha/beta;
- read noise vs gain;
- black level;
- full-scale/saturation level;
- RAW bit depth;
- motion-blur coefficient;
- command delay in frames/time.

### Training/modeling choices

Do not present these as IMX900 specifications:

- blur kernel size;
- smooth-saturation beta;
- motion-proxy depth floor;
- training image resolution;
- environment lighting distribution;
- camera smoothness loss;
- policy architecture.

---

## 5. Policy architecture

### 5.1 Visual input

Use:

[
X_t=[I_t,I_{t-1}].
]

Reasons:

- preserves the existing two-channel visual stem;
- exposes temporal brightness changes;
- supplies looming/motion cues;
- helps infer blur changes.

### 5.2 Flight branch

The flight branch receives visual features and navigation state.

For the main causal experiment it must **not directly receive**
exposure/gain.

Otherwise camera actions could influence the flight network without passing
through image formation.

### 5.3 Camera branch

The camera branch may receive:

- current/previous image features;
- current exposure/gain;
- local velocity/attitude features.

Conditioning on previous camera state is consistent with the adaptive camera-
control design observed in JOCA.

The camera output is exactly:

[
[E,G].
]

No projector power exists in this branch.

---

## 6. Actuator model

Sensor physics and camera actuation are distinct.

The real system will have:

- finite command range;
- command step;
- command-to-effective-frame delay;
- possible driver buffering;
- update-rate limits.

The calibration schema stores command-delay metadata, and the rollout already
implements a frame-delay queue driven by `command_delay_frames`. It also
supports `command_delay_jitter_frames`, sampled once per episode to represent
measured frame-domain timing spread without confusing that hardware effect with
policy smoothing.

The separate `camera_smoothing_alpha` parameter is a policy-command
regularizer, not a measured IMX900 actuator law. Main experiment configs set it
to 0.0; temporal smoothness is handled explicitly by `coef_cam_smooth` unless
an ablation intentionally enables command smoothing.

After hardware characterization:

1. measure command latency/jitter;
2. replace the provisional zero-frame delay in the profile;
3. set real exposure/gain command steps in the profile;
4. keep policy smoothing only if it is intentionally part of the method;
5. add measured jitter only if it materially affects performance.

---

## 7. Training stages

### Stage A — camera gradient unit tests

Verify:

- exposure autograd vs finite difference;
- gain autograd vs finite difference;
- measured LUT mapping remains piecewise differentiable;
- noise reproducibility with fixed random tensors;
- saturation gradient behavior;
- quantization STE.

### Stage B — fixed grayscale navigation

Use nominal fixed exposure/gain.

Compare:

- real grayscale input;
- zero-image/blind input.

Do not train active camera control until grayscale navigation beats the blind
control.

### Stage C — stress environment

Introduce:

- dark;
- bright;
- dark-to-bright;
- bright-to-dark;
- fast close gate approach.

Verify that no single fixed camera setting dominates all conditions.

### Stage D — non-task-gradient baselines

Required:

- fixed nominal;
- random-static;
- mean AE;
- gradient AE.

### Stage E — primary causal camera experiment

Start from the same successful frozen flight checkpoint.

Compare:

1. learned-detached;
2. learned-differentiable.

Everything except the sensor gradient must match.

This is the core experiment.

### Stage F — optional additional baseline

Only after Stage E is understood, add a JOCA-style derivative-free/local-search
camera correction as a separate method.

Do not use it as hidden supervision in the main result.

### Stage G — joint fine-tuning

Optional.

Report the simpler frozen-flight experiment even if joint fine-tuning improves
absolute performance.

---

## 8. Losses

Primary objective:

[
L
=
L_{nav}
+
lambda_{Delta c}L_{Delta c}.
]

Do not make handcrafted image-quality loss the main camera supervision.

Image metrics such as:

- saturation fraction;
- dark fraction;
- entropy;
- gradient magnitude;
- blur proxy;
- SNR proxy;

are diagnostics unless explicitly evaluated as an ablation.

---

## 9. Evaluation

Navigation:

- success rate;
- collision rate;
- time to goal;
- minimum clearance;
- trajectory efficiency;
- flight smoothness.

Camera:

- exposure trajectory;
- gain trajectory;
- saturation/dark fractions;
- image-motion proxy;
- blur strength;
- noise estimate;
- command/effective latency.

Causal:

- full vs detached under matched seeds;
- navigation-loss gradient norm reaching exposure/gain;
- exposure vs speed/scene-depth relation;
- reaction around illumination transitions.

Sim-to-real:

- real vs simulated intensity response;
- noise variance vs mean;
- real vs simulated saturation;
- real vs simulated blur;
- real vs simulated command latency.

---

## 10. DeepLens decision gate

Do not add DeepLens merely for completeness.

Consider it only if held-out real IMX900 imagery shows a meaningful residual
error due to:

- PSF;
- distortion;
- vignetting;
- defocus.

If required:

1. characterize the lens;
2. use DeepLens offline;
3. fit a compact differentiable PSF/distortion/vignetting surrogate;
4. use that surrogate in the training loop.

---

## 11. Failure criteria

Stop and diagnose before increasing model complexity if:

- grayscale navigation does not beat the blind baseline;
- autograd disagrees with finite differences;
- exposure/gain have weak real image effect;
- one fixed setting dominates every stress condition;
- learned camera actions saturate at bounds;
- full and detached are indistinguishable in deliberately constructed
  illumination/motion conflicts;
- calibrated simulator curves disagree strongly on held-out real data.

---

## 12. Final intended real system

~~~text
e-con e-CAM37M_CUONX / IMX900
       |
       | MIPI CSI-2
       v
Jetson Orin NX
       |
       +--> timestamped monochrome frame
       |
       +--> camera policy --> exposure/gain command
       |
       +--> flight policy --> MAVROS / PX4 --> NxtPX4 v2
~~~

The simulation and real system should share the same conceptual camera action:

[
[E,G].
]

Only the simulator uses the differentiable surrogate.

# Technical Plan — Differentiable Grayscale Active Sensing on IMX900

## 0. Objective

Replace the D455-inspired active-depth line with a simpler, physically interpretable, deployable problem:

> Learn a camera-control policy that adjusts exposure and gain so that a monocular grayscale navigation policy performs better under changing illumination and motion, using gradients through a differentiable image-formation model.

The scientific target is not “make prettier images.” It is:

\[
\min_{\theta_f,\theta_c}\;\mathbb{E}\left[L_{nav}\right]
\]

with

\[
u_t = \pi_f(I_t,s_t;\theta_f),\qquad
c_{t+1}=\pi_c(I_t,s_t,c_t;\theta_c),\qquad
c_t=[T_t,G_t],
\]

and

\[
I_t=\mathcal{C}_{\phi}(I_t^{ideal},T_t,G_t,\xi_t,m_t),
\]

where \(\mathcal C_\phi\) is differentiable with respect to camera parameters.

The key gradient is:

\[
\frac{\partial L_{nav}}{\partial \theta_c}
=
\frac{\partial L_{nav}}{\partial I_t}
\frac{\partial I_t}{\partial (T_t,G_t)}
\frac{\partial (T_t,G_t)}{\partial \theta_c}.
\]

## 1. What remains from DiffPhysDrone

Keep:

- CUDA rigid-body/quadrotor dynamics.
- Collision checking and obstacle geometry.
- State generation and target-relative observations.
- BPTT rollout/training infrastructure where compatible.
- Current recurrent flight-policy structure as a starting point.
- Current camera-policy branch concept.
- Rerun/W&B evaluation infrastructure.
- Real-flight PX4/MAVROS/UART architecture.

Replace:

- D455 \`power/exposure/gain\` semantics.
- \`render_diff_depth\`.
- depth health/fill/hole losses as camera objectives.
- scene-specific \`glare/specular/dark\` depth heuristics.
- depth preprocessing \`near/far\` channels.
- D455 teacher/relabel pipeline as a required mechanism.

## 2. Rendering architecture

### 2.1 Ideal grayscale renderer

The existing CUDA ray tracer already computes ray intersections and contains a helper that can return hit normals. Reuse that geometry.

Target output:

\[
I^{ideal}\in[0,1]^{B\times H\times W}.
\]

Version 1 should use deliberately simple appearance:

\[
I^{ideal}(x)=\rho(x)\,[L_a + L_d\max(0,n(x)^Tl)]
\]

with:

- ambient term \(L_a\),
- one directional or area-light approximation \(L_d\),
- hit normal \(n\),
- object albedo/texture \(\rho\).

Add texture because monocular navigation through uniformly colored geometry is unnecessarily ambiguous. Prefer procedural textures initially:

- checkerboard,
- stripe/noise texture,
- randomized low-frequency texture,
- material albedo randomized per obstacle.

Do not start with physically based path tracing. The research variable is camera control, not photorealism.

### 2.2 Differentiable camera model

Create a pure-PyTorch module, provisionally:

\`sensors/differentiable_gray_camera.py\`

API concept:

~~~python
image, aux = camera(
    irradiance=ideal_gray,
    exposure01=exposure01,
    gain01=gain01,
    motion=motion_state,
    noise_sample=noise_sample,
)
~~~

Physical mappings must be centralized:

~~~text
exposure01 -> exposure_us
gain01     -> physical gain / dB / sensor control units
~~~

Never scatter these mappings across trainer/environment/model.

#### Exposure integration

A minimal linear response:

\[
q = k_e\,T\,E_{ideal}
\]

where \(T\) is physical exposure time.

#### Shot noise

Use a differentiable reparameterized approximation:

\[
q_s = q + \sqrt{\max(q,\epsilon)}\,\sigma_s\,\epsilon_s,\qquad
\epsilon_s\sim\mathcal{N}(0,1).
\]

For strict deterministic gradient tests, supply fixed \(\epsilon_s\).

#### Read noise

\[
q_r=q_s+\sigma_r(G)\epsilon_r.
\]

The gain-dependent read-noise curve is calibrated from real data rather than assumed.

#### Gain

\[
q_g=g(G)\,q_r.
\]

Use a monotonic calibrated mapping. If the driver reports gain in dB, preserve an explicit conversion layer rather than treating the normalized command as linear gain.

#### Saturation

Real evaluation can use hard clipping. Training needs a gradient-friendly approximation near saturation.

Options, in preferred order:

1. straight-through estimator around hard clipping;
2. calibrated smooth shoulder;
3. soft clipping only during early training, hard/ST later.

Do not allow a soft saturation function to make heavily overexposed pixels unrealistically informative.

#### Quantization

Train with either:

- no explicit quantization in v1,
- uniform noise approximation,
- or straight-through rounding.

Evaluate with the true selected output bit-depth when available.

### 2.3 Motion blur

Motion blur is essential. Without it, the trivial optimum can become “maximum exposure.”

Version 1 may use a differentiable approximate blur strength:

\[
b=\mathrm{clip}(k_b T\,m,0,1)
\]

where \(m\) is derived from translational/angular motion. Then

\[
I_{blur}=(1-b)I+b\,K(I)
\]

for a small blur operator \(K\).

Version 2 should integrate multiple pose samples across the exposure interval. That is more physically meaningful, but do not block v1 on it.

### 2.4 Illumination domain randomization

The active-camera policy needs scenarios where a fixed exposure/gain is suboptimal.

At minimum randomize:

- global illumination intensity,
- gate/background contrast,
- bright-to-dark transition,
- dark-to-bright transition,
- directional lighting,
- optional local bright source,
- motion/speed.

Avoid reviving the old hand-coded three-class D455 scene semantics as the primary design. Lighting variation should arise from rendered appearance parameters.

## 3. Policy architecture

### 3.1 Visual input

Preserve the existing 2-channel CNN stem initially:

\[
X_t=[I_t,I_{t-1}].
\]

This is deliberate:

- minimal architecture churn;
- temporal luminance change is directly observable;
- looming/motion cues are available;
- blur is easier to infer than from a single frame.

Normalize calibrated sensor output consistently, preferably to \([-1,1]\) after black-level handling.

### 3.2 Flight policy

Keep the current recurrent state-fusion design first. Do not simultaneously redesign the flight policy and sensor model unless grayscale flight fails for a clearly architectural reason.

### 3.3 Camera policy

Change camera output/state from 3-D to 2-D:

~~~text
old: power, exposure, gain
new: exposure, gain
~~~

The camera branch should see:

- current/previous image features,
- current exposure/gain,
- local velocity/angular-motion cues.

It should not need privileged illumination labels.

### 3.4 Camera update dynamics

The real sensor cannot be treated as an infinitely fast continuous actuator.

Model:

- parameter quantization,
- command latency,
- minimum hold duration if measured,
- optional first-order lag,
- slew-rate limits.

Do not guess these values permanently. First implement configurable placeholders, then replace with measurements from the real IMX900 stack.

## 4. Training plan

### Stage 0 — gradient unit test

Before navigation:

- fixed synthetic irradiance image;
- vary exposure/gain;
- compare PyTorch autograd with finite differences;
- check gradient signs in dark, nominal, and near-saturation regions.

Pass criterion: gradients are finite, stable, and directionally sensible.

### Stage 1 — grayscale flight only

Camera fixed at a nominal setting. Train only the flight policy.

Goal: prove that monocular grayscale navigation itself works.

Do not train camera control until this stage is reliable.

### Stage 2 — camera policy with frozen flight policy

Freeze the trained flight policy.

Train the camera policy through the differentiable camera model.

This is the cleanest causal experiment because the flight policy cannot co-adapt to hide a broken camera-control mechanism.

### Stage 3 — detached control

Use the exact same camera-policy architecture and camera model but break the parameter-to-image gradient path.

This is the key ablation.

Preferred implementation:

~~~python
exposure_for_sensor = exposure if use_sensor_grad else exposure.detach()
gain_for_sensor = gain if use_sensor_grad else gain.detach()
~~~

Do not detach the whole image tensor if that changes unrelated training behavior.

### Stage 4 — joint fine-tuning

Only after Stage 2 clearly works:

- unfreeze selected flight layers,
- use a lower LR,
- retain the camera smoothness/actuator constraints.

Joint fine-tuning is an optimization enhancement, not evidence for the core claim.

## 5. Losses

Keep the camera objective as task-driven as possible.

Primary:

\[
L=L_{nav}+\lambda_{\Delta c}L_{\Delta c}+\lambda_{bounds}L_{bounds}.
\]

Where:

- \(L_{nav}\): goal/collision/trajectory/smooth-flight losses already justified by the navigation task;
- \(L_{\Delta c}\): camera switching/slew regularizer;
- \(L_{bounds}\): only if needed for safe numerical behavior.

Avoid a large hand-designed image-quality loss in the main method. It would blur the causal claim.

Image-quality metrics may be logged for analysis:

- saturation fraction,
- underexposure fraction,
- gradient magnitude,
- image entropy,
- temporal blur proxy,
- SNR proxy.

They should not silently become the main supervision.

## 6. Baselines

Required:

1. Fixed nominal exposure/gain.
2. Random-static exposure/gain.
3. Classical auto exposure.
4. Learned camera policy with detached sensor gradient.
5. Differentiable camera policy.
6. Oracle local grid/search for analysis only.

Recommended classical AE baselines:

- mean-intensity target;
- gradient-based exposure selection;
- optionally percentile-based brightness control.

Use existing active-exposure literature implementations where licensing/integration permits, or faithfully reproduce the published rule with explicit attribution.

## 7. Evaluation metrics

Navigation:

- success rate;
- collision rate;
- time to goal;
- minimum obstacle clearance;
- trajectory efficiency;
- control smoothness.

Camera:

- exposure/gain trajectories;
- update frequency;
- saturation ratio;
- dark-pixel ratio;
- image-gradient statistics;
- estimated/real blur width;
- camera-command latency;
- frame age at policy inference.

Causal/diagnostic:

- difference between differentiable and detached under identical seeds;
- gradient norm from navigation loss to exposure/gain;
- correlation between speed and selected exposure;
- response around illumination transitions;
- success conditioned on illumination and speed.

## 8. Real deployment

Runtime frequency targets are not fixed until measured. The system should support:

- camera streaming asynchronously;
- timestamped frames;
- camera-parameter commands with timestamps;
- policy inference on latest valid frame;
- PX4 control independently at the required control rate;
- logging of requested and, when available, effective exposure/gain.

Do not assume every command applies on the next frame. Measure command-to-effective-frame latency.

## 9. Failure criteria

Stop and diagnose before scaling complexity if any of the following occurs:

- fixed-camera grayscale navigation does not work;
- autograd and finite-difference camera gradients disagree;
- different exposure/gain commands do not measurably alter real images;
- camera command latency is too large/variable for the planned policy rate;
- differentiable and detached methods are indistinguishable even in deliberately constructed illumination transitions;
- the learned policy always saturates exposure/gain at one bound.

## 10. Out-of-scope for v1

- RGB white balance/color pipeline;
- full physically based ray tracing;
- differentiating geometry/pose through the renderer;
- modeling all IMX900 ISP internals;
- HDR multi-exposure fusion;
- event cameras;
- active illumination;
- D455 depth as a policy input;
- simultaneously learning optics.

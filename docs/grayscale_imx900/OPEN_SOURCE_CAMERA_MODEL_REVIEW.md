# Open-Source Camera Model Review and Adoption Decision

> Review date: 2026-09-22
>
> Scope: End2endImaging, DeepLens, and JOCA, with emphasis on what should
> actually enter the DiffPhysDrone grayscale/IMX900 implementation.

## Executive decision

Do **not** make any of the three projects a hard runtime dependency.

Use them as follows:

| Project | Role in DiffPhysDrone | Runtime dependency? |
|---|---|---:|
| End2endImaging | sensor/noise/ISP modeling reference | No |
| DeepLens | optional offline optics fidelity / PSF reference | No |
| JOCA | task-driven camera-control and experimental-design reference | No |

The production camera model in this repository is therefore an
IMX900-specific lightweight surrogate:

[
I_t =
F_{IMX900}(I_t^{ideal}, E_t, G_t, m_t; Theta_{calib})
]

where the numerical parameter set (Theta_{calib}) is loaded from a real
camera calibration profile rather than copied from another camera.

Current implementation:

- `sensors/imx900_calibration.py`
- `sensors/imx900_camera.py`
- `configs/calibration/imx900_provisional.json`

The default profile is marked `calibrated: false` and exists only to support
software development before the hardware is characterized.

---

## 1. End2endImaging

Repository:

https://github.com/vccimaging/End2endImaging

Relevant source:

`end2end_imaging/sensor/mono_sensor.py`

License observed in the repository:

Apache-2.0.

### What its code actually provides

`MonoSensor` organizes a monochrome camera around:

- bit depth;
- black level;
- read-noise standard deviation;
- shot-noise `alpha` and `beta`;
- base ISO;
- optional spectral response;
- black-level compensation and gamma ISP.

Its shot-noise standard deviation is modeled approximately as

[
sigma_{shot}
=
alphasqrt{max(I-B,0)}+eta.
]

The implementation then combines shot/read noise with an ISO-dependent gain
term.

This is a useful modeling pattern because it separates sensor parameters from
the rest of the imaging pipeline and supports loading a sensor configuration
from JSON.

### Why it cannot be imported unchanged

The current `MonoSensor` is not an online exposure/gain controller model for
our problem:

1. It has no explicit exposure-time action (T_e) that participates in
   dynamic image formation and motion blur.
2. The source explicitly fixes `gain_analog = 1.0` in its current noise
   model and primarily uses ISO/digital gain.
3. Its current low-ISO noise implementation rejects ISO above 800.
4. It uses hard `torch.round` for quantization and hard `torch.clip`
   after noise. These are appropriate for realistic forward simulation but
   are not sufficient by themselves for stable gradients with respect to
   online exposure/gain actions.
5. Its numerical defaults are not IMX900 measurements.

### What we adopt

We adopt the **structure**, not the coefficients:

- calibration profile separate from model code;
- shot/read-noise decomposition;
- black-level and bit-depth representation;
- sensor-specific configuration loaded from a file.

This directly motivated `IMX900Calibration`.

### What we deliberately change

For active camera control we require:

- explicit differentiable exposure time;
- explicit differentiable gain;
- reparameterized stochastic noise;
- a gradient-friendly saturation surrogate;
- STE quantization when quantization is enabled;
- motion blur coupled to exposure;
- later: measured actuator latency.

---

## 2. DeepLens

Repository:

https://github.com/vccimaging/DeepLens

License observed in the repository:

Apache-2.0.

Current package metadata inspected during this review:

- package: `deeplens-core`;
- version: 2.5.4;
- Python requirement: >=3.12,<3.13;
- PyTorch dependency: 2.10.0.

### What DeepLens is good at

DeepLens is a differentiable **optics** engine rather than merely an exposure
controller. Its documented capabilities include:

- geometric ray tracing;
- PSF analysis;
- MTF;
- distortion;
- spatially varying and depth-dependent aberrations;
- wave/ray-wave optics;
- neural PSF surrogate representations;
- end-to-end optics/algorithm co-design.

This is useful if the dominant sim-to-real gap later comes from the lens.

### Why it is not in the v1 runtime loop

Our online camera action is only:

[
a_t^{cam}=[E_t,G_t].
]

The lens is fixed during flight. Putting full differentiable optics in every
BPTT step would therefore:

- increase memory and compute cost;
- complicate the CUDA/PyTorch environment;
- blur the scientific question;
- introduce a Python/PyTorch environment migration with little immediate
  benefit.

The project already has the geometry needed for the primary causal experiment.

### Future use

If real IMX900 data shows that lens effects dominate the sim-to-real gap,
DeepLens may be used **offline** to model or fit:

- PSF blur;
- spatially varying sharpness;
- distortion;
- vignetting;
- defocus.

The preferred deployment path is then to distill those effects into a small
runtime surrogate rather than invoking a full optical simulator per frame.

DeepLens is therefore a Phase-2 fidelity tool, not a v1 dependency.

---

## 3. JOCA

Repository:

https://github.com/RoboticImaging/JOCA

Paper:

Task-Driven Joint Optimisation of Camera Hardware and Adaptive Camera Control
Algorithms, WACV 2026.

### Why JOCA is the closest conceptual prior work

The public CARLA code explicitly performs dynamic exposure/gain control for a
downstream perception task.

In `CARLA Experiment/camera.py`:

- the controller predicts two normalized values;
- they are mapped to exposure and gain ranges;
- image intensity is scaled by exposure and gain ratios;
- a gain-dependent shot/read-noise model is applied;
- CARLA motion blur is coupled to the selected exposure.

Its noise code states that the coefficients correspond to Basler DaA1280
noise characteristics. This is an important precedent for our design:
**camera-specific measured coefficients should replace arbitrary simulator
constants**.

JOCA's adaptive controller also conditions on previous camera parameters,
which supports our decision to let the camera branch observe its own current
exposure/gain.

### DF-Grad / perturbation correction

The public joint training code also contains a genetic-algorithm perturbation
path used to search exposure corrections and add a correction loss.

This is important prior art, but it should **not** be the main DiffPhysDrone
method.

Our core question is whether an explicit sensor gradient helps closed-loop UAV
navigation. Adding a search/teacher correction into the main method would make
that causal question ambiguous.

Recommended use:

- main experiment: learned-detached vs learned-differentiable;
- optional later baseline: JOCA-style derivative-free/local-search camera
  correction;
- do not use it to rescue the primary result unless reported as a separate
  method.

### Licensing caution

At the time of this review, a root `LICENSE` file was not found through the
GitHub repository contents API. Therefore:

- do not vendor or copy JOCA implementation code into this repository;
- reimplement required baselines from the paper/observable behavior;
- cite the paper/repository;
- confirm licensing separately before reusing code beyond reference study.

This does not affect using JOCA as scientific prior work.

---

## 4. Resulting architecture

The final v1 architecture is:

~~~text
CUDA geometry
   |
   | ray-hit distance + exact normal (internal only)
   v
ideal grayscale irradiance
   |
   v
IMX900DifferentiableCamera
   |
   |-- calibration profile
   |     exposure mapping
   |     gain mapping / LUT
   |     signal scale
   |     shot alpha/beta
   |     read-noise curve
   |     black level
   |     saturation level
   |     bit depth
   |     blur coefficient
   |     actuator metadata
   |
   |-- differentiable training surrogate
   |     exposure integration
   |     gain
   |     reparameterized noise
   |     smooth/STE saturation
   |     STE quantization
   |     motion blur
   v
[current gray, previous gray]
   |
   +--> flight policy --> acceleration
   |
   +--> camera policy --> [exposure, gain]
~~~

The flight policy must not receive exposure/gain directly in the main causal
experiment.

---

## 5. Camera model equations used by this repository

The current surrogate deliberately keeps the parameterization compact.

Exposure:

[
T=T(E),qquad
q=s_E I_{ideal}rac{T}{T_{ref}}.
]

Gain:

[
S=G(G_{cmd})q.
]

Shot-noise standard deviation in output space:

[
sigma_{shot}
=
Gleft(
alphasqrt{q+epsilon}+eta
ight).
]

Read-noise standard deviation:

[
sigma_{read}
=
sigma_{r0}G^{p_r}.
]

Reparameterized noisy signal:

[
S_n
=
S+
sigma_{shot}epsilon_s+
sigma_{read}epsilon_r.
]

Then:

[
I=
Qleft(
Satleft(
rac{S_n+B}{S_{max}}
ight)
ight).
]

Motion blur is applied before exposure integration using the current
lightweight UAV-specific image-motion proxy.

These equations are **surrogate equations**. Their coefficients are not
claimed to be intrinsic IMX900 constants until fitted and validated.

---

## 6. What is calibrated vs what is a training surrogate

### Must come from the real e-con/IMX900 stack

- exposure min/max/step;
- normalized policy action -> exposure mapping;
- gain mapping or LUT;
- signal/exposure response scale;
- black level;
- shot-noise coefficients;
- read-noise versus gain;
- saturation/full-scale behavior;
- RAW bit depth and quantization;
- exposure-dependent motion-blur scale;
- command-to-effective-frame latency;
- effective parameter update quantization.

### Remains a modeling/training choice

- blur kernel implementation in v1;
- smooth saturation beta;
- characteristic-depth floor in the motion proxy;
- image resolution used by the network;
- domain-randomization ranges;
- camera-policy update regularization.

Do not report the latter group as measured camera specifications.

---

## 7. Novelty boundary after this review

Do not claim:

- first task-driven exposure control;
- first differentiable camera control;
- first joint downstream-task/camera optimization;
- first learned exposure/gain controller.

Working project claim:

> Real-camera-calibrated differentiable exposure/gain control for closed-loop
> monocular quadrotor navigation, evaluated with a matched sensor-gradient
> ablation and deployed on an IMX900-based onboard camera.

Potential differentiators relative to JOCA:

- closed-loop aerial navigation dynamics;
- camera action changes future observations and flight states;
- UAV motion/exposure/blur interaction;
- real IMX900 calibration;
- onboard small-quadrotor deployment;
- direct full-vs-detached sensor-gradient experiment.

Re-run the literature search before manuscript submission.


---

## 8. Implementation consequences in this branch

The review is now reflected in code rather than being documentation-only.

### 8.1 End2endImaging-inspired parts adopted

Implemented in `IMX900Calibration` / `IMX900DifferentiableCamera`:

- physical sensor parameters live in a calibration profile instead of scattered
  experiment flags;
- shot/read noise are separated;
- black level, saturation/full scale, and bit depth are explicit;
- stochastic noise uses reparameterized samples so exposure/gain remain useful
  gradient variables;
- forward quantization can use an STE rather than hard-breaking the action
  gradient.

Additional calibration support added after the review:

- exposure action -> microseconds can be `linear` or an observed LUT;
- gain action -> effective gain can be `linear`, `log`, or an observed LUT;
- read noise vs effective gain can be a compact power law or an observed LUT;
- an optional monotonic response LUT can represent an unavoidable fixed ISP
  response when a RAW/linear path cannot be used.

These LUTs are deliberately piecewise differentiable so measured camera curves
can replace assumptions without changing the policy interface.

### 8.2 JOCA-inspired parts adopted

Implemented or retained:

- camera action is exactly normalized `[exposure, gain]`;
- the camera controller observes its own current actuator state;
- the primary objective remains the downstream navigation objective;
- low-light / bright / transition / fast-motion conditions form the camera
  stress benchmark;
- a matched `full` vs `detached` sensor-gradient experiment remains the
  primary causal test.

Not adopted in the main method:

- GA/DF-Grad correction targets;
- search-based camera teacher supervision.

Those remain optional later baselines so the main experiment can answer one
clean question: does the explicit exposure/gain image-formation gradient help?

### 8.3 DeepLens decision

No runtime dependency was added.

If future held-out real data demonstrates a dominant lens-driven residual, the
preferred route is:

`DeepLens offline -> fitted compact PSF/distortion/vignetting surrogate -> runtime`.

Do not put a full optical simulator inside every UAV BPTT frame without evidence
that lens fidelity is the limiting sim-to-real error.

### 8.4 Real-camera handoff

The intended hardware handoff is now profile-driven:

`real e-con/IMX900 characterization -> measured JSON profile -> same camera
surrogate and policy interface`.

The software model should not need another structural rewrite when the camera
arrives; hardware work should mostly replace provisional profile values with
measured mappings/curves and actuator timing.

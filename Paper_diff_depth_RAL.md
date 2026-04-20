# Differentiable Active Depth Perception for Quadrotor Navigation under Backlit Degradation

> RAL manuscript draft.  
> Scope: this paper describes only the current `diff_depth` branch corresponding to `configs/paper_final_full.args`.  
> Excluded from this version: teacher-student learning, dMPC, policy intent output, TBPTT-enabled training, complex multi-scene narrative, and the broader Nature-style scenario story.

## Abstract

Depth cameras are widely used in small aerial robots because they provide compact geometric observations for obstacle avoidance. However, practical active depth sensors, such as Intel RealSense D-series cameras, are not passive measurement devices: their output depends strongly on controllable physical registers including laser power, exposure time, and analog or digital gain. Conventional visuomotor policies usually treat depth images as exogenous observations and leave these sensing parameters fixed or governed by a hand-designed auto-exposure routine. This separation between perception and control becomes brittle when the robot enters lighting conditions where the sensing process itself is part of the task difficulty.

This paper studies a differentiable active depth perception framework for quadrotor navigation. The central idea is to expose active depth sensor parameters to the policy and to train the perception-control loop end-to-end through a differentiable approximation of the depth acquisition process. At each control step, a CUDA geometric renderer first produces an ideal depth map from the current drone pose and a fixed obstacle map. A differentiable depth sensor model then transforms the ideal geometry into a degraded depth observation as a function of normalized laser power, exposure, and gain. The policy consumes the depth image, proprioceptive state, target-relative state, and current sensor parameters, and outputs both a flight command and the next active sensing parameters. The resulting closed loop is

\[
\text{geometry} \rightarrow \text{differentiable depth sensor}
\rightarrow \text{policy network}
\rightarrow \text{camera update and control}
\rightarrow \text{quadrotor dynamics}
\rightarrow \text{loss and backpropagation}.
\]

The method is evaluated in two fixed-map settings: a base obstacle field and a backlit Sun Glare environment. The base scene tests whether differentiable active depth control preserves standard obstacle avoidance performance. The Sun Glare scene tests whether the policy can adapt sensing parameters when entering a strong backlight region where depth quality is locally degraded. The experimental protocol compares the proposed differentiable active depth policy against non-differentiable perception baselines, fixed-camera policies, and an ego-planner style baseline. Results are reported using navigation success, collision rate, trajectory efficiency, depth fill quality, local glare-region reliability, and learned camera parameter statistics. This draft provides the complete algorithmic formulation and experimental structure; numerical results are intentionally left as placeholders for subsequent training and evaluation.

## Keywords

Differentiable perception, active depth sensing, quadrotor navigation, depth camera simulation, end-to-end visuomotor control, sensor-parameter adaptation, differentiable physics.

## 1. Introduction

Small aerial robots operate under strict sensing, computation, and control constraints. A quadrotor must perceive nearby obstacles, select a safe motion, and execute control commands at high frequency while carrying limited onboard sensing hardware. Depth cameras are particularly attractive for this setting because they provide geometric structure directly, avoiding the need to infer scale from monocular images. In many learning-based navigation pipelines, the depth camera is modeled as a fixed observation function. The policy receives a depth image \(D_t\) and produces a control command \(u_t\). The sensor is treated as an external black box:

\[
D_t = \mathcal{S}(x_t),
\qquad
u_t = \pi_\theta(D_t, o_t),
\]

where \(x_t\) denotes the robot state, \(o_t\) denotes low-dimensional proprioceptive or goal-relative state, and \(\pi_\theta\) is a neural policy.

This view is incomplete for active depth cameras. A RealSense-style active stereo depth sensor depends on physical and firmware-level parameters such as laser emitter power, exposure time, and gain. These parameters affect signal-to-noise ratio, motion blur, invalid depth holes, range limits, and sensitivity to environmental infrared illumination. Therefore the correct observation model is closer to

\[
D_t = \mathcal{S}(x_t, c_t; \phi),
\qquad
c_t = [p_t, e_t, g_t],
\]

where \(p_t\) is laser power, \(e_t\) is exposure, \(g_t\) is gain, and \(\phi\) denotes sensor and scene parameters. In difficult lighting, the sensor parameters are not incidental. They determine whether the robot sees enough geometry to avoid obstacles.

Traditional systems usually handle this problem by fixing camera parameters, using an external auto-exposure heuristic, or tuning parameters offline. These choices are simple but have two limitations. First, a heuristic auto-exposure objective is not necessarily aligned with the downstream navigation objective. A depth image that appears visually bright may still be unreliable for obstacle avoidance. Second, the correct sensing action depends on the robot motion. A long exposure can improve weak-signal depth in static scenes but can produce motion blur for a fast quadrotor. A high laser power can improve active stereo structure under some conditions but may waste energy or worsen specular artifacts in others.

This paper explores a different formulation: the policy directly controls both motion and sensing, and the active depth sensor is placed inside the differentiable training loop. Instead of supervising camera parameters with hand-labeled actions, we train the policy with navigation and perception-reliability objectives. The policy is free to discover whether changing laser power, exposure, or gain improves the objective in a given scene.

The current paper is deliberately narrow. It does not claim to solve all real-world depth camera effects. It focuses on a minimal, reproducible setting that isolates one question:

**Can an end-to-end differentiable active depth pipeline learn useful sensor-parameter adaptation for quadrotor obstacle avoidance, especially when entering a backlit region that degrades local depth reliability?**

To answer this, we implement a fixed small-map simulator with two experimental scenes:

1. **Base scene:** a fixed \(10\,\mathrm{m} \times 10\,\mathrm{m}\) obstacle field with six tall cuboid pillars arranged to require lateral obstacle avoidance.
2. **Sun Glare scene:** a fixed backlit scene where a strong light source is projected into the camera view after the drone enters a specified region, causing local depth degradation.

The policy is trained using the current `diff_depth` pipeline. The configuration uses direct action output, a differentiable Python/Torch depth sensor model after CUDA geometric rendering, a recurrent CNN-based policy, and full backpropagation through time over the rollout. Teacher-student learning, dMPC, and policy-intent output are disabled in this paper version.

### 1.1 Contributions

This manuscript makes the following contributions.

1. **Differentiable active depth observation model.**  
   We formulate a D455-style active depth sensor approximation in which ideal rendered depth is converted into noisy, invalid, motion-blurred, and light-degraded depth through differentiable Torch operations. The model explicitly depends on laser power, exposure, and gain.

2. **Closed-loop active perception-control training.**  
   We train a recurrent visuomotor policy that outputs both flight commands and active depth sensor parameters. Gradients flow through the sensor model, neural network, control decoding, and differentiable quadrotor dynamics.

3. **Minimal backlit evaluation environment.**  
   We define a compact base obstacle field and a Sun Glare variant that tests whether active sensing adaptation helps navigation when the local depth observation becomes unreliable under backlight.

4. **Evaluation protocol against non-differentiable perception and planning baselines.**  
   We provide a complete experimental framework comparing differentiable active depth control with fixed-camera learning, non-differentiable sensor adaptation, and an ego-planner style baseline.

5. **Transparent loss and metric design.**  
   We describe the full training objective, including task losses, collision avoidance, camera smoothness, power regularization, energy/blur/noise proxies, depth fill-rate regularization, and local Sun Glare reliability.

## 2. Related Work

### 2.1 Depth-Based Quadrotor Navigation

Depth cameras have been widely used for aerial navigation because they provide metric geometric observations. Classical pipelines often reconstruct local occupancy maps or Euclidean signed distance fields, then use sampling-based or optimization-based planners to generate safe trajectories. Such systems separate perception, mapping, planning, and control. This decomposition is effective in structured settings but can be sensitive to sensor failures. If the depth map contains holes, flying pixels, or locally invalid measurements, the planner may either become overly conservative or fail to represent obstacles correctly.

Learning-based depth navigation replaces some of these modules with neural policies. A policy can consume depth images and proprioceptive state to output velocity, acceleration, or waypoint commands. Compared with classical pipelines, neural policies can be trained to be robust to noise and partial observability. However, most such policies assume the depth image is produced by a fixed camera model. They may randomize image noise during training, but the sensor control registers themselves are usually not part of the policy action.

Our work differs by making depth camera parameters part of the closed-loop action. The robot does not only react to a depth image; it also changes how future depth images are acquired.

### 2.2 Active Vision and Sensor Parameter Control

Active vision studies how an agent can change its sensing process to improve task performance. This includes camera motion, fixation, zoom, exposure, illumination, and viewpoint selection. In robotics, active perception often focuses on view planning or next-best-view exploration. For active depth cameras, another important dimension is physical sensor control: emitter power, exposure, and gain.

Hand-designed auto-exposure algorithms optimize low-level image statistics, such as mean intensity or saturation ratio. For active stereo depth, this may be insufficient. A bright image is not necessarily a reliable depth image, and a reliable depth image is not necessarily optimal for fast flight. For example, increasing exposure can improve signal strength but also increases motion blur:

\[
\text{blur} \propto \|v_t\|\,t_{\mathrm{exp}},
\]

where \(\|v_t\|\) is camera speed and \(t_{\mathrm{exp}}\) is effective exposure time. Thus sensor control should be coupled to robot motion and downstream safety.

The present method treats sensor parameters as policy outputs and optimizes them through differentiable task losses rather than hand-coded exposure rules.

### 2.3 Differentiable Rendering and Differentiable Simulation

Differentiable rendering provides gradients from image-space objectives to scene, camera, or material parameters. In robotics, differentiable simulation similarly provides gradients from task losses to control actions and policy parameters. The proposed pipeline combines both ideas in a practical form: CUDA kernels provide fast geometric depth and quadrotor dynamics, while differentiable Torch operations model the sensor corruption process.

The pipeline does not require a fully differentiable mesh renderer with respect to all scene geometry. Instead, the current goal is to provide gradients from perception reliability losses to the policy-controlled camera parameters and from trajectory losses to policy-controlled flight actions. This design is sufficient for active sensing policy optimization:

\[
\frac{\partial \mathcal{L}}{\partial \theta}
=
\sum_t
\frac{\partial \mathcal{L}}{\partial D_t}
\frac{\partial D_t}{\partial c_t}
\frac{\partial c_t}{\partial \theta}
+
\sum_t
\frac{\partial \mathcal{L}}{\partial x_{t+1}}
\frac{\partial x_{t+1}}{\partial u_t}
\frac{\partial u_t}{\partial \theta}
+ \cdots .
\]

### 2.4 End-to-End Visuomotor Policies

End-to-end visuomotor control maps sensory inputs directly to control actions. Recurrent neural networks are commonly used because navigation is partially observable: the robot may not see all obstacles at once, and the sensor may be temporarily degraded. In our implementation, a convolutional depth encoder is fused with a low-dimensional state encoder, followed by a GRUCell. The policy head outputs direct acceleration-domain commands and a camera head outputs active sensing parameters.

Unlike approaches that train a separate planner or teacher, this RAL version focuses on the direct end-to-end branch. There is no teacher-student distillation and no differentiable MPC module in the described experiments.

## 3. Method

### 3.1 Problem Formulation

We consider a quadrotor navigating from a fixed start position to a fixed goal while avoiding obstacles. The state at time \(t\) is

\[
x_t =
\{p_t, v_t, R_t, a_t, c_t\},
\]

where \(p_t \in \mathbb{R}^3\) is position, \(v_t \in \mathbb{R}^3\) is velocity, \(R_t \in SO(3)\) is attitude, \(a_t\) is the previous applied acceleration-like command, and

\[
c_t = [p^{\mathrm{cam}}_t, e_t, g_t] \in [0,1]^3
\]

is the active depth camera state consisting of normalized laser power, exposure, and gain. To avoid notation conflict with position \(p_t\), we use \(p^{\mathrm{cam}}_t\) for camera power.

At each time step the environment produces a depth observation

\[
D_t, Q_t = \mathcal{R}_{\phi}(x_t, c_t, \mathcal{M}),
\]

where \(D_t \in \mathbb{R}^{H \times W}\) is the degraded depth image, \(Q_t \in [0,1]^{H \times W}\) is a differentiable quality map, \(\mathcal{M}\) is the obstacle map, and \(\phi\) are sensor and scene parameters.

The policy is

\[
(y_t, \hat{c}_{t+1}, h_{t+1})
=
\pi_\theta(D_t, s_t, h_t),
\]

where \(y_t \in \mathbb{R}^6\) contains a raw flight-control output and an auxiliary velocity prediction, \(\hat{c}_{t+1}\in[0,1]^3\) is the raw next camera command, \(s_t\) is a low-dimensional state vector, and \(h_t\) is the recurrent hidden state.

The camera state is updated by an exponential moving average:

\[
c_{t+1}
=
\alpha c_t + (1-\alpha)\hat{c}_{t+1},
\qquad
\alpha = 0.7,
\]

with a stop-gradient on \(c_t\) in implementation to keep the camera update numerically stable while retaining the current-step gradient from \(\hat{c}_{t+1}\) to the policy.

The flight command is decoded into an acceleration-like action \(u_t\), and the differentiable quadrotor dynamics advance the state:

\[
x_{t+1} = f_{\mathrm{quad}}(x_t, u_t, \Delta t_t).
\]

The training objective minimizes a weighted rollout loss:

\[
\min_{\theta}
\mathbb{E}_{\mathcal{M}, \phi}
\left[
\sum_{t=0}^{T-1}
\mathcal{L}_{\mathrm{task}}(x_t, u_t)
+
\mathcal{L}_{\mathrm{cam}}(c_t, D_t, Q_t)
\right].
\]

In the current configuration, \(T=80\), the control frequency is approximately \(15\,\mathrm{Hz}\), and the policy is trained with full backpropagation through the rollout. TBPTT is disabled.

### 3.2 Fixed-Map Environments

The simulator uses a fixed small map of size \(10\,\mathrm{m} \times 10\,\mathrm{m}\), centered around the origin. The start and goal are

\[
p_{\mathrm{start}} = [-5, 0, 1.5]^\top,
\qquad
p_{\mathrm{goal}} = [5, 0, 1.5]^\top .
\]

The base scene contains six tall cuboid pillars. Each pillar has half-width \(0.25\,\mathrm{m}\) and half-height \(1.5\,\mathrm{m}\). The layout alternates laterally around the centerline to require nontrivial obstacle avoidance:

\[
\begin{aligned}
(-3.80,  0.10, 1.5),\quad
(-2.20, -0.80, 1.5),\quad
(-0.60,  0.50, 1.5),\\
( 1.00, -0.80, 1.5),\quad
( 2.60,  0.50, 1.5),\quad
( 4.20, -0.50, 1.5).
\end{aligned}
\]

The Sun Glare scene uses a simplified backlit environment. It is not intended to reproduce every optical property of sunlight. Instead, it creates a minimal active-depth failure mode: after the drone enters a specified \(x\)-region, a bright source is projected into the camera image and increases the local ambient infrared term, washout penalty, and validity threshold. The light source is represented by a world-space anchor:

\[
p_{\mathrm{sun}} = [7.2, 0.0, 1.8]^\top .
\]

The projected image-space glare mask is

\[
M_{\mathrm{sun}}(u,v)
=
\exp
\left(
-\frac{1}{2}
\left[
\left(\frac{u-u_s}{\sigma_u}\right)^2
+
\left(\frac{v-v_s}{\sigma_v}\right)^2
\right]
\right),
\]

where \((u_s,v_s)\) is the projected sun anchor. A spatial gate activates the effect when the drone enters the backlit zone:

\[
G_{\mathrm{zone}}(x)
=
\sigma
\left(
\frac{x - x_{\mathrm{enter}}}{\tau_{\mathrm{zone}}}
\right).
\]

The final glare strength is

\[
M_{\mathrm{glare}} = M_{\mathrm{sun}}\,G_{\mathrm{zone}}\,\mathbb{I}_{\mathrm{visible}} .
\]

This scene is intentionally simple so that a real-world counterpart can be built with a fixed D455 camera, a small number of obstacles, and a strong backlight or IR source near the goal direction.

### 3.3 Camera Parameter Semantics

The policy outputs normalized camera parameters:

\[
p^{\mathrm{cam}}, e, g \in [0,1].
\]

These are mapped to semantic sensor quantities. Exposure is mapped to an effective exposure time:

\[
t_{\mathrm{exp}}(e)
=
\mathrm{clip}
\left(
t_{\min} + t_{\mathrm{span}} e,\,
t_{\mathrm{eff,min}},\,
t_{\mathrm{eff,max}}
\right).
\]

In the current configuration:

\[
t_{\min}=0.25,\quad
t_{\mathrm{span}}=2.75,\quad
t_{\mathrm{eff,min}}=0.25,\quad
t_{\mathrm{eff,max}}=3.0.
\]

Gain is mapped to a semantic ISO-like gain:

\[
G(g)
=
G_0 + G_s g^\gamma,
\]

with

\[
G_0=1.0,\quad
G_s=10.0,\quad
\gamma=1.2 .
\]

Laser power is already represented in normalized form. The default D455-like nominal power is configured as

\[
p^{\mathrm{cam}}_0 = 0.416667,
\]

corresponding to the ratio \(150/360\) in the user-measured hardware range.

### 3.4 Differentiable Depth Sensor Model

The rendering pipeline has two stages.

First, a CUDA geometric renderer computes the ideal depth:

\[
Z_t = \mathcal{G}_{\mathrm{cuda}}(x_t, \mathcal{M}).
\]

Second, a differentiable Torch sensor model converts \(Z_t\) into a degraded depth observation:

\[
D_t,Q_t
=
\mathcal{S}_{\phi}
\left(
Z_t,\,
c_t,\,
x_t,\,
\mathcal{E}
\right),
\]

where \(Q_t\) is the reliability or quality map and \(\mathcal{E}\) contains scene-dependent lighting and material effects.

The model is not a pixel-perfect D455 simulator. Its purpose is to preserve the major causal relationships needed for policy learning:

1. Higher laser power increases active signal and range but costs energy.
2. Longer exposure increases signal but increases motion blur.
3. Higher gain increases amplification but increases noise.
4. Strong ambient infrared can wash out active stereo patterns.
5. Depth discontinuities are prone to flying pixels.
6. Low quality increases invalid depth probability.

#### 3.4.1 Edge and Frontality Proxy

For each depth image \(Z\), local near and far depth are estimated by max pooling:

\[
Z_{\mathrm{far}}
=
\mathrm{MaxPool}(Z),
\qquad
Z_{\mathrm{near}}
=
-\mathrm{MaxPool}(-Z).
\]

The edge strength is

\[
E
=
\mathrm{clip}
\left(
\frac{k_E (Z_{\mathrm{far}}-Z_{\mathrm{near}})}
{Z + b_E},
0,
1.5
\right).
\]

A frontality proxy is

\[
F
=
\exp(-k_F E).
\]

This term reduces active stereo reliability near discontinuities.

#### 3.4.2 Ambient IR and Material Terms

The base ambient infrared level is modeled as

\[
A
=
\left(
0.12
+0.55 A_{\mathrm{amb}}
+0.25 A_{\mathrm{dir}}
+0.18 A_{\mathrm{air}}
\right)
\left(
1+1.5\beta_{\mathrm{fog}}
\right).
\]

The material albedo proxy is

\[
\rho
=
\mathrm{clip}(0.25 + 0.75\rho_{\mathrm{obs}}, 0.1, 1.0).
\]

Specularity is represented as

\[
\kappa = \rho_{\mathrm{spec}}.
\]

In the Sun Glare scene, the ambient term is modified by the projected glare mask:

\[
A'
=
A + a_{\mathrm{glare}} M_{\mathrm{glare}}.
\]

The active signal multiplier is also modified:

\[
\mu_{\mathrm{active}}
=
\mathrm{clip}_{\min}
\left(
1
-d_{\mathrm{active}}M_{\mathrm{glare}}
+r_{\mathrm{active}}p^{\mathrm{cam}}M_{\mathrm{glare}},
0.05
\right).
\]

This expresses the intended physical trade-off: strong backlight degrades the active pattern, but higher laser power can partially recover the active signal.

#### 3.4.3 Active and Passive Signal

The active depth signal is

\[
S_{\mathrm{active}}
=
\frac{
k_a
p^{\mathrm{cam}}
t_{\mathrm{exp}}
\rho
F
\exp(-\beta_{\mathrm{fog}} Z)
}
{Z^2 + b_a}
\mu_{\mathrm{active}} .
\]

The passive signal is

\[
S_{\mathrm{passive}}
=
t_{\mathrm{exp}}
A'
\left(b_p + k_p E\right)
\left(b_\rho + k_\rho \rho\right)
\sqrt{G(g)}
\mu_{\mathrm{passive}} .
\]

The active range gate is

\[
R_{\mathrm{active}}
=
r_0
+Z_{\max}
\left(
\eta_0+\eta_1\sqrt{p^{\mathrm{cam}}t_{\mathrm{exp}}}
\right)
+\eta_g \log G(g),
\]

and the passive range gate is

\[
R_{\mathrm{passive}}
=
r_p
+Z_{\max}
\left(
\eta_e+\eta_A t_{\mathrm{exp}}A'
\right).
\]

Soft range masks are

\[
\Gamma_a(Z)
=
\sigma
\left(
\frac{R_{\mathrm{active}}-Z}{w_a}
\right),
\qquad
\Gamma_p(Z)
=
\sigma
\left(
\frac{R_{\mathrm{passive}}-Z}{w_p}
\right).
\]

The combined signal is

\[
S
=
S_{\mathrm{active}}\Gamma_a
+\lambda_p S_{\mathrm{passive}}\Gamma_p.
\]

#### 3.4.4 Washout, SNR, and Quality

Strong ambient infrared can wash out active stereo. We model washout as

\[
W
=
\frac{A'}{S_{\mathrm{active}} + b_W}.
\]

Specular bloom is

\[
B_{\mathrm{spec}}
=
\kappa p^{\mathrm{cam}}
\left(0.6 + 0.4 A'\right)
\left(1+E\right).
\]

Motion blur proxy is

\[
M
=
\mathrm{clip}
\left(
\|v_t\|\,t_{\mathrm{exp}}\,k_M,
0,
1.25
\right).
\]

The sensor SNR proxy is

\[
\mathrm{SNR}
=
\frac{S}
{
0.08
+\lambda_A A'
+\lambda_G G(g)
+\lambda_B B_{\mathrm{spec}}
+\lambda_M M
}.
\]

The quality map is

\[
Q
=
\mathrm{clip}
\left(
\sigma
\left[
k_Q\mathrm{SNR}
+k_P S_{\mathrm{passive}}
-k_W W
-k_B B_{\mathrm{spec}}
-k_M M E
-k_R R_{\mathrm{far}}
\right]
+\Delta Q_{\mathrm{scene}},
0,
1
\right),
\]

where

\[
R_{\mathrm{far}}
=
\max
\left(
\frac{Z}{R_{\mathrm{active}}+\epsilon}-0.9,
0
\right).
\]

In Sun Glare, the local quality adjustment includes a degradation term and a power-rescue term:

\[
\Delta Q_{\mathrm{glare}}
=
-k_{\mathrm{glare}} P_{\mathrm{glare}}
+k_{\mathrm{rescue}} R_{\mathrm{power}},
\]

with

\[
P_{\mathrm{glare}}
=
M_{\mathrm{glare}}
\frac{b_g + k_e t_{\mathrm{exp}}}
{b_p + k_p p^{\mathrm{cam}}},
\]

\[
R_{\mathrm{power}}
=
M_{\mathrm{glare}}
\frac{p^{\mathrm{cam}}}
{b_r + k_r t_{\mathrm{exp}}}.
\]

This does not directly reward high power. It only makes high power useful if it improves the local quality objective, while the loss still penalizes excessive power.

#### 3.4.5 Motion Blur, Flying Pixels, Noise, and Invalid Depth

Directional blur is estimated using neighboring depth pixels. Let \(Z_h\) and \(Z_v\) be horizontal and vertical local blur approximations. The camera-frame motion weights are

\[
w_h
=
\frac{|v_y^{\mathrm{cam}}|}
{|v_y^{\mathrm{cam}}|+|v_z^{\mathrm{cam}}|+\epsilon},
\qquad
w_v
=
\frac{|v_z^{\mathrm{cam}}|}
{|v_y^{\mathrm{cam}}|+|v_z^{\mathrm{cam}}|+\epsilon}.
\]

The directional blur depth is

\[
Z_{\mathrm{dir}}
=
w_h Z_h + w_v Z_v.
\]

The blended depth is

\[
Z_{\mathrm{blur}}
=
(1-\alpha_M)Z + \alpha_M Z_{\mathrm{dir}},
\qquad
\alpha_M = \mathrm{clip}(k_{\mathrm{blend}}M,0,\alpha_{\max}).
\]

Flying pixel probability is modeled as

\[
F_{\mathrm{fly}}
=
\mathrm{clip}
\left[
\left(b_f + k_f(1-Q)\right)
E
\left(b_m + k_m(M+B_{\mathrm{spec}})\right),
0,
1
\right].
\]

The corrupted pre-noise depth is

\[
Z_{\mathrm{corr}}
=
Z_{\mathrm{blur}}
+F_{\mathrm{fly}}
\left(
Z_{\mathrm{far}}-Z_{\mathrm{blur}}
\right).
\]

The noise standard deviation is

\[
\sigma_Z
=
\mathrm{clip}
\left(
\sigma_{\mathrm{read}}(1+\lambda_G G)
+
\frac{k_{\mathrm{sig}}(1+\lambda_R (Z/Z_{\max})^2)}
{S+b_\sigma}
+
k_{\mathrm{mot}}M(b_E+k_EE)
+
k_{\mathrm{spec}}B_{\mathrm{spec}},
\sigma_{\min},
\sigma_{\max}
\right).
\]

The noisy depth is

\[
\tilde{Z}
=
Z_{\mathrm{corr}}
+\epsilon_Z\sigma_Z,
\qquad
\epsilon_Z\sim\mathcal{N}(0,1).
\]

The validity mask is differentiable:

\[
V
=
\sigma
\left(
\frac{Q-\tau_{\mathrm{valid}}}{s_{\mathrm{valid}}}
\right).
\]

The final depth and quality are

\[
D = \mathrm{clip}(\tilde{Z}, Z_{\min}, Z_{\max})V,
\qquad
Q_{\mathrm{out}} = QV.
\]

### 3.5 Depth Preprocessing for the Policy

The policy receives a processed single-channel depth tensor. Invalid depths are not treated as near obstacles. Let \(D\) be the rendered depth image and \(Z_{\min}, Z_{\max}\) be the valid depth range. The valid mask is

\[
\Omega(i,j)=\mathbb{I}[D(i,j)\ge Z_{\min}].
\]

The safe depth is

\[
D_{\mathrm{safe}}
=
\begin{cases}
\mathrm{clip}(D,Z_{\min},Z_{\max}), & \Omega=1,\\
Z_{\max}, & \Omega=0.
\end{cases}
\]

The inverse-depth normalized input is

\[
I
=
\frac{
D_{\mathrm{safe}}^{-1} - Z_{\max}^{-1}
}
{
Z_{\min}^{-1}-Z_{\max}^{-1}
}
\Omega.
\]

The image is adaptively max-pooled to the neural input resolution:

\[
I_{\mathrm{nn}}
=
\mathrm{AdaptiveMaxPool}(I, H_{\mathrm{nn}},W_{\mathrm{nn}}).
\]

In the current configuration:

\[
H\times W = 48\times 64,
\qquad
H_{\mathrm{nn}}\times W_{\mathrm{nn}} = 24\times 32.
\]

Finally:

\[
\bar{I}=2I_{\mathrm{nn}}-1.
\]

### 3.6 Policy Network

The policy has three input components:

1. processed depth image \(\bar{I}_t\),
2. low-dimensional state \(s_t\),
3. recurrent hidden state \(h_t\).

The state vector contains local velocity, target-relative velocity, body up vector, safety margin, and current camera state:

\[
s_t =
\left[
v_t^{\mathrm{local}},
v_{\mathrm{target},t}^{\mathrm{local}},
R_t[:,2],
m,
2c_t-1
\right].
\]

With odometry enabled and camera state included, the state dimension is \(13\).

#### 3.6.1 Depth Encoder

The depth image is encoded by three convolutional layers:

\[
\begin{aligned}
F_1 &= \phi(\mathrm{Conv}_{3\times3}^{1\rightarrow 32}(\bar{I})),\\
F_2 &= \phi(\mathrm{Conv}_{3\times3,s=2}^{32\rightarrow 64}(F_1)),\\
F_3 &= \phi(\mathrm{Conv}_{3\times3}^{64\rightarrow 128}(F_2)).
\end{aligned}
\]

The feature is pooled and linearly projected:

\[
z_D
=
W_D
\mathrm{Flatten}
\left(
\mathrm{AdaptiveAvgPool}_{3\times6}(F_3)
\right)
\in \mathbb{R}^{192}.
\]

#### 3.6.2 State Encoder and Gated Fusion

The state encoder is

\[
z_s = W_s s_t \in \mathbb{R}^{192}.
\]

Both depth and state features are layer-normalized:

\[
\hat{z}_D=\mathrm{LN}(z_D),
\qquad
\hat{z}_s=\mathrm{LN}(z_s).
\]

A learned gate balances visual and state features:

\[
\lambda_t
=
\sigma
\left(
W_2 \phi(W_1[\hat{z}_D,\hat{z}_s])
\right),
\]

\[
z_t
=
\phi
\left(
\lambda_t\odot \hat{z}_D
+
(1-\lambda_t)\odot \hat{z}_s
\right).
\]

This prevents either modality from trivially dominating the other throughout training.

#### 3.6.3 Recurrent Memory

The recurrent update is

\[
\tilde{h}_{t+1}
=
\mathrm{GRUCell}(z_t,h_t).
\]

A residual stabilizer is applied:

\[
h_{t+1}
=
\mathrm{LN}
\left(
\tilde{h}_{t+1}
+0.1 f_{\mathrm{res}}(\tilde{h}_{t+1})
\right).
\]

The recurrent state helps when depth observations are partially invalid or when the robot must remember recently observed geometry.

#### 3.6.4 Flight and Camera Heads

The flight head outputs a six-dimensional vector:

\[
y_t = W_u\phi(h_{t+1})\in\mathbb{R}^6.
\]

It is reshaped into two 3D vectors after transformation by the local frame \(R_{\mathrm{local}}\):

\[
(a_t^{\mathrm{pred}}, v_t^{\mathrm{pred}})
=
R_{\mathrm{local}}\,
\mathrm{reshape}(y_t).
\]

The direct acceleration command is

\[
u_t
=
\mathrm{clip}
\left(
(a_t^{\mathrm{pred}}-g_{\mathrm{std}})\eta_{\mathrm{thr}}
+g_{\mathrm{std}},
-u_{\max},
u_{\max}
\right),
\]

where \(u_{\max}=20.0\).

The camera head outputs

\[
\hat{c}_{t+1}
=
\sigma(W_c\phi(h_{t+1}))
\in [0,1]^3.
\]

The three channels correspond to laser power, exposure, and gain.

### 3.7 Closed-Loop Differentiable Training

The rollout loop is:

1. Render differentiable depth:

   \[
   (D_t,Q_t)=\mathcal{R}_{\phi}(x_t,c_t,\mathcal{M}).
   \]

2. Compute soft fill and local reliability metrics from \(Q_t\).

3. Construct state vector \(s_t\).

4. Evaluate policy:

   \[
   (y_t,\hat{c}_{t+1},h_{t+1})=\pi_\theta(D_t,s_t,h_t).
   \]

5. Update camera state:

   \[
   c_{t+1}=0.7c_t+0.3\hat{c}_{t+1}.
   \]

6. Decode direct flight action \(u_t\).

7. Advance differentiable quadrotor dynamics:

   \[
   x_{t+1}=f_{\mathrm{quad}}(x_t,u_t,\Delta t_t).
   \]

8. Accumulate losses and backpropagate through the complete rollout.

The effective control time step includes an exposure-dependent delay proxy:

\[
\Delta t_t
=
\Delta t_{\mathrm{base}}
+0.01\,t_{\mathrm{exp}}(e_t).
\]

This couples sensing latency to control.

The full gradient contains two important paths:

\[
\frac{\partial \mathcal{L}}{\partial \theta}
\supset
\frac{\partial \mathcal{L}_{\mathrm{perception}}}{\partial Q_t}
\frac{\partial Q_t}{\partial c_t}
\frac{\partial c_t}{\partial \theta},
\]

and

\[
\frac{\partial \mathcal{L}}{\partial \theta}
\supset
\frac{\partial \mathcal{L}_{\mathrm{task}}}{\partial x_{t+1}}
\frac{\partial x_{t+1}}{\partial u_t}
\frac{\partial u_t}{\partial \theta}.
\]

Thus the policy can improve both motion and sensing using task-level feedback.

### 3.8 Algorithmic Summary

Algorithm 1 summarizes the training procedure implemented by the current `diff_depth` branch. The key feature is that the active depth sensor is evaluated inside the rollout, before the policy action is computed, and that the sensor-quality losses are accumulated together with the navigation losses before backpropagation.

**Algorithm 1: Differentiable active depth rollout training**

**Input:** fixed obstacle map \(\mathcal{M}\), scene profile \(\phi\), policy parameters \(\theta\), rollout length \(T\), batch size \(B\), initial camera state \(c_0\), initial recurrent state \(h_0\).  
**Output:** updated policy parameters \(\theta\).

1. Initialize batched quadrotor states:

   \[
   x_0^{1:B} \leftarrow p_{\mathrm{start}}, v_0, R_0 .
   \]

2. Initialize camera parameters:

   \[
   c_0^{1:B} \leftarrow [p_0,0.5,0.5],
   \qquad
   p_0=0.416667 .
   \]

3. For \(t=0,\ldots,T-1\):

   \[
   Z_t \leftarrow \mathcal{G}_{\mathrm{cuda}}(x_t,\mathcal{M})
   \]

   \[
   D_t,Q_t,\Psi_t
   \leftarrow
   \mathcal{S}_{\phi}(Z_t,c_t,x_t),
   \]

   where \(\Psi_t\) denotes auxiliary differentiable sensor statistics such as fill rate, hole rate, local glare mask, local glare quality, blur proxy, and noise proxy.

4. Preprocess the degraded depth:

   \[
   \bar{I}_t \leftarrow \mathrm{PreprocessDepth}(D_t).
   \]

5. Construct low-dimensional state:

   \[
   s_t \leftarrow
   [
   v_t^{\mathrm{local}},
   v_{\mathrm{target},t}^{\mathrm{local}},
   R_t[:,2],
   m_t,
   2c_t-1
   ].
   \]

6. Evaluate the recurrent policy:

   \[
   y_t,\hat{c}_{t+1},h_{t+1}
   \leftarrow
   \pi_\theta(\bar{I}_t,s_t,h_t).
   \]

7. Decode flight command and update camera state:

   \[
   u_t \leftarrow \mathrm{DecodeAction}(y_t),
   \]

   \[
   c_{t+1}
   \leftarrow
   0.7\,\mathrm{stopgrad}(c_t)+0.3\,\hat{c}_{t+1}.
   \]

8. Advance dynamics:

   \[
   x_{t+1}\leftarrow f_{\mathrm{quad}}(x_t,u_t,\Delta t_t).
   \]

9. Accumulate per-step losses:

   \[
   \mathcal{L}
   \leftarrow
   \mathcal{L}
   +\mathcal{L}_{\mathrm{task}}(x_t,u_t)
   +\mathcal{L}_{\mathrm{cam}}(c_t,\hat{c}_t)
   +\mathcal{L}_{\mathrm{depth}}(\Psi_t).
   \]

10. Backpropagate through the unrolled computation graph:

    \[
    \theta
    \leftarrow
    \mathrm{AdamW}
    \left(
    \theta,
    \nabla_\theta \mathcal{L}
    \right).
    \]

This algorithm differs from a conventional depth-policy training loop in two ways. First, the depth observation is not a fixed input distribution; it is a differentiable function of policy-controlled camera registers. Second, the camera command is not supervised by a target register trajectory. It is optimized only through the navigation and perception-reliability objectives.

### 3.9 Training Objective

The total loss is

\[
\mathcal{L}
=
\lambda_v\mathcal{L}_v
+\lambda_{\mathrm{avoid}}\mathcal{L}_{\mathrm{avoid}}
+\lambda_{\mathrm{coll}}\mathcal{L}_{\mathrm{coll}}
+\lambda_{\mathrm{acc}}\mathcal{L}_{\mathrm{acc}}
+\lambda_{\mathrm{jerk}}\mathcal{L}_{\mathrm{jerk}}
+\lambda_{\mathrm{cam}}\mathcal{L}_{\mathrm{cam}}
+\lambda_{\mathrm{depth}}\mathcal{L}_{\mathrm{depth}} .
\]

The current configuration uses:

\[
\lambda_v=2.5,\quad
\lambda_{\mathrm{avoid}}=4.0,\quad
\lambda_{\mathrm{coll}}=10.0,\quad
\lambda_{\mathrm{acc}}=0.1,\quad
\lambda_{\mathrm{jerk}}=0.2.
\]

#### 3.9.1 Velocity Tracking

Let \(\bar{v}_t\) be a window-averaged velocity:

\[
\bar{v}_t
=
\frac{1}{K}
\sum_{k=0}^{K-1} v_{t+k}.
\]

The velocity loss is a smooth L1 loss:

\[
\mathcal{L}_v
=
\mathrm{SmoothL1}
\left(
\|\bar{v}_t-v^{\mathrm{target}}_t\|_2,
0
\right).
\]

#### 3.9.2 Action Smoothness

Acceleration regularization is

\[
\mathcal{L}_{\mathrm{acc}}
=
\frac{1}{TB}
\sum_{t,b}
\|u_{t,b}\|_2^2.
\]

Jerk regularization is

\[
\mathcal{L}_{\mathrm{jerk}}
=
\frac{1}{TB}
\sum_{t,b}
\left\|
\frac{u_{t,b}-u_{t-1,b}}{\Delta t}
\right\|_2^2.
\]

The implementation scales action differences by the nominal control frequency.

#### 3.9.3 Obstacle Avoidance and Collision Loss

Let \(d_{t,b}\) be the signed distance from robot \(b\) to the nearest obstacle surface minus safety margin. A soft avoidance barrier is

\[
\mathcal{L}_{\mathrm{avoid}}
=
\mathbb{E}
\left[
w_{t,b}
\left(
\max(0,1-d_{t,b})
\right)^2
\right].
\]

The collision loss is

\[
\mathcal{L}_{\mathrm{coll}}
=
\mathbb{E}
\left[
w_{t,b}
\mathrm{softplus}(-32d_{t,b})
\right].
\]

The weight \(w_{t,b}\) increases when the robot approaches obstacles rapidly.

#### 3.9.4 Camera Smoothness and Range Regularization

Let \(\hat{c}_t=[\hat{p}_t,\hat{e}_t,\hat{g}_t]\) be raw camera head output. Smoothness is

\[
\mathcal{L}_{\mathrm{cam,smooth}}
=
\mathbb{E}_t
\left[
\|\hat{c}_t-\hat{c}_{t-1}\|_2^2
\right].
\]

Power regularization has a deadband around the nominal D455-like value \(p_0\):

\[
\mathcal{L}_{\mathrm{power,reg}}
=
\mathbb{E}
\left[
\max
\left(
0,\,
|\hat{p}_t-p_0|-\delta_p
\right)^2
\right].
\]

In the current configuration:

\[
p_0=0.416667,\qquad
\delta_p=0.18.
\]

Exposure and gain range regularization is

\[
\mathcal{L}_{\mathrm{cam,range}}
=
\mathbb{E}
\left[
(\hat{e}_t-0.5)^2
+(\hat{g}_t-0.5)^2
\right].
\]

The camera smoothness, power regularization, and range weights are currently

\[
\lambda_{\mathrm{cam,smooth}}=100,\quad
\lambda_{\mathrm{power,reg}}=100,\quad
\lambda_{\mathrm{cam,range}}=1.
\]

#### 3.9.5 Depth Sensor Losses

Energy proxy:

\[
\mathcal{L}_{\mathrm{power}}
=
\mathbb{E}
\left[
\max(0,p^{\mathrm{cam}}_t-p_{\mathrm{thr}})^2
\right],
\]

where

\[
p_{\mathrm{thr}}=0.416667.
\]

Blur proxy:

\[
\mathcal{L}_{\mathrm{blur}}
=
\mathbb{E}
\left[
\left(
\|v_t\|_2 t_{\mathrm{exp}}(e_t)
\right)^2
\right].
\]

Noise proxy:

\[
\mathcal{L}_{\mathrm{noise}}
=
\mathbb{E}
\left[
g_t^2
\right].
\]

Soft fill-rate loss uses a differentiable fill proxy:

\[
F_t
=
\frac{1}{HW}
\sum_{i,j}
\sigma
\left(
\frac{Q_t(i,j)-q_{\min}}{\tau_q}
\right).
\]

The fill loss is

\[
\mathcal{L}_{\mathrm{fill}}
=
\max(0,F_{\min}-F_t)^2.
\]

In the current configuration:

\[
F_{\min}=0.25.
\]

For Sun Glare, a local quality term is computed inside the glare mask:

\[
\bar{Q}_{\mathrm{glare}}
=
\frac{
\sum_{i,j} M_{\mathrm{glare}}(i,j) Q(i,j)
}
{
\sum_{i,j} M_{\mathrm{glare}}(i,j)+\epsilon
}.
\]

The local Sun Glare reliability loss is

\[
\mathcal{L}_{\mathrm{glare}}
=
\max
\left(
0,\,
Q_{\mathrm{target}}-\bar{Q}_{\mathrm{glare}}
\right)^2.
\]

The current configuration sets

\[
Q_{\mathrm{target}}=0.1,
\qquad
\lambda_{\mathrm{glare}}=30.0.
\]

This term does not supervise a camera action. It does not state that power should increase or exposure should decrease. It only defines a local reliability target. If the simulator physics make power useful under glare, gradients can encourage the policy to use it; if power is not useful, the power penalties discourage unnecessary high power.

The depth-related coefficients are:

\[
\lambda_{\mathrm{power}}=20,\quad
\lambda_{\mathrm{blur}}=0.1,\quad
\lambda_{\mathrm{noise}}=5,\quad
\lambda_{\mathrm{fill}}=30.
\]

### 3.10 Optimization and Runtime Configuration

The current `paper_final_full.args` branch uses:

| Category | Value |
|---|---|
| Batch size | \(150\) in the current args file |
| Rollout length | \(80\) steps |
| Optimizer | AdamW |
| Learning rate | \(5\times 10^{-5}\) |
| Scheduler | Cosine annealing |
| AMP | enabled by default |
| Depth render size | \(64\times48\) |
| Policy depth input size | \(32\times24\) |
| Depth range | \(0.3\,\mathrm{m}\) to \(6.0\,\mathrm{m}\) |
| Camera angle | \(20^\circ\) |
| Sensor backend | `diff_depth=python` |
| Enabled scene in training args | `sun_glare` |
| Direct control | enabled |
| dMPC | disabled |
| Policy intent output | disabled |
| Teacher-student training | disabled |
| TBPTT | disabled |

If GPU memory is limited, the batch size may be reduced without changing the method. The algorithmic description is independent of this hardware-dependent batch setting.

### 3.11 Implementation Correspondence

The implementation is organized so that each paper module has a direct code counterpart. This is important for reproducibility because the proposed method is not only a network architecture but a full differentiable closed-loop system.

| Paper module | Implementation file | Main responsibility |
|---|---|---|
| Runtime arguments and configuration | `config.py`, `configs/paper_final_full.args` | Defines rollout length, depth resolution, camera semantics, sensor-model parameters, loss weights, and enabled scene |
| Environment construction | `train_utils.py` | Creates training and evaluation environments with the selected scene profiles |
| Fixed map and scene effects | `env_cuda.py` | Defines the fixed obstacle layout, Sun Glare scene effect, material proxies, and scene-local masks |
| CUDA geometric rendering | `src/quadsim.cpp`, `src/quadsim_kernel.cu`, `env_cuda.py` | Computes ideal depth and quadrotor simulation primitives |
| Differentiable sensor model | `env_cuda.py` | Converts ideal depth into degraded active depth, quality, invalid masks, and sensor statistics |
| Sensor rollout helpers | `rollout_ops.py` | Initializes and updates camera state, renders sensors, computes shared proxies and fill statistics |
| Neural policy | `model.py` | Implements CNN depth encoder, state encoder, gated fusion, GRU memory, flight head, and camera head |
| Loss computation | `losses.py` | Computes task losses, camera regularizers, depth reliability losses, and Sun Glare local quality loss |
| Training loop | `trainer.py`, `main_cuda.py` | Unrolls simulation, accumulates losses, logs statistics, and updates policy parameters |
| Evaluation and visualization export | `eval.py`, `eval.sh`, `rerun_vis.py` | Evaluates trained checkpoints and exports trajectories, depth, quality, scene masks, and camera time series |

The current version uses CUDA for fast geometric depth and dynamics, but the differentiable active sensor corruption is implemented in PyTorch. Therefore, the sensor gradients required by the camera head are generated by the PyTorch operations in `env_cuda.py`, not by differentiating through the CUDA geometric renderer with respect to scene geometry. This separation is intentional: the paper studies active register adaptation, not geometry optimization.

### 3.12 What Is and Is Not End-to-End Differentiable

The phrase "end-to-end differentiable" can be ambiguous. In this paper it means that gradients used for training flow through the closed-loop computation from losses to policy parameters, including the differentiable dependence of depth quality on camera parameters. More precisely, the following paths are active:

\[
\mathcal{L}_{\mathrm{depth}}
\rightarrow
Q_t,D_t
\rightarrow
c_t
\rightarrow
\hat{c}_t
\rightarrow
\theta ,
\]

\[
\mathcal{L}_{\mathrm{task}}
\rightarrow
x_{t+1}
\rightarrow
u_t
\rightarrow
y_t
\rightarrow
\theta .
\]

The following paths are not claimed:

\[
\frac{\partial Z_t}{\partial \mathcal{M}},
\qquad
\frac{\partial Z_t}{\partial \text{obstacle geometry}},
\qquad
\frac{\partial \mathcal{L}}{\partial \text{real camera firmware}} .
\]

The geometric depth \(Z_t\) is treated as a rendered observation from the current state and fixed map. The differentiable component needed for active perception is the transformation from ideal geometry and camera registers to degraded depth reliability:

\[
(Z_t,c_t,x_t,\phi)
\mapsto
(D_t,Q_t,\Psi_t).
\]

This distinction makes the method practical. A fully differentiable physical simulation of a D455, including stereo matching firmware, projector pattern physics, sensor saturation, rolling shutter, and infrared material BRDFs, would be far more complex and would still require calibration. The proposed surrogate focuses on the causal derivatives most relevant to policy learning:

\[
\frac{\partial Q}{\partial p^{\mathrm{cam}}},
\qquad
\frac{\partial Q}{\partial e},
\qquad
\frac{\partial Q}{\partial g}.
\]

## 4. Experiments

This section defines the experimental framework. Numerical results are left blank and should be filled after running the training and evaluation scripts.

### 4.1 Research Questions

The experiments are designed to answer four questions.

**Q1: Does differentiable active depth perception preserve obstacle avoidance in a clean base scene?**  
The active camera should not destabilize standard navigation.

**Q2: Does differentiable active depth perception improve navigation under backlit Sun Glare degradation?**  
The policy should maintain depth reliability and avoid obstacles after entering the glare region.

**Q3: Does the policy learn meaningful camera-parameter adaptation?**  
In Sun Glare, we expect nontrivial changes in power, exposure, and gain, correlated with the glare region and local depth quality.

**Q4: Is differentiability important compared with non-differentiable sensing or classical planning baselines?**  
The proposed method should outperform methods that do not receive gradient information through the active sensor model.

### 4.2 Scenes

#### 4.2.1 Base

The base scene uses the fixed six-pillar obstacle field. The drone starts at \((-5,0,1.5)\) and aims for \((5,0,1.5)\). This scene evaluates basic navigation, obstacle clearance, trajectory smoothness, and whether active sensor control introduces unnecessary camera variation.

Expected qualitative behavior:

1. The drone follows an S-like trajectory around the pillars.
2. Depth fill remains stable.
3. Camera parameters remain close to nominal values unless the policy finds small beneficial adjustments.

#### 4.2.2 Sun Glare

The Sun Glare scene introduces a backlit region. When the drone approaches the exit direction, the projected light source produces local degradation in the depth quality map. This tests whether active sensing helps recover reliable geometry.

Expected qualitative behavior:

1. The drone continues through the backlit region instead of stopping before it.
2. Exposure may decrease to suppress ambient washout and motion blur.
3. Power may increase if stronger active signal improves local glare quality.
4. Gain may change depending on the learned noise-signal trade-off.

### 4.3 Compared Methods

The proposed evaluation should include the following methods.

#### Method A: Ours, Differentiable Active Depth

This is the full method described in Section 3. The policy controls flight and camera parameters. The sensor model is differentiable with respect to power, exposure, and gain. The policy is trained using navigation losses and perception reliability losses.

#### Method B: Fixed-Camera Depth Policy

The network architecture is the same, but camera parameters are fixed:

\[
c_t = [0.416667,0.5,0.5].
\]

The policy controls only flight. This baseline tests whether active camera control is necessary.

#### Method C: Non-Differentiable Active Depth Policy

The policy may output camera parameters, but gradients through the sensor model are detached:

\[
\frac{\partial D_t}{\partial c_t}=0,
\qquad
\frac{\partial Q_t}{\partial c_t}=0.
\]

This baseline tests whether differentiability is necessary for learning useful sensor adaptation. It may still learn camera behavior indirectly through delayed task rewards, but without direct sensor-quality gradients.

#### Method D: Heuristic Auto-Exposure Baseline

A hand-coded controller adjusts exposure or gain based on global depth fill or image quality statistics. For example:

\[
e_{t+1}
=
\mathrm{clip}
\left(
e_t + k_e(F^\star-F_t),
0,
1
\right),
\]

with fixed power. This baseline tests whether a simple sensor heuristic can replace end-to-end active perception.

#### Method E: Ego-Planner Style Baseline

An ego-planner style method uses the current depth map to construct a local obstacle representation and plans a collision-free trajectory with fixed camera settings. It does not learn active sensor control. This baseline represents modular geometry-based navigation.

For fairness, all baselines should use the same map, start, goal, sensor resolution, and depth range.

### 4.4 Baseline Implementation Details

The baselines should be implemented in a way that isolates the scientific variable of interest. The goal is not to make weak baselines, but to determine which component contributes to performance.

#### 4.4.1 Fixed-Camera Policy

The fixed-camera policy should keep the same network architecture as the proposed method except for the camera-update branch. Two implementations are acceptable:

1. Keep the camera head in the network but ignore its output during rollout.
2. Remove the camera head and use the same recurrent flight-control body.

The first option is easier because it changes fewer files. During rollout:

\[
c_t=[p_0,0.5,0.5],
\qquad
\forall t .
\]

The loss should remove camera smoothness and camera range terms because there is no learnable camera action:

\[
\lambda_{\mathrm{cam,smooth}}
=
\lambda_{\mathrm{power,reg}}
=
\lambda_{\mathrm{cam,range}}
=0.
\]

The depth sensor losses can still be reported as metrics, but they should not update camera parameters. This baseline answers whether the active camera branch is needed at all.

#### 4.4.2 Non-Differentiable Active Depth Policy

The non-differentiable baseline keeps camera outputs but detaches the sensor response:

\[
D_t^{\mathrm{detach}} = \mathrm{stopgrad}(D_t),
\qquad
Q_t^{\mathrm{detach}} = \mathrm{stopgrad}(Q_t)
\]

with respect to \(c_t\). A careful implementation should still allow the policy to receive depth as an observation:

\[
\bar{I}_t = \mathrm{PreprocessDepth}(D_t^{\mathrm{detach}}),
\]

but the gradient

\[
\frac{\partial Q_t}{\partial c_t}
\]

should be zero. This baseline is stronger than fixed-camera because the policy can still vary camera parameters, but it must learn their usefulness only through delayed navigation outcomes.

The expected failure mode is noisy or weak camera adaptation. If this method performs similarly to the proposed method, the paper should be conservative and state that differentiability is not essential in the tested environment.

#### 4.4.3 Heuristic Auto-Exposure and Auto-Gain

The heuristic baseline should be deliberately simple and reproducible. A reasonable global-fill controller is:

\[
e_{t+1}
=
\mathrm{clip}
\left(
e_t
+k_e(F^\star-F_t),
0,
1
\right),
\]

\[
g_{t+1}
=
\mathrm{clip}
\left(
g_t
+k_g(F^\star-F_t),
0,
1
\right),
\]

\[
p^{\mathrm{cam}}_{t+1}=p_0 .
\]

A stronger variant may also control power:

\[
p^{\mathrm{cam}}_{t+1}
=
\mathrm{clip}
\left(
p^{\mathrm{cam}}_t
+k_p(F^\star-F_t),
0,
1
\right).
\]

However, the paper should clearly state which variant is used. If the heuristic controls power, it should use the same energy penalty or a comparable saturation limit so that it cannot trivially keep power at the maximum.

The main limitation of a global heuristic is that it observes only a scalar fill statistic. It cannot distinguish whether the invalid pixels are in the local Sun Glare region, near an obstacle boundary, or in irrelevant background. This is precisely why the proposed method uses a learned policy and local reliability terms.

#### 4.4.4 Ego-Planner Style Baseline

The ego-planner style baseline should represent a modular navigation stack:

\[
D_t
\rightarrow
\text{local occupancy}
\rightarrow
\text{trajectory optimization}
\rightarrow
\text{tracking control}.
\]

For a fair comparison in the current simulator, the planner should use the same degraded depth observation and fixed camera settings as the fixed-camera baseline. The planner may use a conservative inflation radius around obstacles. If the planner receives ground-truth obstacle positions, that must be reported separately as an oracle planner and should not be compared directly as a perception baseline.

The expected limitation is that fixed-depth modular planning can become conservative when Sun Glare reduces local depth validity. If the local map near the obstacle is incomplete, the planner may either stop, choose an unsafe path, or over-inflate unknown space.

### 4.5 Metrics

#### Navigation Metrics

| Metric | Definition |
|---|---|
| Success rate | Fraction of episodes reaching the goal without collision |
| Collision rate | Fraction of episodes with any obstacle collision |
| Minimum clearance | Minimum signed distance to obstacles over the rollout |
| Time to goal | Number of steps or seconds required to reach goal |
| Path length | Integrated traveled distance |
| Mean speed | Average \(\|v_t\|_2\) |
| Control effort | \(\sum_t \|u_t\|_2^2\) |
| Jerk | \(\sum_t \|u_t-u_{t-1}\|_2^2\) |

#### Perception Metrics

| Metric | Definition |
|---|---|
| Depth fill rate | Fraction or soft fraction of valid/reliable depth pixels |
| Hole rate | \(1-\) fill rate |
| Quality mean | Mean depth quality \(Q\) |
| Invalid rate | Mean invalid probability |
| Local glare quality | Mean \(Q\) inside glare mask |
| Local glare invalid rate | Mean invalid probability inside glare mask |

#### Active Camera Metrics

| Metric | Definition |
|---|---|
| Power mean/std/min/max | Statistics of \(p^{\mathrm{cam}}_t\) |
| Exposure mean/std/min/max | Statistics of \(e_t\) |
| Gain mean/std/min/max | Statistics of \(g_t\) |
| Energy proxy | \(\mathbb{E}[p_t^2]\) |
| Blur proxy | \(\mathbb{E}[(\|v_t\|t_{\mathrm{exp}})^2]\) |
| Noise proxy | \(\mathbb{E}[g_t^2]\) |
| Camera smoothness | \(\mathbb{E}[\|c_t-c_{t-1}\|^2]\) |

#### Event-Aligned Camera Metrics

For the Sun Glare scene, global statistics alone are insufficient. A policy can have a high power standard deviation for reasons unrelated to backlight. Therefore, the central metric should be event-aligned around the glare-entry time.

Let

\[
t_{\mathrm{entry}}
=
\min\{t\mid x_t>x_{\mathrm{enter}}\}.
\]

Define pre-entry and post-entry windows:

\[
\mathcal{T}_{\mathrm{pre}}
=
[t_{\mathrm{entry}}-K_{\mathrm{pre}},t_{\mathrm{entry}}),
\]

\[
\mathcal{T}_{\mathrm{post}}
=
[t_{\mathrm{entry}},t_{\mathrm{entry}}+K_{\mathrm{post}}].
\]

Then report:

\[
\Delta p
=
\mathbb{E}_{t\in\mathcal{T}_{\mathrm{post}}}[p_t^{\mathrm{cam}}]
-
\mathbb{E}_{t\in\mathcal{T}_{\mathrm{pre}}}[p_t^{\mathrm{cam}}],
\]

\[
\Delta e
=
\mathbb{E}_{t\in\mathcal{T}_{\mathrm{post}}}[e_t]
-
\mathbb{E}_{t\in\mathcal{T}_{\mathrm{pre}}}[e_t],
\]

\[
\Delta g
=
\mathbb{E}_{t\in\mathcal{T}_{\mathrm{post}}}[g_t]
-
\mathbb{E}_{t\in\mathcal{T}_{\mathrm{pre}}}[g_t].
\]

This makes the result interpretable: camera adaptation is meaningful only if it is temporally aligned with the physical degradation event and if it improves local perception or navigation.

### 4.6 Experimental Protocol

For each method and each scene:

1. Train with the corresponding training configuration.
2. Evaluate on fixed seeds and, optionally, randomized sensor model parameters.
3. Record full trajectories, depth images, quality maps, invalid masks, scene masks, and camera parameters.
4. Report mean and standard deviation over evaluation episodes.
5. Visualize representative trajectories and camera-parameter time series.

For Sun Glare, align time series by the zone-entry event:

\[
t_{\mathrm{entry}}
=
\min\{t\mid x_t > x_{\mathrm{enter}}\}.
\]

Then plot camera parameters as functions of

\[
\Delta t = t-t_{\mathrm{entry}}.
\]

This reveals whether camera adaptation occurs specifically when entering the backlit region.

#### 4.6.1 Training Protocol

The main training run should use the active configuration in `configs/paper_final_full.args`. For a clean paper comparison, each baseline should be trained from scratch with only the minimum required changes. The recommended training protocol is:

1. Use the same random seed set for all learning-based methods.
2. Use the same rollout length \(T=80\), depth resolution \(64\times48\), and neural depth input \(32\times24\).
3. Use the same optimizer, learning rate, scheduler, and gradient clipping.
4. Select checkpoints by validation success rate or by a fixed iteration count, not by cherry-picking visual behavior.
5. Report at least three seeds for every learned method if computationally feasible.

If only one seed is available in the first draft, the paper should explicitly call the result preliminary and avoid making strong statistical claims.

#### 4.6.2 Evaluation Protocol

Evaluation should be deterministic unless the experiment explicitly studies robustness. Recommended settings are:

1. Freeze the obstacle layout.
2. Freeze the Sun Glare anchor and zone gate.
3. Disable training-time sensor parameter randomization for the main table.
4. Run an additional robustness table with small sensor-model randomization.
5. Export every evaluated rollout for visualization.

For each episode, record:

\[
\{x_t,u_t,c_t,D_t,Q_t,M_{\mathrm{glare},t},F_t\}_{t=0}^{T-1}.
\]

The evaluation should mark an episode as successful if:

1. the drone reaches a goal-radius threshold,
2. no collision occurs,
3. the final speed is not excessively high,
4. the drone does not stop permanently before the glare region.

The stop-before-glare metric is especially important because a policy may reduce loss by refusing to enter the difficult region. A practical definition is:

\[
\mathrm{StopBeforeGlare}=1
\]

if

\[
\max_t x_t < x_{\mathrm{enter}}+\epsilon_x
\]

and

\[
\frac{1}{K}\sum_{t=T-K}^{T-1}\|v_t\| < v_{\mathrm{stop}} .
\]

#### 4.6.3 Rerun Visualization Protocol

The qualitative figures should be produced from the same evaluation rollouts used in the tables. For each selected rollout, the Rerun visualization should include:

1. top-down trajectory in the fixed map,
2. drone pose and direction,
3. six obstacle pillars,
4. Sun Glare anchor and glare cone or projected mask,
5. degraded depth image,
6. quality map \(Q_t\),
7. invalid or hole map,
8. local glare mask \(M_{\mathrm{glare}}\),
9. time series of power, exposure, gain, speed, fill rate, and local glare quality.

The intended visual evidence is not that the 3D viewer becomes physically bright like a photorealistic renderer. The intended evidence is that the simulation state contains the correct scene-local degradation: the glare mask appears when the drone enters the backlit region, depth reliability decreases locally, and the learned camera parameters change near the same event.

### 4.7 Main Quantitative Tables

#### Table 1: Base Scene Navigation

| Method | Success ↑ | Collision ↓ | Min Clearance ↑ | Time to Goal ↓ | Path Length ↓ | Control Effort ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Ours: Differentiable Active Depth | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Fixed-Camera Depth Policy | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Non-Differentiable Active Depth | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Heuristic Auto-Exposure | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Ego-Planner Style Baseline | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |

Expected interpretation: In the base scene, the proposed active sensor policy should match or exceed fixed-camera navigation while not producing unnecessary high energy usage.

#### Table 2: Sun Glare Scene Navigation

| Method | Success ↑ | Collision ↓ | Stop-before-glare Rate ↓ | Local Glare Quality ↑ | Local Glare Invalid ↓ | Time to Goal ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Ours: Differentiable Active Depth | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Fixed-Camera Depth Policy | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Non-Differentiable Active Depth | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Heuristic Auto-Exposure | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Ego-Planner Style Baseline | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |

Expected interpretation: Under backlit degradation, fixed-camera and modular baselines may either lose local depth reliability or become conservative. The differentiable active method should improve local glare quality and maintain navigation success.

#### Table 3: Camera Adaptation in Sun Glare

| Method | Power Mean | Power Std | Power Max | Exposure Mean | Exposure Std | Gain Mean | Energy Proxy | Blur Proxy | Noise Proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Fixed-Camera | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Non-Differentiable Active | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Heuristic AE | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |

Expected interpretation: The proposed method should show environment-correlated camera changes rather than constant parameters. Importantly, the desired result is not simply maximal power. The result should show a trade-off between local reliability, energy, blur, and noise.

#### Table 4: Event-Aligned Camera Response in Sun Glare

| Method | \(\Delta\) Power | \(\Delta\) Exposure | \(\Delta\) Gain | \(\Delta\) Local Quality | \(\Delta\) Fill Rate | Success ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Ours | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Fixed-Camera | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Non-Differentiable Active | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |
| Heuristic AE | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` |

Expected interpretation: The strongest evidence for active perception is not a large global standard deviation, but a positive event-aligned improvement in local quality and navigation success.

### 4.8 Ablation Studies

#### Ablation A: Remove Differentiable Sensor Gradient

Detach \(D_t\) and \(Q_t\) with respect to \(c_t\). This tests whether gradients through the sensor model are essential.

Expected result: camera adaptation becomes weaker or less correlated with glare entry.

#### Ablation B: Fixed Power

Set

\[
p^{\mathrm{cam}}_t=p_0
\]

and allow only exposure and gain to vary.

Expected result: if power is genuinely useful in Sun Glare, local glare quality or success rate should decrease relative to the full model.

#### Ablation C: Remove Local Glare Quality Loss

Set

\[
\lambda_{\mathrm{glare}}=0.
\]

Expected result: the policy may rely more on global fill rate or may learn to avoid entering the glare region. This ablation is important to determine whether local perception reliability improves the active sensing behavior.

#### Ablation D: Remove Camera State from Observation

Do not append \(2c_t-1\) to the low-dimensional state.

Expected result: camera control may become less stable because the policy does not observe the current sensor state.

#### Ablation E: Remove Sensor Randomization

Disable small training-time randomization of grouped sensor parameters. This tests whether the learned camera policy overfits to a narrow sensor model.

Expected result: training performance may improve, but robustness to calibration mismatch may decrease.

#### Ablation F: Remove Power Rescue in Sensor Physics

Remove the Sun Glare active-signal recovery term:

\[
r_{\mathrm{active}}p^{\mathrm{cam}}M_{\mathrm{glare}}.
\]

Expected result: if power no longer improves local quality under glare, the learned policy should stop increasing power. This is an important sanity check: it verifies that power changes are caused by the modeled physical utility of power, not by an accidental logging artifact or direct action reward.

#### Table 5: Ablation Summary

| Variant | Success ↑ | Collision ↓ | Local Glare Quality ↑ | \(\Delta\) Power | Energy Proxy ↓ | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Full model | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | Main result |
| No sensor gradient | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | Tests differentiability |
| Fixed power | `<待填>` | `<待填>` | `<待填>` | `0` | `<待填>` | Tests power utility |
| No local glare loss | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | Tests local reliability term |
| No camera state obs | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | Tests observability |
| No sensor randomization | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | Tests robustness |
| No power rescue physics | `<待填>` | `<待填>` | `<待填>` | `<待填>` | `<待填>` | Sanity check for power mechanism |

### 4.9 Qualitative Figures

The final paper should include the following figures.

#### Figure 1: System Overview

A block diagram:

\[
\text{CUDA geometry}
\rightarrow
\text{differentiable sensor}
\rightarrow
\text{depth preprocessing}
\rightarrow
\text{CNN-GRU policy}
\rightarrow
\text{flight and camera heads}
\rightarrow
\text{quadrotor dynamics}
\rightarrow
\text{loss}.
\]

#### Figure 2: Base Scene Trajectories

Show top-down trajectories for all compared methods. The proposed method should follow a smooth collision-free path around the six pillars.

#### Figure 3: Sun Glare Scene Visualization

Show:

1. drone trajectory,
2. glare region,
3. obstacle layout,
4. representative depth image,
5. quality map,
6. invalid mask.

#### Figure 4: Camera Parameters Around Glare Entry

Plot

\[
p^{\mathrm{cam}}_t,\quad e_t,\quad g_t
\]

against \(t-t_{\mathrm{entry}}\). This figure is central for demonstrating active sensing adaptation.

#### Figure 5: Local Quality and Navigation Outcome

Plot local glare quality and invalid rate over time. Compare fixed-camera and differentiable active depth policies.

#### Figure 6: Ablation of Power Utility

Show Sun Glare rollouts with and without the power-rescue term in the sensor model. If the proposed explanation is correct, removing the physical usefulness of power should reduce or eliminate learned power increase.

### 4.10 Expected Claims and Careful Interpretation

The strongest scientifically defensible claim is:

> A differentiable active depth model enables a policy to optimize sensing parameters jointly with motion, producing task-relevant camera adaptation in a backlit obstacle-avoidance scene.

The paper should avoid claiming:

1. that the simulator is a perfect D455 digital twin,
2. that the learned behavior is guaranteed to transfer without calibration,
3. that the local Sun Glare loss directly proves spontaneous emergence of a universal sensing strategy,
4. that high power is always the correct response.

Instead, the correct interpretation is:

1. The model captures physically meaningful causal trends.
2. The policy receives no direct action supervision for camera parameters.
3. Camera behavior emerges from the differentiable sensor model, navigation objective, and perception-reliability losses.
4. Sim-to-real transfer should be validated by D455 calibration data and real scene tests.

## 5. Discussion

### 5.1 Why Differentiability Matters

In a non-differentiable pipeline, the policy can still discover camera adaptation through trial and error, but the credit assignment problem is difficult. A change in power or exposure affects future depth, which affects future control, which affects future collisions or success. The gradient signal is indirect and delayed.

In the proposed pipeline, the local depth quality loss provides a direct gradient:

\[
\frac{\partial \mathcal{L}_{\mathrm{glare}}}{\partial p^{\mathrm{cam}}}
=
\frac{\partial \mathcal{L}_{\mathrm{glare}}}{\partial \bar{Q}_{\mathrm{glare}}}
\frac{\partial \bar{Q}_{\mathrm{glare}}}{\partial Q}
\frac{\partial Q}{\partial p^{\mathrm{cam}}}.
\]

Similarly,

\[
\frac{\partial \mathcal{L}_{\mathrm{blur}}}{\partial e}
=
2\|v\|^2 t_{\mathrm{exp}}
\frac{\partial t_{\mathrm{exp}}}{\partial e}.
\]

These gradients make it easier to learn sensor control policies that trade off reliability, blur, noise, and energy.

### 5.2 Why Local Reliability Is Used in Sun Glare

Global fill rate can hide localized failure. If only a small but task-critical region is affected by glare, the global fill rate may remain acceptable:

\[
F_{\mathrm{global}}
=
\frac{1}{HW}
\sum_{i,j} Q(i,j).
\]

However, obstacle avoidance depends on whether the relevant obstacle boundary is visible. Therefore local reliability is measured inside the glare mask:

\[
\bar{Q}_{\mathrm{glare}}
=
\frac{\sum M_{\mathrm{glare}}Q}{\sum M_{\mathrm{glare}}+\epsilon}.
\]

This term is not an action label. It does not encode "increase power." It encodes "maintain useful local depth reliability where the sensor is degraded." The policy can choose the register combination that best satisfies this objective under the differentiable sensor model.

### 5.3 Limitations

The current method has several limitations.

1. The geometric depth renderer is fast but simplified.
2. The D455-style sensor model is a surrogate, not a full physical simulation.
3. The Sun Glare scene is intentionally minimal and does not cover all real sunlight conditions.
4. The local quality loss is scene-aware in the current implementation. A more general formulation should use a unified scene-local reliability term across multiple physical degradation types.
5. The current policy uses direct action output; no model-predictive planning is used in this RAL version.
6. Real-world transfer requires calibration of exposure, gain, laser power, depth noise, invalid depth, and backlight response.

### 5.4 Path Toward Real-World Validation

A real-world validation should reproduce the minimal Sun Glare setup:

1. D455 mounted on a fixed drone or moving rig.
2. A simple obstacle near the backlit region.
3. A strong visible or infrared light source near the goal direction.
4. Controlled sweeps over laser power, exposure, and gain.
5. Measurement of depth fill, invalid rate, and local obstacle boundary reliability.

The simulator parameters should then be adjusted so that the qualitative response matches hardware:

\[
\frac{\partial \bar{Q}_{\mathrm{glare}}}{\partial p^{\mathrm{cam}}},
\quad
\frac{\partial \bar{Q}_{\mathrm{glare}}}{\partial e},
\quad
\frac{\partial \bar{Q}_{\mathrm{glare}}}{\partial g}
\]

should have the same signs and approximate relative magnitudes as the real sensor in the operating region.

### 5.5 Threats to Validity

This section lists the main threats to validity and how the paper should address them.

#### 5.5.1 Sensor Model Fidelity

The largest threat is that the differentiable sensor model may not match a real D455. A RealSense D455 depth image is produced by an active stereo system with proprietary firmware, stereo matching, projector pattern interaction, exposure control, invalidation logic, and filtering. The proposed model does not reproduce this full pipeline.

The paper should therefore avoid claiming pixel-level realism. The correct claim is causal realism in the operating regime:

\[
\mathrm{sign}
\left(
\frac{\partial Q_{\mathrm{sim}}}{\partial c}
\right)
\approx
\mathrm{sign}
\left(
\frac{\partial Q_{\mathrm{real}}}{\partial c}
\right),
\]

for

\[
c\in\{p^{\mathrm{cam}},e,g\}.
\]

The simulator is useful for policy learning if it preserves which camera adjustments improve or degrade local depth reliability near the decision boundary. This can be tested by collecting static D455 calibration data under backlight and comparing:

\[
\Delta Q_{\mathrm{real}}(p,e,g)
\quad\text{vs.}\quad
\Delta Q_{\mathrm{sim}}(p,e,g).
\]

#### 5.5.2 Scene-Specific Local Loss

The Sun Glare local quality loss uses a scene mask. A reviewer may argue that this injects scene knowledge. The correct response is:

1. The mask does not supervise camera actions.
2. The mask identifies where perception reliability matters in the synthetic scene.
3. The policy still chooses power, exposure, and gain through optimization.
4. The ablation without local quality loss is reported to quantify its effect.

In the RAL version, this is acceptable as a controlled diagnostic experiment. For a broader paper, the local reliability term should be generalized to task-relevant regions, such as obstacle boundaries, predicted collision corridors, or planner attention maps:

\[
\bar{Q}_{\mathrm{task}}
=
\frac{
\sum_{i,j} W_{\mathrm{task}}(i,j)Q(i,j)
}
{
\sum_{i,j} W_{\mathrm{task}}(i,j)+\epsilon
}.
\]

#### 5.5.3 Reward Hacking and Conservative Stopping

A learned policy may reduce loss by stopping before the difficult glare region. This is not successful navigation. The evaluation therefore includes stop-before-glare rate and time-to-goal. The training objective should also ensure that progress toward the goal remains important.

The paper should report failure modes explicitly:

1. collision inside glare,
2. stopping before glare,
3. oscillating near an obstacle,
4. reaching the goal with excessive energy,
5. changing camera parameters without perception improvement.

#### 5.5.4 Power Change Interpretation

The goal is not to show that power always increases. The goal is to show task-relevant adaptation. A power increase is meaningful only if:

\[
\Delta p>0,
\qquad
\Delta \bar{Q}_{\mathrm{glare}}>0,
\qquad
\Delta \mathrm{Success}>0,
\]

relative to baselines or ablations. If power changes but local quality and success do not improve, it should not be presented as evidence of active perception.

The ablation that removes power utility in the Sun Glare sensor physics is important because it checks whether learned power changes disappear when power no longer helps. This is a scientific control, not a cosmetic test.

#### 5.5.5 Sim-to-Real Transfer

Even if the simulation result is strong, real deployment requires:

1. mapping normalized exposure to D455 exposure microseconds,
2. mapping normalized power to D455 laser power register,
3. choosing a gain range that does not saturate or destabilize depth,
4. validating depth invalidation trends under backlight,
5. measuring latency and dropped frames,
6. enforcing safety limits for power, speed, and obstacle clearance.

The current paper may present real-world calibration as future work if hardware experiments are not yet complete. If hardware results are included, they should be reported as a separate validation section rather than mixed with simulation metrics.

### 5.6 Reproducibility Checklist

For the final submission, the following details should be included.

| Item | Required detail |
|---|---|
| Code branch | Commit hash and branch name |
| Configuration | Full `paper_final_full.args` snapshot |
| Hardware | GPU model, CUDA version, PyTorch version |
| Training budget | Number of iterations, batch size, rollout length, wall-clock time |
| Random seeds | Seeds for each reported run |
| Checkpoint selection | Last checkpoint or validation-selected checkpoint |
| Scene geometry | Start, goal, obstacle centers, obstacle dimensions |
| Sun Glare parameters | Sun anchor, zone gate, mask width, ambient add, active drop/recovery |
| Sensor semantics | Exposure mapping, gain mapping, power nominal value |
| Loss weights | All task, camera, depth, and glare coefficients |
| Evaluation scripts | Exact `eval.sh` command and checkpoint path |
| Visualization | Rerun export settings and representative rollout IDs |

This checklist is especially important because the paper studies a closed-loop system. Small changes in sensor loss, camera smoothing, or obstacle placement can alter the learned behavior.

## 6. Conclusion

This paper presents a differentiable active depth perception framework for quadrotor navigation. The method places a D455-style depth sensor model inside the end-to-end training loop, allowing a recurrent visuomotor policy to control both flight actions and active depth parameters. The differentiable sensor model captures key causal relationships among laser power, exposure, gain, ambient infrared, motion blur, depth noise, invalid pixels, and local quality. The policy is trained with navigation, collision avoidance, control smoothness, energy, blur, noise, fill-rate, and local Sun Glare reliability losses.

The proposed RAL version intentionally focuses on a minimal experimental setting: a fixed base obstacle map and a Sun Glare variant. This narrow scope is useful because it isolates the main scientific question: whether differentiable active perception can produce task-relevant sensor adaptation without hand-labeling camera actions. The experimental framework compares the method against fixed-camera policies, non-differentiable active sensing, heuristic auto-exposure, and ego-planner style baselines. Quantitative results remain to be filled after training and evaluation.

The expected outcome is not simply that the policy increases power. The desired behavior is more precise: the policy should regulate power, exposure, and gain only when such regulation improves local depth reliability and navigation performance under the energy, blur, and noise trade-offs imposed by the loss. If validated in both simulation and D455 hardware tests, this would support the broader claim that active sensor registers should be treated as part of the robot control loop rather than as fixed camera settings.

## Appendix A. Current Configuration Summary

| Parameter | Value |
|---|---|
| `--scenarios` | `sun_glare` |
| `--batch_size` | `150` |
| `--num_iters` | `5000` |
| `--timesteps` | `80` |
| `--depth_width` | `64` |
| `--depth_height` | `48` |
| `--depth_nn_width` | `32` |
| `--depth_nn_height` | `24` |
| `--depth_min_valid` | `0.3` |
| `--depth_max_range` | `6.0` |
| `--include_camera_state_in_obs` | enabled |
| `--diff_sensor_impl` | `diff_depth=python` |
| `--use_dmpc` | disabled |
| `--policy_output_intent` | disabled |
| `--enable_teacher_student_training` | disabled |
| `--tbptt_enable` | disabled |
| `--coef_v` | `2.5` |
| `--coef_obj_avoidance` | `4.0` |
| `--coef_collide` | `10.0` |
| `--coef_cam_smooth` | `100` |
| `--coef_power_reg` | `100` |
| `--cam_power_reg_deadband` | `0.18` |
| `--cam_power_nominal` | `0.416667` |
| `--cam_power_penalty_threshold` | `0.416667` |
| `--coef_diff_depth_power` | `20` |
| `--coef_diff_depth_blur` | `0.1` |
| `--coef_diff_depth_noise` | `5` |
| `--coef_diff_depth_fill` | `30` |
| `--diff_depth_min_fill_rate` | `0.25` |
| `--coef_sun_glare_local_quality` | `30` |
| `--sun_glare_local_quality_target` | `0.1` |

## Appendix B. Symbols

| Symbol | Meaning |
|---|---|
| \(x_t\) | Robot state |
| \(p_t\) | Robot position |
| \(v_t\) | Robot velocity |
| \(R_t\) | Robot attitude |
| \(c_t\) | Camera state |
| \(p^{\mathrm{cam}}_t\) | Normalized laser power |
| \(e_t\) | Normalized exposure |
| \(g_t\) | Normalized gain |
| \(D_t\) | Degraded depth observation |
| \(Q_t\) | Differentiable depth quality map |
| \(Z_t\) | Ideal geometric depth |
| \(\mathcal{M}\) | Obstacle map |
| \(\pi_\theta\) | Neural policy |
| \(u_t\) | Flight control action |
| \(h_t\) | Recurrent hidden state |
| \(M_{\mathrm{glare}}\) | Sun Glare image mask |
| \(F_t\) | Soft depth fill rate |
| \(\bar{Q}_{\mathrm{glare}}\) | Local glare-region quality |

## Appendix C. Result Placeholders to Fill

Before submission, fill the following:

1. Training curves for total loss and loss shares.
2. Base scene success and collision table.
3. Sun Glare success and collision table.
4. Camera parameter statistics.
5. Glare-entry aligned power/exposure/gain plots.
6. Depth, quality, invalid mask, and scene mask visualizations.
7. Baseline implementation details for fixed-camera, non-differentiable active sensing, heuristic AE, and ego-planner.
8. Real D455 calibration details if hardware validation is included.

## Appendix D. Suggested Result Paragraph Templates

The following paragraphs can be completed after the experiments are run.

### D.1 Base Scene Result Template

In the base scene, the proposed differentiable active depth policy achieved a success rate of `<待填>` with a collision rate of `<待填>`. Its minimum clearance was `<待填>`, and its average time to goal was `<待填>`. Compared with the fixed-camera policy, the proposed method produced `<待填>` navigation performance while maintaining camera parameters near the nominal operating region. This indicates that adding active camera control does not destabilize navigation in clean conditions.

### D.2 Sun Glare Result Template

In the Sun Glare scene, the differentiable active depth policy achieved a success rate of `<待填>`, compared with `<待填>` for the fixed-camera policy and `<待填>` for the non-differentiable active baseline. The local glare quality increased from `<待填>` to `<待填>` around glare entry, while the stop-before-glare rate decreased from `<待填>` to `<待填>`. These results suggest that differentiable sensor feedback improves task-relevant perception under backlit degradation.

### D.3 Camera Adaptation Template

Event-aligned analysis shows that the policy changed camera parameters near the glare-entry event. The mean power change was \(\Delta p=\)<`待填`>, the mean exposure change was \(\Delta e=\)<`待填`>, and the mean gain change was \(\Delta g=\)<`待填`>. These changes were accompanied by a local quality improvement of `<待填>` and a success-rate improvement of `<待填>`. Therefore, the camera response was not merely global variation; it was temporally aligned with the physical degradation region.

### D.4 Ablation Template

Removing the differentiable sensor gradient reduced `<待填>` and weakened the event-aligned camera response. Fixing power reduced local glare quality by `<待填>`, indicating that power contributed to perception recovery in this scene. Removing the local glare quality term caused `<待填>`, suggesting that global fill rate alone was insufficient to focus the policy on the task-relevant degraded region.

## Appendix E. Minimal LaTeX Conversion Notes

For an IEEE RAL submission, this Markdown draft can be converted into LaTeX with the following structure:

1. `\section{Introduction}`
2. `\section{Related Work}`
3. `\section{Method}`
4. `\section{Experiments}`
5. `\section{Discussion}`
6. `\section{Conclusion}`

Long implementation tables and result templates should be moved to an appendix or supplementary material. The main paper should keep only:

1. the system overview figure,
2. the sensor model equations most relevant to camera control,
3. one algorithm box,
4. two main result tables,
5. one event-aligned camera response plot,
6. one ablation table.

The final RAL version should compress the method section while preserving the critical equations:

\[
D,Q=\mathcal{S}_{\phi}(Z,c,x),
\qquad
c_{t+1}=0.7c_t+0.3\hat{c}_{t+1},
\]

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{task}}
+\mathcal{L}_{\mathrm{cam}}
+\mathcal{L}_{\mathrm{depth}}
+\mathcal{L}_{\mathrm{glare}}.
\]

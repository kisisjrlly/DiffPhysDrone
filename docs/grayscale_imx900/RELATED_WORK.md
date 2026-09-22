# Related Work and Open-Source Survey

> Purpose: give future Codex/developers the research context and prevent unsupported novelty claims.

## 1. Differentiable imaging / sensor models

### End2endImaging

Repository:
https://github.com/vccimaging/End2endImaging

Useful component:
`end2end_imaging/sensor/mono_sensor.py`

Why it matters:

- PyTorch sensor abstraction;
- monochrome response;
- read-noise and shot-noise model;
- black level;
- bit depth;
- gamma/ISP blocks;
- intended for differentiable end-to-end imaging research.

Important limitation for DiffPhysDrone:

- it is not a pre-calibrated IMX900 digital twin;
- its current `MonoSensor` does not directly provide the exact online exposure/gain actuator model required here;
- hard `round` / `clip` operations in realistic forward simulation are not sufficient by themselves for stable task-gradient camera control.

Use it as a structural/reference implementation, not as ground-truth IMX900 behavior.

### Differentiable rendering frameworks

Potential references:

- Mitsuba 3 / Dr.Jit: https://www.mitsuba-renderer.org/
- redner: https://github.com/BachiLi/redner
- nvdiffrast: https://github.com/NVlabs/nvdiffrast
- PyTorch3D: https://github.com/facebookresearch/pytorch3d
- Kornia: https://github.com/kornia/kornia

For this project, a heavy differentiable renderer is not required in v1 because the main gradient of interest is with respect to camera parameters, not scene geometry.

## 2. Active exposure control for robotics

### Active Exposure Control for Robust Visual Odometry in HDR Environments

Project:
https://github.com/uzh-rpg/active_camera_exposure_control

This line of work is important as a classical/robotics baseline. It demonstrates that exposure control matters for downstream geometric vision and provides exposure-selection strategies based on image statistics/gradients.

Implication for our novelty:
“camera exposure matters for robotics” is not novel.

### Noise-Aware Camera Exposure Control

Project:
https://github.com/UkcheolShin/Noise-AwareCameraExposureControl

Relevant ideas:

- joint exposure/gain control;
- noise-aware image-quality reasoning;
- real-camera data and calibration-oriented evaluation.

Use as a reference for:
- gain/exposure action definition;
- real-camera sweep protocol;
- classical learned image-quality-based controller baseline.

### Reinforcement-learning exposure control

Project:
https://github.com/shuyanguni/drl_exposure_ctrl

Relevant because exposure can be learned without analytical camera gradients.

Implication:
A learned camera policy alone is not the contribution. The differentiable sensor-gradient path and downstream closed-loop navigation experiment must be isolated.

## 3. Direct task-driven camera control: important novelty boundary

### JOCA — Task-Driven Joint Optimisation of Camera Hardware and Adaptive Camera Control Algorithms

Paper:
https://openaccess.thecvf.com/content/WACV2026/html/Yan_JOCA_Task-Driven_Joint_Optimisation_of_Camera_Hardware_and_Adaptive_Camera_WACV_2026_paper.html

Authors: Chengyang Yan, Mitch Bryson, Donald G. Dansereau. WACV 2026.

JOCA is the closest known prior work found in the 2026 literature search. It jointly optimizes fixed camera hardware parameters, an adaptive camera-control network, and downstream perception. Its experiments explicitly include dynamic exposure/gain, low light, and motion blur; it introduces DF-Grad to handle non-differentiable image effects.

This changes the novelty boundary substantially. DiffPhysDrone should not present generic “task-driven differentiable exposure/gain control” as new.

### TaCOS — Task-Specific Camera Optimization with Simulation

Paper:
https://openaccess.thecvf.com/content/WACV2025/html/Yan_TaCOS_Task-Specific_Camera_Optimization_with_Simulation_WACV_2025_paper.html

TaCOS (WACV 2025) is another important precursor on simulation-based task-specific camera co-design. It focuses on camera design rather than the exact closed-loop UAV problem here, but it belongs in the final related-work chain.

Therefore this project must **not** claim:

- first differentiable task-driven camera control;
- first downstream-task optimization of exposure/gain;
- first adaptive camera settings via differentiable image formation.

A defensible project-specific contribution should instead center on some combination of:

1. closed-loop **aerial navigation** rather than static/per-frame vision;
2. camera action affecting future observations while flight dynamics evolve;
3. a real-camera-calibrated monochrome IMX900 model;
4. onboard deployment on a small quadrotor;
5. controlled differentiable-vs-detached causal ablation in navigation;
6. exposure/SNR versus motion-blur trade-off under drone motion.

Before manuscript submission, re-run a literature search because this area is moving quickly.

## 4. Camera exposure and visual localization/SLAM

Also survey before writing the final paper:

- exposure control for visual odometry/SLAM;
- HDR-aware camera control;
- photometric calibration and camera response estimation;
- auto-exposure for robotics;
- active vision sensor-parameter selection.

These are necessary to correctly position the navigation contribution.

## 5. Real-camera image formation / noise calibration

Useful concepts/keywords:

- photon transfer curve (PTC);
- temporal dark noise;
- conversion gain;
- read noise;
- shot noise;
- black level;
- full-well/saturation;
- EMVA 1288;
- camera response function;
- motion transfer function / edge spread for blur.

The real-camera calibration in this project should be described as sensor characterization, not as a complete CMOS device simulation.

## 6. Hardware references

### e-con Systems e-CAM37M_CUONX

Official product page:
https://www.e-consystems.com/nvidia-cameras/jetson-orin-nx-cameras/sony-imx900-global-shutter-monochrome-camera.asp

Relevant advertised properties to verify against the exact purchased SKU and driver release:

- Sony IMX900;
- monochrome;
- global shutter;
- MIPI CSI-2;
- Jetson Orin NX / Orin Nano support;
- exposure/gain controls;
- RAW output modes;
- Linux/V4L2 integration.

Do not hard-code numerical ranges until the exact hardware/driver reports them.

### Sony IMX900

Use Sony Semiconductor official documentation/product pages for sensor-level specifications where available.

Do not infer camera-module behavior solely from the bare sensor datasheet; the e-con module and driver define the actual usable control interface.

### DAMIAO DM-ORIN NX V2.X carrier

Critical facts from the user's hardware manual:

- target carrier for Jetson Orin NX/Nano;
- 39.2 g;
- 12–28 V input, explicitly supports 6S;
- two 22-pin 0.5-mm FPC CIS connectors;
- camera circuitry described as matching the original/reference arrangement;
- MIPI data lanes, camera I2C, MCLK, PWDN and 3.3 V are exposed.

Compatibility with e-con's exact camera cable/DT overlay still needs confirmation.

## 7. What to borrow vs what not to borrow

Borrow:

- MonoSensor structure/noise terminology from End2endImaging;
- exposure-control baselines from UZH;
- calibration ideas from noise-aware camera-control work;
- task-driven differentiable optimization framing from recent JOCA-like work, with proper attribution;
- standard PTC/noise characterization practice.

Do not blindly copy:

- arbitrary noise coefficients from another camera;
- RGB ISP assumptions for a monochrome sensor;
- exposure/gain ranges from unrelated sensors;
- an external renderer if the current CUDA renderer is sufficient;
- task losses that turn the method into image-quality optimization rather than navigation optimization.

## 8. Novelty statement to use internally

A safe internal working hypothesis is:

> We study real-camera-calibrated differentiable exposure/gain control for closed-loop monocular quadrotor navigation, and isolate the value of the sensor gradient through a matched differentiable-vs-detached experiment before deploying the policy on an IMX900-based onboard camera.

This is a hypothesis/positioning statement, not a final novelty claim. Revalidate against literature before submission.

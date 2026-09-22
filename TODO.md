# Grayscale / IMX900 TODO

This file tracks the active branch only. The D455 research line is intentionally
not maintained here.

## P0 — local integration gate

- [ ] Rebuild CUDA extension after removal of D455 kernels:
  `pip install -e src` in the `mappo-mpc` environment.
- [ ] Run all grayscale unit tests.
- [ ] Run `python tools/test_gray_env_render.py`.
- [ ] Visually inspect nominal/dark/bright smoke images.
- [ ] Confirm no NaN/Inf and sensible saturation/dark fractions.

## P1 — prove monocular navigation

- [ ] Train `gray_gate_fixed`.
- [ ] Train matched `gray_gate_blind`.
- [ ] Require fixed-gray navigation to materially outperform zero-image control.
- [ ] Verify the policy reacts to randomized slit location rather than only
  following target-state information.

## P2 — construct the active-camera trade-off

- [x] Add nominal/dark/bright illumination.
- [x] Add dark-to-bright and bright-to-dark transitions.
- [x] Keep exposure-dependent motion blur in the camera model.
- [ ] Tune lighting ranges so no single fixed exposure/gain dominates all cases.
- [ ] Add speed-conditioned analysis and transition plots.
- [x] Add classical mean-intensity AE baseline.
- [x] Add a gradient/image-detail heuristic AE baseline.

## P3 — isolate the differentiable-sensor contribution

- [x] Support `sensor_grad_mode=full|detached`.
- [x] Add frozen-flight `train_camera_only`.
- [x] Add matched configs `gray_camera_full` and `gray_camera_detached`.
- [ ] Start both from the same successful flight checkpoint.
- [ ] Log/verify camera-branch gradient norm.
- [ ] Compare success/collision by illumination and speed.
- [ ] Verify learned exposure/gain trajectories are physically interpretable.
- [ ] Reject degenerate solutions that pin both controls to a bound.

## P4 — real IMX900

- [ ] Confirm DAMIAO CSI pinout/device-tree compatibility with e-con.
- [ ] Record JetPack/L4T/e-con driver versions.
- [ ] Enumerate real V4L2/e-con exposure/gain controls and effective ranges.
- [ ] Measure command-to-effective-frame latency.
- [ ] Implement `tools/realflight/imx900_camera_node.py`.
- [ ] Add dark-frame / exposure / gain / PTC / motion-blur calibration tools.
- [ ] Fit simulator parameters from measured data.
- [ ] Validate held-out real-vs-sim response curves.

## P5 — paper-grade experiments

- [ ] Fixed nominal.
- [ ] Random static.
- [ ] Classical AE.
- [ ] Learned detached.
- [ ] Differentiable camera policy.
- [ ] Optional oracle local grid search.
- [ ] Real-drone deployment after bench and tethered safety stages.

# Tools Overview — Grayscale / IMX900 Branch

This branch no longer contains the legacy D455 active-depth calibration,
teacher/relabel, depth-probe, or sensor-semantics tools.

## Current validation flow

Before training:

```bash
python -m pytest -q \
  tests/test_differentiable_gray_camera.py \
  tests/test_ideal_gray_renderer.py \
  tests/test_model_gray_mode.py \
  tests/test_gray_rollout_helpers.py

python tools/test_gray_env_render.py
```

The smoke test writes nominal/dark/bright ideal and sensor images under
`logs/gray_smoke/`.

## Training sequence

```bash
# 1. Prove grayscale navigation uses vision.
TASK=gray_gate_fixed bash run.sh
TASK=gray_gate_blind bash run.sh

# 2. Stress the fixed camera under changing illumination.
TASK=gray_gate_mixed_fixed bash run.sh

# 3. Evaluate classical AE on the same flight checkpoint.
CONFIG=configs/gray_camera_mean_ae.args CKPT=<flight_ckpt> bash eval.sh
CONFIG=configs/gray_camera_gradient_ae.args CKPT=<flight_ckpt> bash eval.sh

# 4. Train camera-only from a successful flight checkpoint.
TASK=gray_camera_full \
RUN_EXTRA_ARGS="--resume checkpoint/<flight>/checkpointXXXX.pth" \
bash run.sh

# 5. Matched no-sensor-gradient ablation.
TASK=gray_camera_detached \
RUN_EXTRA_ARGS="--resume checkpoint/<flight>/checkpointXXXX.pth" \
bash run.sh
```

The next tool family to grow is `tools/grayscale_calibration/`, which maps the
real e-con IMX900 exposure/gain/noise/blur/latency behavior into the
differentiable camera model.

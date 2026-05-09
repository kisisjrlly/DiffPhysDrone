#!/bin/bash
set -euo pipefail

# DAgger-style camera relabeling:
# 1. Roll out the current pretrained learned camera online.
# 2. Re-label those visited states with the differentiable teacher optimizer.
# 3. Pretrain the camera head again from the current pretrained checkpoint.
# 4. Eval the new camera checkpoint against the old pretrained and flight-only checkpoints.

PY=${PYTHON_BIN:-/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python}
CONFIG=${CONFIG:-configs/slit_active_sensing.args}
BASE_PRETRAIN_CKPT=${BASE_PRETRAIN_CKPT:-checkpoint/closed_loop_teacher_camera_policy_v3d_full/camera_head_pretrained_best.pth}
FLIGHTONLY_CKPT=${FLIGHTONLY_CKPT:-checkpoint/2026-05-07-00-58-11/checkpoint0014.pth}
WORK_DIR=${WORK_DIR:-paper/experiment/results/closed_loop_teacher_camera_policy_v3d_dagger_full}
PRETRAIN_DIR=${PRETRAIN_DIR:-checkpoint/closed_loop_teacher_camera_policy_v3d_dagger_full}
EVAL_DIR=${EVAL_DIR:-paper/experiment/results/pretrain_dagger_vs_flightonly_eval_20260507}
STAGE=${STAGE:-all}
DEVICE=${DEVICE:-cuda}

if [ ! -f "$BASE_PRETRAIN_CKPT" ]; then
  echo "[error] base pretrained checkpoint not found: $BASE_PRETRAIN_CKPT" >&2
  exit 1
fi
if [ ! -f "$FLIGHTONLY_CKPT" ]; then
  echo "[error] flight-only checkpoint not found: $FLIGHTONLY_CKPT" >&2
  exit 1
fi

"$PY" -u tools/run_camera_teacher_pipeline.py \
  --stage "$STAGE" \
  --teacher_source closed_loop_diffopt \
  --config "$CONFIG" \
  --flight_checkpoint "$BASE_PRETRAIN_CKPT" \
  --work_dir "$WORK_DIR" \
  --pretrain_dir "$PRETRAIN_DIR" \
  --scenarios glare specular dark \
  --rollouts_per_scene 4 \
  --collect_batch_size 12 \
  --timesteps 80 \
  --teacher_steps 120 \
  --teacher_lr 0.08 \
  --teacher_every 1 \
  --rollout_camera_mode learned \
  --no-teacher_camera_ema \
  --coef_nominal_when_healthy 0.075 \
  --nominal_fill_margin 0.25 \
  --coef_diff_depth_fill 50 \
  --coef_diff_depth_power 0.0 \
  --coef_diff_depth_blur 0.00015 \
  --coef_diff_depth_noise 0.0007 \
  --coef_cam_smooth 0.005 \
  --pretrain_epochs 120 \
  --pretrain_batch_size 8 \
  --pretrain_lr 5e-4 \
  --pretrain_weight_decay 0.0 \
  --temporal_smooth 0.0 \
  --eval_modes learned \
  --eval_episodes 30 \
  --device "$DEVICE" \
  "$@"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "eval" ]; then
  "$PY" -u tools/run_checkpoint_eval_suite.py \
    --out_dir "$EVAL_DIR" \
    --methods pretrained dagger flightonly \
    --episodes_per_scene 100 \
    --pretrained_ckpt "$BASE_PRETRAIN_CKPT" \
    --dagger_ckpt "$PRETRAIN_DIR/camera_head_pretrained_best.pth" \
    --flightonly_ckpt "$FLIGHTONLY_CKPT"

  "$PY" -u tools/diagnose_pretrain_camera_trace.py \
    --eval_dir "$EVAL_DIR" \
    --dataset "$WORK_DIR/camera_teacher_dataset.pt" \
    --pretrained_ckpt "$PRETRAIN_DIR/camera_head_pretrained_best.pth" \
    --offline_method_label dagger
fi

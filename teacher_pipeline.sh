#!/bin/bash
set -euo pipefail

PY=${PYTHON_BIN:-/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python}
CONFIG=${CONFIG:-configs/slit_active_sensing.args}
FLIGHT_CKPT=${FLIGHT_CKPT:?set FLIGHT_CKPT=checkpoint/.../checkpointXXXX.pth}
WORK_DIR=${WORK_DIR:-paper/experiment/results/camera_teacher_pipeline}
PRETRAIN_DIR=${PRETRAIN_DIR:-checkpoint/camera_teacher_pipeline}
STAGE=${STAGE:-all}
DEVICE=${DEVICE:-cuda}

"$PY" -u tools/run_camera_teacher_pipeline.py \
  --stage "$STAGE" \
  --teacher_source rollout_local \
  --config "$CONFIG" \
  --flight_checkpoint "$FLIGHT_CKPT" \
  --work_dir "$WORK_DIR" \
  --pretrain_dir "$PRETRAIN_DIR" \
  --scenarios glare specular dark \
  --rollouts_per_scene 12 \
  --collect_batch_size 12 \
  --timesteps 80 \
  --teacher_steps 50 \
  --teacher_lr 0.10 \
  --coef_nominal_when_healthy 0.5 \
  --nominal_fill_margin 0.12 \
  --pretrain_epochs 40 \
  --pretrain_batch_size 8 \
  --pretrain_lr 2e-4 \
  --eval_modes fixed randfix learned learned_detached \
  --eval_episodes 300 \
  --device "$DEVICE" \
  "$@"

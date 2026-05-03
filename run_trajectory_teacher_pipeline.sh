#!/bin/bash
set -euo pipefail

PY=${PYTHON_BIN:-/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python}
CONFIG=${CONFIG:-configs/slit_active_sensing.args}
FLIGHT_CKPT=${FLIGHT_CKPT:-}
WORK_DIR=${WORK_DIR:-paper/experiment/results/trajectory_teacher_camera_policy_randfix_20260503}
PRETRAIN_DIR=${PRETRAIN_DIR:-checkpoint/trajectory_teacher_camera_policy_randfix_20260503}
STAGE=${STAGE:-all}
DEVICE=${DEVICE:-cuda}

if [ -z "$FLIGHT_CKPT" ]; then
  echo "[error] set FLIGHT_CKPT=checkpoint/.../checkpointXXXX.pth" >&2
  exit 1
fi

"$PY" -u tools/run_camera_teacher_pipeline.py \
  --stage "$STAGE" \
  --teacher_source trajectory_diffopt \
  --config "$CONFIG" \
  --flight_checkpoint "$FLIGHT_CKPT" \
  --work_dir "$WORK_DIR" \
  --pretrain_dir "$PRETRAIN_DIR" \
  --scenarios glare specular dark \
  --slots left right \
  --rollouts_per_scene 12 \
  --trajectory_xs=-1.20,-0.90,-0.60,-0.35,-0.18,-0.05,0.10,0.35,0.70,1.05,1.35 \
  --trajectory_x_jitter 0.035 \
  --teacher_steps 120 \
  --teacher_lr 0.08 \
  --diffopt_random_restarts 4 \
  --diffopt_randfix_k 24 \
  --pretrain_epochs 80 \
  --pretrain_batch_size 8 \
  --pretrain_lr 1e-4 \
  --temporal_smooth 0.005 \
  --eval_modes fixed randfix learned \
  --eval_episodes 90 \
  --device "$DEVICE" \
  "$@"

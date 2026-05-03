#!/bin/bash
set -euo pipefail

PY=${PYTHON_BIN:-/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python}
CONFIG=${CONFIG:-configs/paper_final_full.args}
FLIGHT_CKPT=${FLIGHT_CKPT:-checkpoint/2026-05-03-15-28-53/checkpoint0014.pth}
TEACHER_SOURCE=${TEACHER_SOURCE:-closed_loop_diffopt}
WORK_DIR=${WORK_DIR:-paper/experiment/results/closed_loop_teacher_camera_policy_randfix_20260504}
PRETRAIN_DIR=${PRETRAIN_DIR:-checkpoint/closed_loop_teacher_camera_policy_randfix_20260504}
STAGE=${STAGE:-all}
DEVICE=${DEVICE:-cuda}

"$PY" -u tools/run_camera_teacher_pipeline.py \
  --stage "$STAGE" \
  --teacher_source "$TEACHER_SOURCE" \
  --config "$CONFIG" \
  --flight_checkpoint "$FLIGHT_CKPT" \
  --work_dir "$WORK_DIR" \
  --pretrain_dir "$PRETRAIN_DIR" \
  --scenarios glare specular dark \
  --slots left right \
  --rollouts_per_scene 12 \
  --rollout_camera_mode fixed_random_static \
  --no-teacher_camera_ema \
  --trajectory_xs=-1.20,-0.90,-0.60,-0.35,-0.18,-0.05,0.10,0.35,0.70,1.05,1.35 \
  --trajectory_x_jitter 0.035 \
  --teacher_steps 120 \
  --teacher_lr 0.08 \
  --diffopt_random_restarts 4 \
  --diffopt_randfix_k 24 \
  --pretrain_epochs 180 \
  --pretrain_batch_size 8 \
  --pretrain_lr 1e-3 \
  --pretrain_weight_decay 0.0 \
  --temporal_smooth 0.0 \
  --eval_modes fixed randfix learned \
  --eval_episodes 90 \
  --device "$DEVICE" \
  "$@"

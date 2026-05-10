# RESUME_CKPT=checkpoint/closed_loop_teacher_camera_policy_v3d_full/camera_head_pretrained_best.pth \
# RUN_TRAIN=1 MODES="flightonly" bash run_train_modes.sh


# bash run_train_modes.sh


# RUN_TRAIN=1 MODES="fix randfix nondiff zero" bash run_train_modes.sh

# RESUME_CKPT=checkpoint/closed_loop_teacher_camera_policy_semantics_v3_dagger/camera_head_pretrained_best.pth \
# RUN_TRAIN=1 MODES="flightonly" bash run_train_modes.sh



# flightonly: checkpoint/2026-05-07-00-58-11/checkpoint0014.pth
# randfix: checkpoint/2026-05-07-01-26-30/checkpoint0014.pth
# nondiff: checkpoint/2026-05-07-01-57-25/checkpoint0014.pth


PY=/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python
TAG=light_v3d_t60_20260509_fix_20260509_223206
FIXED_CKPT=checkpoint/2026-05-09-22-32-06/checkpoint0014.pth


$PY -u tools/run_camera_teacher_pipeline.py \
  --stage all \
  --teacher_source closed_loop_diffopt \
  --config configs/slit_active_sensing.args \
  --flight_checkpoint $FIXED_CKPT \
  --work_dir paper/experiment/results/closed_loop_teacher_camera_policy_${TAG} \
  --pretrain_dir checkpoint/closed_loop_teacher_camera_policy_${TAG} \
  --scenarios glare specular dark \
  --rollouts_per_scene 4 \
  --collect_batch_size 12 \
  --timesteps 60 \
  --teacher_steps 120 \
  --teacher_lr 0.08 \
  --teacher_every 1 \
  --rollout_camera_mode fixed \
  --teacher_camera_ema \
  --coef_nominal_when_healthy 0.075 \
  --nominal_fill_margin 0.25 \
  --pretrain_epochs 180 \
  --pretrain_batch_size 8 \
  --pretrain_lr 1e-3 \
  --pretrain_weight_decay 0.0 \
  --temporal_smooth 0.0 \
  --eval_modes fixed learned \
  --eval_episodes 90 \
  --device cuda \
  --force_collect \
  --force_pretrain \
  --coef_diff_depth_fill 50 \
  --coef_diff_depth_power 0.0 \
  --coef_diff_depth_blur 0.00015 \
  --coef_diff_depth_noise 0.0007 \
  --coef_cam_smooth 0.005


BASE_PRETRAIN_CKPT=checkpoint/closed_loop_teacher_camera_policy_${TAG}/camera_head_pretrained_best.pth
DAGGER_TAG=${TAG}_dagger
RESUME_CKPT=checkpoint/closed_loop_teacher_camera_policy_${DAGGER_TAG}/camera_head_pretrained_best.pth

$PY -u tools/run_camera_teacher_pipeline.py \
  --stage all \
  --teacher_source closed_loop_diffopt \
  --config configs/slit_active_sensing.args \
  --flight_checkpoint $BASE_PRETRAIN_CKPT \
  --work_dir paper/experiment/results/closed_loop_teacher_camera_policy_${DAGGER_TAG} \
  --pretrain_dir checkpoint/closed_loop_teacher_camera_policy_${DAGGER_TAG} \
  --scenarios glare specular dark \
  --rollouts_per_scene 4 \
  --collect_batch_size 12 \
  --timesteps 60 \
  --teacher_steps 120 \
  --teacher_lr 0.08 \
  --teacher_every 1 \
  --rollout_camera_mode learned \
  --no-teacher_camera_ema \
  --coef_nominal_when_healthy 0.075 \
  --nominal_fill_margin 0.25 \
  --pretrain_epochs 120 \
  --pretrain_batch_size 8 \
  --pretrain_lr 5e-4 \
  --pretrain_weight_decay 0.0 \
  --temporal_smooth 0.0 \
  --eval_modes learned \
  --eval_episodes 30 \
  --device cuda \
  --coef_diff_depth_fill 50 \
  --coef_diff_depth_power 0.0 \
  --coef_diff_depth_blur 0.00015 \
  --coef_diff_depth_noise 0.0007 \
  --coef_cam_smooth 0.005


RESUME_CKPT=checkpoint/closed_loop_teacher_camera_policy_light_v3d_t60_20260509_dagger/camera_head_pretrained_best.pth \
MODES="flightonly" \ 
RUN_TRAIN=1 \
bash run_train_modes.sh

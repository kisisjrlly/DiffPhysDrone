# RESUME_CKPT=checkpoint/closed_loop_teacher_camera_policy_v3d_full/camera_head_pretrained_best.pth \
# RUN_TRAIN=1 MODES="flightonly" bash run_train_modes.sh


# bash run_train_modes.sh


RUN_TRAIN=1 MODES="fix randfix nondiff zero" bash run_train_modes.sh

RESUME_CKPT=checkpoint/closed_loop_teacher_camera_policy_semantics_v3_dagger/camera_head_pretrained_best.pth \
RUN_TRAIN=1 MODES="flightonly" bash run_train_modes.sh



# flightonly: checkpoint/2026-05-07-00-58-11/checkpoint0014.pth
# randfix: checkpoint/2026-05-07-01-26-30/checkpoint0014.pth
# nondiff: checkpoint/2026-05-07-01-57-25/checkpoint0014.pth
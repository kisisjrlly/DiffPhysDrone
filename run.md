# For multi-agemt
python main_cuda.py $(cat configs/multi_agent.args)
# For single-agemt，basic mode
python main_cuda.py $(cat configs/single_agent.args)
# For single-agemt，camera mode
python main_cuda.py $(cat configs/single_agent.args)
# For wall-slit (narrow gap sideways flight)
python main_cuda.py $(cat configs/wall_slit.args)
# Evaluate wall-slit
python eval_wall_slit.py --resume checkpoint0004.pth --ellipsoid_collision --num_episodes 200
tensorboard --logdir runs
python main_cuda.py $(cat configs/single_agent.args) --wandb_disabled

# ===== Paper.md modes =====
# Paper: optical perception losses (camera control + blur/noise)
python main_cuda.py $(cat configs/paper_optical.args)
# Paper: unified control (camera + camera obs)
python main_cuda.py $(cat configs/paper_unified.args)
# Paper: full teacher-student two-phase training
python main_cuda.py $(cat configs/paper_gdac.args)
# Evaluate with camera-control model
python eval_wall_slit.py --resume checkpoint0004.pth --ellipsoid_collision --include_camera_state_in_obs --num_episodes 200

scp -r -p 46936 root@10.130.145.237:/root/code/DiffPhysDrone/checkpoint/2026-03-27-10-23-25 //home/zhaoguodong/work/code/DiffPhysDrone/checkpoint
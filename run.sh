# For multi-agemt
python main_cuda.py $(cat configs/multi_agent.args)
# For single-agemt，basic mode
python main_cuda.py $(cat configs/single_agent.args)
# For single-agemt，diff cam mode
python main_cuda.py $(cat configs/single_agent.args) --diff_cam
# For wall-slit (narrow gap sideways flight)
python main_cuda.py $(cat configs/wall_slit.args)
# Evaluate wall-slit
python eval_wall_slit.py --resume checkpoint0004.pth --ellipsoid_collision --num_episodes 200
tensorboard --logdir runs
python main_cuda.py $(cat configs/single_agent.args) --wandb_disabled

# ===== Paper.md modes =====
# Paper: optical perception losses (diff_cam + blur/noise/defocus)
python main_cuda.py $(cat configs/paper_optical.args)
# Paper: unified control (camera deltas in action space + camera obs)
python main_cuda.py $(cat configs/paper_unified.args)
# Paper: full G-DAC two-phase training (teacher-student)
python main_cuda.py $(cat configs/paper_gdac.args)
# Evaluate with unified control model
python eval_wall_slit.py --resume checkpoint0004.pth --ellipsoid_collision --paper_unified_control --paper_cam_obs --num_episodes 200
# For multi-agemt
python main_cuda.py $(cat configs/multi_agent.args)
# For single-agemt
python main_cuda.py $(cat configs/single_agent.args)
# For wall-slit (narrow gap sideways flight)
python main_cuda.py $(cat configs/wall_slit.args)
# Evaluate wall-slit
python eval_wall_slit.py --resume checkpoint0004.pth --ellipsoid_collision --num_episodes 200
tensorboard --logdir runs
python main_cuda.py $(cat configs/single_agent.args) --wandb_disabled
# For multi-agemt
python main_cuda.py $(cat configs/multi_agent.args)
# For single-agemt
python main_cuda.py $(cat configs/single_agent.args)
tensorboard --logdir runs
python main_cuda.py $(cat configs/single_agent.args) --wandb_disabled
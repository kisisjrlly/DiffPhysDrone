python3 paper/experiment/run_ral_eval_suite.py \
  --ours_ckpt /home/zhaoguodong/work/code/DiffPhysDrone/checkpoint/2026-04-22-10-25-45/checkpoint0017.pth \
  --fixed_ckpt /home/zhaoguodong/work/code/DiffPhysDrone/checkpoint/2026-04-22-11-52-18/checkpoint0017.pth \
  --nondiff_ckpt /home/zhaoguodong/work/code/DiffPhysDrone/checkpoint/2026-04-22-13-00-37/checkpoint0049.pth \
  --episodes_per_condition 20 > logs/paper_experiment_eval.log 2>&1

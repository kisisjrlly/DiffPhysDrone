# Reproduce Commands

This result directory was generated for the split-stem flight-only checkpoint:

`checkpoint/2026-05-10-02-45-09/checkpoint0014.pth`

The source training run is:

`wandb/run-20260510_024510-bbxkm6bj`

## Full Checkpoint Eval Suite

```bash
python3 -u tools/run_checkpoint_eval_suite.py \
  --config configs/slit_active_sensing.args \
  --out_dir paper/paper_tex/results \
  --episodes_per_scene 300 \
  --scenarios glare specular dark \
  --seed 42 \
  --device cuda \
  --methods pretrained dagger flightonly fixed randfix nondiff zero \
  --pretrained_ckpt checkpoint/closed_loop_teacher_camera_policy_light_v3d_t60_20260509/camera_head_pretrained_best.pth \
  --dagger_ckpt checkpoint/closed_loop_teacher_camera_policy_light_v3d_t60_20260509_dagger/camera_head_pretrained_best.pth \
  --flightonly_ckpt checkpoint/2026-05-10-02-45-09/checkpoint0014.pth \
  --fixed_ckpt checkpoint/2026-05-09-22-32-06/checkpoint0014.pth \
  --randfix_ckpt checkpoint/2026-05-09-22-35-26/checkpoint0014.pth \
  --nondiff_ckpt checkpoint/2026-05-09-22-34-42/checkpoint0014.pth \
  --zero_ckpt checkpoint/2026-05-09-16-48-43/checkpoint0014.pth
```

## DAgger / Online-Offline Camera Diagnosis

```bash
python3 -u tools/diagnose_pretrain_camera_trace.py \
  --eval_dir paper/paper_tex/results \
  --dataset paper/paper_tex/results/camera_teacher_dataset.pt \
  --config configs/slit_active_sensing.args \
  --pretrained_ckpt checkpoint/closed_loop_teacher_camera_policy_light_v3d_t60_20260509_dagger/camera_head_pretrained_best.pth \
  --offline_method_label dagger \
  --device cuda \
  --batch_size 12
```

## Journal Tables And Aggregate Figures

```bash
python3 tools/make_journal_assets.py \
  --eval_dir paper/paper_tex/results \
  --out_dir paper/paper_tex/journal_assets
```

## Figure 5 Depth Observation Sequences

```bash
python3 -u tools/export_journal_depth_sequences.py \
  --config configs/slit_active_sensing.args \
  --eval_dir paper/paper_tex/results \
  --out_dir paper/paper_tex/journal_assets \
  --scenarios glare dark specular \
  --slot far_right \
  --target_local_x=-1.20,-0.75,-0.35,-0.08,0.18 \
  --seed 42 \
  --device cuda \
  --flightonly_ckpt checkpoint/2026-05-10-02-45-09/checkpoint0014.pth \
  --fixed_ckpt checkpoint/2026-05-09-22-32-06/checkpoint0014.pth \
  --randfix_ckpt checkpoint/2026-05-09-22-35-26/checkpoint0014.pth
```

## Single-Method Sanity Check

This is the shorter command used to confirm the new flight-only checkpoint alone.

```bash
python3 -u tools/run_checkpoint_eval_suite.py \
  --config configs/slit_active_sensing.args \
  --out_dir paper/paper_tex/results/eval_rerun_success_check \
  --episodes_per_scene 300 \
  --scenarios glare specular dark \
  --seed 42 \
  --device cuda \
  --methods flightonly \
  --flightonly_ckpt checkpoint/2026-05-10-02-45-09/checkpoint0014.pth
```

## Training-Curve Figure

`tools/make_journal_assets.py` only draws `fig2_training_convergence` when
`raw/wandb_export_*_{loss,success_rate,collision_rate}.csv` files are present in
the eval directory. Those history CSV files were not available locally for this
run; only `wandb-summary.json` is present under the W&B run directory.

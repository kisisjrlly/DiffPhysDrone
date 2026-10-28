# Method-selection evidence, 2026-10-29

Read [the report](../METHOD_SELECTION_2026_10_29.md) first. These are single-training-seed development comparisons in a provisional simulator. None establishes learned sensing/motion coordination, a winning learning method, or hardware readiness.

All training and evaluation conditions are complete. `summary.json` is the compact review entry point; it does not hide the failed arms.

## Evidence files

- `screening.json`: exploratory 30-update run, with its original limitations.
- `cold_study.json`: matched 150-update H4/H8/H16/H120 study, executed source hashes, actor/critic initialization, optimizer, configuration, calibration snapshot, checkpoint hashes, update curves, paired CPU holdout episodes, critic quality, camera interventions, wiring and task-objective diagnostics.
- `warm_study.json`: bounded H16 follow-up with a common return-pretrained critic; includes the extra data/fitting cost and paired evaluation. The follow-up and independent evaluations are complete.
- `cold_source.patch`, `warm_source.patch`: exact executed source/config differences from Git `2e74ba2`. Apply with `git apply --unidiff-zero`: these are zero-context patches, verified by reconstruction. They preserve the distinction between the original cold experiment and later optional critic-resume support.
- `scripts/*.py.txt`: exact study-specific diagnostic scripts, including critic pretraining and the original cold evaluator. They are preserved as text to avoid silently changing the executable source snapshots during a paired experiment.

The original binary checkpoints and full runtime artifacts remain under the local `logs/` directories named in the report. Checkpoints are identified by SHA-256, not published as new flight releases. The common actor is the disclosed flight-only imitation warm start with SHA-256 `b1fa7436da59db48c64d6d043ecda8909b269007f3d38db1f4e09535abf039ec`.

## Reproduce the cold experiment

Use a separate checkout at `2e74ba2`, apply `cold_source.patch`, and verify the resulting files against `source_snapshots` in `cold_study.json`. Use the project's CUDA extension and verified Python environment:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
STUDY_PYTHON=/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python
for H in 4 8 16 120; do
  "$STUDY_PYTHON" tools/run_intent_pilot.py \
    --config "configs/gray_method_selection_h${H}.args" \
    --out_dir "logs/method_selection_20261029/h${H}" \
    --variants joint camera_gradient_cut \
    --iterations 150 --batch_size 4 \
    --eval_episodes 40 --eval_seed 800000 --wandb_offline
done
```

Output directories must be new. The common actor checkpoint must be present and hash-verified. These commands regenerate training, not just evaluation.

The original cold evaluation is archived in `scripts/evaluate_study.py.txt`; it used CPU for both arms, saved updates 50/100, and 100 independent final episodes. The maintained evaluator in the current branch performs the same forward evaluation and adds calibration/cache guards:

```bash
"$STUDY_PYTHON" tools/evaluate_learning_method_study.py \
  --study_dir logs/method_selection_20261029 \
  --horizons 4 8 16 120 --device cpu --interventions
```

The maintained evaluator reads each run's actual saved `args.json` and calibration snapshot. It refuses mixed evaluation devices or changed cached checkpoints. CPU and CUDA episode sequences must never be combined as if they were the same paired set.

## Reproduce the warm-critic diagnostic

The collection/fitting script is preserved in `scripts/warm_critic.py.txt`. Run it as a Python file from the repository root after the configs and common checkpoint are available. It records train/validation episode separation, data/critic hashes and fitting cost in `logs/method_selection_20261029/critic_warmup/manifest.json`.

Use a separate checkout with `warm_source.patch` applied before launching the follow-up:

```bash
"$STUDY_PYTHON" tools/run_intent_pilot.py \
  --config configs/gray_method_selection_h16.args \
  --out_dir logs/method_selection_20261029_warm/h16 \
  --variants joint camera_gradient_cut \
  --iterations 150 --batch_size 4 \
  --eval_episodes 40 --eval_seed 800000 --wandb_offline \
  --critic_resume logs/method_selection_20261029/critic_warmup/critic.pth
"$STUDY_PYTHON" tools/evaluate_learning_method_study.py \
  --study_dir logs/method_selection_20261029_warm \
  --horizons 16 --device cpu --interventions
```

Only the critic initialization changes relative to cold H16. Critic validation R² on the common actor distribution does not certify its derivatives or its accuracy after the actor changes. No exposure/gain labels are used. The warm-up data, fitting updates and CPU time are additional resources, reported explicitly.

The executed training revision was dirty `2e74ba2`; the patches and source hashes bind each result to the actual executed code, rather than pretending it ran from a later clean commit. W&B recording was offline throughout.

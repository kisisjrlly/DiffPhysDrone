# Tools Overview

This directory contains training, evaluation and paper-asset utilities. For the
final active-sensing paper figures, use one canonical entrypoint:

```bash
bash run_exp_eval.sh
```

It extracts checkpoints from the latest training logs, runs
`tools/run_checkpoint_eval_suite.py`, regenerates aggregate journal assets with
`tools/make_journal_assets.py`, and exports qualitative depth sequences with
`tools/export_journal_depth_sequences.py`.

The current final result directory is:

```text
paper/experiment/results/final_semantics_v3_eval_20260508
```

## Paper Asset Scripts

- `run_sensor_semantics_probe_suite.py`: current pre-training/pre-paper sensor
  semantics gate. It runs the controlled opening probe, actual rollout-state
  probes from checkpoints, and a final-figure-equivalent qualitative depth
  export. Run this before accepting simulator changes or before trusting final
  qualitative figures. The hard gate checks both material-mask leakage into the
  back-wall cue and same-rollout-pose good/bad camera sweep signal; the current
  checkpoint qualitative contrast is diagnostic only after simulator changes,
  because learned-camera checkpoints are stale until retrained.
- `make_journal_assets.py`: current official generator for paper figures,
  extended-data figures, LaTeX tables, captions and the asset README.
- `export_journal_depth_sequences.py`: current official generator for the
  qualitative raw-depth/observed-depth sequence panels.
- `run_final_eval_from_logs.py`: extracts final checkpoints from training logs
  and orchestrates eval plus asset regeneration. `run_exp_eval.sh` is the shell
  wrapper for this script.
- `paper_asset_utils.py`: private shared data/statistics helpers used by the
  journal asset generator.
- `export_paper_result_assets.py`: diagnostic export for quick inspection of
  trajectories, traces and metrics. Do not use its `paper_assets` output as the
  final manuscript figure set.
- `make_probe_paper_figures.py`: older depth-probe experiment figure script,
  separate from the final active-sensing evaluation suite.

In short: for the current paper, regenerate evaluation data with
`bash run_exp_eval.sh`. If eval has already been run and only figures need to be
rebuilt, use:

```bash
SKIP_EVAL=1 bash run_exp_eval.sh
```

Before a simulator/sensor semantic change enters teacher data, training, or
paper figures, run:

```bash
/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python -u tools/run_sensor_semantics_probe_suite.py \
  logs/2026-05-08-00-44-21-slit_active_sensing_auto_fix-cam-fixed_grad-detached_depth-depth.log \
  logs/2026-05-08-01-17-20-slit_active_sensing_auto_randfix-cam-fixed_random_static_grad-detached_depth-depth.log \
  logs/2026-05-08-03-04-26-slit_active_sensing_auto_flightonly-cam-learned_grad-detached_depth-depth.log \
  --config configs/slit_active_sensing.args \
  --out_dir paper/experiment/results/sensor_semantics_probe_suite_<tag> \
  --python /home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python \
  --device cuda
```

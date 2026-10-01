# IMX900 experiment runbook

The camera-only path restores only flight-policy parameters from the common
flight checkpoint. It then resets every camera-controller parameter from the
requested seed, warm-starts `cam_stem` from the flight stem, and sets the
camera head to the configured initial exposure/gain. Each run writes
`initialization_manifest.json` with separate flight and camera parameter
hashes.

Training rollouts record the post-action state. Collision and goal checks use
that state, including the last action, and terminal samples are masked from
later navigation and smoothness losses. The collision loss uses the nearest
geometry vector per batch item and its transition is aligned with the action
that produced it.

## Fixed response surface

Use the mixed-light mean-AE flight checkpoint with paired episode seeds:

```bash
/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python \
  tools/evaluate_fixed_response_surface.py \
  --resume checkpoint/2026-09-24-11-35-14/final.pth \
  --out_dir logs/imx900_review/fixed_response_surface \
  --episodes_per_scenario 25 \
  --scenarios dark bright dark_to_bright bright_to_dark
```

The output contains `episodes.csv`, `summary.json`, and a calibration snapshot.
The summary reports mean-best, worst-case-best, per-scenario results, and
episode-bootstrap intervals. The simulator profile remains provisional.

## Matched causal runs

The explicit driver runs seeds 101, 202, 303, 404, and 505 as full/detached
pairs, evaluates each checkpoint, and writes an experiment manifest:

```bash
/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python \
  tools/run_full_detached_seeds.py \
  --flight_checkpoint checkpoint/2026-09-24-11-35-14/final.pth \
  --out_dir logs/imx900_full_detached_5seeds
```

The driver is a one-shot experiment command; it does not create a scheduler.

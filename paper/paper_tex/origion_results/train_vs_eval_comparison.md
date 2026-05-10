# Train vs Eval Comparison

This file records the consistency check for the latest split-stem flight-only
checkpoint:

`checkpoint/2026-05-10-02-45-09/checkpoint0014.pth`

## W&B Training Summary

Source:

`wandb/run-20260510_024510-bbxkm6bj/files/wandb-summary.json`

Final summary values:

| metric | value |
|---|---:|
| success_rate | 0.750 |
| collision_rate | 0.250 |
| charts/goal_dist | 0.155 |
| cam/power_mean | 0.561 |
| cam/exposure_mean | 0.437 |
| cam/gain_mean | 0.430 |

## Offline Eval Summary

Source:

`paper/experiment/results/final_dagger_flightonly_eval_20260510_splitstem/combined_report.md`

Flight-only eval over 900 episodes:

| metric | value |
|---|---:|
| success | 0.751 |
| collision | 0.249 |
| fill | 0.975 |
| final dist | 0.550 |
| power/exposure/gain | 0.577/0.415/0.407 |

Flight-only eval by scene:

| scene | n | success | collision | fill | final dist | power/exposure/gain |
|---|---:|---:|---:|---:|---:|---:|
| dark | 300 | 0.820 | 0.180 | 0.977 | 0.428 | 0.602/0.607/0.597 |
| glare | 300 | 0.670 | 0.330 | 0.958 | 0.697 | 0.704/0.213/0.223 |
| specular | 300 | 0.763 | 0.237 | 0.990 | 0.525 | 0.426/0.425/0.401 |

## Interpretation

For this checkpoint, the training summary and offline eval are aligned: both put
flight-only success at about 0.75 and collision at about 0.25. The previous
train/eval mismatch appears to have been caused by the earlier shared-stem
flight-only adaptation changing the feature semantics seen by the frozen camera
branch. The split-stem run keeps camera semantics fixed while allowing the flight
visual stem to adapt, and the offline eval now preserves the W&B advantage.

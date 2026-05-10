# Pretrained Camera Trace Diagnosis

## Result

The offline teacher-dataset check asks whether the checkpoint can reproduce `dark`/`glare` labels on its supervised data.  The online eval rows ask whether that separation survives closed-loop rollout.

| source | method | glare near p/e/g | dark near p/e/g | glare-dark L1 | per-param diff |
|---|---|---:|---:|---:|---:|
| offline teacher dataset | dagger | 0.578/0.289/0.317 | 0.582/0.596/0.584 | 0.193 | 0.004/0.308/0.267 |
| online eval | dagger | 0.750/0.097/0.128 | 0.661/0.706/0.687 | 0.419 | 0.089/0.609/0.560 |
| online eval | fixed | 0.500/0.500/0.500 | 0.500/0.500/0.500 | 0.000 | 0.000/0.000/0.000 |
| online eval | flightonly | 0.738/0.099/0.127 | 0.661/0.688/0.677 | 0.405 | 0.077/0.589/0.550 |
| online eval | nondiff | 0.412/0.368/0.508 | 0.411/0.369/0.508 | 0.000 | 0.001/0.000/0.000 |
| online eval | pretrained | 0.725/0.084/0.116 | 0.665/0.705/0.689 | 0.418 | 0.060/0.621/0.573 |
| online eval | randfix | 0.525/0.503/0.456 | 0.547/0.487/0.449 | 0.015 | 0.023/0.016/0.007 |
| online eval | zero | 0.500/0.500/0.500 | 0.500/0.500/0.500 | 0.000 | 0.000/0.000/0.000 |

## Interpretation

- The offline row checks whether supervised camera learning can represent the teacher labels on the relabeled dataset.
- Offline glare-dark L1 for `dagger` is `0.193`.  Values around `0.18-0.22` mean the dataset and fitted camera head have a clear dark/glare distinction.
- Online glare-dark L1 for `dagger` is `0.419`: clear separation.
- Online glare-dark L1 for `fixed` is `0.000`: weak separation.
- Online glare-dark L1 for `flightonly` is `0.405`: clear separation.
- Online glare-dark L1 for `nondiff` is `0.000`: weak separation.
- Online glare-dark L1 for `pretrained` is `0.418`: clear separation.
- Online glare-dark L1 for `randfix` is `0.015`: weak separation.
- Online glare-dark L1 for `zero` is `0.000`: weak separation.

## Next Checks

1. If the DAgger checkpoint still has weak online separation, save a small batch of online `(depth_obs, state, camera_state, camera_motion_state)` tensors and run the teacher optimizer on those exact states.  That isolates whether the online dark state is truly ambiguous or merely underrepresented.
2. If the DAgger checkpoint restores online separation, use it as the resume checkpoint for the next flight-only run.
3. The immediate target is online glare-dark near L1 above about `0.12`, with dark exposure/gain clearly higher than glare.

Detailed phase rows: `paper/experiment/results/final_dagger_flightonly_eval_20260510_splitstem/pretrain_online_offline_phase_summary.csv`.

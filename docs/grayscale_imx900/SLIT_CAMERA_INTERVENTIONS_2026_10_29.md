# Narrow-Slit Camera Intervention Evaluation (2026-10-29)

## Question

Does camera control affect navigation on the original single-wall narrow-slit
task, and does the trained camera policy improve on a conventional mean-AE
controller under the same frozen flight policy?

## Setup

- Scene: `single_wall_slit`; no pole or double-gate scene was used.
- Flight policy: frozen mean-AE flight checkpoint
  `2026-09-24-11-35-14/final.pth`.
- Learned camera policy: seed-101 full-gradient camera checkpoint from the
  five-seed experiment. Its non-camera parameters were hash-checked against
  the common flight checkpoint before evaluation.
- Conditions: learned full, mean-AE, fixed nominal `(E=0.10, G=0.02)`, fixed low
  `(0.03, 0.00)`, fixed high `(0.35, 0.15)`, and the previously selected fixed
  response `(0.00, 0.80)`.
- Evaluation: four lighting conditions, 100 episodes per condition, 400 paired
  episode seeds per method (`900000-900399`). All methods used identical seeds.
- Simulator calibration is provisional (`calibrated: false`). These results
  make no real-camera or real-flight claim.

## Results

| Camera condition | Success | Collision | Timeout |
| --- | ---: | ---: | ---: |
| Learned full | 283/400 (70.75%) | 27.50% | 1.75% |
| Mean-AE | 262/400 (65.50%) | 33.50% | 1.00% |
| Fixed nominal | 22/400 (5.50%) | 93.25% | 1.25% |
| Fixed low | 108/400 (27.00%) | 70.75% | 2.25% |
| Fixed high | 90/400 (22.50%) | 76.25% | 1.25% |
| Previously selected fixed response | 49/400 (12.25%) | 86.75% | 1.00% |

The paired learned-minus-mean-AE success difference is **+5.25 percentage
points**, with a lighting-stratified episode-bootstrap 95% interval of
**[+2.50, +8.00] points**. The paired episodes were 28 learned-only successes
and 7 mean-AE-only successes.

| Lighting condition | Learned full | Mean-AE | Difference |
| --- | ---: | ---: | ---: |
| Dark | 71/100 (71%) | 73/100 (73%) | -2 pp |
| Bright | 69/100 (69%) | 62/100 (62%) | +7 pp |
| Dark to bright | 75/100 (75%) | 73/100 (73%) | +2 pp |
| Bright to dark | 68/100 (68%) | 54/100 (54%) | +14 pp |

## Conclusion

Camera adaptation matters in this narrow-slit simulator task: all tested fixed
settings performed substantially worse than the adaptive controllers. The
trained full-gradient camera policy also improves on mean-AE, but the overall
gain is modest and uneven across lighting conditions; mean-AE is 2 points
better in the dark-only condition, while the largest gain is after bright-to-
dark transitions.

This is evidence for the value of adaptive camera control, not proof that
differentiable training is the best learning method. The episode interval is
conditional on one trained camera checkpoint and does not measure variation
across training seeds. The tested fixed settings do not exhaust the E/G space,
so no claim is made that fixed control has no optimum. The project’s separate
3000-iteration grayscale-vs-blind gate is documented in
`REVIEW_2026_09_24.md` and passes for its stated provisional-simulator scope.

## Next step

Keep the original narrow-slit task. The next decisive experiment is a matched
training-method comparison on this task: same flight initialization, policy
capacity, camera initialization, data, optimizer budget, and evaluation seeds;
change only whether navigation gradients pass through the camera model. Report
results per training seed, and retain mean-AE and fixed-camera conditions as
navigation baselines. Do not treat the current one-checkpoint comparison as
that method-level result.

## Artifacts and provenance

- Evaluator: `tools/evaluate_slit_camera_interventions.py`.
- Full local results: `logs/slit_camera_interventions_20261029/`.
- Manifest SHA-256: `f4674117aabe2b2d1df9af9dfaeb8e98818ffafe57ab4e6e1c3b2785146079d0`.
- Summary SHA-256: `a22a98d76297a3355e86213e12e1af0eef675ffd3921b0e066eebb0ff121998d`.
- Source revision before adding the evaluator: `361fe0a3284e64b376903e1619547cc33aeb6641`.
- Flight-parameter SHA-256 shared by learned and baseline runs:
  `2dec643b359dc2cb6ba4f040f7947adcd10f58af4f0b60e5ed5152e6baa0ec0e`.

The CUDA run emitted PyTorch's warn-only determinism notice for `nanmedian`.
Seeds were reset per episode and all methods were evaluated on the same paired
episodes; exact bitwise determinism is not claimed.

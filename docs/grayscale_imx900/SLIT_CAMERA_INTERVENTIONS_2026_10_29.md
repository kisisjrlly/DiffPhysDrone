# Narrow-Slit Camera Intervention Evaluation (2026-10-29)

## Question

Does camera control affect navigation on the original single-wall narrow-slit
task? Does opening the differentiable camera path help, and how does the trained
controller compare with mean-AE and fixed settings?

## Setup

- Scene: `single_wall_slit`; no pole or double-gate scene was used.
- Flight policy: frozen mean-AE flight checkpoint
  `2026-09-24-11-35-14/final.pth`.
- Learned camera policy: seed-101 full-gradient camera checkpoint from the
  five-seed experiment. Its non-camera parameters were hash-checked against
  the common flight checkpoint before evaluation.
- Matched detached camera policy: same seed, same initial camera and flight
  parameter hashes, and same 3000-iteration training configuration; only
  `sensor_grad_mode` differs.
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

The separate matched full-versus-detached evaluation reused the same 400
episode seeds. Full achieved **283/400 (70.75%)** and detached **22/400
(5.50%)**: **+65.25 percentage points**, with a lighting-stratified
episode-bootstrap 95% interval of **[+60.75, +69.75] points**. There were 261
full-only successes and no detached-only successes. The detached controller
remained near its initial settings (mean `E=0.1001`, `G=0.0200`); full adjusted
both settings.

| Lighting condition | Learned full | Learned detached | Difference |
| --- | ---: | ---: | ---: |
| Dark | 71/100 (71%) | 7/100 (7%) | +64 pp |
| Bright | 69/100 (69%) | 9/100 (9%) | +60 pp |
| Dark to bright | 75/100 (75%) | 3/100 (3%) | +72 pp |
| Bright to dark | 68/100 (68%) | 3/100 (3%) | +65 pp |

| Lighting condition | Learned full | Mean-AE | Difference |
| --- | ---: | ---: | ---: |
| Dark | 71/100 (71%) | 73/100 (73%) | -2 pp |
| Bright | 69/100 (69%) | 62/100 (62%) | +7 pp |
| Dark to bright | 75/100 (75%) | 73/100 (73%) | +2 pp |
| Bright to dark | 68/100 (68%) | 54/100 (54%) | +14 pp |

## Conclusion

In the original narrow-slit simulator task, camera adaptation matters. The
matched full checkpoint substantially outperforms detached sensing on all four
lighting conditions, supporting the claim that navigation gradients through
the camera model let this camera policy learn useful changes. The fixed
settings also perform poorly. Against conventional mean-AE, however, the
full-gradient policy's gain is only 5.25 points and varies by lighting;
mean-AE is 2 points better in darkness, while full is 14 points better after a
bright-to-dark transition.

This supports differentiable camera learning over the matched detached control
for this training setup; it does not show that differentiable training is the
best learning method overall. Detached receives no substitute task-learning
signal for camera control, so this is a gradient-path test rather than a fair
comparison against reinforcement learning or another optimizer. The confidence
intervals are conditional on one trained seed and do not measure variation
across training runs. The tested fixed settings do not exhaust the E/G space,
so no claim is made that fixed control has no optimum. The project's separate
3000-iteration grayscale-vs-blind gate is documented in
`REVIEW_2026_09_24.md` and passes for its stated provisional-simulator scope.

## Next step

Keep the original narrow-slit task and the current BPTT method as the reference.
To answer whether a different learning method is better, run one focused
short-horizon actor-critic comparison. Use the same critic, actor capacity,
flight and camera initialization, training interaction budget, and held-out
seeds in both arms; only the differentiable camera-gradient path should differ.
This gives detached sensing a genuine trial-and-error learning signal and
avoids treating it as a weak no-camera-learning control. Do not add VQ-VAE,
world models, or a broad RL benchmark before this comparison answers the
specific question.

## Artifacts and provenance

- Evaluator: `tools/evaluate_slit_camera_interventions.py`.
- Full local results: `logs/slit_camera_interventions_20261029/`.
- Matched gradient-path results: `logs/slit_camera_full_detached_20261029/`.
- First-run manifest SHA-256:
  `3b663570a958fecc8279108bd0a8fbb0fb9ab06a7380c7eee6710a1de42771e5`.
- Summary SHA-256: `a22a98d76297a3355e86213e12e1af0eef675ffd3921b0e066eebb0ff121998d`.
- Matched full/detached manifest SHA-256:
  `14747a1e4c7891f319712070f9d32f11ddb4ec160adad5f21a3c8ef3bdf7cf94`.
- Matched full/detached summary SHA-256:
  `2325cef714376aaeb4a12a37def447f423f2fd485677db4ad69edf22be5a4b54`.
- Source revision before adding the evaluator: `361fe0a3284e64b376903e1619547cc33aeb6641`.
- Revision used by the matched full/detached evaluation:
  `300c8a42e635524e0b62951bf8aa132c22d55d35`.
- Full and detached checkpoint SHA-256 values:
  `f40bc25ae444c6140bd821be11cdf3cdd95e7aa2d1f6f8a94b2648aa4833d954` and
  `e46885ea99d77bd0eadd5e0967068f28ab4b3126fd9a9518caab3d6fc7da20af`.
- Flight-parameter SHA-256 shared by learned and baseline runs:
  `2dec643b359dc2cb6ba4f040f7947adcd10f58af4f0b60e5ed5152e6baa0ec0e`.

The CUDA run emitted PyTorch's warn-only determinism notice for `nanmedian`.
Seeds were reset per episode and all methods were evaluated on the same paired
episodes; exact bitwise determinism is not claimed.

# IMX900 camera-model OOD screening (2026-10-02)

This is an engineering screen of the frozen seed-101 checkpoint. It changes
only the evaluation-time provisional sensor model and uses 25 episodes per
scenario (100 total, four lighting scenarios). It is not a new training-seed
statistical result and does not establish hardware calibration.

| Variant | Full | Detached | Fixed | Full-detached |
|---|---:|---:|---:|---:|
| nominal | 76% | 4% | 17% | +72 pp |
| hard saturation | 76% | 4% | 17% | +72 pp |
| STE saturation | 76% | 4% | 17% | +72 pp |
| blur 0.5x | 76% | 4% | 17% | +72 pp |
| blur 1.5x | 73% | 4% | 17% | +69 pp |
| noise 0.5x | 79% | 4% | 21% | +75 pp |
| noise std x2 | 58% | 4% | 10% | +54 pp |
| delay 1 control step (~66.7 ms at 15 Hz) | 73% | 4% | 17% | +69 pp |
| delay 2 control steps (~133.3 ms at 15 Hz) | 56% | 4% | 17% | +52 pp |
| exposure response 0.9x | 73% | 4% | 19% | +69 pp |
| exposure response 1.1x | 76% | 4% | 11% | +72 pp |
| linear gain mapping | 75% | 4% | 6% | +71 pp |
| gain max 1.1x | 76% | 4% | 11% | +72 pp |

The causal direction remains positive in every screened condition. The two
engineering risks are accumulated noise and command delay: both reduce full
success materially while leaving detached near failure. They should be
included in measured-profile retraining and timing validation before HIL.
Exposure command-to-us mapping perturbations were added to the evaluator after
this screening run and have not been evaluated in this table. The nominal
provisional calibration is still `calibrated: false`; no result here authorizes
real flight.

Machine-readable output: `logs/imx900_camera_ood_screen_20261002/ood_manifest.json`.

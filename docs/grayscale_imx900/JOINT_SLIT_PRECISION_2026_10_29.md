# Single-slit precision and tracking follow-up — 2026-10-29

## Decision

The candidate remains **unready for a camera/flight coordination experiment**.
The earlier intuition that a light transition and slit would suffice was too
confident. Additional tests still do not demonstrate a need for a
motion-dependent exposure choice. No camera/flight policies were trained in
this follow-up, and the frozen checkpoints and original benchmark are intact.

## Separate reference tracking from visual sensing

Used 64 random slit centers and identical current-disturbance and future-noise
streams across each width. Reset seed: 291101; dynamics seed: 291102. The
reference controller uses true geometry and is diagnostic only. A second arm
subtracts the known current simulator disturbance, without knowing the next
random innovation. Both arms use the actual command queue and collision model.

| Slit width | Uncompensated success | Known-current-disturbance compensation |
| --- | ---: | ---: |
| 30cm (original) | 48/64 | 61/64 |
| 36cm | 62/64 | 64/64 |
| 40cm | 63/64 | 64/64 |
| 48cm | 64/64 | 64/64 |

These data support a tracking/disturbance confound in the 30cm slit, which
leaves 3cm clearance per side for a 24cm-diameter body. They are not evidence
that blind navigation solves the task, that disturbances explain all failures,
or that a new learned controller cannot pass the narrow slit. The wider
geometries are separately labelled diagnostics and do not replace the original.

## Remove exact-pose/texture knowledge from the image test

`tools/audit_slit_precision.py` fits a fixed-regularization linear classifier on
one image set and tests on an independent set. Each set has 64 examples per
class; train/test nuisance seeds are disjoint. It does not train a flight or
camera policy. All images use the actual 48x36 policy preprocessing resolution.

The 192-condition screen covers:

- Two slit-position differences: 5mm and 20mm.
- Dark and spatial bright-to-dark illumination, at x=-1.0m and -0.4m.
- Stationary, translation+1.5rad/s yaw, translation+3rad/s yaw stress.
- Exposure 890/2470/4840/6815us, fixed gain and matched mean-signal gain.
- Unknown residual yaw uniform in +/-0.25 degrees, random material texture
  phase, illumination scale in [0.92,1.08], and sensor noise.

Texture phase is a world material property that stays fixed through exposure
integration. Its new optional renderer argument defaults to zero, preserving
the prior scenes. Common nuisance/noise draws are reused across camera/motion
settings; train/test examples remain separate. Ridge regularization and
train-only feature scaling are fixed without using the test labels for tuning.

Selected results, 5mm cue at x=-1m in uniform darkness, fixed gain:

| Motion | 890us | 2470us | 4840us | 6815us |
| --- | ---: | ---: | ---: | ---: |
| Stationary | 93.75% | 99.22% | 100% | 100% |
| Translation + 1.5rad/s yaw | 91.41% | 99.22% | 100% | 100% |

No short-minus-long paired interval excluded zero on the positive side in
this screen. The intervals are descriptive episode/nuisance intervals, not
training-seed uncertainty. Many conditions saturate at perfect decoder
performance, so these tests cannot quantify small residual benefits.

## Check subpixel integration

The renderer normally samples one ray per sensor pixel. To avoid attributing
precise edge cues entirely to this sampling, repeated a focused 32-condition
screen at 2x spatial resolution in each dimension. Integrated the resulting
subpixel irradiance to the original 96x72 sensor dimensions **before** applying
the same photon/read noise, then used the same 48x36 policy resolution. This
does not introduce a measured lens PSF or change sensor noise through an
unreported resolution increase.

At x=-1m with a 5mm cue in darkness, fixed gain:

| Motion | 890us | 2470us | 4840us | 6815us |
| --- | ---: | ---: | ---: | ---: |
| Stationary | 96.88% | 100% | 100% | 100% |
| Translation + 1.5rad/s yaw | 95.31% | 100% | 100% | 100% |

Short exposure again showed no clear advantage. This reduces one sampling
concern but does not prove spatial convergence or model real optics. Nuisance
variation is modest, trajectories are locally constant-twist, gain/noise
coefficients remain provisional, and the diagnostic classifier may differ
substantially from a recurrent navigator.

## What this means for the next implementation

The observed limitations currently precede choosing an RL or BPTT algorithm.
Retain the original scene as the exposure-adaptation baseline. Keep the
independent wider-slit geometry available to test tracking separately, without
silently turning its easier success rate into a narrow-slit result.

Before training a stronger joint-sensing claim, require a physical
counterfactual: at the same illumination and task, some plausible change of
motion must improve a decision-relevant measurement enough to offset its
time/control cost. A different best exposure is one possible observation, not
a necessary theoretical condition. Occlusion, viewpoint or genuinely fast
image motion may provide such a mechanism, but none was established here.
Do not claim that slowing improves this task from a nonzero motion gradient.
Do not exaggerate blur/noise just to create a preferred result. Separately
align the camera acquisition time with pending flight commands before testing
an intent-conditioned camera head.

## Artifacts and reproduction

Tracked results are in `joint_slit_precision_20261029/`, with source and
calibration hashes. Local source snapshots are under
`logs/joint_slit_20261029/precision_followup/`.

```bash
/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python tools/audit_slit_tracking.py \
  --out_dir logs/slit_tracking_new --device cuda
/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python tools/audit_slit_precision.py \
  --out_dir logs/slit_precision_new --device cuda
/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python tools/audit_slit_precision.py \
  --out_dir logs/slit_precision_area2_new --device cuda --spatial_supersample 2 --focused
```

Full project test suite: **143 passed**. These are provisional-simulator
diagnostics, not measured IMX900, learned coordination or real-flight results.

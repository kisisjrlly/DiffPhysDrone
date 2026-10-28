# Post-measurement and live-guidance gate — 2026-10-28

This report records the two checks required before considering a longer
full-vs-score-function experiment. Both use the sparse-pole `parallax_pair`
scene and the provisional IMX900 simulator profile (`calibrated: false`).

## 1. External post-measurement diagnostic

The common flight-only checkpoint was evaluated with a new episode-seed range
`799000–799039` and 40 episodes. `forced_probe` keeps the camera fixed and
applies a geometry-independent lateral motion for the first 24 control steps;
the actor then flies normally. This is a conditional diagnostic, not learned
sensing.

| condition | all | near-pole sign − | near-pole sign + |
| --- | ---: | ---: | ---: |
| nominal | 10/40 (25.0%) | 0/23 (0.0%) | 10/17 (58.8%) |
| external probe, then actor | 33/40 (82.5%) | 20/23 (87.0%) | 13/17 (76.5%) |

The result answers the narrow post-measurement question: after an external
view-changing motion, the shared flight actor can usually complete both mirror
layouts. It does not show that the actor learned to choose that motion.

Evidence is in:

```text
logs/verification_20261028/post_measurement_common_v2/
```

The checkpoint SHA-256 is
`b1fa7436da59db48c64d6d043ecda8909b269007f3d38db1f4e09535abf039ec`.

## 2. Matched live-guidance pilot

The pilot uses the same common actor, camera initialization, critic
initialization, replay dataset/targets, seed (`101`), 20 updates, batch size 4,
and 40 paired evaluation episodes (`799100–799139`) in both arms. The first
24 control steps have no flight target. No exposure/gain target is supplied.
After step 24, a simulator-geometry teacher supervises only flight actions;
this is privileged training-time guidance and is not constructed at deployment.

| condition | all | near-pole sign − | near-pole sign + |
| --- | ---: | ---: | ---: |
| full pathwise | 8/40 (20.0%) | 0/23 (0.0%) | 8/17 (47.1%) |
| sensing-cut + score function | 8/40 (20.0%) | 0/23 (0.0%) | 8/17 (47.1%) |

The initial common actor on the same evaluation seeds was 10/40, with 0/23
on sign −. Thus the short matched update did not recover the missing mirror
layout. The camera did not learn a meaningful measurement action: across
episodes its exposure range averaged about `7.6e-4` and its gain range about
`3.2e-4`, around the initial `E=0.08, G=0.05`.

The guidance term was active (roughly 8–18% of the logged weighted loss in the
full arm), so this is not a silent configuration failure. The result instead
shows that supervision introduced only after the 24-step unlabelled phase did
not repair the missing initial view-change behavior.

Evidence and resource audit:

```text
logs/short_ac_guidance_20261028_v2/pilot_manifest.json
docs/grayscale_imx900/actor_critic_20261028/guidance_pilot_audit_v2.json
```

Both arms have matching flight/camera/critic initialization and replay/source
hashes. W&B was recorded locally in offline mode; it was not synced online.

## Decision

The post-measurement gate passes, but the live-guidance gate fails. The
current scene, common actor, and short-horizon training protocol therefore do
not yet establish learned flight-for-measurement or an advantage of the
full-pathwise arm. Do not increase the training budget or start a longer
full-vs-score-function comparison on this setup. A follow-up must first change
the learning/task design so that the initial unlabelled measurement phase is
survivable and receives a task-relevant learning signal, then rerun the same
matched checks.

## Implementation fix made during the gate

`PoleNavigationTeacher.reference()` now explicitly handles batched per-episode
times. The previous `(B,)` versus `(B, segments-1)` comparison crashed the
guidance pilot at batch size 4. A regression test covers the batched target
shape and finite output.

# Single-slit joint-sensing audit — 2026-10-29

## Decision

**Do not start a new training comparison on this prototype yet.** The proposed
single-slit scene has plausible physics but has not demonstrated the central
measurement/control conflict. This supersedes the immediate actor-critic
training recommendation in `SLIT_CAMERA_INTERVENTIONS_2026_10_29.md`.

The intended scientific question is whether a drone can choose how to measure
for its upcoming control decision, and choose motion that improves that
measurement. Neither a nonzero gradient nor a frozen-flight camera success
rate answers this question.

## Problems found in the existing work

1. In the frozen `single_wall_slit` renderer, transition brightness multiplies
   the entire image according to the **observer's x position**. It does not
   depict a stationary world-lit shadow boundary visible in advance. The old
   results remain valid for that simulator; an anticipatory-lighting claim is
   unsupported.
2. Its motion blur interpolates between a sharp image and a fixed 5x5 average.
   The proxy is detached speed/depth; rotation and blur direction are absent.
   It cannot train a flight action through motion-induced observation changes.
3. The original full/detached study freezes flight weights. It establishes
   useful camera adaptation for a frozen controller, not learned cooperative
   flight/measurement behavior. Detached lacks an alternative camera task
   learning signal, so the large difference is not an RL-method comparison.
4. The reference velocity loss points directly toward the final goal. A
   temporary slowdown for measurement incurs a tracking penalty. This may be
   a legitimate time pressure but must be distinguished from a final
   success/time objective; its effect has not been isolated.
5. Camera and flight actions have different timelines. At 15Hz with the frozen
   zero-additional-delay profile, a camera command first affects the next
   image (~66.7ms), while a flight command enters a two-step queue (~133.3ms)
   before actuator filtering. Passing the current raw action to a camera head
   does not by itself align flight intent to the effective exposure. The
   camera timeline input exists, but an explicit projection through the pending
   flight queue is not implemented.
6. `flight_max_speed` sets the reference velocity; it is not a physical speed
   clamp. Diagnostics and future reports must use realized speed, not call the
   configured reference speed a hard bound.

The development process previously emphasized executable pipelines and
gradient connectivity without these scene/claim checks. Switching from slit
to gate/pole probes made results harder to interpret. The remedy is to bind
each next experiment to a stated physical decision, rather than add another
architecture or training sweep first.

## Implemented diagnostic prototype

New opt-in config: `configs/gray_joint_slit.args`, scene `joint_slit`.

- Retains the original two slit-wall boxes and background wall, a 0.30m slit,
  sphere radius 0.12m, random lateral center, and fixed final goal.
- Adds corridor side walls/floor/ceiling to close bypass routes. These are
  visible geometry used identically for ray tracing and collision.
- Uses a stationary spatial illumination field evaluated at each ray-hit
  point. The same frame can contain bright foreground and a dark slit wall.
- Extends the approach from -1.5m to -2.5m and deadline from 4s to 6s. At the
  original short approach, bright foreground surfaces were often outside the
  view. These changes are explicit and do not alter frozen experiment args.
- Uses existing translational/angular shutter integration, 9 temporal samples,
  original provisional radiometry (100–8000us), and hard forward/STE backward.
- Uses the existing approximate fixed-depth image-warp Jacobian for pose,
  labelled `warp_ste`; no claim of an exact visibility derivative.
- No privileged geometry, slit labels, light fields or oracle commands are
  supplied to the actor. Diagnostic oracle results are not learned policies.

## Measurements

The reproducible final audit is `logs/joint_slit_20261029/scene_audit_v2/`.
The earlier exploratory `scene_audit/` used an uncontrolled disturbance seed
and is not the final comparison. Reset seed is 291030; dynamics seed is 291031.
All runs retain `calibrated: false`.

### Reachability and reference tracking

The same waypoint tracker was evaluated on 32 random slit positions with the
actual action queue, body clearance, disturbances and collision rules:

| Diagnostic controller | Success | Collision |
| --- | ---: | ---: |
| Geometry-aware waypoint tracker | 20/32 | 12/32 |
| Same tracker, subtracting the known current simulator disturbance | 29/32 | 3/32 |

The second controller is **privileged and diagnostic only**. It does not know
future random disturbance innovations. This supports a substantial disturbance
or tracking contribution, but does not prove a complete failure diagnosis or
intrinsic unreachability. The slit leaves only 3cm clearance on each side. A
geometrically collision-free path exists even for the extreme lateral centers.
Reliable execution across the random layouts has not been established.

On the uncompensated witness's active trajectory, speed p50/p95/max was
0.766/1.420/1.523 m/s, and angular speed was 0.314/1.492/2.076 rad/s.

### Task-relevant visual cue

At x=-1.0m and -0.4m, compared two slit centers differing by 5cm. The optimistic
decoder knows the true pose/motion and both noiseless templates. It receives
noisy sensor images downsampled to the actor's actual 48x36 input. This measures
cue availability, not a trained navigator's accuracy or robustness to unknown
texture/pose. Each class has 128 independent noise realizations.

256 conditions cover four light cases, four exposure settings (890, 2470,
4840, 6815us), two gains protocols (fixed gain and equal pre-saturation mean
signal via gain), two distances, and four motion states (static, translation,
translation+1rad/s yaw, translation+3rad/s yaw stress). The 3rad/s state is a
separate stress condition and exceeds the witness maximum.

**No useful shift toward short exposure appeared in the sampled dark or
bright-to-dark states.** Long exposure remained preferred by the template
signal/noise metric, including when the mean signal was matched with gain.
For example, at x=-0.4m in darkness and matched signal:

| Motion | Template separation at 890us | At 6815us |
| --- | ---: | ---: |
| Static | 159.2 | 758.6 |
| Translation + 1rad/s yaw | 159.3 | 762.5 |
| Translation + 3rad/s yaw stress | 159.6 | 731.5 |

These separations are not success rates or cross-training confidence measures.
The known-state decoder reaches 100% sampled accuracy in all dark-condition
settings; the task may still be difficult for a learned controller with pose,
texture and model uncertainty. At high brightness, long exposures erase the
cue by saturation; the decoder then correctly reports 50% chance performance.
The existing clear finding is an illumination/exposure tradeoff, not a
demonstrated movement/exposure tradeoff sufficient for cooperative learning.

Policy-resolution irradiance MAE for 9 vs 33 shutter samples was 2.15e-5 at a
selected close, moving dark state. This local sampling check does not validate
all motions or substitute for real-camera calibration.

### Gradient paths

The real 24-step rollout audit preserves identical forward trajectories for
each gradient cut. Camera navigation-gradient norm is 0.0291 with full sensing
and zero at the camera-boundary cut. The motion-to-observation contribution
to the first action gradient is 0.000728. Full-vs-motion-cut flight gradient
cosine is 0.9999997 at this untrained test state.

This confirms connected paths, not their optimization utility. The motion
contribution is small in this probe and cannot be treated as evidence that
joint learning will improve navigation.

## Next experiment boundary

1. First establish a reliable flight reference at fixed settings. Investigate
   tracking/disturbance errors explicitly; a separately declared wider slit
   may be used to separate these errors from camera learning. Preserve the
   original narrow-slit benchmark and report each geometry separately.
2. At fixed illumination, test whether real-scale changes of motion alter a
   task-relevant measurement choice. Add pose/appearance nuisance variation
   to the optimistic template diagnostic. Consider plausible shutter range,
   resolution and approach motion from the actual target camera/task. Do not
   amplify blur or noise merely to manufacture an advantage. A different best
   exposure is one sufficient screen, not a necessary theorem for cooperation.
3. Align flight intent with the camera's effective acquisition time using the
   pending flight commands and actuator response. Verify that predicted motion
   matches executed motion before attributing a camera decision to intent.
4. Compare success under equal deadlines and time/energy tradeoffs; avoid
   silently changing the velocity objective or supplying image-quality labels.
5. Only after these checks support a measurable cooperative benefit, compare
   trained joint/full, motion-gradient-cut, no-intent, classical AE, and a
   learned nondifferentiable baseline that has a genuine task-learning signal.
   Keep network capacity and resources matched. More seeds are not the current
   bottleneck.

If physical screening continues to show that ordinary adaptive exposure solves
the useful part of the task, keep the narrower camera-learning claim. Do not
promote the scene into evidence for a stronger coordination claim.

## Reproduce and inspect

```bash
/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python tools/audit_joint_slit.py \
  --out_dir logs/joint_slit_audit_new --device cuda --noise_samples 128
/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python tools/verify_intent_closed_loop.py \
  --config configs/gray_joint_slit.args --out_dir logs/joint_slit_gradient_new \
  --device cuda --rollout_steps 24
/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python -m pytest -q tests
```

Tracked numerical tables, provenance and preview are in
`joint_slit_audit_20261029/`. Full oracle trajectories and source snapshots
remain in the local run directories. Project tests: **142 passed**.

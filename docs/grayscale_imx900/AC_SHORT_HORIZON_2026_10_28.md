# Short-horizon actor-critic implementation audit — 2026-10-28

## Purpose

The earlier full-vs-sensing-cut implementation could produce a zero camera
gradient in the cut arm. That is a valid causal intervention, but it is not a
fair learned-camera control: the cut arm had no estimator for a camera action.
This branch therefore adds an opt-in short-horizon actor-critic comparison.

The experiment is deliberately small. It checks that both conditions can learn
camera commands under the same protocol; it is not a performance claim and it
does not replace the frozen provisional-simulator result.

## Estimator

At a 16-control-step boundary, the rollout is truncated. The task costs are the
existing velocity, avoidance, collision, action, jerk, terminal and camera
regularization terms with the existing terminal mask. A 128-128-1 latent critic
bootstraps the next segment. Critic weights are detached in the actor objective,
while the segment-end latent retains its gradient into the preceding segment.

The full condition uses the pathwise image-formation derivative. The
`nondiff_sensing` condition cuts exposure/gain and pose/motion image gradients,
then uses the same Gaussian camera-logit exploration and a score-function return
estimator. Flight remains trained by the differentiable dynamics path in both
conditions. Camera exploration is in logit space with standard deviation 0.25.
The critic has 33,153 training-only parameters; the deployed actor remains the
same 200,605-parameter network.

## Verification

- `pytest -q`: **130 passed**.
- `tools/verify_gray_navigation_gradient.py`: full camera gradient
  `0.00358485`, sensing-cut gradient `0`; forward loss is identical.
- `tools/verify_intent_closed_loop.py --config configs/gray_joint_parallax_ac.args`:
  full camera gradient `0.0282079`, camera-boundary cut `0`, forward trajectories
  identical.
- Configuration preflight rejects stale replay datasets before the first GPU
  run. The tests cover sampled-logit semantics, masked terminal costs, segment
  truncation, score-function sign, critic gradients, and guidance exclusions.

## Matched pilot

Command configuration: `configs/gray_joint_parallax_ac.args`, common flight-only
imitation checkpoint, 20 optimizer updates, batch 4, 120 control steps, seed
101, and 16 paired nominal evaluation episodes with seeds 798000–798015.
The provisional calibration remains `calibrated: false`.

| condition | training updates | evaluation success | collision |
| --- | ---: | ---: | ---: |
| full pathwise | 20 | 8/16 (50%) | 8/16 |
| sensing-cut + score function | 20 | 8/16 (50%) | 8/16 |
| common actor before update | — | 8/16 (50%) | 8/16 |

The paired difference is zero on these particular episodes. The degenerate
episode-bootstrap interval [0, 0] reflects identical paired outcomes and cannot
establish population equivalence. Both conditions kept the common actor's initial
behavior: the near-pole-positive layouts succeeded 8/9 and the mirrored
near-pole-negative layouts succeeded 0/7. Exposure and gain stayed essentially
at the initialization (`E≈0.080`, `G≈0.050`). Therefore this pilot shows that
the score-function camera path is wired and numerically trainable; it does not
show learned measurement selection, mirrored-layout recovery, or an advantage
for differentiable sensing.

Online W&B initialization timed out during one attempt. The successful pilot
was recorded in offline W&B runs and in
`logs/short_ac_20261028_pilot_v4/pilot_manifest.json`; it can be synchronized
later with `wandb sync`. The manifest verifies matched actor/critic hashes,
replay targets, source hashes and evaluation seeds.

## Relation to nearby work

This implementation follows established first-order actor-critic ideas rather
than claiming algorithmic novelty. Relevant examples are:

- [Adaptive Horizon Actor-Critic for Policy Learning in Contact-Rich
  Differentiable Simulation](https://arxiv.org/abs/2405.17784), which uses a
  value bootstrap to limit error from stiff long rollouts.
- [Learning Vision-Based Agile Gap Traversal: Differentiable Simulation with a
  Warm-Started Critic](https://arxiv.org/abs/2609.30696), which combines a
  critic with quasi-analytical policy gradients for visual gap traversal.
- [VisFly-Lab: Unified Differentiable Framework for First-Order Reinforcement
  Learning of Quadrotor Control](https://arxiv.org/abs/2603.21123), which uses
  value-based auxiliary objectives and visited-state initialization to improve
  differentiable quadrotor training.

The project-specific question remains whether the **camera action** benefits
from the image-formation gradient when both arms have a legitimate learning
estimator. The current pilot leaves that question open.

## Limits of this comparison

The nondifferentiable condition cuts both E/G and motion-to-image derivatives;
this is not an E/G-only causal intervention. Flight still uses model gradients,
so the comparator is not model-free RL or a tuned PPO baseline. The critic
observes the actor's latent memory, which is only an approximate sufficient
state for delayed camera commands and hidden scene depth. Hard termination,
masked-loss normalization, STE gradients and fixed-horizon bootstrapping also
make this an approximate estimator, not an unbiased gradient guarantee.

The implementation retains all segment graphs until the end of the episode;
it limits gradient horizon but does not yet have the memory scaling of a
streamed SHAC implementation. Resource comparisons must count actual wall time
and memory as well as environment transitions. The pilot used equal transition
and update budgets, not equal wall-clock compute.

The online literature metadata above was checked through OpenAlex. Direct
arXiv access timed out in this session; the new preprints have not been audited
against their full methods or confirmed as journal publications.

## Next experiment

Do not automatically extend this 20-update pilot and do not claim superiority.
Keep scene physics and sensors unchanged. First validate post-measurement
flight competence on both mirrored layouts, with the same externally imposed
view change. Then inspect whether balanced, identical flight-only guidance on
live actor observations corrects the missing navigation skill without labeling
initial measurement motion or E/G. Any teaching resources must be disclosed and
shared. Only after that gate should a longer estimator comparison be run, with
fixed budgets, camera-action movement and layout-conditioned outcomes. The
existing frozen IMX900 simulator results remain unchanged.

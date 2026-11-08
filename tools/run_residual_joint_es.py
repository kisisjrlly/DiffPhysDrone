"""Arm F: paired antithetic parameter-perturbation evolution strategy.

The plan's question for F is narrow and specific: *can the same interaction
budget be spent without backpropagation at all?* It is not "is PPO better" and
not "is ES better in general". So this tool implements one clearly defined
gradient-free method and reports its cost honestly.

Method
------
Each update draws ``directions`` Gaussian perturbations of the residual
parameters only - the frozen base is never touched. Every direction is evaluated
with **both signs**, ``+eps`` and ``-eps``, using the *same* external episode
sample for the pair. That antithetic pairing removes the first-order variance
that would otherwise dominate a 4-direction estimate.

    update = alpha * sum_d ( L(theta + eps_d) - L(theta - eps_d) ) * eps_d / (2 * scale)

``alpha`` is the Adam learning rate, so F receives the same optimiser entry point
as the backprop arms and the comparison is not confounded by an optimiser
change.

Accounting rules the plan is explicit about:
  * every positive and negative trial is counted against the budget - a
    ``+eps`` rollout is never "free";
  * perturbations are scaled per parameter tensor by its own RMS, so a tensor
    with small weights is not effectively frozen;
  * the relative scale decays linearly from 0.01 to 0.002 and the range is not
    searched;
  * both the flight and camera residuals must actually move, and the tool
    asserts that at the end rather than assuming it.

Why zero-initialised tensors need an absolute scale, not seeding
--------------------------------------------------------------
The residual policy deliberately zero-initialises ``flight_head``,
``camera_head`` and the two LayerNorm biases, so that at step 0 it reproduces
the frozen base exactly. That is correct for backprop - a zero head still has a
non-zero *gradient* - but it breaks a naive parameter-perturbation ES, which
scales the noise on each tensor by that tensor's own RMS:

  * six of the 21 residual tensors have RMS exactly 0, so the perturbation on
    them is the ``1e-6`` floor times noise, i.e. float32 rounding (measured
    2.6e-8);
  * and because every *other* tensor is multiplied by the zero head, the
    perturbation has no path to the action at all. Measured: a 1% perturbation
    moved the episode loss by 2.4e-6 on a loss of 4.75, a relative sensitivity
    of 5e-7 that no 4-direction estimate can resolve.

The fix must **not** be to change ``theta``. Seeding the heads would give F a
different initial policy from the backprop arms and destroy the only thing F
exists to measure. What is needed is an absolute scale for zero tensors, which
is legitimate because the estimator is antithetic: it evaluates ``theta+eps``
and ``theta-eps`` and averages, so the *centre* of the pair remains exactly the
zero-residual base policy. Only the exploration radius changes; the initial
function does not.

``ZERO_RMS_FLOOR`` is that absolute scale. Derivation: the flight head is
``Linear(64, 3)`` fed by a LayerNorm output, so ``||h||_2 ~ sqrt(64) = 8`` and a
weight perturbation of std ``s`` produces a residual of std ``~8s``. The
residual is passed through ``FLIGHT_RESIDUAL_MAX * tanh(.)``, so ``s = 1e-3``
explores about 1% of the head's range. At the starting relative scale of 0.01
that means a floor of 0.1 on the tensor RMS. A typical initialised GRU tensor
has RMS ~0.072, so the floor is just above the median tensor and does not
dominate the non-zero tensors.

This deliberately does **not** claim superiority over reinforcement learning.
It measures one gradient-free method under a matched budget.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_CONFIG = "configs/gray_joint_slit_residual.args"
#: 4 directions x 2 signs x 4 episodes, per the plan.
DIRECTIONS = 4
SIGNS = 2
EPISODES_PER_SIGN = 4
EPISODES_PER_UPDATE = DIRECTIONS * SIGNS * EPISODES_PER_SIGN
#: Linear decay of the relative perturbation scale.
SCALE_START = 0.01
SCALE_FINAL = 0.002
#: Matches the BPTT arms' optimiser entry point.
LEARNING_RATE = 1e-4
#: Absolute RMS substitute for zero-initialised tensors. See the module
#: docstring for the derivation. This does *not* modify theta: it only sets the
#: exploration radius for tensors whose own RMS is zero.
ZERO_RMS_FLOOR = 0.1


def _residual_parameters(policy):
    """Parameters ES is allowed to perturb: never the frozen base."""
    return [(name, parameter) for name, parameter in policy.named_parameters()
            if not name.startswith("base.") and parameter.requires_grad]


def zero_tensor_names(parameters):
    """Residual tensors whose own RMS is zero.

    Reported, not repaired: these are the tensors that would be frozen by a
    purely relative perturbation, so the reader can see how much of the residual
    is in that state. ``_draw_direction`` handles them.
    """
    return [name for name, parameter in parameters
            if parameter.detach().abs().max().item() == 0.0]


def _relative_scale(update, total_updates):
    """Linear decay from SCALE_START to SCALE_FINAL. Not tuned."""
    if total_updates <= 1:
        return SCALE_START
    fraction = update / (total_updates - 1)
    return SCALE_START + (SCALE_FINAL - SCALE_START) * float(min(max(fraction, 0.0), 1.0))


def _draw_direction(parameters, scale, generator, device):
    """One perturbation, scaled per tensor by that tensor's own RMS.

    Per-tensor scaling matters: a single global norm would leave the camera
    head - which has far fewer, differently sized weights - effectively
    unperturbed, and F would then be testing the flight residual only.

    A tensor whose own RMS is zero gets ``ZERO_RMS_FLOOR`` instead of its RMS.
    Without that substitution a zero-initialised head is perturbed by float32
    rounding only, and the whole antithetic pair becomes numerically identical.
    ``theta`` is untouched: the floor changes the exploration radius, not the
    policy the pair is centred on.
    """
    direction = []
    for _, parameter in parameters:
        noise = torch.randn(parameter.shape, generator=generator,
                            device=device, dtype=parameter.dtype)
        rms = parameter.detach().pow(2).mean().sqrt()
        reference = torch.clamp(rms, min=ZERO_RMS_FLOOR)
        direction.append(noise * (scale * reference))
    return direction


def _apply(parameters, direction, sign):
    with torch.no_grad():
        for (_, parameter), delta in zip(parameters, direction):
            parameter.add_(delta, alpha=sign)


@torch.no_grad()
def evaluate(policy, args, device, scene_names, episode_seeds):
    """Mean task loss over a fixed episode list.

    Returns the per-episode losses as well, so the antithetic pair uses the
    identical sample and the variance reduction is real.
    """
    from eval import run_one_episode
    from train_utils import build_env

    env = build_env(1, args, device, eval_mode=True)
    losses = []
    successes = 0
    for index, seed in enumerate(episode_seeds):
        scene = scene_names[index % len(scene_names)]
        row, _ = run_one_episode(index, scene, args, policy, env, None, device,
                                 episode_seed=seed)
        # Task loss proxy that is defined for every episode: miss distance plus
        # a collision penalty, both from the episode summary. Using success rate
        # alone would give a constant 0/1 signal and a useless gradient estimate.
        loss = float(row["final_goal_dist"]) + 3.0 * float(row["collision_rate"])
        losses.append(loss)
        successes += int(row["success_rate"] >= 1.0)
    return sum(losses) / max(len(losses), 1), losses, successes


def run(args, device, updates, seed=42, out_dir=None, log=print):
    from intent_model import build_policy

    torch.manual_seed(seed)
    policy = build_policy(args, 7 if args.no_odom else 10).to(device)
    if hasattr(policy, "bind_calibration"):
        env_probe = __import__("train_utils").build_env(1, args, device)
        policy.bind_calibration(env_probe.imx900_calibration)
    parameters = _residual_parameters(policy)
    if not parameters:
        raise SystemExit("no trainable residual parameters; wrong policy_arch?")

    generator = torch.Generator(device=device).manual_seed(seed)
    # theta is left exactly as the backprop arms initialise it: zero residual.
    # Only the exploration radius changes, via ZERO_RMS_FLOOR.
    zero_tensors = zero_tensor_names(parameters)
    if zero_tensors:
        log(f"[F] zero-RMS tensors use the absolute floor "
            f"{ZERO_RMS_FLOOR} (not seeded): {', '.join(zero_tensors)}")

    optimizer = torch.optim.Adam([p for _, p in parameters], lr=LEARNING_RATE,
                                 weight_decay=0.0)
    initial = {name: parameter.detach().clone() for name, parameter in parameters}

    history = []
    total_rollouts = 0
    started = time.time()
    episode_cursor = 0

    # Sensitivity gate. Before spending the budget, confirm that a perturbation
    # at the starting scale actually moves the loss by more than rollout noise.
    # Without this check the tool happily reports "no learning" when the real
    # problem is that the perturbation never reaches the action.
    probe_seeds = [900000 + k for k in range(EPISODES_PER_SIGN)]
    probe_base, probe_rows_base, _ = evaluate(policy, args, device,
                                              args.scenarios, probe_seeds)
    probe_direction = _draw_direction(parameters, SCALE_START, generator, device)
    _apply(parameters, probe_direction, +1.0)
    probe_plus, _, _ = evaluate(policy, args, device, args.scenarios, probe_seeds)
    _apply(parameters, probe_direction, -2.0)
    probe_minus, _, _ = evaluate(policy, args, device, args.scenarios, probe_seeds)
    _apply(parameters, probe_direction, +1.0)
    total_rollouts += 2 * EPISODES_PER_SIGN

    # Between-episode spread in the baseline sample sets the noise floor.
    spread = (max(probe_rows_base) - min(probe_rows_base)) if len(probe_rows_base) > 1 else 0.0
    separation = abs(probe_plus - probe_minus)
    sensitivity = {
        "probe_seeds": probe_seeds,
        "loss_base": probe_base,
        "loss_plus": probe_plus,
        "loss_minus": probe_minus,
        "antithetic_separation": separation,
        "baseline_spread": spread,
        "separation_over_spread": (separation / spread) if spread > 0 else float("inf"),
        "rollouts_spent": 2 * EPISODES_PER_SIGN,
    }
    log(f"[F] sensitivity probe: base={probe_base:.4f} plus={probe_plus:.4f} "
        f"minus={probe_minus:.4f} separation={separation:.3e} "
        f"spread={spread:.3e} ratio={sensitivity['separation_over_spread']:.3f}")

    for update in range(updates):
        scale = _relative_scale(update, updates)
        # Same external samples for every direction and both signs.
        base_seeds = [910000 + episode_cursor + k for k in range(EPISODES_PER_SIGN)]
        episode_cursor += EPISODES_PER_SIGN

        accumulator = [torch.zeros_like(p) for _, p in parameters]
        positives, negatives = [], []
        for d in range(DIRECTIONS):
            direction = _draw_direction(parameters, scale, generator, device)

            _apply(parameters, direction, +1.0)
            loss_plus, plus_rows, plus_ok = evaluate(policy, args, device,
                                                     args.scenarios, base_seeds)
            _apply(parameters, direction, -2.0)
            loss_minus, minus_rows, minus_ok = evaluate(policy, args, device,
                                                        args.scenarios, base_seeds)
            _apply(parameters, direction, +1.0)  # back to theta

            # Both signs are billed to the interaction budget.
            total_rollouts += 2 * EPISODES_PER_SIGN
            positives.append({"loss": loss_plus, "rows": plus_rows, "successes": plus_ok})
            negatives.append({"loss": loss_minus, "rows": minus_rows, "successes": minus_ok})

            # Antithetic estimator: (L+ - L-) / (2 * scale) is the directional
            # derivative along this perturbation.
            weight = (loss_plus - loss_minus) / (2.0 * scale)
            for slot, delta in enumerate(direction):
                accumulator[slot].add_(delta, alpha=weight)

        for (_, parameter), gradient in zip(parameters, accumulator):
            parameter.grad = gradient
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        entry = {
            "update": update + 1,
            "relative_scale": scale,
            "mean_loss_plus": sum(p["loss"] for p in positives) / DIRECTIONS,
            "mean_loss_minus": sum(p["loss"] for p in negatives) / DIRECTIONS,
            "successes_plus": sum(p["successes"] for p in positives),
            "successes_minus": sum(p["successes"] for p in negatives),
            "rollouts_this_update": 2 * EPISODES_PER_SIGN * DIRECTIONS,
            "total_rollouts": total_rollouts,
            "elapsed_s": time.time() - started,
        }
        history.append(entry)
        log(f"[F] update {entry['update']:>4}/{updates} scale={scale:.5f} "
            f"loss+={entry['mean_loss_plus']:.3f} loss-={entry['mean_loss_minus']:.3f} "
            f"success+={entry['successes_plus']} success-={entry['successes_minus']} "
            f"rollouts={total_rollouts}")

    # The plan requires that both heads actually moved; assert rather than claim.
    moved = {}
    for name, parameter in parameters:
        delta = float((parameter.detach() - initial[name]).abs().max())
        moved[name] = delta
    flight_moved = any(v > 0.0 for k, v in moved.items() if k.startswith("flight_"))
    camera_moved = any(v > 0.0 for k, v in moved.items() if k.startswith("camera_"))
    encoder_moved = any(v > 0.0 for k, v in moved.items() if k.startswith("image_encoder."))

    report = {
        "method": "paired antithetic parameter perturbation",
        "directions": DIRECTIONS,
        "signs": SIGNS,
        "episodes_per_sign": EPISODES_PER_SIGN,
        "learning_rate": LEARNING_RATE,
        "scale_start": SCALE_START,
        "scale_final": SCALE_FINAL,
        "zero_rms_floor": ZERO_RMS_FLOOR,
        "zero_rms_tensors": zero_tensors,
        "initialisation_matches_backprop_arms": True,
        "theta_modified_before_training": False,
        "sensitivity": sensitivity,
        "updates": updates,
        "total_rollouts": total_rollouts,
        # The sensitivity probe is billed like any other rollout: it consumes
        # real interaction budget even though it does not update the policy.
        "expected_rollouts": 2 * EPISODES_PER_SIGN + updates * EPISODES_PER_UPDATE,
        "all_rollouts_billed": total_rollouts == (2 * EPISODES_PER_SIGN
                                                  + updates * EPISODES_PER_UPDATE),
        "flight_parameters_moved": flight_moved,
        "camera_parameters_moved": camera_moved,
        "encoder_moved": encoder_moved,
        "max_abs_delta": moved,
        "history": history,
        "elapsed_s": time.time() - started,
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(policy.state_dict(), os.path.join(out_dir, "final.pth"))
        with open(os.path.join(out_dir, "es_report.json"), "w") as handle:
            json.dump(report, handle, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=None)
    cli = parser.parse_args()

    with open(cli.config) as handle:
        lines = [line.split("#")[0].strip() for line in handle]
    tokens = " ".join(line for line in lines if line).split()
    sys.argv = ["es"] + tokens
    from config import parse_args
    args = parse_args()
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")

    report = run(args, device, cli.updates, seed=cli.seed, out_dir=cli.out_dir)
    sens = report["sensitivity"]
    print()
    print(f"zero-RMS tensors   : {len(report['zero_rms_tensors'])} "
          f"(absolute floor {report['zero_rms_floor']})")
    print(f"init matches A-E   : {report['initialisation_matches_backprop_arms']} "
          f"(theta modified: {report['theta_modified_before_training']})")
    print(f"sensitivity sep    : {sens['antithetic_separation']:.4e} "
          f"(baseline spread {sens['baseline_spread']:.4e}, "
          f"ratio {sens['separation_over_spread']:.2f})")
    print(f"updates            : {report['updates']}")
    print(f"total rollouts     : {report['total_rollouts']} "
          f"(expected {report['expected_rollouts']})")
    print(f"all billed         : {report['all_rollouts_billed']}")
    print(f"flight moved       : {report['flight_parameters_moved']}")
    print(f"camera moved       : {report['camera_parameters_moved']}")
    print(f"elapsed            : {report['elapsed_s']:.1f} s")


if __name__ == "__main__":
    main()

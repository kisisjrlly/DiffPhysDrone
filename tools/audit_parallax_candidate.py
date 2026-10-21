#!/usr/bin/env python3
"""Audit-only two-pole candidate: equal initial images, different depth order.

This is not yet a selectable training scene. Geometry overrides are explicit.
The privileged route tracker checks feasibility; it does not learn sensing.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from config import set_global_seed
from rollout_ops import render_gray_sensor
from tools.intent_common import load_intent_args
from tools.pole_navigation_teacher import PoleNavigationTeacher
from train_utils import build_env, write_run_provenance


def candidate_env(batch=2, seed=754001, *, jitter=False):
    args = load_intent_args("configs/gray_joint_poles.args", batch_size=batch,
                           pole_count=2, pole_lateral_range=.1, pole_lateral_jitter=0.,
                           pole_arena_half_width=.38, gray_texture_strength=0.,
                           gray_enable_noise=False, scenarios=["nominal"])
    set_global_seed(seed)
    env = build_env(batch, args, torch.device("cpu"))
    env.reset("nominal")
    env.gray_light_factor.fill_(1)
    depth = torch.tensor([1., 1.8])[None].expand(batch, -1).clone()
    if jitter:
        depth += torch.empty_like(depth).uniform_(-.05, .05)
    sign = (torch.arange(batch) % 2 * 2 - 1).float()
    env.cyl = torch.stack([depth + args.simple_start_x,
                           depth * .1 * torch.stack([sign, -sign], -1), depth * .06], -1)
    return env, args


@torch.no_grad()
def image_audit():
    env, args = candidate_env()
    stats, pictures = {}, {}
    for name, x, y in (("initial", -2., 0.), ("sideways_12cm", -2., .12),
                       ("forward_15cm", -1.85, 0.)):
        env.p[:, 0], env.p[:, 1] = x, y
        image, _ = render_gray_sensor(env, torch.full((2,), .08), torch.full((2,), .05))
        image = F.adaptive_avg_pool2d(image, (args.gray_nn_height, args.gray_nn_width))
        difference = (image[0] - image[1]).abs()
        stats[name] = {"policy_image_mae": float(difference.mean()),
                       "policy_image_max": float(difference.max()),
                       "different_policy_pixels_over_1e3": int((difference > .001).sum())}
        pictures[name] = image[:, 0].numpy()
    assert stats["initial"]["policy_image_max"] < 1e-4
    assert stats["sideways_12cm"]["policy_image_mae"] > .01
    return stats, pictures


@torch.no_grad()
def feasibility_audit(count=64):
    env, args = candidate_env(count, jitter=True)
    teacher = PoleNavigationTeacher(env, args, clearance_margin=.06)
    queue = [env.act] * 2
    alive = torch.ones(count, dtype=torch.bool)
    reached, collision = ~alive, ~alive
    min_clearance = torch.full((count,), 100.)
    for step in range(args.timesteps):
        _, command = teacher.command(env, step / args.base_control_freq)
        queue.append(command)
        env.run(queue[step], 1 / args.base_control_freq, env.p_target - env.p)
        clearance = env.transition_clearance()
        min_clearance = torch.minimum(min_clearance, torch.where(alive, clearance, torch.full_like(clearance, 100)))
        hit = clearance <= args.collision_clearance
        goal = (env.p - env.p_target).norm(dim=-1) < .35
        reached |= alive & ~hit & goal
        collision |= alive & hit
        alive &= ~(hit | goal)
    return {"episodes": count, "successes": int(reached.sum()), "collisions": int(collision.sum()),
            "timeouts": int(alive.sum()), "min_clearance_m": float(min_clearance.min()),
            "max_reference_duration_s": float(teacher.ends[:, -1].max()),
            "planned_clearance_m": .06, "privileged_tracker": True}, args


def figure(pictures, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from PIL import Image
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.1), gridspec_kw={"width_ratios": [.8, 1.6, 1.6]})
    env, _ = candidate_env()
    for row in range(2):
        ax = axes[row, 0]
        ax.axvspan(-.38, .38, color="#f1f3f5")
        for x, y, radius in env.cyl[row].tolist():
            ax.add_patch(Circle((y, x + 2), radius, color="#343a40"))
        ax.plot(0, 0, marker="^", color="#177e89", markersize=8)
        ax.set(xlim=(-.45, .45), ylim=(-.15, 2.1), xlabel="Lateral (m)", ylabel="Forward (m)")
        ax.set_xticks([-.3, 0, .3])
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal")
        ax.set_title(f"World {'AB'[row]}", fontsize=10)
        for col, name in ((1, "initial"), (2, "sideways_12cm")):
            axes[row, col].imshow(pictures[name][row], cmap="gray", vmin=0, vmax=.45, interpolation="nearest")
            axes[row, col].axis("off")
            axes[row, col].set_title("At start" if col == 1 else "After moving 12 cm", fontsize=10)
    fig.suptitle("Same first image; moving reveals which pole is nearer", fontsize=13)
    fig.text(.5, .01, "Design audit, not learned flight. Same display range [0, 0.45] for every image.", ha="center", fontsize=9, color="#5b6068")
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(out / "parallax_candidate.png", dpi=150, facecolor="white")
    plt.close(fig)
    with Image.open(out / "parallax_candidate.png") as img:
        img.thumbnail((1050, 700))
        img.convert("RGB").save(out / "parallax_candidate_preview.jpg", quality=86)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--episodes", type=int, default=64)
    cli = p.parse_args()
    if cli.episodes < 2:
        raise ValueError("audit requires both depth orders")
    out = Path(cli.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    stats, pictures = image_audit()
    routes, args = feasibility_audit(cli.episodes)
    report = {"status": "audit-only candidate; not yet a training scene", "calibrated": False,
              "definition": {"near_depth_m": 1., "far_depth_m": 1.8, "depth_jitter_m": .05,
                             "bearing_y_over_depth": .1, "radius_over_depth": .06,
                             "arena_half_width_m": .38, "uniform_albedo": True},
              "image_audit": stats, "feasibility": routes,
              "image_audit_limit": "No image noise; static views at specified poses. Not an executed sensing maneuver.",
              "initial_no_common_safe_lane": {"center_allowed_abs_y_m": .26,
                                              "minimum_union_obstacle_extent_abs_y_m": .272,
                                              "reason": "At the shared near-pole plane the two mirrored body-expanded obstacles cover every permitted lateral center position."},
              "claim": "Initial optical ambiguity and geometric/dynamic feasibility only; not learned coordination."}
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    write_run_provenance(out, args)
    figure(pictures, out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

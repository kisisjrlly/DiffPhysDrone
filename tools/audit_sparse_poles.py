#!/usr/bin/env python3
"""Audit a simple sensing/navigation task before comparing learning methods.

All masks and geometry in this tool are diagnostic only, never actor inputs.
The plotted trajectory is a privileged feasibility witness, not learned flight.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch
import torch.nn.functional as F

from config import set_global_seed
from intent_model import build_policy
from render.box_scene import box_clearance
from render.intent_scene import render_exposure
from render.pole_scene import trace_pole_scene, swept_cylinder_clearance
from render.pose import rotation_exp
from tools.audit_intent_scene import executed_dynamics_audit
from tools.intent_common import load_intent_args
from train_utils import build_env, write_run_provenance


def plan_diagnostic_path(env):
    """Shortest safe polyline over a small lateral lattice; privileged audit."""
    poles = env.cyl[0].cpu().numpy()
    stages = [np.array([[env.simple_start_x, 0.]])]
    ys = np.linspace(-env.pole_arena_half_width + env.drone_radius + .08,
                     env.pole_arena_half_width - env.drone_radius - .08, 19)
    stages.extend(np.column_stack([np.full_like(ys, x), ys]) for x in poles[:, 0])
    stages.append(np.array([[env.simple_goal_x, 0.]]))
    costs, routes = [0.], [[stages[0][0]]]
    for candidates in stages[1:]:
        new_costs, new_routes = [], []
        for point in candidates:
            options = []
            for cost, route in zip(costs, routes):
                a = route[-1]
                delta = point - a
                fraction = np.clip(((poles[:, :2] - a) * delta).sum(1) / (delta @ delta), 0, 1)
                nearest = a + fraction[:, None] * delta
                gap = np.linalg.norm(nearest - poles[:, :2], axis=1) - poles[:, 2] - env.drone_radius
                if gap.min() >= .14:
                    options.append((cost + np.linalg.norm(delta), route + [point]))
            if options:
                cost, route = min(options, key=lambda pair: pair[0])
                new_costs.append(cost)
                new_routes.append(route)
        if not new_routes:
            raise AssertionError("diagnostic lattice found no feasible route")
        costs, routes = new_costs, new_routes
    points = np.array(routes[int(np.argmin(costs))])
    # Quintic rest-to-rest segments: max |d²s/du²| = 10/sqrt(3).
    # Leave tracking headroom under the 3.5m/s² command bound.
    durations = np.sqrt((10 / np.sqrt(3)) * np.abs(np.diff(points, axis=0)).max(1) / 2.9)
    xyz = torch.as_tensor(np.column_stack([points, np.full(len(points), env.simple_slit_center_z)]),
                          device=env.p.device, dtype=env.p.dtype)[None]
    return xyz, durations.tolist()


def geometric_route_audit(env, xyz, durations):
    points = []
    for a, b in zip(xyz[0, :-1], xyz[0, 1:]):
        u = torch.linspace(0, 1, 101, device=xyz.device)
        points.append(a + (10 * u**3 - 15 * u**4 + 6 * u**5)[:, None] * (b - a))
    path = torch.cat(points)
    clear = swept_cylinder_clearance(path[:-1], path[1:], env.cyl.expand(len(path) - 1, -1, -1), env.drone_radius)
    bound = box_clearance(path[:, None], env.voxels, env.drone_radius)
    return {"duration_s": sum(durations), "min_clearance_m": float(torch.minimum(clear.min(), bound.min())),
            "max_reference_acceleration_component_m_s2": 2.9,
            "privileged_diagnostic_only": True}


def blind_lane_screen(args, device, count=2048):
    """Fixed paths under exact geometry, ignoring dynamics: optimistic screen.

    Not a trained blind baseline or an upper bound on all blind controllers.
    Quantifies the obvious 'always go to the same side' shortcut.
    """
    set_global_seed(721211)
    env = build_env(count, args, device)
    env.reset("nominal")
    rows = []
    for y in np.linspace(-args.pole_arena_half_width + env.drone_radius + .01,
                          args.pole_arena_half_width - env.drone_radius - .01, 41):
        # Move laterally before the first pole; return after the final one.
        points = [(args.simple_start_x, 0.), (-.8, y),
                  (args.simple_goal_x - .8, y), (args.simple_goal_x, 0.)]
        gap = torch.full((count,), 100., device=device)
        for a, b in zip(points[:-1], points[1:]):
            start = env.p.new_tensor([*a, 1.5])[None].expand(count, -1)
            end = env.p.new_tensor([*b, 1.5])[None].expand(count, -1)
            gap = torch.minimum(gap, swept_cylinder_clearance(start, end, env.cyl, env.drone_radius))
        path_length = sum(np.linalg.norm(np.asarray(b) - a) for a, b in zip(points[:-1], points[1:]))
        rows.append({"fixed_lateral_lane_m": y, "geometrically_collision_free_fraction": float((gap > 0).float().mean()),
                     "path_length_m": float(path_length)})
    return {"layout_seed": 721211, "layouts": count, "paths": rows,
            "interpretation": "Optimistic fixed-path geometry screen; no dynamics, not a learned blind benchmark."}


def observation_audit(args, device):
    set_global_seed(721212)
    env = build_env(1, args, device)
    env.reset("dark")
    env.cyl[0, :, 0] = env.cyl.new_tensor([0., 1.5, 3.])[:args.pole_count]
    env.cyl[0, :, 1] = env.cyl.new_tensor([0., -.48, .48])[:args.pole_count]
    env.p = env.p.new_tensor([[-2., 0., 1.5]])
    env.R = torch.eye(3, device=device)[None]
    env.R_old = env.R.clone()
    env.v.zero_()
    exposures = [.02, .08, .25, .60, .95]
    pictures, rows = {}, []
    def pool(image):
        return F.adaptive_avg_pool2d(image, (args.gray_nn_height, args.gray_nn_width))
    generator = torch.Generator(device=device).manual_seed(721213)
    shape = (128, 1, env.height, env.width)
    shot = torch.randn(shape, device=device, generator=generator)
    read = torch.randn(shape, device=device, generator=generator)
    for scenario in ("dark", "bright"):
        env.current_scene_names = [scenario]
        for exposure in exposures:
            samples, variances, masks = [], [], []
            for side in (-1, 1):
                env.cyl[0, 0, 1] = side * .10
                image = render_exposure(env, env.p.new_tensor([exposure]))[0]
                _, _, ids = trace_pole_scene(env, env.p, env.R)
                masks.append(pool((ids == env.voxels.shape[1]).float()[:, None]) > .02)
                noisy, _ = env.gray_camera(image.expand(128, -1, -1, -1), image.new_full((128,), exposure),
                                           image.new_full((128,), .05), motion=None,
                                           shot_noise=shot, read_noise=read, enable_noise=True)
                observations = pool(noisy)
                samples.append(observations.mean(0))
                variances.append(observations.var(0))
            roi = (masks[0] | masks[1])[0]
            signal = ((samples[0] - samples[1]).square() / (variances[0] + variances[1]).clamp_min(1e-8))[roi].mean().sqrt()
            rows.append({"scenario": scenario, "exposure01": exposure,
                         "exposure_us": float(env.gray_camera.calibration.exposure_to_us(env.p.new_tensor([exposure]))),
                         "pole_position_signal_over_noise_in_roi": float(signal),
                         "roi_policy_pixels": int(roi.sum())})
            pictures[(scenario, exposure)] = observations[0, 0].cpu().numpy()
    # Show blur honestly at physically specified motion, without a multiplier.
    env.current_scene_names = ["dark"]
    env.cyl[0, 0, 1] = .10
    env.v = env.v.new_tensor([[1.2, 1., 0.]])
    env.R_old = env.R @ rotation_exp(env.p.new_tensor([[0., 0., -3. * env.last_control_dt]]))
    integrations = {}
    for n in (3, 9, 17):
        env.gray_temporal_samples = n
        integrations[n] = render_exposure(env, env.p.new_tensor([.95]))[0]
    convergence = {f"{n}_vs_17_irradiance_mae": float((im - integrations[17]).abs().mean())
                   for n, im in integrations.items() if n != 17}
    fx = args.gray_nn_width / (2 * args.fov_x_half_tan)
    return {"pose": [-2., 0., 1.5], "alternative_pole_y_m": [-.10, .10],
            "figure_camera_images": "single noisy frames at policy resolution; not noise-averaged images",
            "gain01": .05, "independent_noise_samples_per_condition": 128,
            "policy_resolution_wh": [args.gray_nn_width, args.gray_nn_height],
            "diameter_policy_pixels_at_2m": [fx * 2 * args.pole_radius_min / 2, fx * 2 * args.pole_radius_max / 2],
            "noise_seed": 721213, "rows": rows,
            "temporal_sample_convergence": convergence,
            "nominal_yaw_blur_pixels_at_max_exposure": fx * 3 * env.gray_camera.calibration.exposure_us_max * 1e-6,
            "interpretation": "Local pixel discriminability, not learned obstacle detection or navigation."}, pictures


def audit(out_dir, device="cuda"):
    device = torch.device(device)
    args = load_intent_args("configs/gray_joint_poles.args", batch_size=1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_global_seed(721210)
    env = build_env(1, args, device)
    env.reset("dark_to_bright")
    scene = env.get_scene_effects_for_env()
    xyz, durations = plan_diagnostic_path(env)
    geometry = geometric_route_audit(env, xyz, durations)
    assert geometry["min_clearance_m"] > 0 and sum(durations) < args.timesteps / args.base_control_freq
    dynamics = executed_dynamics_audit(env, args, local_points=xyz, durations=durations)
    assert dynamics["success"] == [True] and dynamics["collision"] == [False]
    # Independent feasibility layouts, not a claim that a learned actor succeeds.
    checks = []
    for seed in range(721220, 721236):
        set_global_seed(seed)
        e = build_env(1, args, device)
        e.reset("dark")
        route, times = plan_diagnostic_path(e)
        record = geometric_route_audit(e, route, times)
        record["seed"] = seed
        result = executed_dynamics_audit(e, args, local_points=route, durations=times)
        record.update(success=result["success"][0], collision=result["collision"][0],
                      executed_min_clearance_m=result["min_transition_clearance_m"][0])
        checks.append(record)
    lane_screen = blind_lane_screen(args, device)
    observation, pictures = observation_audit(args, device)
    parameters = sum(p.numel() for p in build_policy(args).parameters())
    result = {"scene_layout": args.scene_layout, "scene_seed": 721210, "scene": scene,
              "calibrated": False, "actor_parameter_count": parameters,
              "geometric_witness": geometry, "executed_privileged_witness": dynamics,
              "independent_feasibility_checks": checks,
              "blind_fixed_path_screen": lane_screen, "observation_audit": observation,
              "claim": "Scene implementation and diagnostic feasibility only; learned coordination and algorithm superiority untested."}
    (out / "scene_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    write_run_provenance(out, args)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13, 7), constrained_layout=True, facecolor="white")
    grid = fig.add_gridspec(3, 5, height_ratios=[1.4, 1, 1])
    ax = fig.add_subplot(grid[0, :3])
    for x, y, r in scene["poles_x_y_radius"]:
        ax.add_patch(Circle((x, y), r, color="#4a5665"))
        ax.add_patch(Circle((x, y), r + .12, fill=False, ls=":", ec="#8d96a0"))
    path = np.array(dynamics["local_trajectory"])[:, 0, :2]
    ax.plot(path[:, 0], path[:, 1], color="#286b91", label="Privileged feasibility witness")
    ax.scatter([args.simple_start_x, args.simple_goal_x], [0, 0], c=["#286b91", "#b45b38"], zorder=3)
    ax.axhline(-args.pole_arena_half_width, c="#bcc3cb"); ax.axhline(args.pole_arena_half_width, c="#bcc3cb")
    ax.set(xlim=(-2.3, 5.5), ylim=(-args.pole_arena_half_width - .1, args.pole_arena_half_width + .1), xlabel="Forward (m)", ylabel="Lateral (m)",
           title="Three round poles; no gate or prescribed avoidance side")
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="box")
    ax = fig.add_subplot(grid[0, 3:])
    for scenario, color in (("dark", "#286b91"), ("bright", "#b45b38")):
        rows = [r for r in observation["rows"] if r["scenario"] == scenario]
        ax.plot([r["exposure_us"] / 1000 for r in rows],
                [r["pole_position_signal_over_noise_in_roi"] for r in rows], "o-", color=color, label=scenario)
    ax.set(xlabel="Exposure (ms)", ylabel="Pole-position signal / noise", title="At the actor's actual 48 × 36 resolution")
    ax.legend(frameon=False)
    for row, scenario in enumerate(("dark", "bright"), 1):
        for col, exposure in enumerate([.02, .08, .25, .60, .95]):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(pictures[(scenario, exposure)], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 1:
                us = next(r["exposure_us"] for r in observation["rows"] if r["exposure01"] == exposure)
                ax.set_title(f"{us / 1000:.2f} ms")
            if col == 0:
                ax.set_ylabel(scenario.capitalize())
    fig.suptitle("Sparse-pole scene audit — provisional simulator; trajectory is not a learned policy", fontsize=13)
    fig.savefig(out / "scene_audit.png", dpi=150)
    fig.set_size_inches(10, 5.4)
    fig.savefig(out / "scene_audit_preview.jpg", dpi=95, pil_kwargs={"quality": 85})
    plt.close(fig)
    print(json.dumps({"output": str(out), "geometry": geometry,
                      "executed_success": dynamics["success"],
                      "independent_feasibility_successes": sum(r["success"] for r in checks),
                      "independent_feasibility_count": len(checks),
                      "best_fixed_lane_geometric_fraction": max(r["geometrically_collision_free_fraction"] for r in lane_screen["paths"]),
                      "actor_parameters": parameters, "observation": observation}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="logs/sparse_poles/scene_audit")
    p.add_argument("--device", default="cuda")
    cli = p.parse_args()
    with torch.no_grad():
        audit(cli.out_dir, cli.device)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit geometry, radiometry and temporal blur before interpreting training.

Oracle waypoints and image masks below are diagnostic only. They are never
fed to the policy or used as training labels.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from config import set_global_seed
from render.box_scene import box_clearance, swept_box_collision, trace_boxes
from render.intent_scene import build_intent_gates, render_boxes_irradiance, render_exposure
from render.pose import rotation_exp
from rollout_ops import build_local_frame, decode_action_direct, render_gray_sensor
from tools.intent_common import load_intent_args
from train_utils import build_env, write_run_provenance


def feasible_path(env):
    """Smooth geometric feasibility witness, not an expert-trained policy."""
    yc = env.gate_centers_y[0].detach().cpu().numpy()
    points = np.array([[env.simple_start_x, 0], [float(env.gate_x[0]), yc[0]],
                       [float(env.gate_x[1]), yc[1]], [env.simple_goal_x, 0]])
    durations = [2.2, 2.6, 2.2]
    path, accelerations, speeds = [], [], []
    for a, b, duration in zip(points[:-1], points[1:], durations):
        u = np.linspace(0, 1, 151)
        s = 10 * u**3 - 15 * u**4 + 6 * u**5
        ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
        dds = (60 * u - 180 * u**2 + 120 * u**3) / duration**2
        path.append(a + s[:, None] * (b - a))
        accelerations.append(dds[:, None] * (b - a))
        speeds.append(np.linalg.norm(ds[:, None] * (b - a), axis=1))
    path = np.concatenate(path)
    xyz = torch.as_tensor(np.column_stack([path, np.full(len(path), env.simple_slit_center_z)]),
                          device=env.p.device, dtype=env.p.dtype)[:, None]
    clearance = box_clearance(xyz, env.voxels, env.drone_radius)
    sweeps = swept_box_collision(xyz[:-1, 0], xyz[1:, 0],
                                 env.voxels.expand(len(xyz) - 1, -1, -1),
                                 env.drone_radius)
    return path, {"min_spherical_clearance_m": float(clearance.min()), "duration_s": sum(durations),
                  "conservative_swept_collision": bool(sweeps.any()),
                  "max_acceleration_component_m_s2": float(np.abs(np.concatenate(accelerations)).max()),
                  "peak_speed_m_s": float(np.concatenate(speeds).max()),
                  "note": "Geometric/kinematic witness; does not establish closed-loop policy success."}


@torch.no_grad()
def executed_dynamics_audit(template_env, args):
    """Privileged waypoint tracker checks reachability in the actual dynamics.

    This is deliberately an oracle diagnostic, never actor supervision or a
    comparison method. It uses the identical command queue and collision rule.
    """
    env = copy.deepcopy(template_env)
    batch, dt = env.batch_size, 1 / args.base_control_freq
    z = env.p.new_full((batch,), env.simple_slit_center_z)
    local_points = torch.stack([
        torch.stack([z * 0 + env.simple_start_x, z * 0, z], -1),
        torch.stack([z * 0 + env.gate_x[0], env.gate_centers_y[:, 0], z], -1),
        torch.stack([z * 0 + env.gate_x[1], env.gate_centers_y[:, 1], z], -1),
        torch.stack([z * 0 + env.simple_goal_x, z * 0, z], -1),
    ], 1)
    points = (env.R_scene[:, None] @ local_points[..., None]).squeeze(-1)
    durations = [2.2, 2.6, 2.2]

    def reference(time):
        index = 0
        while index < 2 and time > durations[index]:
            time -= durations[index]
            index += 1
        duration = durations[index]
        u = min(max(time / duration, 0), 1)
        delta = points[:, index + 1] - points[:, index]
        s = 10 * u**3 - 15 * u**4 + 6 * u**5
        ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
        dds = (60 * u - 180 * u**2 + 120 * u**3) / duration**2
        return points[:, index] + s * delta, ds * delta, dds * delta

    queue = [env.act] * 2
    alive = torch.ones(batch, dtype=torch.bool, device=env.p.device)
    collided, reached = ~alive, ~alive
    minimum = env.p.new_full((batch,), 100)
    terminal_time = env.p.new_full((batch,), args.timesteps * dt)
    trajectory, maximum_command = [env.p.clone()], 0.0
    for step in range(args.timesteps):
        horizon = 2 * dt + 1 / env.fixed_pitch_ctl_delay
        p, v, a = reference(step * dt + horizon)
        predicted_p = env.p + env.v * horizon + .5 * env.a * horizon**2
        predicted_v = env.v + env.a * horizon
        command_world = a + 6 * (p - predicted_p) + 4 * (v - predicted_v) + env.fixed_drag_linear * env.v
        frame = build_local_frame(env)
        command_local = (command_world[:, None] @ frame).squeeze(1)
        command = decode_action_direct(command_local, frame, env, batch, args.max_acc_cmd)
        maximum_command = max(maximum_command, float(command_local.clamp(-args.max_acc_cmd, args.max_acc_cmd).abs().max()))
        queue.append(command)
        env.run(queue[step], dt, env.p_target - env.p)
        clearance = env.transition_clearance()
        minimum = torch.minimum(minimum, torch.where(alive, clearance, minimum))
        hit = clearance <= args.collision_clearance
        goal = (env.p - env.p_target).norm(dim=-1) < .35
        collided = collided | (alive & hit)
        reached = reached | (alive & ~hit & goal)
        terminal = alive & (hit | goal)
        terminal_time = torch.where(terminal, terminal_time.new_full((batch,), (step + 1) * dt), terminal_time)
        alive = alive & ~terminal
        trajectory.append(env.p.clone())
    local_trajectory = (env.R_scene_T[None] @ torch.stack(trajectory)[..., None]).squeeze(-1)
    return {
        "success": reached.cpu().tolist(), "collision": collided.cpu().tolist(),
        "min_transition_clearance_m": minimum.cpu().tolist(),
        "terminal_time_s": terminal_time.cpu().tolist(),
        "max_local_acceleration_command_m_s2": maximum_command,
        "local_trajectory": local_trajectory.cpu().tolist(),
        "privileged_oracle": True, "used_for_training": False,
    }


def decision_cue_audit(args, device, exposures):
    """Can the actual sensor distinguish left/right openings at a fixed state?

    Uses common random variates to reduce error in the mean-image difference;
    denominator is the sum of marginal variances for two independent frames.
    The normalized per-pixel signal is a diagnostic, not a classifier result.
    """
    records = {}
    for scenario in ("dark", "bright"):
        env = build_env(1, args, device)
        env.reset(scenario)
        irradiances = []
        for side in (-1, 1):
            centers = env.p.new_tensor([[.65, side * .65]])
            widths = env.p.new_tensor([[.4, .4]])
            build_intent_gates(env, centers=centers, half_widths=widths)
            env.p = env.p.new_tensor([[.65, 0., env.simple_slit_center_z]])
            env.R = torch.eye(3, device=device)[None]
            env.R_old = env.R.clone()
            irradiances.append(render_exposure(env, env.p.new_tensor([.1]))[0])
        rows = []
        generator = torch.Generator(device=device).manual_seed(510)
        shape = (128, 1, env.height, env.width)
        shot = torch.randn(shape, device=device, generator=generator)
        read = torch.randn(shape, device=device, generator=generator)
        for exposure in exposures:
            observations = []
            for image in irradiances:
                noisy, _ = env.gray_camera(image.expand(128, -1, -1, -1), image.new_full((128,), exposure),
                                           image.new_full((128,), .05), motion=None,
                                           shot_noise=shot, read_noise=read, enable_noise=True)
                observations.append((noisy.mean(0), noisy.var(0)))
            (a, va), (b, vb) = observations
            signal = ((a - b).square() / (va + vb).clamp_min(1e-8)).mean().sqrt()
            rows.append({"exposure01": exposure, "left_right_signal_over_noise": float(signal)})
        records[scenario] = rows
    assert records["dark"][-1]["left_right_signal_over_noise"] > records["dark"][0]["left_right_signal_over_noise"]
    assert records["bright"][0]["left_right_signal_over_noise"] > records["bright"][-1]["left_right_signal_over_noise"]
    return records


def audit(out_dir, device="cuda"):
    device = torch.device(device)
    args = load_intent_args(batch_size=1, gray_enable_noise=False)
    set_global_seed(21021)
    env = build_env(1, args, device)
    env.reset("dark_to_bright")
    path, feasibility = feasible_path(env)
    assert feasibility["min_spherical_clearance_m"] > 0
    assert not feasibility["conservative_swept_collision"]
    assert feasibility["max_acceleration_component_m_s2"] <= args.max_acc_cmd
    assert feasibility["duration_s"] <= args.timesteps / args.base_control_freq
    executed = executed_dynamics_audit(env, args)
    assert all(executed["success"]) and not any(executed["collision"])
    original_boxes = env.voxels.clone()
    effects = env.get_scene_effects_for_env()
    env.p = env.p.new_tensor([[-.65, float(env.gate_centers_y[0, 0]), env.simple_slit_center_z]])
    env.R = torch.eye(3, device=device)[None]
    env.R_old = env.R.clone()
    env.v.zero_()
    local_p = (env.R_scene_T @ env.p[:, :, None]).squeeze(-1)
    ideal, _ = render_boxes_irradiance(env, local_p, env.R)
    _, _, ids = trace_boxes(env.voxels, env.R, local_p, env.height, env.width, env._fov_x_half_tan)
    dark_mask, bright_mask = ids < 4, (ids >= 4) & (ids < 8)
    if not bright_mask.any():
        bright_mask = ids == 12  # bright back wall seen through the aperture
    exposures = [.02, .08, .25, .60, .95]
    rows, pictures = [], []
    calibration = env.gray_camera.calibration
    for exposure in exposures:
        e = torch.tensor([exposure], device=device)
        g = torch.tensor([.05], device=device)
        static, _ = render_gray_sensor(env, e, g)
        charge = ideal * calibration.exposure_to_relative(e)[:, None, None, None] * calibration.signal_scale
        factor = calibration.gain_to_factor(g)[:, None, None, None]
        shot = factor * (calibration.shot_noise_alpha * torch.sqrt(charge + 1e-6) + calibration.shot_noise_beta)
        read = calibration.read_noise_std(factor)
        snr = factor * charge / (shot.square() + read.square()).sqrt().clamp_min(1e-8)
        preclip = factor * charge + calibration.black_level
        env.v = env.v.new_tensor([[1.2, 1.0, 0.]])
        # A physically sized, explicit yaw rate (rad/s), not an arbitrary blur multiplier.
        env.R_old = env.R @ rotation_exp(env.v.new_tensor([[0., 0., -3.0 * env.last_control_dt]]))
        moving, aux = render_gray_sensor(env, e, g)
        pictures.append((static[0, 0].cpu().numpy(), moving[0, 0].cpu().numpy()))
        rows.append({"exposure01": exposure, "exposure_us": float(calibration.exposure_to_us(e)),
                     "dark_surface_snr_median": float(snr[:, 0][dark_mask].median()),
                     "bright_surface_saturation_fraction": float((preclip[:, 0][bright_mask] >= calibration.saturation_level).float().mean()),
                     "temporal_blur_irradiance_mae": float(aux["temporal_blur_mae"][0]),
                     "angular_speed_rad_s": float(aux["angular_speed_rad_s"][0])})
        env.v.zero_()
        env.R_old = env.R.clone()
    assert rows[-1]["dark_surface_snr_median"] > rows[0]["dark_surface_snr_median"]
    assert rows[-1]["bright_surface_saturation_fraction"] > rows[0]["bright_surface_saturation_fraction"]
    assert rows[-1]["temporal_blur_irradiance_mae"] > rows[0]["temporal_blur_irradiance_mae"]
    # Integration convergence is reported, never silently interpreted as calibration.
    env.v = env.v.new_tensor([[1.2, 1.0, 0.]])
    env.R_old = env.R @ rotation_exp(env.v.new_tensor([[0., 0., -3.0 * env.last_control_dt]]))
    integrated = {}
    for samples in (3, 9, 17):
        env.gray_temporal_samples = samples
        integrated[samples] = render_exposure(env, torch.tensor([.95], device=device))[0]
    convergence = {f"{samples}_vs_17_irradiance_mae": float((image - integrated[17]).abs().mean())
                   for samples, image in integrated.items() if samples != 17}
    decision_cues = decision_cue_audit(args, device, exposures)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 8), facecolor="white", constrained_layout=True)
    grid = fig.add_gridspec(3, 5, height_ratios=[1.4, 1, 1])
    ax = fig.add_subplot(grid[0, :3])
    for box in original_boxes[0].cpu().numpy():
        if abs(box[2] - env.simple_slit_center_z) < .01:
            ax.add_patch(Rectangle((box[0] - box[3], box[1] - box[4]), 2 * box[3], 2 * box[4], color="#c6ced8"))
    ax.plot(path[:, 0], path[:, 1], color="#236d9c", lw=2, label="Feasible diagnostic path")
    executed_xy = np.asarray(executed["local_trajectory"])[:, 0, :2]
    ax.plot(executed_xy[:, 0], executed_xy[:, 1], color="#bd6337", lw=1.5,
            ls="--", label="Executed oracle (not learned)")
    ax.scatter([env.simple_start_x, env.simple_goal_x], [0, 0], color=["#236d9c", "#bd6337"], zorder=4)
    ax.set(xlim=(args.simple_start_x - .3, args.simple_goal_x + .4), ylim=(-1.9, 1.9),
           xlabel="Forward position (m)", ylabel="Lateral position (m)", title="Independent apertures; fixed goal; world-fixed lighting")
    ax.legend(loc="lower right", frameon=False)
    ax = fig.add_subplot(grid[0, 3:])
    for scenario, color in (("dark", "#236d9c"), ("bright", "#bd6337")):
        ax.plot([r["exposure_us"] / 1000 for r in rows],
                [r["left_right_signal_over_noise"] for r in decision_cues[scenario]],
                "o-", color=color, label=scenario.capitalize())
    ax.set(xlabel="Exposure (ms)", ylabel="Left/right decision signal / noise",
           title="The relevant observation depends on exposure")
    ax.legend(frameon=False)
    for index, (static, moving) in enumerate(pictures):
        for row, image in ((1, static), (2, moving)):
            ax = fig.add_subplot(grid[row, index])
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 1:
                ax.set_title(f"{rows[index]['exposure_us'] / 1000:.2f} ms")
            if index == 0:
                ax.set_ylabel("Static" if row == 1 else "Translation + yaw")
    fig.suptitle("Scene physics audit — provisional IMX900 surrogate; diagnostic images, not a learned policy", fontsize=13)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "scene_audit.png", dpi=160)
    plt.close(fig)
    result = {"calibrated": False, "scene": effects, "feasible_path": feasibility,
              "executed_dynamics_audit": executed,
              "exposure_motion_tradeoff": rows, "temporal_sampling_convergence": convergence,
              "task_relevant_left_right_discriminability": decision_cues,
              "diagnostic_masks_are_not_policy_inputs": True,
              "claim": "Scene exhibits physical sensing tradeoffs; no claim of learned coordination or method superiority."}
    (out / "scene_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    write_run_provenance(out, args)
    brief = {key: value for key, value in result.items() if key != "executed_dynamics_audit"}
    brief["executed_dynamics_audit"] = {key: value for key, value in executed.items() if key != "local_trajectory"}
    print(json.dumps(brief, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="logs/intent_joint/scene_audit")
    parser.add_argument("--device", default="cuda")
    cli = parser.parse_args()
    with torch.no_grad():
        audit(cli.out_dir, cli.device)


if __name__ == "__main__":
    main()

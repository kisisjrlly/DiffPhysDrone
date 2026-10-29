#!/usr/bin/env python3
"""Test scene physics BEFORE training a claim of camera/flight coordination.

Oracle geometry, trajectories, light fields and matched image templates are
diagnostic only; none are actor observations or training targets. The decision
metric is an optimistic known-state template decoder, not learned navigation.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import set_global_seed
from render.intent_scene import render_exposure, render_scene_irradiance
from render.pose import body_angular_velocity, rotation_exp
from render.slit_scene import slit_surface_light
from tools.audit_intent_scene import executed_dynamics_audit
from tools.intent_common import load_intent_args
from train_utils import build_env, write_run_provenance


CONFIG = "configs/gray_joint_slit.args"
SCENARIOS = ("dark", "bright", "dark_to_bright", "bright_to_dark")
EXPOSURES = (.10, .30, .60, .85)
MOTIONS = {
    "stationary": ([0., 0., 0.], 0.),
    "translation": ([.8, .6, 0.], 0.),
    "translation_yaw_1": ([.8, .6, 0.], 1.),
    "translation_yaw_3_stress": ([.8, .6, 0.], 3.),
}


def set_motion(env, position, velocity, yaw_rate):
    env.p = env.voxels.new_tensor([position])
    env.v = env.p.new_tensor([velocity])
    env.R = torch.eye(3, device=env.p.device, dtype=env.p.dtype)[None]
    env.R_old = env.R @ rotation_exp(env.p.new_tensor([[0., 0., -yaw_rate * env.last_control_dt]]))


def set_slit_center(env, center):
    """Preserve everything but the two aperture-edge boxes, for matched images."""
    delta = center - env.slit_center_y[0]
    env.voxels[:, :2, 1] += delta
    env.slit_center_y[:] = center


def policy_resolution(image, args):
    return F.adaptive_avg_pool2d(image, (args.gray_nn_height, args.gray_nn_width))


def gain_for_exposure(env, exposure, mode):
    if mode == "fixed_gain":
        return .02
    # Compare blur at matched *pre-saturation mean signal*: longer exposure
    # collects more photons; shorter exposure is compensated with gain.
    calibration = env.gray_camera.calibration
    if calibration.gain_mapping != "log":
        raise ValueError("this matched-signal diagnostic requires the documented log-gain profile")
    ref = calibration.exposure_to_us(torch.tensor(.85)).item()
    duration = calibration.exposure_to_us(torch.tensor(exposure)).item()
    reference_gain = calibration.gain_to_factor(torch.tensor(.02)).item()
    factor = ref / duration * reference_gain
    factor_min, factor_max = calibration.gain_factor_min, calibration.gain_factor_max
    if not factor_min <= factor <= factor_max:
        raise ValueError("matched brightness requires unavailable physical gain")
    return float(np.log(factor / factor_min) / np.log(factor_max / factor_min))


@torch.no_grad()
def decision_measure(env, args, exposure, gain, samples=64):
    """Discriminate slit centers +/- 2.5cm around 0.65m, not image sharpness.

    Independent shot/read realizations for the two classes. Decoder knows the
    true pose, velocity, sensor and two means: optimistic cue availability only.
    """
    observations, means, temporal_mae = [], [], []
    rng = torch.Generator(device=env.p.device).manual_seed(510)
    saved = float(env.slit_center_y[0])
    for center in (.625, .675):
        set_slit_center(env, center)
        e, g = env.p.new_tensor([exposure]), env.p.new_tensor([gain])
        ideal, aux = render_exposure(env, e, return_aux=True)
        clean, _ = env.gray_camera(ideal, e, g, motion=None, enable_noise=False)
        clean = policy_resolution(clean, args)
        shot = torch.randn((samples, 1, env.height, env.width), generator=rng, device=env.p.device)
        read = torch.randn(shot.shape, generator=rng, device=env.p.device)
        noisy, _ = env.gray_camera(ideal.expand(samples, -1, -1, -1),
                                  e.expand(samples), g.expand(samples), motion=None,
                                  shot_noise=shot, read_noise=read, enable_noise=True)
        observations.append(policy_resolution(noisy, args))
        means.append(clean)
        temporal_mae.append(float(aux["temporal_blur_mae"][0]))
    set_slit_center(env, saved)
    difference = means[1] - means[0]
    mean_image = (means[1] + means[0]) / 2
    # A simple fixed, known-template linear decision. It includes clipping/noise
    # and actual policy downsampling; it supplies NO supervision to the actor.
    scores = [((image - mean_image) * difference).flatten(1).sum(1) for image in observations]
    accuracy = .5 * (((scores[0] < 0).float() + .5 * (scores[0] == 0).float()).mean()
                     + ((scores[1] > 0).float() + .5 * (scores[1] == 0).float()).mean())
    variance = scores[0].var() + scores[1].var()
    dprime = (scores[1].mean() - scores[0].mean()) / (.5 * variance).sqrt().clamp_min(1e-12)
    return {"known_state_template_accuracy": float(accuracy),
            "known_state_template_dprime": float(dprime),
            "policy_resolution_cue_rms": float(difference.square().mean().sqrt()),
            "temporal_irradiance_mae": float(np.mean(temporal_mae))}, means[0][0, 0].cpu().numpy()


@torch.no_grad()
def run_audit(out, device, samples):
    args = load_intent_args(CONFIG, batch_size=1, gray_light_jitter=0., gray_enable_noise=False)
    set_global_seed(291029)
    env = build_env(1, args, device)
    env.reset("bright_to_dark")
    # Feasibility in actual dynamics: randomized slit positions, fixed goal.
    feasible_args = copy.deepcopy(args)
    feasible_args.batch_size = 32
    set_global_seed(291030)
    feasible_env = build_env(32, feasible_args, device)
    feasible_env.reset("bright_to_dark")
    centers = feasible_env.slit_center_y
    z = torch.full_like(centers, env.simple_slit_center_z)
    points = torch.stack([
        torch.stack([z * 0 + args.simple_start_x, z * 0, z], -1),
        torch.stack([z * 0 - .5, centers, z], -1),
        torch.stack([z * 0 + .5, centers, z], -1),
        torch.stack([z * 0 + args.simple_goal_x, z * 0, z], -1),
    ], 1)
    set_global_seed(291031)
    witness = executed_dynamics_audit(feasible_env, args, local_points=points,
                                     durations=[2.6, 1.4, 1.6], record_camera_states=True)
    # A separate known-disturbance diagnostic helps distinguish the reference
    # tracker's shortcomings from intrinsic unreachability of the narrow slit.
    set_global_seed(291031)
    compensated_witness = executed_dynamics_audit(
        feasible_env, args, local_points=points, durations=[2.6, 1.4, 1.6],
        compensate_current_disturbance=True)
    witness["slit_centers_y"] = centers.cpu().tolist()
    witness["episode_reset_seed"] = 291030
    witness["dynamics_seed"] = 291031
    camera_states = witness.pop("camera_states")
    speed, omega = [], []
    for state in camera_states:
        velocity = torch.tensor(state["v"])
        rate = body_angular_velocity(torch.tensor(state["R_old"]), torch.tensor(state["R"]), 1 / args.base_control_freq)
        mask = torch.tensor(state["alive"])
        speed.extend(velocity.norm(dim=-1)[mask].tolist())
        omega.extend(rate.norm(dim=-1)[mask].tolist())
    witness["active_speed_m_s_p50_p95_max"] = np.quantile(speed, [.5, .95, 1]).tolist()
    witness["active_angular_speed_rad_s_p50_p95_max"] = np.quantile(omega, [.5, .95, 1]).tolist()
    witness["reference_speed_is_not_a_hard_speed_limit"] = True

    lighting = {}
    for scenario in SCENARIOS:
        env.reset(scenario)
        set_slit_center(env, .65)
        set_motion(env, [-2.5, 0., 1.5], [0., 0., 0.], 0)
        _, aux = render_scene_irradiance(env, env.p, env.R @ env.R_cam)
        light = slit_surface_light(env, aux["points"])
        lighting[scenario] = {"visible_light_min": float(light.min()), "visible_light_max": float(light.max()),
                              "wall_light": float(slit_surface_light(env, env.p.new_tensor([[[[0., .5, 1.5]]]])))}

    rows, images = [], {}
    # Moderate and close approach. Also include a deliberately high-rate
    # stress state; it must not be confused with the actual-dynamics witness.
    for scenario in SCENARIOS:
        env.reset(scenario)
        set_slit_center(env, .65)
        for x in (-1.0, -.4):
            for motion, (velocity, yaw_rate) in MOTIONS.items():
                set_motion(env, [x, .55, 1.5], velocity, yaw_rate)
                for gain_mode in ("fixed_gain", "matched_signal"):
                    for exposure in EXPOSURES:
                        gain = gain_for_exposure(env, exposure, gain_mode)
                        metrics, image = decision_measure(env, args, exposure, gain, samples)
                        rows.append(dict(scenario=scenario, x=x, motion=motion, gain_mode=gain_mode,
                                         exposure01=exposure, exposure_us=float(env.gray_camera.calibration.exposure_to_us(torch.tensor(exposure))),
                                         gain01=gain, velocity_m_s=velocity, yaw_rate_rad_s=yaw_rate, **metrics))
                        if scenario == "dark" and x == -.4 and gain_mode == "matched_signal":
                            images[(motion, exposure)] = image

    integration = {}
    env.reset("dark")
    set_slit_center(env, .65)
    set_motion(env, [-.4, .55, 1.5], [.8, .6, 0.], 1.)
    integrated = {}
    for count in (9, 17, 33):
        env.gray_temporal_samples = count
        integrated[count] = render_exposure(env, env.p.new_tensor([.85]))[0]
    for count in (9, 17):
        delta = policy_resolution(integrated[count] - integrated[33], args)
        integration[f"{count}_vs_33_policy_irradiance_mae"] = float(delta.abs().mean())
    env.gray_temporal_samples = args.gray_temporal_samples

    # Compare actual frames from identical poses/motions for gradient cuts and
    # nominal settings. Tests and separate rollout wiring audit cover backward.
    summaries = []
    for scenario in SCENARIOS:
        for x in (-1., -.4):
            for motion in MOTIONS:
                group = [r for r in rows if r["scenario"] == scenario and r["x"] == x and
                         r["motion"] == motion and r["gain_mode"] == "matched_signal"]
                best = max(group, key=lambda r: r["known_state_template_dprime"])
                summaries.append({"scenario": scenario, "x": x, "motion": motion,
                                  "best_sampled_exposure01_by_dprime": best["exposure01"] if best["known_state_template_dprime"] > 1e-6 else None,
                                  "best_template_accuracy": best["known_state_template_accuracy"]})

    result = {"scene": env.get_scene_effects_for_env(), "calibrated": False,
              "claim": "Scene physics and optimistic cue audit only; NO learned coordination or algorithm superiority result.",
              "reference_dynamics_witness": witness, "lighting_preview": lighting,
              "disturbance_compensated_dynamics_witness": compensated_witness,
              "cue_definition": "Two slit centers 0.625/0.675m; actual sensor noise followed by 48x36 averaging. Known-pose mean-template decoder, not a policy.",
              "noise_samples_per_class": samples, "motion_conditions": MOTIONS,
              "rows": rows, "sampled_exposure_preferences": summaries,
              "temporal_sampling_check": integration,
              "data_used_for_training": False}
    preference_changes = []
    for scenario in ("dark", "bright_to_dark"):
        for x in (-1., -.4):
            group = [r for r in rows if r["scenario"] == scenario and r["x"] == x
                     and r["gain_mode"] == "matched_signal"]
            stationary = max((r for r in group if r["motion"] == "stationary"),
                             key=lambda r: r["known_state_template_dprime"])
            for motion in MOTIONS:
                if motion == "stationary":
                    continue
                moving = [r for r in group if r["motion"] == motion]
                best = max(moving, key=lambda r: r["known_state_template_dprime"])
                same = next(r for r in moving if r["exposure01"] == stationary["exposure01"])
                if best["exposure01"] < stationary["exposure01"] and best["known_state_template_dprime"] > 1.1 * same["known_state_template_dprime"]:
                    preference_changes.append(dict(scenario=scenario, x=x, motion=motion))
    reliable_witness = all(witness["success"]) and not any(witness["collision"])
    reasons = []
    if not preference_changes:
        reasons.append("No shorter-exposure preference with >10% template-dprime gain under motion in dark / bright-to-dark sampled states.")
    if not reliable_witness:
        reasons.append("Privileged waypoint tracker does not reliably traverse all 32 randomized slits; tracking errors confound navigation results.")
    result["training_gate"] = {
        "passed": bool(preference_changes and reliable_witness),
        "motion_dependent_preference_changes": preference_changes,
        "reliable_dynamics_witness": reliable_witness,
        "reasons": reasons,
        "interpretation": "Exploratory scene-readiness screen; >10% is an engineering heuristic, not a significance test. A pass would still not prove learned coordination or method superiority.",
    }
    (out / "scene_audit.json").write_text(json.dumps(result, indent=2) + "\n")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(4, 4, figsize=(10, 7), layout="constrained")
    for row, motion in enumerate(MOTIONS):
        for col, exposure in enumerate(EXPOSURES):
            ax = axes[row, col]
            ax.imshow(images[(motion, exposure)], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{100 + 7900 * exposure:.0f} us")
            if col == 0:
                ax.set_ylabel(motion.replace("translation_yaw_", "Yaw ").replace("_stress", " rad/s (stress)"))
    fig.suptitle("Dark slit at x=-0.4m | 48x36 input | matched signal via gain | provisional model")
    fig.savefig(out / "policy_input_comparison.png", dpi=130)
    plt.close(fig)
    write_run_provenance(out, args)
    (out / "result_sha256.json").write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                      for p in (out / "scene_audit.json", out / "provenance.json")}, indent=2) + "\n")
    return {"output": str(out), "oracle_successes": sum(witness["success"]), "oracle_episodes": 32,
            "speed": witness["active_speed_m_s_p50_p95_max"],
            "angular_speed": witness["active_angular_speed_rad_s_p50_p95_max"],
            "lighting": lighting, "preferences": summaries, "integration": integration}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--noise_samples", type=int, default=64)
    cli = parser.parse_args()
    if cli.noise_samples < 16:
        raise ValueError("at least 16 noise samples per class are required")
    out = Path(cli.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    print(json.dumps(run_audit(out, torch.device(cli.device), cli.noise_samples), indent=2))


if __name__ == "__main__":
    main()

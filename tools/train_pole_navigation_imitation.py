#!/usr/bin/env python3
"""Common grayscale flight pretraining, with explicit privileged action labels.

No exposure/gain labels and no camera-head training. This is a disclosed warm
start for both later full/nondifferentiable sensing conditions, not evidence
for task-driven camera learning. Dataset generation cost is recorded.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from config import set_global_seed
from eval import run_one_episode
from intent_model import build_policy
from main_cuda import _parameter_hash
from rollout_ops import (build_local_frame, build_state_vector, compute_target_velocity,
                         render_gray_sensor, decode_action_direct, select_policy_gray_obs)
from tools.intent_common import load_intent_args
from tools.pole_navigation_teacher import PoleNavigationTeacher, bounded_probe_command
from tools.evaluate_pole_coordination import initial_probe_override
from train_utils import build_env, write_run_provenance


def dataset_configuration(args):
    keys = ("scene_layout", "pole_count", "pole_radius_min", "pole_radius_max", "pole_spacing",
            "pole_x_jitter", "pole_lateral_range", "pole_lateral_jitter", "pole_arena_half_width",
            "simple_start_x", "simple_goal_x", "simple_slit_center_z", "timesteps", "base_control_freq",
            "gray_width", "gray_height", "gray_nn_width", "gray_nn_height", "cam_angle",
            "fov_x_half_tan", "fixed_camera_exposure", "fixed_camera_gain", "flight_max_speed",
            "max_acc_cmd", "no_odom", "gray_enable_noise", "gray_temporal_samples", "gray_saturation_mode",
            "gray_nominal_ambient", "gray_nominal_diffuse", "gray_light_jitter", "gray_texture_strength",
            "gray_texture_scale", "random_rotation", "random_rotation_max_deg")
    result = {**{key: getattr(args, key) for key in keys},
              "calibration_sha256": hashlib.sha256(Path(args.imx900_calibration).read_bytes()).hexdigest()}
    if args.pole_layout == "parallax_pair":
        result.update({k: getattr(args, k) for k in ("pole_layout", "pole_pair_near_depth", "pole_pair_far_depth",
                                                    "pole_pair_depth_jitter", "pole_pair_bearing", "pole_pair_angular_radius",
                                                    "imitation_initial_probe_steps", "imitation_clearance_margin")})
    return result


@torch.no_grad()
def collect(args, episodes, batch_size, seed, device, policy=None, teacher_probability=1.):
    if episodes % batch_size:
        raise ValueError("episodes must be divisible by collection batch size")
    start = time.perf_counter()
    data = {key: [] for key in ("frames", "state", "action", "valid")}
    successes, collisions = [], []
    for offset in range(0, episodes, batch_size):
        set_global_seed(seed + offset)
        env = build_env(batch_size, args, device)
        env.reset("nominal")
        teacher = PoleNavigationTeacher(env, args, clearance_margin=getattr(args, "imitation_clearance_margin", .14))
        probe_steps = getattr(args, "imitation_initial_probe_steps", 0)
        amplitude = torch.empty(batch_size, device=device).uniform_(.08, .16) * (torch.randint(0, 2, (batch_size,), device=device) * 2 - 1) if probe_steps else None
        queue = [env.act] * 2
        hx, previous = None, None
        alive = torch.ones(batch_size, dtype=torch.bool, device=device)
        success, collision = ~alive, ~alive
        rows = {key: [] for key in data}
        e = env.p.new_full((batch_size,), args.fixed_camera_exposure)
        g = env.p.new_full((batch_size,), args.fixed_camera_gain)
        for step in range(args.timesteps):
            frame, _ = render_gray_sensor(env, e, g, differentiable=False)
            frame = F.adaptive_avg_pool2d(frame, (args.gray_nn_height, args.gray_nn_width))
            state = build_state_vector(env, compute_target_velocity(env.p_target - env.p, env),
                                       build_local_frame(env), e, g, args.no_odom, False)[0]
            if step < probe_steps:
                _, command = bounded_probe_command(env, args, step, amplitude)
                local = torch.zeros(batch_size, 3, device=device)
            else:
                local, command = teacher.command(env, (step - probe_steps) / args.base_control_freq)
            for key, value in (("frames", frame.to(torch.float16)), ("state", state),
                               ("action", local), ("valid", alive & (step >= probe_steps))):
                rows[key].append(value.cpu().clone())
            if policy is not None:
                pair = torch.cat([frame, frame if previous is None else previous], 1).float()
                action, _, hx, _ = policy(state, hx, gray_obs=select_policy_gray_obs(pair, args.policy_gray_mode))
                policy_command = decode_action_direct(action, build_local_frame(env), env, batch_size, args.max_acc_cmd)
                if step >= probe_steps:
                    choose_teacher = torch.rand(batch_size, device=device) < teacher_probability
                    command = torch.where(choose_teacher[:, None], command, policy_command)
                previous = frame
            queue.append(command)
            env.run(queue[step], 1 / args.base_control_freq, env.p_target - env.p)
            hit = env.transition_clearance() <= args.collision_clearance
            goal = (env.p - env.p_target).norm(dim=-1) < .35
            success |= alive & ~hit & goal
            collision |= alive & hit
            alive = alive & ~(hit | goal)
        for key in data:
            data[key].append(torch.stack(rows[key], 1))
        successes.extend(success.cpu().tolist())
        collisions.extend(collision.cpu().tolist())
        print(f"[collect] {offset + batch_size}/{episodes}, collection success {np.mean(successes):.3f}", flush=True)
    packed = {key: torch.cat(value) for key, value in data.items()}
    metadata = {"generation_seed": seed, "episodes": episodes, "steps": args.timesteps,
                "generated_transitions": episodes * args.timesteps,
                "valid_transitions": int(packed["valid"].sum()),
                "generation_wall_time_s": time.perf_counter() - start,
                "collection_success_rate": float(np.mean(successes)), "collection_collision_rate": float(np.mean(collisions)),
                "privileged_acceleration_labels": True, "camera_parameter_labels": False,
                "teacher_execution_probability": teacher_probability,
                "unlabelled_initial_probe_steps": getattr(args, "imitation_initial_probe_steps", 0),
                "initial_probe_motion_labels": False,
                "collection_actor": "current learned policy with teacher mixing" if policy is not None else "teacher only",
                "actor_inputs": "ordinary grayscale, velocity, orientation and goal only",
                "frame_storage": "float16 at policy resolution; single frame, previous frame reconstructed"}
    return packed, metadata


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config", default="configs/gray_poles_navigation.args")
    p.add_argument("--dataset")
    p.add_argument("--aggregate_dataset", help="Append recovery data to the common demonstration dataset.")
    p.add_argument("--resume", help="Continue flight imitation from this actor; camera head stays frozen.")
    p.add_argument("--collection_checkpoint", help="Collect teacher-labelled states visited by this learned actor.")
    p.add_argument("--teacher_probability", type=float, default=1.)
    p.add_argument("--initial_probe_steps", type=int, default=0,
                   help="Collect a geometry-independent initial view change, with its action labels masked out.")
    p.add_argument("--teacher_clearance_margin", type=float, default=.14)
    p.add_argument("--collect_only", action="store_true")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--collection_batch", type=int, default=8)
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=1201)
    p.add_argument("--eval_seed", type=int, default=741000)
    p.add_argument("--eval_episodes", type=int, default=40)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--device", default="cuda")
    p.add_argument("--collection_device", default="cpu")
    p.add_argument("--cpu_threads", type=int, default=4)
    p.add_argument("--wandb_disabled", action="store_true")
    cli = p.parse_args()
    if not 0 <= cli.teacher_probability <= 1:
        raise ValueError("teacher_probability must be in [0,1]")
    if cli.teacher_probability != 1 and not cli.collection_checkpoint:
        raise ValueError("mixed collection requires a collection_checkpoint")
    out = Path(cli.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be fresh")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cli.cpu_threads)
    args = load_intent_args(cli.config, batch_size=cli.batch_size, camera_control_mode="fixed", scenarios=["nominal"])
    if args.pole_layout == "parallax_pair" and cli.initial_probe_steps < 24:
        raise ValueError("Initially ambiguous images require an unlabelled informative probe before any geometry-aware action label.")
    if not 0 <= cli.initial_probe_steps < args.timesteps or cli.teacher_clearance_margin <= 0:
        raise ValueError("invalid initial probe duration or teacher clearance")
    args.imitation_initial_probe_steps = cli.initial_probe_steps
    args.imitation_clearance_margin = cli.teacher_clearance_margin
    args.checkpoint_dir = str(out)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    (out / "command.json").write_text(json.dumps(vars(cli), indent=2) + "\n")
    write_run_provenance(out, args)
    import shutil
    shutil.copy2(args.imx900_calibration, out / "imx900_calibration.json")
    if cli.dataset:
        stored = torch.load(cli.dataset, map_location="cpu")
        data, meta = stored["data"], stored["metadata"]
        if stored["dataset_configuration"] != dataset_configuration(args):
            raise ValueError("dataset configuration does not match the requested navigation task")
        dataset_path = Path(cli.dataset).resolve()
    else:
        policy = None
        if cli.collection_checkpoint:
            policy = build_policy(args).to(cli.collection_device).eval()
            policy.load_state_dict(torch.load(cli.collection_checkpoint, map_location=cli.collection_device), strict=True)
        data, meta = collect(args, cli.episodes, cli.collection_batch, cli.seed + 5000,
                             torch.device(cli.collection_device), policy=policy,
                             teacher_probability=cli.teacher_probability)
        if cli.collection_checkpoint:
            meta["collection_checkpoint_sha256"] = hashlib.sha256(Path(cli.collection_checkpoint).read_bytes()).hexdigest()
        dataset_path = out / "flight_dataset.pt"
        torch.save({"data": data, "metadata": meta, "configuration": vars(args),
                    "dataset_configuration": dataset_configuration(args)}, dataset_path)
    meta = dict(meta, dataset_path=str(dataset_path), dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest())
    if cli.aggregate_dataset:
        extra_path = Path(cli.aggregate_dataset).resolve()
        extra = torch.load(extra_path, map_location="cpu")
        if extra["dataset_configuration"] != dataset_configuration(args):
            raise ValueError("recovery dataset configuration does not match")
        data = {key: torch.cat([value, extra["data"][key]]) for key, value in data.items()}
        meta = {"datasets": [meta, dict(extra["metadata"], dataset_path=str(extra_path),
                                        dataset_sha256=hashlib.sha256(extra_path.read_bytes()).hexdigest())],
                "episodes": len(data["frames"]), "generated_transitions": data["valid"].numel(),
                "valid_transitions": int(data["valid"].sum()),
                "privileged_acceleration_labels": True, "camera_parameter_labels": False}
    (out / "dataset_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    if cli.collect_only:
        print(json.dumps(meta, indent=2), flush=True)
        return
    wandb.init(project="diff-simulation", name=f"poles_common_flight_imitation_{cli.seed}",
               config=dict(vars(cli), privileged_flight_labels=True, camera_labels=False),
               mode="disabled" if cli.wandb_disabled else "online")
    set_global_seed(cli.seed)
    device = torch.device(cli.device)
    model = build_policy(args).to(device)
    if cli.resume:
        model.load_state_dict(torch.load(cli.resume, map_location=device), strict=True)
    else:
        model.initialize_camera_output(args.fixed_camera_exposure, args.fixed_camera_gain)
    model.freeze_camera_for_flight_only()
    camera_hash = _parameter_hash(model, True)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cli.lr)
    generator = torch.Generator().manual_seed(cli.seed)
    train_start = time.perf_counter()
    frames = data["frames"]
    for iteration in range(cli.iterations):
        indices = torch.randint(len(frames), (cli.batch_size,), generator=generator)
        current = frames[indices].to(device=device, dtype=torch.float32)
        previous = torch.cat([current[:, :1], current[:, :-1]], 1)
        images = torch.cat([current, previous], 2)
        state = data["state"][indices].to(device)
        target = data["action"][indices].to(device)
        valid = data["valid"][indices].to(device)
        batch, steps = valid.shape
        encoded = model.encode_observation(state.reshape(-1, state.shape[-1]), images.flatten(0, 1)).reshape(batch, steps, -1)
        hx, predicted = None, []
        for step in range(steps):
            action, hx = model.flight_from_encoded(encoded[:, step], hx)
            predicted.append(action)
        predicted = torch.stack(predicted, 1)
        error = F.smooth_l1_loss(predicted, target, reduction="none").mean(-1)
        loss = (error * valid).sum() / valid.sum().clamp_min(1)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optim.step()
        row = {"iteration": iteration + 1, "flight_imitation_loss": float(loss.detach()),
               "valid_supervised_steps": int(valid.sum())}
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if (iteration + 1) % 25 == 0:
            print(f"[imitation] {iteration + 1}/{cli.iterations}: loss {row['flight_imitation_loss']:.6f}", flush=True)
            wandb.log(row, step=iteration + 1)
        if (iteration + 1) % 200 == 0:
            torch.save(model.state_dict(), out / f"checkpoint{iteration + 1:04d}.pth")
    torch.save(model.state_dict(), out / "final.pth")
    training_seconds = time.perf_counter() - train_start
    assert _parameter_hash(model, True) == camera_hash
    model.eval()
    rows = {}
    for mode in (("gray", "zero", "gray_forced_probe") if args.pole_layout == "parallax_pair" else ("gray", "zero")):
        args.policy_gray_mode, args.batch_size, args.vis_enable = "gray" if mode == "gray_forced_probe" else mode, 1, False
        env = build_env(1, args, device, eval_mode=True)
        results = []
        with torch.no_grad():
            for episode in range(cli.eval_episodes):
                result, _ = run_one_episode(episode, "nominal", args, model, env, None, device,
                                            episode_seed=cli.eval_seed + episode,
                                            action_override=initial_probe_override(args, .12 if episode % 2 else -.12) if mode == "gray_forced_probe" else None)
                results.append(result)
        rows[mode] = results
    summary = {"purpose": "common flight-only imitation warm start", "camera_labels": False,
               "camera_parameters_unchanged": True, "dataset": meta,
               "optimizer_steps": cli.iterations, "training_wall_time_s": training_seconds,
               "supervised_sequence_samples": cli.iterations * cli.batch_size,
               "actor_parameters": sum(p.numel() for p in model.parameters()),
               "checkpoint_sha256": hashlib.sha256((out / "final.pth").read_bytes()).hexdigest(),
               "resume_checkpoint_sha256": hashlib.sha256(Path(cli.resume).read_bytes()).hexdigest() if cli.resume else None,
               "flight_parameter_sha256": _parameter_hash(model, False), "camera_parameter_sha256": camera_hash,
               "eval_seed": cli.eval_seed,
               "evaluations": {k: {"success_rate": float(np.mean([r["success_rate"] for r in v])),
                                    "collision_rate": float(np.mean([r["collision_rate"] for r in v])),
                                    "episodes": v} for k, v in rows.items()},
               "claim": "Disclosed privileged flight pretraining only; no learned-camera or joint-coordination claim."}
    (out / "imitation_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    wandb.log({f"evaluation/{k}_success": v["success_rate"] for k, v in summary["evaluations"].items()})
    wandb.finish()
    print(json.dumps({"training_wall_time_s": training_seconds,
                      "success": {k: v["success_rate"] for k, v in summary["evaluations"].items()}}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a camera-policy teacher dataset from trajectory-level diffopt.

Unlike generate_camera_teacher_dataset.py, this script does not use a trained
flight policy rollout to define labels.  It first optimizes the full p/e/g
camera trajectory on a fixed local path through the wall slit, then records
the observations a camera policy would see while tracking those teacher
targets with the same EMA dynamics used online.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import parse_scenarios, set_global_seed  # noqa: E402
from rollout_ops import (  # noqa: E402
    build_local_frame,
    build_state_vector,
    compute_target_velocity,
    select_policy_depth_obs,
)
from train_utils import build_env  # noqa: E402
from tools.probe_opening_depth_views import (  # noqa: E402
    _build_project_args,
    _parse_float_list,
    _parse_slots,
    _set_pose_look_at,
)
from tools.run_diffopt_camera_benchmark import (  # noqa: E402
    _best_static_sequence,
    _build_trajectory,
    _local_to_world,
    _optimize_trajectory_multistart,
    _parse_triplet,
    _random_settings,
)


def _goal_world(env) -> torch.Tensor:
    fx = env.current_scene_effects or {}
    goal_local = fx.get("geometry_goal_local", None)
    if torch.is_tensor(goal_local):
        local = goal_local[0].to(device=env.device, dtype=torch.float32)
    else:
        local = torch.tensor(
            [
                float(fx.get("geometry_goal_x", getattr(env, "simple_goal_x", 1.5))),
                0.0,
                float(fx.get("geometry_gate_z", getattr(env, "simple_gate_z", 1.5))),
            ],
            device=env.device,
            dtype=torch.float32,
        )
    return _local_to_world(env, (float(local[0]), float(local[1]), float(local[2])))


def _scene_id(env) -> int:
    ids = getattr(env, "current_scene_ids", None)
    if torch.is_tensor(ids):
        return int(ids.reshape(-1)[0].detach().cpu().item())
    return int(getattr(env, "current_scene_id", -1))


def _make_cfg(script_args, project_args):
    cfg = argparse.Namespace(**vars(script_args))
    cfg.diffopt_steps = int(script_args.teacher_steps)
    cfg.diffopt_lr = float(script_args.teacher_lr)
    cfg.power_baseline = float(project_args.cam_power_baseline)
    return cfg


def _collect_sequence(env, project_args, states, teacher_seq: torch.Tensor,
                      nominal, ema_alpha: float, speed_mps: float):
    device = env.device
    power = torch.full((1,), float(nominal[0]), device=device)
    exposure = torch.full((1,), float(nominal[1]), device=device)
    gain = torch.full((1,), float(nominal[2]), device=device)
    goal = _goal_world(env).reshape(1, 3)

    depth_seq, state_seq, cam_state_seq, cam_motion_seq = [], [], [], []
    teacher_out, scene_ids, local_xs, local_ys, phases = [], [], [], [], []

    with torch.no_grad():
        for t, state in enumerate(states):
            _set_pose_look_at(env, state.pose, state.target)
            env.p_target = goal.repeat(env.batch_size, 1).clone()
            direction = torch.nn.functional.normalize(state.target.to(device) - env.p[0], dim=0)
            env.v = direction.reshape(1, 3).repeat(env.batch_size, 1) * float(speed_mps)

            depth_obs, _quality = env.render_diff_depth(power, exposure, gain)
            policy_depth_obs = select_policy_depth_obs(depth_obs, project_args.policy_depth_mode)
            target_v_raw = env.p_target - env.p.detach()
            R = build_local_frame(env)
            target_v = compute_target_velocity(target_v_raw, env)
            state_vec, _local_v, camera_state, camera_motion_state = build_state_vector(
                env,
                target_v,
                R,
                power,
                exposure,
                gain,
                project_args.no_odom,
                project_args.include_camera_state_in_obs,
            )

            target = teacher_seq[t].to(device=device, dtype=torch.float32).clamp(0.001, 0.999)
            depth_seq.append(policy_depth_obs[0].detach().to(torch.float16).cpu())
            state_seq.append(state_vec[0].detach().to(torch.float32).cpu())
            cam_state_seq.append(camera_state[0].detach().to(torch.float32).cpu())
            cam_motion_seq.append(camera_motion_state[0].detach().to(torch.float32).cpu())
            teacher_out.append(target.detach().to(torch.float32).cpu())
            scene_ids.append(torch.tensor(_scene_id(env), dtype=torch.long))
            local_xs.append(torch.tensor(float(state.local_x), dtype=torch.float32))
            local_ys.append(torch.tensor(float(state.local_y), dtype=torch.float32))
            phases.append(str(state.phase))

            power = float(ema_alpha) * power + (1.0 - float(ema_alpha)) * target[0:1]
            exposure = float(ema_alpha) * exposure + (1.0 - float(ema_alpha)) * target[1:2]
            gain = float(ema_alpha) * gain + (1.0 - float(ema_alpha)) * target[2:3]

    return {
        "depth_obs": torch.stack(depth_seq, dim=0),
        "state": torch.stack(state_seq, dim=0),
        "camera_state": torch.stack(cam_state_seq, dim=0),
        "camera_motion_state": torch.stack(cam_motion_seq, dim=0),
        "teacher_camera": torch.stack(teacher_out, dim=0),
        "scene_id": torch.stack(scene_ids, dim=0),
        "local_x": torch.stack(local_xs, dim=0),
        "local_y": torch.stack(local_ys, dim=0),
        "phase": phases,
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_final_full.args")
    parser.add_argument("--out", default="paper/experiment/results/trajectory_diffopt_camera_dataset.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scenarios", nargs="*", default=["glare", "specular", "dark"])
    parser.add_argument("--slots", nargs="*", default=["left", "right"])
    parser.add_argument("--trajectories_per_scene_slot", type=int, default=8)
    parser.add_argument("--xs", default="-1.20,-0.90,-0.60,-0.35,-0.18,-0.05,0.10,0.35,0.70,1.05,1.35")
    parser.add_argument("--x_jitter", type=float, default=0.035)
    parser.add_argument("--path_y_mode", default="slot", choices=["center", "blend", "slot"])
    parser.add_argument("--target_mode", default="opening_then_goal", choices=["opening_then_goal", "opening", "goal"])
    parser.add_argument("--after_wall_margin", type=float, default=0.18)
    parser.add_argument("--lookahead_x", type=float, default=0.80)
    parser.add_argument("--sensor_impl", default="cuda", choices=["cuda", "python"])
    parser.add_argument("--keep_random_rotation", action="store_true")
    parser.add_argument("--speed_mps", type=float, default=1.0)
    parser.add_argument("--nominal_setting", default="0.50,0.50,0.50")
    parser.add_argument("--fixed_setting", default=None)
    parser.add_argument("--diffopt_init", default=None)
    parser.add_argument("--teacher_steps", type=int, default=120)
    parser.add_argument("--teacher_lr", type=float, default=0.08)
    parser.add_argument("--diffopt_random_restarts", type=int, default=4)
    parser.add_argument("--randfix_k", type=int, default=24)
    parser.add_argument("--teacher_ema_alpha", type=float, default=0.7)
    parser.add_argument("--trace_every", type=int, default=1000000)
    parser.add_argument("--health_patch_rows", type=int, default=6)
    parser.add_argument("--health_patch_cols", type=int, default=8)
    parser.add_argument("--health_cvar_frac", type=float, default=0.25)
    parser.add_argument("--fill_target", type=float, default=0.90)
    parser.add_argument("--region_target", type=float, default=0.88)
    parser.add_argument("--edge_target", type=float, default=0.86)
    parser.add_argument("--quality_target", type=float, default=0.58)
    parser.add_argument("--recovery_fill_target", type=float, default=0.92)
    parser.add_argument("--recovery_margin", type=float, default=0.08)
    parser.add_argument("--region_visible_mass", type=float, default=0.006)
    parser.add_argument("--coef_fill", type=float, default=12.0)
    parser.add_argument("--coef_region", type=float, default=8.0)
    parser.add_argument("--coef_edge", type=float, default=5.0)
    parser.add_argument("--coef_quality", type=float, default=1.0)
    parser.add_argument("--coef_nominal", type=float, default=1.5)
    parser.add_argument("--coef_smooth", type=float, default=2.0)
    parser.add_argument("--coef_power", type=float, default=0.05)
    parser.add_argument("--coef_blur", type=float, default=0.01)
    parser.add_argument("--coef_noise", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_known_args()


def main():
    script_args, project_overrides = _parse_args()
    project_args = _build_project_args(Path(script_args.config), project_overrides)
    project_args.diff_sensor_impl["diff_depth"] = str(script_args.sensor_impl)
    if not bool(script_args.keep_random_rotation):
        project_args.random_rotation = False
    project_args.camera_control_mode = "learned"
    set_global_seed(int(script_args.seed), getattr(project_args, "deterministic", False))

    device = torch.device(script_args.device)
    scenarios = parse_scenarios(script_args.scenarios)
    slots = _parse_slots(script_args.slots)
    base_xs = _parse_float_list(script_args.xs)
    nominal = _parse_triplet(script_args.nominal_setting, (0.5, 0.5, 0.5))
    fixed = _parse_triplet(script_args.fixed_setting, (
        float(project_args.fixed_camera_power),
        float(project_args.fixed_camera_exposure),
        float(project_args.fixed_camera_gain),
    ))
    diffopt_init = _parse_triplet(script_args.diffopt_init, nominal)
    cfg = _make_cfg(script_args, project_args)
    rng = np.random.default_rng(int(script_args.seed))

    seqs = []
    total = len(scenarios) * len(slots) * int(script_args.trajectories_per_scene_slot)
    pbar = tqdm(total=total, ncols=90)
    for scene in scenarios:
        for slot in slots:
            for sample_idx in range(int(script_args.trajectories_per_scene_slot)):
                cond_args = copy.deepcopy(project_args)
                cond_args.scenarios = [scene]
                cond_args.sun_glare_eval_slot = slot
                env = build_env(1, cond_args, device, eval_mode=True)
                env.reset(scene_name=scene)
                if float(script_args.x_jitter) > 0.0 and len(base_xs) > 2:
                    jittered = [base_xs[0]]
                    for x in base_xs[1:-1]:
                        jittered.append(float(x) + float(rng.uniform(-script_args.x_jitter, script_args.x_jitter)))
                    jittered.append(base_xs[-1])
                    xs = sorted(jittered)
                else:
                    xs = list(base_xs)
                states = _build_trajectory(
                    env,
                    xs,
                    y_mode=str(script_args.path_y_mode),
                    target_mode=str(script_args.target_mode),
                    after_wall_margin=float(script_args.after_wall_margin),
                    lookahead_x=float(script_args.lookahead_x),
                )
                rand_candidates = _random_settings(env, int(script_args.randfix_k), rng)
                rand_setting, _ = _best_static_sequence(
                    env,
                    cond_args,
                    states,
                    rand_candidates,
                    nominal,
                    float(script_args.speed_mps),
                    cfg,
                    "randfix_best",
                )
                init_candidates = [
                    ("nominal", diffopt_init),
                    ("fixed", fixed),
                    ("randfix_best", (rand_setting.power, rand_setting.exposure, rand_setting.gain)),
                ]
                for ridx in range(max(0, int(script_args.diffopt_random_restarts))):
                    init_candidates.append((
                        f"random_{ridx:02d}",
                        (
                            float(rng.uniform(*env.fixed_random_power_range)),
                            float(rng.uniform(*env.fixed_random_exposure_range)),
                            float(rng.uniform(*env.fixed_random_gain_range)),
                        ),
                    ))
                teacher_seq, _trace, info = _optimize_trajectory_multistart(
                    env,
                    cond_args,
                    states,
                    init_candidates,
                    nominal,
                    float(script_args.speed_mps),
                    cfg,
                    trace_prefix={"scene": scene, "slot": slot, "sample": sample_idx},
                )
                seq = _collect_sequence(
                    env,
                    cond_args,
                    states,
                    teacher_seq,
                    nominal,
                    float(script_args.teacher_ema_alpha),
                    float(script_args.speed_mps),
                )
                seq["slot_id"] = torch.full_like(seq["scene_id"], float(env.supported_slots.index(slot)), dtype=torch.float32)
                seq["teacher_loss"] = torch.full_like(seq["local_x"], float(info.get("loss", 0.0)))
                seqs.append(seq)
                pbar.set_description(f"{scene}/{slot} sample {sample_idx + 1}")
                pbar.update(1)
    pbar.close()

    def stack(key):
        return torch.stack([seq[key] for seq in seqs], dim=0).contiguous()

    dataset = {
        "depth_obs": stack("depth_obs"),
        "state": stack("state"),
        "camera_state": stack("camera_state"),
        "camera_motion_state": stack("camera_motion_state"),
        "teacher_camera": stack("teacher_camera"),
        "scene_id": stack("scene_id"),
        "local_x": stack("local_x"),
        "local_y": stack("local_y"),
        "slot_id": stack("slot_id"),
        "teacher_loss": stack("teacher_loss"),
        "phase": [seq["phase"] for seq in seqs],
        "meta": {
            "teacher_source": "trajectory_diffopt",
            "config": str(script_args.config),
            "scenarios": scenarios,
            "slots": slots,
            "trajectories_per_scene_slot": int(script_args.trajectories_per_scene_slot),
            "xs": base_xs,
            "teacher_steps": int(script_args.teacher_steps),
            "teacher_lr": float(script_args.teacher_lr),
            "diffopt_random_restarts": int(script_args.diffopt_random_restarts),
            "teacher_ema_alpha": float(script_args.teacher_ema_alpha),
            "nominal_setting": nominal,
        },
    }
    out = Path(script_args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out)
    teacher = dataset["teacher_camera"].float()
    print(f"[trajectory-teacher] saved: {out}")
    print(f"[trajectory-teacher] sequences={teacher.shape[0]} timesteps={teacher.shape[1]} samples={teacher.shape[0] * teacher.shape[1]}")
    print(
        "[trajectory-teacher] target mean p/e/g="
        f"{teacher[..., 0].mean():.3f}/{teacher[..., 1].mean():.3f}/{teacher[..., 2].mean():.3f} "
        "std="
        f"{teacher[..., 0].std():.3f}/{teacher[..., 1].std():.3f}/{teacher[..., 2].std():.3f}"
    )


if __name__ == "__main__":
    main()

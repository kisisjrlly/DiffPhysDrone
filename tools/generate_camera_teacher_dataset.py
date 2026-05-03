#!/usr/bin/env python3
"""Generate a supervised camera-teacher dataset from a frozen flight policy.

Stage usage:
1. Train a fixed-camera flight policy.
2. Run this script with that checkpoint.  The flight policy is used for actions
   on the states it actually visits, while p/e/g labels are produced by local
   differentiable camera optimization.
3. Train only the camera head with tools/pretrain_camera_head.py.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from random import normalvariate

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import build_parser, parse_diff_sensor_impl, parse_scenarios, set_global_seed, validate_args  # noqa: E402
from model import Model, load_model_state_dict  # noqa: E402
from rollout_ops import (  # noqa: E402
    build_local_frame,
    build_state_vector,
    compute_depth_fill_health,
    compute_target_velocity,
    decode_action_direct,
    diff_depth_exposure_to_time,
    init_camera_params,
    render_sensors,
    select_policy_depth_obs,
)
from train_utils import build_env  # noqa: E402


def _read_args_file(path: Path) -> list[str]:
    import shlex

    tokens: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            tokens.extend(shlex.split(line))
    return tokens


def _build_project_args(config_path: Path, overrides: list[str]):
    parser = build_parser()
    args = parser.parse_args(_read_args_file(config_path) + list(overrides))
    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.scenarios = parse_scenarios(args.scenarios)
    args.wandb_disabled = True
    args.vis_enable = False
    validate_args(args)
    return args


def _logit01(value: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    value = value.clamp(eps, 1.0 - eps)
    return torch.log(value / (1.0 - value))


def _scene_id_tensor(env):
    ids = getattr(env, "current_scene_ids", None)
    if ids is None:
        return torch.full((env.batch_size,), int(env.current_scene_id), device=env.device, dtype=torch.long)
    return ids.to(device=env.device, dtype=torch.long)


def _local_x(env):
    p_local = torch.bmm(env.R_scene_T, env.p[:, :, None])[:, :, 0]
    return p_local[:, 0]


def _camera_teacher_loss(
    env,
    args,
    peg: torch.Tensor,
    cam_current: torch.Tensor,
    speed: torch.Tensor,
    *,
    nominal: torch.Tensor,
    coef_nominal_when_healthy: float,
    nominal_fill_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    power, exposure, gain = peg.unbind(-1)
    depth, _ = env.render_diff_depth(power, exposure, gain)
    fill = compute_depth_fill_health(
        env,
        depth,
        min_valid_depth=float(args.depth_min_valid),
        patch_rows=int(args.diff_depth_health_patch_rows),
        patch_cols=int(args.diff_depth_health_patch_cols),
        cvar_frac=float(args.diff_depth_health_cvar_frac),
    )
    exp_phys = diff_depth_exposure_to_time(exposure, camera_semantics=env.cam_sem)
    loss_smooth = (peg - cam_current).pow(2).mean(dim=-1)
    loss_power = F.relu(power - float(args.cam_power_baseline)).pow(2)
    loss_blur = (speed * exp_phys).pow(2)
    loss_noise = gain.pow(2)
    loss_fill = F.relu(float(args.diff_depth_min_fill_rate) - fill).pow(2)
    total = (
        float(args.coef_cam_smooth) * loss_smooth
        + float(args.coef_diff_depth_power) * loss_power
        + float(args.coef_diff_depth_blur) * loss_blur
        + float(args.coef_diff_depth_noise) * loss_noise
        + float(args.coef_diff_depth_fill) * loss_fill
    )
    if float(coef_nominal_when_healthy) > 0.0:
        margin = max(float(nominal_fill_margin), 1e-6)
        health = ((fill - float(args.diff_depth_min_fill_rate)) / margin).clamp(0.0, 1.0).detach()
        loss_nominal = (peg - nominal).pow(2).mean(dim=-1)
        total = total + float(coef_nominal_when_healthy) * health * loss_nominal
    terms = {
        "fill": fill.detach(),
        "loss_total_per_env": total.detach(),
        "loss_fill": loss_fill.detach(),
        "loss_power": loss_power.detach(),
        "loss_blur": loss_blur.detach(),
        "loss_noise": loss_noise.detach(),
    }
    return total.mean(), terms


def _optimize_camera_teacher(
    env,
    args,
    cam_current: torch.Tensor,
    *,
    steps: int,
    lr: float,
    nominal: torch.Tensor,
    coef_nominal_when_healthy: float,
    nominal_fill_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    init = cam_current.detach().clamp(0.001, 0.999)
    logits = _logit01(init).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=float(lr))
    last_terms: dict[str, torch.Tensor] = {}
    speed = env.v.norm(2, -1).detach()

    for _ in range(max(1, int(steps))):
        opt.zero_grad(set_to_none=True)
        peg = torch.sigmoid(logits).clamp(0.001, 0.999)
        loss, last_terms = _camera_teacher_loss(
            env,
            args,
            peg,
            cam_current,
            speed,
            nominal=nominal,
            coef_nominal_when_healthy=coef_nominal_when_healthy,
            nominal_fill_margin=nominal_fill_margin,
        )
        loss.backward()
        opt.step()
    return torch.sigmoid(logits).clamp(0.001, 0.999).detach(), last_terms


def _make_model(args, device):
    obs_dim = 7 if args.no_odom else 10
    return Model(
        obs_dim,
        3,
        include_camera_state_in_obs=args.include_camera_state_in_obs,
        use_policy_intent=False,
        depth_nn_width=args.depth_nn_width,
        depth_nn_height=args.depth_nn_height,
        depth_use_pipeline=args.depth_use_pipeline,
        depth_min_valid=args.depth_min_valid,
        depth_max_range=args.depth_max_range,
    ).to(device)


def _parse_script_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_final_full.args")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="paper/experiment/results/camera_teacher_dataset.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--rollouts_per_scene", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--teacher_steps", type=int, default=40)
    parser.add_argument("--teacher_lr", type=float, default=0.12)
    parser.add_argument("--teacher_every", type=int, default=1)
    parser.add_argument("--teacher_camera_ema", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--teacher_ema_alpha", type=float, default=0.7)
    parser.add_argument(
        "--rollout_camera_mode",
        default="fixed",
        choices=["fixed", "fixed_random_static"],
        help=(
            "Camera mode used while collecting closed-loop flight states.  Use "
            "fixed_random_static with --no-teacher_camera_ema to collect states "
            "from the randfix flight-policy distribution."
        ),
    )
    parser.add_argument("--coef_nominal_when_healthy", type=float, default=0.5)
    parser.add_argument("--nominal_fill_margin", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sensor_impl", default=None, choices=["cuda", "python"])
    return parser.parse_known_args()


def main():
    script_args, project_overrides = _parse_script_args()
    args = _build_project_args(Path(script_args.config), project_overrides)
    args.batch_size = int(script_args.batch_size)
    if script_args.timesteps is not None:
        args.timesteps = int(script_args.timesteps)
    if script_args.scenarios:
        args.scenarios = parse_scenarios(script_args.scenarios)
    if script_args.sensor_impl is not None:
        args.diff_sensor_impl["diff_depth"] = str(script_args.sensor_impl)
    set_global_seed(int(script_args.seed), getattr(args, "deterministic", False))

    device = torch.device(script_args.device)
    env_args = copy.deepcopy(args)
    env_args.camera_control_mode = str(script_args.rollout_camera_mode)
    env = build_env(args.batch_size, env_args, device, eval_mode=False)
    model = _make_model(args, device)
    state = torch.load(script_args.checkpoint, map_location=device)
    load_model_state_dict(model, state, allow_missing_camera_adapter=True)
    model.eval()

    all_depth, all_state, all_cam_state, all_cam_motion, all_teacher = [], [], [], [], []
    all_scene_id, all_local_x, all_fill, all_loss = [], [], [], []
    scenes = list(args.scenarios)
    pbar = tqdm(total=len(scenes) * int(script_args.rollouts_per_scene), ncols=90)
    nominal = torch.tensor(
        [float(args.fixed_camera_power), float(args.fixed_camera_exposure), float(args.fixed_camera_gain)],
        device=device,
        dtype=torch.float32,
    ).reshape(1, 3)

    for scene in scenes:
        for rollout_idx in range(int(script_args.rollouts_per_scene)):
            env.reset(scene_name=scene)
            power, exposure, gain = init_camera_params(env, args.batch_size, device)
            act_buffer = [env.act] * 2
            h = None
            cam_h = None
            seq_depth, seq_state, seq_cam_state, seq_cam_motion, seq_teacher = [], [], [], [], []
            seq_scene_id, seq_local_x, seq_fill, seq_loss = [], [], [], []

            for t in range(int(args.timesteps)):
                base_dt = normalvariate(1 / args.base_control_freq, 0.1 / args.base_control_freq)
                exposure_delay = float(diff_depth_exposure_to_time(exposure.mean().detach(), camera_semantics=env.cam_sem)) * 0.01
                ctl_dt = base_dt + exposure_delay
                depth_obs, _ = render_sensors(env, ctl_dt, power, exposure, gain, differentiable=False)
                policy_depth_obs = select_policy_depth_obs(depth_obs, args.policy_depth_mode)
                target_v_raw = env.p_target - env.p.detach()
                R = build_local_frame(env)
                target_v = compute_target_velocity(target_v_raw, env)
                state_vec, _local_v, camera_state, camera_motion_state = build_state_vector(
                    env, target_v, R, power, exposure, gain,
                    args.no_odom, args.include_camera_state_in_obs,
                )

                with torch.no_grad():
                    act_raw, _cam_params, h, cam_h = model(
                        state_vec,
                        h,
                        depth_obs=policy_depth_obs,
                        add_noise=False,
                        cam_hx=cam_h,
                        camera_state=camera_state,
                        camera_motion_state=camera_motion_state,
                    )
                act = decode_action_direct(act_raw.float(), R, env, args.batch_size, args.max_acc_cmd)
                act_buffer.append(act)

                cam_current = torch.stack([power, exposure, gain], dim=-1).detach()
                if t % max(1, int(script_args.teacher_every)) == 0:
                    teacher, terms = _optimize_camera_teacher(
                        env,
                        args,
                        cam_current,
                        steps=int(script_args.teacher_steps),
                        lr=float(script_args.teacher_lr),
                        nominal=nominal.expand(args.batch_size, -1),
                        coef_nominal_when_healthy=float(script_args.coef_nominal_when_healthy),
                        nominal_fill_margin=float(script_args.nominal_fill_margin),
                    )
                else:
                    teacher = cam_current
                    terms = {
                        "fill": torch.zeros((args.batch_size,), device=device),
                        "loss_total_per_env": torch.zeros((args.batch_size,), device=device),
                    }

                seq_depth.append(policy_depth_obs.detach().to(torch.float16).cpu())
                seq_state.append(state_vec.detach().to(torch.float32).cpu())
                seq_cam_state.append(camera_state.detach().to(torch.float32).cpu())
                seq_cam_motion.append(camera_motion_state.detach().to(torch.float32).cpu())
                seq_teacher.append(teacher.detach().to(torch.float32).cpu())
                seq_scene_id.append(_scene_id_tensor(env).detach().cpu())
                seq_local_x.append(_local_x(env).detach().to(torch.float32).cpu())
                seq_fill.append(terms["fill"].detach().to(torch.float32).cpu())
                seq_loss.append(terms["loss_total_per_env"].detach().to(torch.float32).cpu())

                if bool(script_args.teacher_camera_ema):
                    alpha = float(script_args.teacher_ema_alpha)
                    power = alpha * power.detach() + (1.0 - alpha) * teacher[:, 0]
                    exposure = alpha * exposure.detach() + (1.0 - alpha) * teacher[:, 1]
                    gain = alpha * gain.detach() + (1.0 - alpha) * teacher[:, 2]

                env.run(act_buffer[t], ctl_dt, target_v_raw)

            def pack(seq):
                return torch.stack(seq, dim=0).transpose(0, 1).contiguous()

            all_depth.append(pack(seq_depth))
            all_state.append(pack(seq_state))
            all_cam_state.append(pack(seq_cam_state))
            all_cam_motion.append(pack(seq_cam_motion))
            all_teacher.append(pack(seq_teacher))
            all_scene_id.append(pack(seq_scene_id))
            all_local_x.append(pack(seq_local_x))
            all_fill.append(pack(seq_fill))
            all_loss.append(pack(seq_loss))
            pbar.set_description(f"{scene} rollout {rollout_idx + 1}")
            pbar.update(1)
    pbar.close()

    dataset = {
        "depth_obs": torch.cat(all_depth, dim=0),
        "state": torch.cat(all_state, dim=0),
        "camera_state": torch.cat(all_cam_state, dim=0),
        "camera_motion_state": torch.cat(all_cam_motion, dim=0),
        "teacher_camera": torch.cat(all_teacher, dim=0),
        "scene_id": torch.cat(all_scene_id, dim=0),
        "local_x": torch.cat(all_local_x, dim=0),
        "teacher_fill": torch.cat(all_fill, dim=0),
        "teacher_loss": torch.cat(all_loss, dim=0),
        "meta": {
            "config": str(script_args.config),
            "checkpoint": str(script_args.checkpoint),
            "scenarios": scenes,
            "teacher_steps": int(script_args.teacher_steps),
            "teacher_lr": float(script_args.teacher_lr),
            "teacher_camera_ema": bool(script_args.teacher_camera_ema),
            "teacher_ema_alpha": float(script_args.teacher_ema_alpha),
            "rollout_camera_mode": str(script_args.rollout_camera_mode),
            "coef_nominal_when_healthy": float(script_args.coef_nominal_when_healthy),
            "nominal_fill_margin": float(script_args.nominal_fill_margin),
        },
    }
    out = Path(script_args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out)
    print(f"[teacher] saved: {out}")
    print(
        f"[teacher] sequences={dataset['depth_obs'].shape[0]} timesteps={dataset['depth_obs'].shape[1]} "
        f"depth_hw={tuple(dataset['depth_obs'].shape[-2:])}"
    )
    print(
        "[teacher] target mean p/e/g="
        f"{dataset['teacher_camera'][..., 0].mean():.3f}/"
        f"{dataset['teacher_camera'][..., 1].mean():.3f}/"
        f"{dataset['teacher_camera'][..., 2].mean():.3f}"
    )


if __name__ == "__main__":
    main()

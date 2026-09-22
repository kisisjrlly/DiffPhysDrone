"""Evaluation entry point for grayscale/IMX900 active sensing."""

import csv
import os
from random import normalvariate

import torch
from torch.cuda.amp.autocast_mode import autocast

from config import build_parser, parse_scenarios, set_global_seed, validate_args, print_runtime_mode
from controllers.auto_exposure import (
    gradient_auto_exposure_target,
    mean_auto_exposure_target,
)
from model import Model
from rerun_vis import RerunVis
from rollout_ops import (
    render_gray_sensor,
    select_policy_gray_obs,
    build_local_frame,
    build_state_vector,
    compute_target_velocity,
    decode_action_direct,
    init_camera_params,
    update_camera_params,
    compute_camera_param_stats,
)
from train_utils import build_env


def parse_eval_args():
    parser = build_parser()
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--vis_episode_idx", type=int, default=-1)
    parser.add_argument("--eval_trace_csv", type=str, default=None)
    parser.add_argument("--eval_episode_csv", type=str, default=None)
    args = parser.parse_args()
    args.scenarios = parse_scenarios(args.scenarios)
    set_global_seed(args.seed, args.deterministic)
    validate_args(args)
    if args.eval_episodes < 1:
        raise ValueError("--eval_episodes must be >= 1")
    return args


def _append_csv_rows(path, rows):
    if not path or not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _min_clearance_from_vec(vec_now, env):
    dist = torch.norm(vec_now, 2, -1)
    batch = int(env.batch_size)
    if dist.ndim == 1:
        return dist
    if dist.ndim == 2 and dist.shape[1] == batch:
        return dist.min(dim=0).values
    if dist.ndim == 2 and dist.shape[0] == batch:
        return dist.min(dim=1).values
    return dist.reshape(-1, batch).min(dim=0).values


def run_one_episode(ep_idx, scene_name, args, model, env, vis, device, collect_trace=False):
    B = env.batch_size
    use_amp = bool(args.amp and device.type == "cuda")
    env.reset(scene_name=scene_name)
    model.reset()
    h = None
    cam_h = None
    act_buffer = [env.act] * 2
    exposure, gain = init_camera_params(env, B, device)
    prev_gray = None

    collided = torch.zeros(B, dtype=torch.bool, device=device)
    reached = torch.zeros(B, dtype=torch.bool, device=device)
    min_clearance_hist = []
    speed_hist = []
    exposure_hist = []
    gain_hist = []
    sat_hist = []
    dark_hist = []
    blur_hist = []
    light_hist = []
    motion_proxy_hist = []
    char_depth_hist = []
    trace_rows = []

    log_vis = bool(
        vis.enabled
        and (args.vis_episode_idx < 0 or int(ep_idx) == int(args.vis_episode_idx))
    )
    vis_phase = f"episodes/ep_{ep_idx:03d}/student" if log_vis else "student"
    if log_vis:
        j = int(min(max(args.vis_env_idx, 0), B - 1))
        vis.begin_episode(ep_idx, step_base=0)
        vis.log_environment(
            phase=vis_phase,
            balls=env.get_world_balls_for_env(j),
            voxels=env.get_world_voxels_for_env(j),
            cyl=env.get_world_cyl_for_env(j),
            cyl_h=env.get_world_cyl_h_for_env(j),
            start=env.p[j].detach().cpu().numpy(),
            target=env.p_target[j].detach().cpu().numpy(),
            scene_name=env.current_scene_names[j],
            scene_effects=env.get_scene_effects_for_env(j),
            scene_yaw=env.get_scene_yaw_for_env(j),
            step_idx=0,
        )

    stop_reason = "timeout"
    final_goal_dist = torch.norm(env.p_target - env.p, dim=-1).detach()

    for t in range(args.timesteps):
        ctl_dt = normalvariate(
            1.0 / args.base_control_freq,
            0.1 / args.base_control_freq,
        )

        vec_now = env.find_vec_to_nearest_pt()
        clearance = _min_clearance_from_vec(vec_now, env)
        goal_dist = torch.norm(env.p_target - env.p, dim=-1).detach()
        collided |= clearance <= float(args.collision_clearance)
        reached |= goal_dist < 0.35
        if bool(collided.any()):
            stop_reason = "collision"
            break

        gray, aux = render_gray_sensor(
            env,
            exposure,
            gain,
            differentiable=False,
        )
        if prev_gray is None:
            prev_gray = gray
        policy_gray = torch.cat([gray, prev_gray], dim=1)
        policy_gray = select_policy_gray_obs(policy_gray, args.policy_gray_mode)
        prev_gray = gray

        target_v_raw = env.p_target - env.p.detach()
        R = build_local_frame(env)
        target_v = compute_target_velocity(target_v_raw, env)
        state, _, camera_state, camera_motion_state = build_state_vector(
            env,
            target_v,
            R,
            exposure,
            gain,
            args.no_odom,
            args.include_camera_state_in_obs,
        )
        with autocast(enabled=use_amp):
            act_raw, cam_params, h, cam_h = model(
                state,
                h,
                gray_obs=policy_gray,
                cam_hx=cam_h,
                camera_state=camera_state,
                camera_motion_state=camera_motion_state,
            )

        act = decode_action_direct(
            act_raw.float(),
            R,
            env,
            B,
            args.max_acc_cmd,
        )
        if args.camera_control_mode == "mean_ae":
            cam_target = mean_auto_exposure_target(
                gray.detach(), exposure.detach(), gain.detach()
            )
        elif args.camera_control_mode == "gradient_ae":
            cam_target = gradient_auto_exposure_target(
                gray.detach(), exposure.detach(), gain.detach()
            )
        else:
            cam_target = cam_params.float()

        render_exposure, render_gain = exposure, gain
        exposure, gain, _ = update_camera_params(
            cam_target,
            exposure,
            gain,
            env,
        )
        act_buffer.append(act)

        min_clearance_hist.append(clearance.detach())
        speed_hist.append(env.v.norm(2, -1).detach())
        exposure_hist.append(render_exposure.detach())
        gain_hist.append(render_gain.detach())
        for key, hist in (
            ("saturation_fraction", sat_hist),
            ("dark_fraction", dark_hist),
            ("blur_strength", blur_hist),
            ("light_scale", light_hist),
            ("motion_proxy", motion_proxy_hist),
            ("characteristic_depth", char_depth_hist),
        ):
            val = aux.get(key)
            if isinstance(val, torch.Tensor):
                val = val.flatten(1).mean(1) if val.ndim > 1 else val
                hist.append(val.detach())

        if collect_trace:
            p_local = torch.bmm(env.R_scene_T, env.p[:, :, None])[:, :, 0]
            trace_rows.append({
                "episode_idx": ep_idx,
                "step": t,
                "scenario": scene_name,
                "x": float(env.p[0, 0].detach().cpu()),
                "y": float(env.p[0, 1].detach().cpu()),
                "z": float(env.p[0, 2].detach().cpu()),
                "local_x": float(p_local[0, 0].detach().cpu()),
                "local_y": float(p_local[0, 1].detach().cpu()),
                "exposure": float(render_exposure[0].cpu()),
                "gain": float(render_gain[0].cpu()),
                "light_scale": float(aux["light_scale"][0].cpu()),
                "saturation_fraction": float(aux["saturation_fraction"][0].cpu()),
                "dark_fraction": float(aux["dark_fraction"][0].cpu()),
                "blur_strength": float(aux["blur_strength"][0].mean().cpu()),
                "motion_proxy": float(aux["motion_proxy"][0].cpu()),
                "characteristic_depth": float(aux["characteristic_depth"][0].cpu()),
                "goal_dist": float(goal_dist[0].cpu()),
                "clearance": float(clearance[0].cpu()),
            })

        if log_vis and t % max(args.vis_every_steps, 1) == 0:
            j = int(min(max(args.vis_env_idx, 0), B - 1))
            vis.log_step(
                phase=vis_phase,
                step_idx=t,
                pos=env.p[j].detach().cpu().numpy(),
                target=env.p_target[j].detach().cpu().numpy(),
                main_img=gray[j, 0].detach().cpu().numpy(),
                main_img_mode="luma",
                cam=(float(render_exposure[j].cpu()), float(render_gain[j].cpu())),
                scalars={
                    "goal_dist_m": float(goal_dist[j].cpu()),
                    "clearance_m": float(clearance[j].cpu()),
                    "light_scale": float(aux["light_scale"][j].cpu()),
                    "saturation_fraction": float(aux["saturation_fraction"][j].cpu()),
                    "dark_fraction": float(aux["dark_fraction"][j].cpu()),
                    "blur_strength": float(aux["blur_strength"][j].mean().cpu()),
                    "motion_proxy": float(aux["motion_proxy"][j].cpu()),
                    "characteristic_depth": float(aux["characteristic_depth"][j].cpu()),
                },
                drone_R=env.R[j].detach().cpu().numpy(),
                cam_R=env.R_cam[j].detach().cpu().numpy(),
                main_fov_half_tan=float(env._fov_x_half_tan),
                main_hw=(int(env.height), int(env.width)),
                depth_hw=(int(env.height), int(env.width)),
            )

        env.run(act_buffer[t], ctl_dt, target_v_raw)
        final_goal_dist = torch.norm(env.p_target - env.p, dim=-1).detach()
        after_clearance = _min_clearance_from_vec(env.find_vec_to_nearest_pt(), env)
        collided |= after_clearance <= float(args.collision_clearance)
        reached |= final_goal_dist < 0.35
        if bool(collided.any()):
            stop_reason = "collision"
            break

    success = reached & (~collided)
    cam_stats = compute_camera_param_stats(exposure_hist, gain_hist)
    calibration = getattr(env, "imx900_calibration", None)
    row = {
        "scenario": scene_name,
        "camera_profile": getattr(calibration, "profile_name", ""),
        "camera_profile_calibrated": bool(getattr(calibration, "calibrated", False)),
        "success_rate": float(success.float().mean().cpu()),
        "collision_rate": float(collided.float().mean().cpu()),
        "goal_reach_rate": float(reached.float().mean().cpu()),
        "final_goal_dist": float(final_goal_dist.mean().cpu()),
        "avg_speed": float(torch.stack(speed_hist).mean().cpu()) if speed_hist else 0.0,
        "min_clearance": float(torch.stack(min_clearance_hist).min().cpu()) if min_clearance_hist else 0.0,
        "exposure_mean": cam_stats.get("exposure_mean", 0.0),
        "gain_mean": cam_stats.get("gain_mean", 0.0),
        "saturation_fraction": float(torch.stack(sat_hist).mean().cpu()) if sat_hist else 0.0,
        "dark_fraction": float(torch.stack(dark_hist).mean().cpu()) if dark_hist else 0.0,
        "blur_strength": float(torch.stack(blur_hist).mean().cpu()) if blur_hist else 0.0,
        "light_scale": float(torch.stack(light_hist).mean().cpu()) if light_hist else 0.0,
        "motion_proxy": float(torch.stack(motion_proxy_hist).mean().cpu()) if motion_proxy_hist else 0.0,
        "characteristic_depth": float(torch.stack(char_depth_hist).mean().cpu()) if char_depth_hist else 0.0,
        "steps": len(speed_hist),
        "stop_reason": stop_reason,
    }
    return row, trace_rows


def main():
    args = parse_eval_args()
    print_runtime_mode(args)
    if not args.resume:
        raise ValueError("eval requires --resume")
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(args.resume)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = build_env(args.batch_size, args, device, eval_mode=True)
    obs_dim = 7 if args.no_odom else 10
    model = Model(
        obs_dim,
        3,
        include_camera_state_in_obs=args.include_camera_state_in_obs,
        gray_nn_width=args.gray_nn_width,
        gray_nn_height=args.gray_nn_height,
    ).to(device)
    model.load_state_dict(torch.load(args.resume, map_location=device), strict=True)
    model.eval()

    vis = RerunVis(
        enabled=(args.vis_enable and args.vis_backend == "rerun"),
        app_id="DiffPhysDrone-Gray-Eval",
        spawn=args.vis_spawn,
        show_aabb=args.vis_show_aabb,
    )

    if args.eval_episode_csv and os.path.exists(args.eval_episode_csv):
        os.remove(args.eval_episode_csv)
    if args.eval_trace_csv and os.path.exists(args.eval_trace_csv):
        os.remove(args.eval_trace_csv)

    rows = []
    with torch.no_grad():
        for ep in range(args.eval_episodes):
            scene = args.scenarios[ep % len(args.scenarios)]
            row, trace = run_one_episode(
                ep,
                scene,
                args,
                model,
                env,
                vis,
                device,
                collect_trace=bool(args.eval_trace_csv),
            )
            rows.append(row)
            _append_csv_rows(args.eval_episode_csv, [row])
            _append_csv_rows(args.eval_trace_csv, trace)

    print("[eval] overall summary")
    for key in (
        "success_rate",
        "collision_rate",
        "final_goal_dist",
        "avg_speed",
        "saturation_fraction",
        "dark_fraction",
        "blur_strength",
        "motion_proxy",
        "characteristic_depth",
    ):
        print(f"  {key:<22}: {sum(float(r[key]) for r in rows) / len(rows):.4f}")


if __name__ == "__main__":
    main()

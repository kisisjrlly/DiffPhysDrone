"""
DiffPhysDrone evaluation entry point.

目的：
- 复用训练时同一套参数文件与传感器/控制逻辑
- 仅做前向推理，不做训练、不计算 loss、不写 wandb
- 支持 Rerun 实时可视化
"""

import argparse
import os
from random import normalvariate
import time

import torch
from torch.cuda.amp.autocast_mode import autocast

from config import (
    build_parser,
    parse_diff_sensor_impl,
    normalize_sensor_mode,
    resolve_sensor_flags,
    set_global_seed,
    validate_args,
    print_runtime_mode,
)
from lqr import build_velocity_tracking_linear_system, solve_batched_dlqr
from model import Model
from rerun_vis import RerunVis
from rollout_ops import (
    render_sensors,
    build_local_frame,
    build_state_vector,
    compute_target_velocity,
    decode_action_direct,
    decode_action_lqr,
    update_camera_params,
)
from train_utils import build_env, make_yaw_drift_R


def parse_eval_args():
    parser = build_parser()
    parser.add_argument('--eval_episodes', type=int, default=1,
                        help='评估 episode 数（每个 episode 重置一次环境并做 timesteps 步推理）')
    args = parser.parse_args()

    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.sensor_mode = normalize_sensor_mode(args.sensor_mode)
    set_global_seed(args.seed, args.deterministic)

    sensor_flags = resolve_sensor_flags(args)
    validate_args(args, sensor_flags)

    if args.eval_episodes < 1:
        raise ValueError('--eval_episodes 必须 >= 1')

    return args, sensor_flags


def run_one_episode(ep_idx, args, sensor_flags, model, env, vis, device):
    sf = sensor_flags
    B = env.batch_size
    use_amp = bool(args.amp and device.type == 'cuda')

    use_depth_only = sf['use_depth_only']
    use_camera_luma = sf['use_camera_luma']
    use_depth_aux = sf['use_depth_aux']
    use_diff_depth = sf['use_diff_depth']
    use_camera_control = sf['use_camera_control']
    effective_include_camera_state = sf['effective_include_camera_state']

    env.reset()
    model.reset()

    if vis.enabled:
        vis.begin_iter(ep_idx)
        j = int(min(max(args.vis_env_idx, 0), B - 1))
        # 从环境中提取缩放参数（用于动态AABB计算）
        y_stretch_j = getattr(env, '_current_y_stretch', None)
        scale_j = getattr(env, '_current_scale', None)
        vis.log_environment(
            phase='student',
            balls=env.balls[j].detach().cpu().numpy(),
            voxels=env.voxels[j].detach().cpu().numpy(),
            cyl=env.cyl[j].detach().cpu().numpy(),
            cyl_h=env.cyl_h[j].detach().cpu().numpy(),
            start=env.p[j].detach().cpu().numpy(),
            target=env.p_target[j].detach().cpu().numpy(),
            y_stretch=y_stretch_j,
            scale=scale_j,
        )

    h = None
    act_buffer = [env.act] * 2
    target_v_raw = env.p_target - env.p
    yaw_drift_R = make_yaw_drift_R(B, device) if args.yaw_drift else None

    cam_fov = torch.full((B,), env._fov_x_half_tan, device=device)
    cam_exposure = torch.full((B,), 0.5, device=device)
    cam_iso = torch.full((B,), 0.5, device=device)

    # 与训练推理路径一致（目前训练中也是固定 1/15 构造 LQR 离散系统）
    A_lqr, B_lqr = build_velocity_tracking_linear_system(B, 1 / 15, device)

    min_margin_hist = []
    speed_hist = []

    for t in range(args.timesteps):
        print("timestep:", t)
        base_dt = normalvariate(1 / args.base_control_freq, 0.1 / args.base_control_freq)
        exposure_delay = (
            float(env.cam_sem.exposure_to_time(cam_exposure.mean().detach())) * 0.01
            if use_camera_control else 0.015
        )
        ctl_dt = base_dt + exposure_delay

        main_obs, depth_obs = render_sensors(
            env, ctl_dt, cam_fov, cam_exposure, cam_iso,
            use_depth_only, use_camera_luma, use_diff_depth,
            use_depth_aux, use_camera_control,
            differentiable=False,
        )

        vec_now = env.find_vec_to_nearest_pt()
        min_margin_now = (torch.norm(vec_now, 2, -1) - env.margin)
        min_margin_hist.append(min_margin_now)

        if args.yaw_drift and yaw_drift_R is not None:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ yaw_drift_R, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()

        env.run(act_buffer[t], ctl_dt, target_v_raw)

        R = build_local_frame(env)
        target_v = compute_target_velocity(target_v_raw, env)
        state, local_v = build_state_vector(
            env, target_v, R, cam_fov, cam_exposure, cam_iso,
            args.no_odom, effective_include_camera_state, use_camera_control,
        )

        if args.policy_output_intent:
            with autocast(enabled=use_amp):
                act_raw, cam_params, h, intent = model(
                    state, h, return_intent=True,
                    main_obs=main_obs, depth_obs=depth_obs,
                    add_noise=False,
                )
            act_raw = act_raw.float()
            intent = intent.float()
        else:
            with autocast(enabled=use_amp):
                act_raw, cam_params, h = model(
                    state, h,
                    main_obs=main_obs, depth_obs=depth_obs,
                    add_noise=False,
                )
            act_raw = act_raw.float()
            intent = None

        if cam_params is not None:
            cam_params = cam_params.float()

        cam_fov, cam_exposure, cam_iso, _ = update_camera_params(
            cam_params, cam_fov, cam_exposure, cam_iso, env,
        )

        if args.use_dmpc and args.policy_output_intent and intent is not None:
            act_final, _ = decode_action_lqr(
                intent, R, env, local_v, B,
                A_lqr, B_lqr,
                args.lqr_horizon, args.lqr_reg, args.max_acc_cmd,
                args.inject_depth_into_lqr, args.depth_safe_dist, args.depth_repel_gain,
                vec_now, solve_batched_dlqr,
            )
        else:
            act_final, _ = decode_action_direct(act_raw, R, env, B, args.max_acc_cmd)

        act_buffer.append(act_final)
        speed_hist.append(env.v.norm(2, -1))

        if vis.enabled:
            j = int(min(max(args.vis_env_idx, 0), B - 1))
            cam_vals = None
            if use_camera_control:
                cam_vals = (
                    float(cam_fov[j].detach().cpu()),
                    float(cam_exposure[j].detach().cpu()),
                    float(cam_iso[j].detach().cpu()),
                )

            main_img_np = main_obs[j].detach().cpu().numpy() if main_obs is not None else None
            main_img_mode = 'luma' if use_camera_luma else 'depth'
            depth_img_np = depth_obs[j].detach().cpu().numpy() if depth_obs is not None else None

            vis.log_step(
                phase='student',
                step_idx=t,
                pos=env.p[j].detach().cpu().numpy(),
                target=env.p_target[j].detach().cpu().numpy(),
                depth=(main_obs[j].detach().cpu().numpy() if (main_obs is not None and use_depth_only) else None),
                cam=cam_vals,
                main_img=main_img_np,
                main_img_mode=main_img_mode,
                depth_img=depth_img_np,
                drone_R=env.R[j].detach().cpu().numpy(),
                cam_R=env.R_cam[j].detach().cpu().numpy(),
                main_fov_half_tan=(float(cam_fov[j].detach().cpu()) if use_camera_control else float(env._fov_x_half_tan)),
                main_hw=(int(env.height), int(env.width)),
                depth_hw=(int(env.depth_height), int(env.depth_width)),
            )
        time.sleep(1)

    min_margin_all = torch.stack(min_margin_hist).amin(dim=0)
    success_mask = min_margin_all > 0
    success_rate = float(success_mask.float().mean().detach().cpu())

    speed_all = torch.stack(speed_hist)
    avg_speed = float(speed_all.mean().detach().cpu())
    max_speed = float(speed_all.max().detach().cpu())

    print(
        f"[eval] episode={ep_idx + 1}/{args.eval_episodes} "
        f"success_rate={success_rate:.3f} avg_speed={avg_speed:.3f} max_speed={max_speed:.3f}"
    )

    if vis.enabled:
        vis.log_train_scalars({
            'eval_success_rate': success_rate,
            'eval_avg_speed': avg_speed,
            'eval_max_speed': max_speed,
        }, iter_idx=ep_idx)


def main():
    args, sensor_flags = parse_eval_args()
    sf = sensor_flags

    if not args.resume:
        raise ValueError('评估必须提供 --resume <checkpoint_path>')
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f'checkpoint 不存在: {args.resume}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("\n" + "=" * 30 + " Eval Configuration " + "=" * 30)
    for k, v in vars(args).items():
        print(f"{k:<30}: {v}")
    print("=" * 80 + "\n")
    print_runtime_mode(args, sf)

    env = build_env(args.batch_size, args, sf, device)

    obs_dim = 7 if args.no_odom else 10
    main_channels = 1
    in_channels = main_channels + (1 if sf['use_depth'] else 0)
    model = Model(
        obs_dim, 6,
        include_camera_state_in_obs=sf['effective_include_camera_state'],
        in_channels=in_channels,
        use_policy_intent=args.policy_output_intent,
        intent_dim=9,
        main_in_channels=main_channels,
        enable_camera_head=sf['use_camera_control'],
        depth_nn_width=args.depth_nn_width,
        depth_nn_height=args.depth_nn_height,
        diff_depth_use_pipeline=args.diff_depth_use_pipeline,
        sensor_mode=args.sensor_mode,
    ).to(device)

    print(f"[eval] loading checkpoint: {args.resume}")
    state_dict = torch.load(args.resume, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print('[eval][warn] missing_keys:', missing)
    if unexpected:
        print('[eval][warn] unexpected_keys:', unexpected)

    model.eval()

    vis = RerunVis(
        enabled=(args.vis_enable and args.vis_backend == 'rerun'),
        app_id='DiffPhysDrone-Eval',
        spawn=args.vis_spawn,
    )

    with torch.no_grad():
        for ep_idx in range(args.eval_episodes):
            run_one_episode(ep_idx, args, sf, model, env, vis, device)

    print('[eval] done.')


if __name__ == '__main__':
    main()

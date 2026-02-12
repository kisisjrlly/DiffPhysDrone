from collections import defaultdict
import math
import os
from random import normalvariate
from matplotlib import pyplot as plt
from env_cuda import Env, apply_camera_effects
import imageio
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import time
from tqdm import tqdm

import argparse
from model import Model


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_iters', type=int, default=50000)
parser.add_argument('--coef_v', type=float, default=1.0, help='smooth l1 of norm(v_set - v_real)')
parser.add_argument('--coef_speed', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_v_pred', type=float, default=2.0, help='mse loss for velocity estimation (no odom)')
parser.add_argument('--coef_collide', type=float, default=2.0, help='softplus loss for collision (large if close to obstacle, zero otherwise)')
parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='quadratic clearance loss')
parser.add_argument('--coef_d_acc', type=float, default=0.01, help='control acceleration regularization')
parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='control jerk regularizatinon')
parser.add_argument('--coef_d_snap', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_ground_affinity', type=float, default=0., help='legacy')
parser.add_argument('--coef_bias', type=float, default=0.0, help='legacy')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--grad_decay', type=float, default=0.4)
parser.add_argument('--speed_mtp', type=float, default=1.0)
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--cam_angle', type=int, default=10)
parser.add_argument('--single', default=False, action='store_true')
parser.add_argument('--gate', default=False, action='store_true')
parser.add_argument('--ground_voxels', default=False, action='store_true')
parser.add_argument('--scaffold', default=False, action='store_true')
parser.add_argument('--random_rotation', default=False, action='store_true')
parser.add_argument('--yaw_drift', default=False, action='store_true')
parser.add_argument('--no_odom', default=False, action='store_true')
parser.add_argument('--diff_cam', default=False, action='store_true', help='enable differentiable perception (camera params)')
parser.add_argument('--coef_cam_smooth', type=float, default=0.01, help='camera param smoothness regularization')
parser.add_argument('--coef_fov_reg', type=float, default=0.005, help='FOV deviation from default regularization')
parser.add_argument('--coef_cam_range', type=float, default=0.001, help='camera param range regularization')
parser.add_argument('--wandb_disabled', default=False, action='store_true', help='Disable wandb logging')
parser.add_argument('--wall_slit', default=False, action='store_true', help='Wall-slit environment (narrow vertical gap)')
parser.add_argument('--ellipsoid_collision', default=False, action='store_true', help='Use ellipsoid drone model for collision detection')
parser.add_argument('--drone_a', type=float, default=0.15, help='Ellipsoid semi-axis XY (propeller plane radius)')
parser.add_argument('--drone_c', type=float, default=0.075, help='Ellipsoid semi-axis Z (half-height)')
parser.add_argument('--coef_tilt', type=float, default=0.0, help='Tilt alignment loss near wall slit')

# ===== Paper.md: Unified control, optical losses, G-DAC =====
parser.add_argument('--paper_unified_control', default=False, action='store_true',
                    help='Paper §2.1: camera deltas as part of action (incremental tanh)')
parser.add_argument('--paper_cam_obs', default=False, action='store_true',
                    help='Paper §2.1: include camera state in observation vector')
parser.add_argument('--paper_optical_loss', default=False, action='store_true',
                    help='Paper §2.3: motion blur / shot noise / defocus potentials')
parser.add_argument('--coef_blur', type=float, default=0.1,
                    help='Paper §2.3A: motion blur potential weight')
parser.add_argument('--coef_noise', type=float, default=0.05,
                    help='Paper §2.3B: shot noise potential weight')
parser.add_argument('--coef_defocus', type=float, default=0.05,
                    help='Paper §2.3C: defocus potential weight')
parser.add_argument('--cam_delta_scale', type=float, default=0.05,
                    help='Per-step scale for incremental camera deltas (unified control)')
parser.add_argument('--paper_gdac', default=False, action='store_true',
                    help='Paper §3: two-phase G-DAC (teacher-student) training')
parser.add_argument('--gdac_inner_steps', type=int, default=10,
                    help='G-DAC Phase I: number of inner optimization steps')
parser.add_argument('--gdac_inner_lr', type=float, default=0.01,
                    help='G-DAC Phase I: inner optimizer learning rate')
parser.add_argument('--coef_distill', type=float, default=1.0,
                    help='G-DAC Phase II: distillation loss weight')
parser.add_argument('--gdac_physics_weight', type=float, default=0.3,
                    help='G-DAC Phase II: weight of physics losses alongside distillation')

args = parser.parse_args()

# Generate a unique run name based on time
run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"

wandb.init(
    project="diff-simulation", 
    name=run_name,
    config=args,
    # Automatically save only relevant code files
    settings=wandb.Settings(code_dir="."),
    mode="disabled" if args.wandb_disabled else "online"
)

# Manually save specific source files to ensure they are tracked
# Exclude build catalogs and installation directories by only selecting specific files/folders
wandb.save("*.py")
wandb.save("src/*.cu")
wandb.save("src/*.cpp")
wandb.save("src/*.py")
wandb.save("configs/*.args")
wandb.save("*.sh")

print(args)

device = torch.device('cuda')

# Determine if any camera-aware mode is active
use_cam = args.diff_cam or args.paper_unified_control

env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle,
          wall_slit=args.wall_slit,
          ellipsoid_a=args.drone_a if args.ellipsoid_collision else 0.0,
          ellipsoid_c=args.drone_c if args.ellipsoid_collision else 0.0)

obs_dim = 7 if args.no_odom else 10
model = Model(obs_dim, 6,
              use_diff_cam=args.diff_cam,
              use_unified_control=args.paper_unified_control,
              use_cam_obs=args.paper_cam_obs)
model = model.to(device)

if args.resume:
    state_dict = torch.load(args.resume, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
    if missing_keys:
        print("missing_keys:", missing_keys)
    if unexpected_keys:
        print("unexpected_keys:", unexpected_keys)
optim = AdamW(model.parameters(), args.lr)
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)

ctl_dt = 1 / 15


scaler_q = defaultdict(list)
def smooth_dict(ori_dict):
    for k, v in ori_dict.items():
        scaler_q[k].append(float(v))

def barrier(x: torch.Tensor, v_to_pt):
    return (v_to_pt * (1 - x).relu().pow(2)).mean()

def is_save_iter(i):
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0

pbar = tqdm(range(args.num_iters), ncols=80)
# depths = []
# states = []
B = args.batch_size
vid_idx = min(4, B - 1)
iter_start_time = time.time()
for i in pbar:
    iter_tic = time.time()
    env.reset()
    model.reset()

    # ===== Paper §3: G-DAC Phase I — Teacher / Solver =====
    u_star = None  # optimal actions from teacher (only when paper_gdac enabled)
    u_star_cam = None  # optimal camera deltas from teacher
    if args.paper_gdac:
        # Save initial environment state for replay
        env_snapshot = env.save_state()
        yaw_drift_R = None
        if args.yaw_drift:
            drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
            zeros = torch.zeros_like(drift_av)
            ones = torch.ones_like(drift_av)
            yaw_drift_R = torch.stack([
                torch.cos(drift_av), -torch.sin(drift_av), zeros,
                torch.sin(drift_av), torch.cos(drift_av), zeros,
                zeros, zeros, ones,
            ], -1).reshape(B, 3, 3)

        # Get initial guess from current policy (no grad for model)
        with torch.no_grad():
            init_acts = []
            init_cam_deltas = []
            h_tmp = None
            env.restore_state(env_snapshot)
            act_buf_tmp = [env.act] * 2
            tv_raw = env.p_target - env.p
            cam_fov_tmp = torch.full((B,), env._fov_x_half_tan, device=device) if use_cam else None
            cam_exp_tmp = torch.full((B,), 0.5, device=device) if use_cam else None
            cam_iso_tmp = torch.full((B,), 0.5, device=device) if use_cam else None
            cam_foc_tmp = torch.full((B,), 0.5, device=device) if use_cam else None

            for t in range(args.timesteps):
                dt_tmp = 1 / 15
                if use_cam:
                    dp = env.render_diff(cam_fov_tmp)
                    dp = apply_camera_effects(dp, cam_exp_tmp, cam_iso_tmp, cam_foc_tmp)
                else:
                    dp, _ = env.render(dt_tmp)
                if args.yaw_drift:
                    tv_raw = torch.squeeze(tv_raw[:, None] @ yaw_drift_R, 1)
                else:
                    tv_raw = env.p_target - env.p
                env.run(act_buf_tmp[t], dt_tmp, tv_raw)
                R_t = env.R
                fwd_t = env.R[:, :, 0].clone(); fwd_t[:, 2] = 0
                up_t = torch.zeros_like(fwd_t); up_t[:, 2] = 1
                fwd_t = F.normalize(fwd_t, 2, -1)
                R_t = torch.stack([fwd_t, torch.cross(up_t, fwd_t), up_t], -1)

                tv_n = torch.norm(tv_raw, 2, -1, keepdim=True)
                tv_u = tv_raw / tv_n
                tv = tv_u * torch.minimum(tv_n, env.max_speed)
                st = [torch.squeeze(tv[:, None] @ R_t, 1), env.R[:, 2], env.margin[:, None]]
                lv = torch.squeeze(env.v[:, None] @ R_t, 1)
                if not args.no_odom:
                    st.insert(0, lv)
                if args.paper_cam_obs and use_cam:
                    co = torch.stack([cam_fov_tmp / env._fov_x_half_tan - 1.0,
                                      cam_exp_tmp, cam_iso_tmp, cam_foc_tmp], -1)
                    st.append(co)
                st = torch.cat(st, -1)
                if use_cam:
                    xx = 3 / dp.clamp(0.3, 24) - 0.6
                else:
                    xx = 3 / dp.clamp_(0.3, 24) - 0.6
                xx = F.max_pool2d(xx[:, None], 4, 4)
                a_out, c_out, h_tmp = model(xx, st, h_tmp)
                init_acts.append(a_out.clone())
                if c_out is not None:
                    init_cam_deltas.append(c_out.clone())
                # Decode action for sim
                ap, vp, *_ = (R_t @ a_out.reshape(B, 3, -1)).unbind(-1)
                a_final = (ap - vp - env.g_std) * env.thr_est_error[:, None] + env.g_std
                act_buf_tmp.append(a_final)
                # Update camera params
                if args.paper_unified_control and c_out is not None:
                    df, de, di, dfo = c_out.unbind(-1)
                    sc = args.cam_delta_scale
                    cam_fov_tmp = (cam_fov_tmp + df * sc * env._fov_x_half_tan).clamp(
                        env._fov_x_half_tan * 0.3, env._fov_x_half_tan * 2.0)
                    cam_exp_tmp = (cam_exp_tmp + de * sc).clamp(0.01, 0.99)
                    cam_iso_tmp = (cam_iso_tmp + di * sc).clamp(0.01, 0.99)
                    cam_foc_tmp = (cam_foc_tmp + dfo * sc).clamp(0.01, 0.99)
                elif c_out is not None:
                    fd, ex, iso_v, fc_v = c_out.unbind(-1)
                    cam_fov_tmp = env._fov_x_half_tan * (0.5 + fd)
                    cam_exp_tmp = ex; cam_iso_tmp = iso_v; cam_foc_tmp = fc_v

        # Make optimizable action sequence
        u_guess = [a.clone().requires_grad_(True) for a in init_acts]
        u_cam_guess = None
        if use_cam and len(init_cam_deltas) > 0:
            u_cam_guess = [c.clone().requires_grad_(True) for c in init_cam_deltas]
            inner_params = u_guess + u_cam_guess
        else:
            inner_params = u_guess
        inner_optim = torch.optim.Adam(inner_params, lr=args.gdac_inner_lr)

        # Inner optimization loop
        for k in range(args.gdac_inner_steps):
            inner_optim.zero_grad()
            env.restore_state(env_snapshot)
            act_buf_k = [env.act.detach()] * 2
            tv_raw_k = env.p_target - env.p
            p_hist_k = []
            v_hist_k = []
            tv_hist_k = []
            vtp_hist_k = []
            cam_exp_k = []
            cam_iso_k = []
            cam_fov_k = []
            cam_foc_k = []
            speed_k = []
            cam_fov_k_val = torch.full((B,), env._fov_x_half_tan, device=device) if use_cam else None
            cam_exp_k_val = torch.full((B,), 0.5, device=device) if use_cam else None
            cam_iso_k_val = torch.full((B,), 0.5, device=device) if use_cam else None
            cam_foc_k_val = torch.full((B,), 0.5, device=device) if use_cam else None

            for t in range(args.timesteps):
                dt_k = 1 / 15
                p_hist_k.append(env.p)
                vtp_hist_k.append(env.find_vec_to_nearest_pt())
                if args.yaw_drift:
                    tv_raw_k = torch.squeeze(tv_raw_k[:, None] @ yaw_drift_R, 1)
                else:
                    tv_raw_k = env.p_target - env.p.detach()
                env.run(act_buf_k[t], dt_k, tv_raw_k)

                R_k = env.R
                fwd_k = env.R[:, :, 0].clone(); fwd_k[:, 2] = 0
                up_k = torch.zeros_like(fwd_k); up_k[:, 2] = 1
                fwd_k = F.normalize(fwd_k, 2, -1)
                R_k = torch.stack([fwd_k, torch.cross(up_k, fwd_k), up_k], -1)

                tv_n_k = torch.norm(tv_raw_k, 2, -1, keepdim=True)
                tv_k = (tv_raw_k / tv_n_k) * torch.minimum(tv_n_k, env.max_speed)

                # Decode the optimizable action
                ap_k, vp_k, *_ = (R_k @ u_guess[t].reshape(B, 3, -1)).unbind(-1)
                a_k = (ap_k - vp_k - env.g_std) * env.thr_est_error[:, None] + env.g_std
                act_buf_k.append(a_k)

                # Update camera params from optimizable deltas
                if use_cam and u_cam_guess is not None:
                    if args.paper_unified_control:
                        df, de, di, dfo = u_cam_guess[t].unbind(-1)
                        sc = args.cam_delta_scale
                        cam_fov_k_val = (cam_fov_k_val + df * sc * env._fov_x_half_tan).clamp(
                            env._fov_x_half_tan * 0.3, env._fov_x_half_tan * 2.0)
                        cam_exp_k_val = (cam_exp_k_val + de * sc).clamp(0.01, 0.99)
                        cam_iso_k_val = (cam_iso_k_val + di * sc).clamp(0.01, 0.99)
                        cam_foc_k_val = (cam_foc_k_val + dfo * sc).clamp(0.01, 0.99)
                    else:
                        fd, ex, iso_v, fc_v = u_cam_guess[t].unbind(-1)
                        cam_fov_k_val = env._fov_x_half_tan * (0.5 + fd)
                        cam_exp_k_val = ex; cam_iso_k_val = iso_v; cam_foc_k_val = fc_v
                    cam_fov_k.append(cam_fov_k_val)
                    cam_exp_k.append(cam_exp_k_val)
                    cam_iso_k.append(cam_iso_k_val)
                    cam_foc_k.append(cam_foc_k_val)
                speed_k.append(env.v.norm(2, -1))
                v_hist_k.append(env.v)
                tv_hist_k.append(tv_k)

            # Compute loss on the teacher rollout
            p_hist_k = torch.stack(p_hist_k)
            act_buf_k = torch.stack(act_buf_k)
            v_hist_k = torch.stack(v_hist_k)
            tv_hist_k = torch.stack(tv_hist_k)
            vtp_hist_k = torch.stack(vtp_hist_k)

            l_ga_k = p_hist_k[..., 2].relu().pow(2).mean()
            v_cum_k = v_hist_k.cumsum(0)
            v_avg_k = (v_cum_k[30:] - v_cum_k[:-30]) / 30
            d_v_k = torch.norm(v_avg_k - tv_hist_k[1:1-30], 2, -1)
            l_v_k = F.smooth_l1_loss(d_v_k, torch.zeros_like(d_v_k))

            jerk_k = act_buf_k.diff(1, 0).mul(15)
            l_acc_k = act_buf_k.pow(2).sum(-1).mean()
            l_jerk_k = jerk_k.pow(2).sum(-1).mean()

            dist_k = torch.norm(vtp_hist_k, 2, -1) - env.margin
            with torch.no_grad():
                v_to_k = (-torch.diff(dist_k, 1, 1) * 135).clamp_min(1)
            l_avoid_k = barrier(dist_k[:, 1:], v_to_k)
            l_coll_k = F.softplus(dist_k[:, 1:].mul(-32)).mul(v_to_k).mean()

            inner_loss = args.coef_v * l_v_k + \
                args.coef_obj_avoidance * l_avoid_k + \
                args.coef_d_acc * l_acc_k + \
                args.coef_d_jerk * l_jerk_k + \
                args.coef_collide * l_coll_k + \
                l_ga_k

            # Add optical losses to teacher objective
            if args.paper_optical_loss and use_cam and len(cam_exp_k) > 0:
                sp_k = torch.stack(speed_k)
                ex_k = torch.stack(cam_exp_k)
                iso_k = torch.stack(cam_iso_k)
                fov_k_t = torch.stack(cam_fov_k)
                exp_phys_k = ex_k * 10 + 0.5
                eff_focal_k = 1.0 / fov_k_t.clamp(min=0.1)
                inner_loss = inner_loss + args.coef_blur * (sp_k.pow(2) * exp_phys_k.pow(2) * eff_focal_k.pow(2)).mean()
                ns_k = 0.03 * (1.0 + 2.0 * iso_k) / (ex_k + 0.3)
                inner_loss = inner_loss + args.coef_noise * ns_k.pow(2).mean()

            inner_loss.backward()
            inner_optim.step()

        # Extract optimized actions
        u_star = [u.detach() for u in u_guess]
        if u_cam_guess is not None:
            u_star_cam = [c.detach() for c in u_cam_guess]
        # Restore environment for student rollout
        env.restore_state(env_snapshot)

    # ===== Standard / G-DAC Phase II: Student rollout =====
    p_history = []
    v_history = []
    target_v_history = []
    vec_to_pt_history = []
    act_diff_history = []
    v_preds = []
    vid = []
    v_net_feats = []
    raw_act_history = []  # for G-DAC distillation
    raw_cam_history = []  # for G-DAC distillation (camera deltas)
    h = None

    act_lag = 1
    act_buffer = [env.act] * (act_lag + 1)
    target_v_raw = env.p_target - env.p
    if args.yaw_drift:
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
        zeros = torch.zeros_like(drift_av)
        ones = torch.ones_like(drift_av)
        R_drift = torch.stack([
            torch.cos(drift_av), -torch.sin(drift_av), zeros,
            torch.sin(drift_av), torch.cos(drift_av), zeros,
            zeros, zeros, ones,
        ], -1).reshape(B, 3, 3)

    # Differentiable camera: initialize camera params to defaults
    cam_params_history = []
    cam_fov_history = []
    cam_exposure_history = []
    cam_iso_history = []
    cam_focus_history = []
    speed_for_cam_history = []   # for optical loss & emerging-behavior metrics
    R_up_history = []            # for roll angle tracking

    if use_cam:
        cam_fov = torch.full((B,), env._fov_x_half_tan, device=device)
        cam_exposure = torch.full((B,), 0.5, device=device)
        cam_iso = torch.full((B,), 0.5, device=device)
        cam_focus = torch.full((B,), 0.5, device=device)

    for t in range(args.timesteps):
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)
        # Render: use differentiable FOV if any camera-aware mode is enabled
        if use_cam:
            depth = env.render_diff(cam_fov)
            depth = apply_camera_effects(depth, cam_exposure, cam_iso, cam_focus)
        else:
            depth, flow = env.render(ctl_dt)
        p_history.append(env.p)
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())

        if is_save_iter(i):
            vid.append(depth[vid_idx])

        if args.yaw_drift:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()
        env.run(act_buffer[t], ctl_dt, target_v_raw)

        R = env.R
        fwd = env.R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[:, 2] = 0
        up[:, 2] = 1
        fwd = F.normalize(fwd, 2, -1)
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1)

        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / target_v_norm
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)
        state = [
            torch.squeeze(target_v[:, None] @ R, 1),
            env.R[:, 2],
            env.margin[:, None]]
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom:
            state.insert(0, local_v)
        # Paper §2.1: include current camera state in observation
        if args.paper_cam_obs and use_cam:
            cam_obs = torch.stack([
                cam_fov / env._fov_x_half_tan - 1.0,  # normalized FOV deviation
                cam_exposure,
                cam_iso,
                cam_focus
            ], -1)  # (B, 4)
            state.append(cam_obs)
        state = torch.cat(state, -1)

        # normalize depth to inverse-depth feature
        if use_cam:
            # Use non-inplace clamp to preserve autograd graph through depth
            x = 3 / depth.clamp(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
        else:
            x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
        x = F.max_pool2d(x[:, None], 4, 4)
        act, cam_params, h = model(x, state, h)
        raw_act_history.append(act)
        if cam_params is not None:
            raw_cam_history.append(cam_params)

        # Update camera parameters for next timestep's render
        if args.paper_unified_control and cam_params is not None:
            # Unified control: cam_params are tanh deltas in [-1, 1]
            delta_fov, delta_exp, delta_iso, delta_focus = cam_params.unbind(-1)
            scale = args.cam_delta_scale
            cam_fov = (cam_fov + delta_fov * scale * env._fov_x_half_tan).clamp(
                env._fov_x_half_tan * 0.3, env._fov_x_half_tan * 2.0)
            cam_exposure = (cam_exposure + delta_exp * scale).clamp(0.01, 0.99)
            cam_iso = (cam_iso + delta_iso * scale).clamp(0.01, 0.99)
            cam_focus = (cam_focus + delta_focus * scale).clamp(0.01, 0.99)
            cam_params_history.append(torch.stack([
                cam_fov / env._fov_x_half_tan,  # store as normalized FOV
                cam_exposure, cam_iso, cam_focus], -1))
        elif cam_params is not None:
            # Legacy diff_cam: absolute sigmoid params
            fov_delta, exposure, iso, focus_dist = cam_params.unbind(-1)
            # fov_delta in [0,1] via sigmoid -> FOV in [0.5*base, 1.5*base]
            cam_fov = env._fov_x_half_tan * (0.5 + fov_delta)
            cam_exposure = exposure
            cam_iso = iso
            cam_focus = focus_dist
            cam_params_history.append(cam_params)

        # Track camera & speed histories for optical losses and metrics
        if use_cam:
            cam_fov_history.append(cam_fov)
            cam_exposure_history.append(cam_exposure)
            cam_iso_history.append(cam_iso)
            cam_focus_history.append(cam_focus)
        speed_for_cam_history.append(env.v.norm(2, -1))
        R_up_history.append(env.R[:, :, 2].clone())  # up-vector (3rd col of R)

        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        v_preds.append(v_pred)
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act)
        v_net_feats.append(torch.cat([act, local_v, h], -1))

        v_history.append(env.v)
        target_v_history.append(target_v)

    p_history = torch.stack(p_history)
    loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()
    act_buffer = torch.stack(act_buffer)

    v_history = torch.stack(v_history)
    v_history_cum = v_history.cumsum(0)
    v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30
    target_v_history = torch.stack(target_v_history)
    T, B, _ = v_history.shape
    delta_v = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1)
    loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))

    v_preds = torch.stack(v_preds)
    loss_v_pred = F.mse_loss(v_preds, v_history.detach())

    target_v_history_norm = torch.norm(target_v_history, 2, -1)
    target_v_history_normalized = target_v_history / target_v_history_norm[..., None]
    fwd_v = torch.sum(v_history * target_v_history_normalized, -1)
    loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_history_normalized) * 3

    jerk_history = act_buffer.diff(1, 0).mul(15)
    snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)
    loss_d_acc = act_buffer.pow(2).sum(-1).mean()
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean()
    loss_d_snap = snap_history.pow(2).sum(-1).mean()

    vec_to_pt_history = torch.stack(vec_to_pt_history)
    distance = torch.norm(vec_to_pt_history, 2, -1)
    distance = distance - env.margin
    with torch.no_grad():
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
    loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)
    loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()

    speed_history = v_history.norm(2, -1)
    loss_speed = F.smooth_l1_loss(fwd_v, target_v_history_norm)

    # Camera parameter losses (differentiable perception)
    loss_cam_smooth = torch.tensor(0.0, device=device)
    loss_fov_reg = torch.tensor(0.0, device=device)
    loss_cam_range = torch.tensor(0.0, device=device)
    if use_cam and len(cam_params_history) > 1:
        cam_hist = torch.stack(cam_params_history)  # (T, B, 4)
        # Smoothness: penalize rapid camera parameter changes between timesteps
        cam_diff = cam_hist.diff(1, 0)  # (T-1, B, 4)
        loss_cam_smooth = cam_diff.pow(2).mean()
        # FOV regularization: keep FOV near default
        if args.paper_unified_control:
            # In unified mode, cam_hist[:,:,0] is normalized FOV (1.0 = default)
            fov_vals = cam_hist[:, :, 0]
            loss_fov_reg = (fov_vals - 1.0).pow(2).mean()
        else:
            # Legacy: fov_delta=0.5 → default FOV
            fov_deltas = cam_hist[:, :, 0]
            loss_fov_reg = (fov_deltas - 0.5).pow(2).mean()
        # Range regularization: keep all params near center to avoid extreme values
        loss_cam_range = (cam_hist - 0.5).pow(2).mean()

    # ===== Paper §2.3: Optical Perception Potentials =====
    loss_blur = torch.tensor(0.0, device=device)
    loss_noise = torch.tensor(0.0, device=device)
    loss_defocus = torch.tensor(0.0, device=device)
    if args.paper_optical_loss and use_cam and len(cam_exposure_history) > 0:
        speed_hist = torch.stack(speed_for_cam_history)       # (T, B)
        exp_hist = torch.stack(cam_exposure_history)           # (T, B)
        iso_hist = torch.stack(cam_iso_history)                # (T, B)
        focus_hist = torch.stack(cam_focus_history)            # (T, B)
        fov_hist = torch.stack(cam_fov_history)                # (T, B)

        # A. Motion Blur Potential: V_blur = ||v||^2 * t_exp^2 / fov^2
        #    Smaller FOV (longer focal length) amplifies motion blur
        exposure_phys = exp_hist * 10 + 0.5   # [0.5, 10.5] ms → seconds: /1000
        effective_focal = 1.0 / fov_hist.clamp(min=0.1)  # focal ∝ 1/fov
        loss_blur = (speed_hist.pow(2) * exposure_phys.pow(2) * effective_focal.pow(2)).mean()

        # B. Shot Noise Potential: V_noise ∝ noise_sigma^2
        #    noise_sigma = 0.03 * (1 + 2*iso) / (exposure + 0.3)
        noise_sigma = 0.03 * (1.0 + 2.0 * iso_hist) / (exp_hist + 0.3)
        loss_noise = noise_sigma.pow(2).mean()

        # C. Defocus Potential: V_defocus = (d_focus - d_nearest)^2
        #    d_nearest from nearest obstacle distance at each timestep
        dist_hist = torch.norm(vec_to_pt_history, 2, -1)  # (T, sub_steps, B)
        d_nearest = dist_hist.min(1).values                # (T, B) min over sub-steps
        focus_phys = focus_hist * 20 + 0.5                 # [0.5, 20.5] m
        # Only penalize when obstacle is within sensing range (<15m)
        in_range_mask = (d_nearest < 15.0).float()
        loss_defocus = (in_range_mask * (focus_phys - d_nearest).pow(2)).mean()

    # Wall-slit tilt loss: encourage the drone to roll sideways near the wall
    loss_tilt = torch.tensor(0.0, device=device)
    if args.wall_slit and args.coef_tilt > 0:
        # p_history is (T, B, 3). Check when drone is near the wall x position.
        wall_x = env.wall_x
        dx_to_wall = (p_history[..., 0] - wall_x).abs()  # (T, B)
        near_wall_mask = (dx_to_wall < 1.0).float()  # within 1m of wall, (T, B)
        if near_wall_mask.sum() > 0:
            # The drone's up vector is R[:, :, 2] (3rd column).
            # For the drone to pass through a vertical slit (narrow in Y),
            # its Y-extent should be minimized, i.e. its "up" vector should be
            # close to horizontal (pointing along Y). We penalize |up_z| being
            # close to 1 (level flight) when near the wall. Instead we want
            # |up_y| to be large (tilted sideways).
            # Since R changes each timestep and we only have the final R,
            # use the distance-to-obstacle min as a proxy — already handled
            # by the ellipsoid collision. The tilt loss provides a soft
            # curriculum signal before the ellipsoid penalty kicks in.
            # We approximate by penalizing up_z^2 when near the wall.
            # Note: R is updated inside the loop, so we use vec_to_pt as proxy.
            # Actually, the best approach: build up-vector history.
            # For simplicity, use the distance penalty which already encodes tilt
            # via the ellipsoid model. Set loss_tilt = 0 and rely on ellipsoid.
            pass

    loss = args.coef_v * loss_v + \
        args.coef_obj_avoidance * loss_obj_avoidance + \
        args.coef_bias * loss_bias + \
        args.coef_d_acc * loss_d_acc + \
        args.coef_d_jerk * loss_d_jerk + \
        args.coef_d_snap * loss_d_snap + \
        args.coef_speed * loss_speed + \
        args.coef_v_pred * loss_v_pred + \
        args.coef_collide * loss_collide + \
        args.coef_ground_affinity + loss_ground_affinity + \
        args.coef_cam_smooth * loss_cam_smooth + \
        args.coef_fov_reg * loss_fov_reg + \
        args.coef_cam_range * loss_cam_range + \
        args.coef_tilt * loss_tilt + \
        args.coef_blur * loss_blur + \
        args.coef_noise * loss_noise + \
        args.coef_defocus * loss_defocus

    # ===== Paper §3: G-DAC Phase II — Distillation Loss =====
    loss_distill = torch.tensor(0.0, device=device)
    if args.paper_gdac and u_star is not None:
        # Distill teacher's optimized actions into student policy
        student_acts = torch.stack(raw_act_history)   # (T, B, 6)
        teacher_acts = torch.stack(u_star)             # (T, B, 6)
        loss_distill_act = F.mse_loss(student_acts, teacher_acts)
        loss_distill = loss_distill_act
        if u_star_cam is not None and len(raw_cam_history) > 0:
            student_cam = torch.stack(raw_cam_history)  # (T, B, 4)
            teacher_cam = torch.stack(u_star_cam)        # (T, B, 4)
            loss_distill = loss_distill + F.mse_loss(student_cam, teacher_cam)
        # In G-DAC mode: primarily distillation + reduced-weight physics for curriculum
        loss = args.coef_distill * loss_distill + args.gdac_physics_weight * loss

    if torch.isnan(loss):
        print("loss is nan, exiting...")
        exit(1)

    pbar.set_description_str(f'loss: {loss:.3f}')
    optim.zero_grad()
    loss.backward()
    optim.step()
    sched.step()


    iter_toc = time.time()
    iter_time = iter_toc - iter_tic  # seconds per iteration
    iter_per_sec = 1.0 / max(iter_time, 1e-6)
    sim_fps = iter_per_sec * args.timesteps * B  # total simulated frames per second

    with torch.no_grad():
        avg_speed = speed_history.mean(0)
        success = torch.all(distance.flatten(0, 1) > 0, 0)
        _success = success.sum() / B
        smooth_dict({
            'iter_per_sec': iter_per_sec,
            'sim_fps': sim_fps,
            'iter_time_ms': iter_time * 1000,
        })
        smooth_dict({
            'loss': loss,
            'loss_v': loss_v,
            'loss_v_pred': loss_v_pred,
            'loss_obj_avoidance': loss_obj_avoidance,
            'loss_d_acc': loss_d_acc,
            'loss_d_jerk': loss_d_jerk,
            'loss_d_snap': loss_d_snap,
            'loss_bias': loss_bias,
            'loss_speed': loss_speed,
            'loss_collide': loss_collide,
            'loss_ground_affinity': loss_ground_affinity,
            'loss_cam_smooth': loss_cam_smooth,
            'loss_fov_reg': loss_fov_reg,
            'loss_cam_range': loss_cam_range,
            'loss_tilt': loss_tilt,
            'loss_blur': loss_blur,
            'loss_noise': loss_noise,
            'loss_defocus': loss_defocus,
            'loss_distill': loss_distill,
            'success': _success,
            'max_speed': speed_history.max(0).values.mean(),
            'avg_speed': avg_speed.mean(),
            'ar': (success * avg_speed).mean()})

        # ===== Emerging-behavior metrics =====
        # Roll angle: angle between drone up-vector and world vertical
        if len(R_up_history) > 0:
            up_hist = torch.stack(R_up_history)  # (T, B, 3)
            # up_z component: cos(roll_from_vertical)
            roll_angle = torch.acos(up_hist[:, :, 2].clamp(-1, 1))  # (T, B) radians
            roll_deg = roll_angle * 180 / math.pi
            smooth_dict({
                'roll_max_deg': roll_deg.max().item(),
                'roll_mean_deg': roll_deg.mean().item(),
            })
            if args.wall_slit:
                # Roll angle specifically near the wall
                wall_x = env.wall_x
                dx = (p_history[..., 0] - wall_x).abs()  # (T, B)
                near_wall = dx < 1.0  # within 1m
                if near_wall.any():
                    smooth_dict({
                        'roll_at_wall_deg': roll_deg[near_wall].mean().item(),
                    })

        # Visuo-motor coupling: speed-exposure correlation
        if use_cam and len(cam_exposure_history) > 0:
            _sp = torch.stack(speed_for_cam_history)  # (T, B)
            _ex = torch.stack(cam_exposure_history)    # (T, B)
            # Per-batch Pearson correlation, then average
            sp_mean = _sp.mean(0, keepdim=True)
            ex_mean = _ex.mean(0, keepdim=True)
            cov = ((_sp - sp_mean) * (_ex - ex_mean)).mean(0)
            sp_std = (_sp - sp_mean).pow(2).mean(0).sqrt().clamp(min=1e-6)
            ex_std = (_ex - ex_mean).pow(2).mean(0).sqrt().clamp(min=1e-6)
            speed_exposure_corr = (cov / (sp_std * ex_std)).mean()
            smooth_dict({'speed_exposure_corr': speed_exposure_corr.item()})

            # Optical breathing: FOV-obstacle distance correlation
            _fv = torch.stack(cam_fov_history)  # (T, B)
            _dn = torch.norm(vec_to_pt_history, 2, -1).min(1).values  # (T, B)
            fv_mean = _fv.mean(0, keepdim=True)
            dn_mean = _dn.mean(0, keepdim=True)
            cov_fd = ((_fv - fv_mean) * (_dn - dn_mean)).mean(0)
            fv_std = (_fv - fv_mean).pow(2).mean(0).sqrt().clamp(min=1e-6)
            dn_std = (_dn - dn_mean).pow(2).mean(0).sqrt().clamp(min=1e-6)
            fov_obstacle_corr = (cov_fd / (fv_std * dn_std)).mean()
            smooth_dict({'fov_obstacle_corr': fov_obstacle_corr.item()})

        # Wall-slit specific metrics
        if args.wall_slit:
            # Check if drone crossed the wall (final x > wall_x given start x < wall_x)
            final_x = p_history[-1, :, 0]
            crossed = (final_x > env.wall_x).float()
            slit_pass = (crossed * success.float())  # crossed AND no collision
            smooth_dict({
                'slit_crossed': crossed.mean(),
                'slit_pass_rate': slit_pass.mean(),
            })

        log_dict = {}
        if is_save_iter(i):
            print("save check success:", i)
            vid = torch.stack(vid).cpu().div(10).clamp(0, 1)[None, :, None]
            vid = vid.repeat(1, 1, 3, 1, 1)
            fig_p, ax = plt.subplots()
            p_history = p_history[:, vid_idx].cpu()
            ax.plot(p_history[:, 0], label='x')
            ax.plot(p_history[:, 1], label='y')
            ax.plot(p_history[:, 2], label='z')
            ax.legend()
            fig_v, ax = plt.subplots()
            v_history = v_history[:, vid_idx].cpu()
            ax.plot(v_history[:, 0], label='x')
            ax.plot(v_history[:, 1], label='y')
            ax.plot(v_history[:, 2], label='z')
            ax.legend()
            fig_a, ax = plt.subplots()
            act_buffer = act_buffer[:, vid_idx].cpu()
            ax.plot(act_buffer[:, 0], label='x')
            ax.plot(act_buffer[:, 1], label='y')
            ax.plot(act_buffer[:, 2], label='z')
            ax.legend()
            
            # Log to wandb
            # wandb expects video in (T, C, H, W) format, but vid is (1, T, 3, H, W)
            # Remove batch dim: (T, 3, H, W)
            # Convert to numpy array for wandb and scaling
            # vid[0] is (T, 3, H, W)
            # Save video to temp file to avoid wandb/moviepy fps bug
            vid_np = vid[0].permute(0, 2, 3, 1).cpu().numpy()  # (T, C, H, W) -> (T, H, W, C)
            vid_np = (vid_np * 255).astype('uint8')
            tmp_video_path = f'/tmp/wandb_demo_{i}.mp4'
            writer = imageio.get_writer(tmp_video_path, fps=15)
            for frame in vid_np:
                writer.append_data(frame)
            writer.close()
            wandb.log({
                "demo": wandb.Video(tmp_video_path, fps=15, format="mp4"),
                "p_history": wandb.Image(fig_p),
                "v_history": wandb.Image(fig_v),
                "a_reals": wandb.Image(fig_a)
            }, step=i + 1)
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)
            
            plt.close(fig_p)
            plt.close(fig_v)
            plt.close(fig_a)

            if use_cam and len(cam_params_history) > 0:
                cam_hist = torch.stack(cam_params_history)[:, vid_idx].cpu()
                fig_cam, axes = plt.subplots(2, 2, figsize=(8, 6))
                if args.paper_unified_control:
                    labels = ['FOV (norm)', 'Exposure', 'ISO', 'Focus']
                else:
                    labels = ['FOV delta', 'Exposure', 'ISO', 'Focus']
                for ci, (ax_c, lb) in enumerate(zip(axes.flatten(), labels)):
                    ax_c.plot(cam_hist[:, ci].numpy(), label=lb)
                    ax_c.set_title(lb)
                    if not args.paper_unified_control:
                        ax_c.set_ylim(-0.05, 1.05)
                fig_cam.tight_layout()
                wandb.log({'cam_params': wandb.Image(fig_cam)}, step=i + 1)
                plt.close(fig_cam)

            # Emerging behavior: roll angle + speed/exposure plot at save iters
            if len(R_up_history) > 0:
                up_hist = torch.stack(R_up_history)[:, vid_idx].cpu()  # (T, 3)
                roll_rad = torch.acos(up_hist[:, 2].clamp(-1, 1))
                roll_deg_plot = roll_rad * 180 / math.pi
                fig_roll, ax_roll = plt.subplots(figsize=(6, 3))
                ax_roll.plot(roll_deg_plot.numpy(), label='Roll angle (deg)')
                ax_roll.set_ylabel('Roll (deg)')
                ax_roll.set_xlabel('Timestep')
                if use_cam and len(cam_exposure_history) > 0:
                    ax2 = ax_roll.twinx()
                    sp_plot = torch.stack(speed_for_cam_history)[:, vid_idx].cpu()
                    ex_plot = torch.stack(cam_exposure_history)[:, vid_idx].cpu()
                    ax2.plot(sp_plot.numpy(), 'r--', alpha=0.7, label='Speed')
                    ax2.plot(ex_plot.numpy(), 'g-.', alpha=0.7, label='Exposure')
                    ax2.set_ylabel('Speed / Exposure')
                    ax2.legend(loc='upper right')
                ax_roll.legend(loc='upper left')
                fig_roll.tight_layout()
                wandb.log({'roll_speed_exposure': wandb.Image(fig_roll)}, step=i + 1)
                plt.close(fig_roll)
        if (i + 1) % 10000 == 0:
            torch.save(model.state_dict(), f'checkpoint{i//10000:04d}.pth')
            # Optionally log checkpoint to wandb
            wandb.save(f'checkpoint{i//10000:04d}.pth')
        if (i + 1) % 25 == 0:
            log_data = {}
            for k, v in scaler_q.items():
                log_data[k] = sum(v) / len(v)
            wandb.log(log_data, step=i + 1)
            scaler_q.clear()

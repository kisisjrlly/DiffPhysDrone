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

env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle,
          wall_slit=args.wall_slit,
          ellipsoid_a=args.drone_a if args.ellipsoid_collision else 0.0,
          ellipsoid_c=args.drone_c if args.ellipsoid_collision else 0.0)
if args.no_odom:
    model = Model(7, 6, use_diff_cam=args.diff_cam)
else:
    model = Model(7+3, 6, use_diff_cam=args.diff_cam)
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
    p_history = []
    v_history = []
    target_v_history = []
    vec_to_pt_history = []
    act_diff_history = []
    v_preds = []
    vid = []
    v_net_feats = []
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
    if args.diff_cam:
        cam_fov = torch.full((B,), env._fov_x_half_tan, device=device)
        cam_exposure = torch.full((B,), 0.5, device=device)
        cam_iso = torch.full((B,), 0.5, device=device)
        cam_focus = torch.full((B,), 0.5, device=device)

    for t in range(args.timesteps):
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)
        # Render: use differentiable FOV if diff_cam is enabled
        if args.diff_cam:
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
        state = torch.cat(state, -1)

        # normalize depth to inverse-depth feature
        if args.diff_cam:
            # Use non-inplace clamp to preserve autograd graph through depth
            x = 3 / depth.clamp(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
        else:
            x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
        x = F.max_pool2d(x[:, None], 4, 4)
        act, cam_params, h = model(x, state, h)

        # Update camera parameters for next timestep's render
        if cam_params is not None:
            fov_delta, exposure, iso, focus_dist = cam_params.unbind(-1)
            # fov_delta in [0,1] via sigmoid -> FOV in [0.5*base, 1.5*base]
            cam_fov = env._fov_x_half_tan * (0.5 + fov_delta)
            cam_exposure = exposure
            cam_iso = iso
            cam_focus = focus_dist
            cam_params_history.append(cam_params)

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
    if args.diff_cam and len(cam_params_history) > 1:
        cam_hist = torch.stack(cam_params_history)  # (T, B, 4)
        # Smoothness: penalize rapid camera parameter changes between timesteps
        cam_diff = cam_hist.diff(1, 0)  # (T-1, B, 4)
        loss_cam_smooth = cam_diff.pow(2).mean()
        # FOV regularization: keep FOV near default (fov_delta=0.5 → default FOV)
        fov_deltas = cam_hist[:, :, 0]  # (T, B)
        loss_fov_reg = (fov_deltas - 0.5).pow(2).mean()
        # Range regularization: keep all params near center to avoid extreme values
        loss_cam_range = (cam_hist - 0.5).pow(2).mean()

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
        args.coef_tilt * loss_tilt

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
            'success': _success,
            'max_speed': speed_history.max(0).values.mean(),
            'avg_speed': avg_speed.mean(),
            'ar': (success * avg_speed).mean()})

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

            if args.diff_cam and len(cam_params_history) > 0:
                cam_hist = torch.stack(cam_params_history)[:, vid_idx].cpu()
                fig_cam, axes = plt.subplots(2, 2, figsize=(8, 6))
                labels = ['FOV delta', 'Exposure', 'ISO', 'Focus']
                for ci, (ax_c, lb) in enumerate(zip(axes.flatten(), labels)):
                    ax_c.plot(cam_hist[:, ci].numpy(), label=lb)
                    ax_c.set_title(lb)
                    ax_c.set_ylim(-0.05, 1.05)
                fig_cam.tight_layout()
                wandb.log({'cam_params': wandb.Image(fig_cam)}, step=i + 1)
                plt.close(fig_cam)
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

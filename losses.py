"""Minimal losses for the active-sensing simulation branch."""
import torch
import torch.nn.functional as F


def velocity_tracking_loss(v_hist: torch.Tensor, tv_hist: torch.Tensor, win: int = 12):
    if v_hist.shape[0] <= win:
        return torch.zeros((), device=v_hist.device, dtype=v_hist.dtype)
    v_cum = v_hist.cumsum(0)
    v_avg = (v_cum[win:] - v_cum[:-win]) / win
    tv_ref = tv_hist[win:]
    m = min(v_avg.shape[0], tv_ref.shape[0])
    if m <= 0:
        return torch.zeros((), device=v_hist.device, dtype=v_hist.dtype)
    delta_v = torch.norm(v_avg[:m] - tv_ref[:m], 2, -1)
    return F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))


def barrier(x: torch.Tensor, v_to_pt):
    return (v_to_pt * (1 - x).relu().clamp(max=5.0).pow(2)).mean()


def compute_physics_losses(v_chunk, tv_chunk, act_chunk, vec_chunk, p_chunk,
                           margin, prev_act_tail, win=12):
    loss_v = velocity_tracking_loss(v_chunk, tv_chunk, win=win)
    act_for_smooth = torch.cat([prev_act_tail[None], act_chunk], 0)
    jerk = act_for_smooth.diff(1, 0).mul(15)
    loss_d_acc = act_chunk.pow(2).sum(-1).mean()
    loss_d_jerk = jerk.pow(2).sum(-1).mean()

    dist = torch.norm(vec_chunk + 1e-6, 2, -1) - margin
    with torch.no_grad():
        v_to = (-torch.diff(dist, 1, 1) * 135).clamp_min(1)
    loss_avoid = barrier(dist[:, 1:], v_to)
    loss_collide = F.softplus(dist[:, 1:].clamp(min=-3.0).mul(-32)).mul(v_to).mean()
    return {
        'loss_v': loss_v,
        'loss_d_acc': loss_d_acc,
        'loss_d_jerk': loss_d_jerk,
        'loss_avoid': loss_avoid,
        'loss_collide': loss_collide,
    }


def _infer_loss_device(*items):
    for item in items:
        if isinstance(item, torch.Tensor):
            return item.device
    return torch.device('cpu')


def compute_camera_losses(cam_hist, power_seq, exposure_seq, gain_seq, speed_seq,
                          fill_rate_seq=None, min_fill_rate=0.0,
                          camera_semantics=None, power_baseline: float = 0.5,
                          cam_initial=None):
    _ = exposure_seq, gain_seq, speed_seq, fill_rate_seq, min_fill_rate, camera_semantics
    device = _infer_loss_device(cam_hist, power_seq, cam_initial)
    result = {
        'loss_cam_smooth': torch.zeros((), device=device),
        'loss_diff_depth_power': torch.zeros((), device=device),
        'loss_diff_depth_blur': torch.zeros((), device=device),
        'loss_diff_depth_noise': torch.zeros((), device=device),
        'loss_diff_depth_fill': torch.zeros((), device=device),
    }
    if cam_hist is not None:
        cam_for_smooth = cam_hist
        if cam_initial is not None:
            init = cam_initial.to(device=cam_hist.device, dtype=cam_hist.dtype)
            if init.ndim == cam_hist.ndim - 1:
                init = init.unsqueeze(0)
            cam_for_smooth = torch.cat([init.detach(), cam_hist], dim=0)
        if cam_for_smooth.shape[0] > 1:
            result['loss_cam_smooth'] = cam_for_smooth.diff(1, 0).pow(2).mean()
    if power_seq is not None:
        result['loss_diff_depth_power'] = F.relu(power_seq - float(power_baseline)).pow(2).mean()
    return result


def aggregate_loss(physics_losses, camera_losses, args):
    loss = (
        args.coef_v * physics_losses['loss_v']
        + args.coef_obj_avoidance * physics_losses['loss_avoid']
        + args.coef_d_acc * physics_losses['loss_d_acc']
        + args.coef_d_jerk * physics_losses['loss_d_jerk']
        + args.coef_collide * physics_losses['loss_collide']
        + args.coef_cam_smooth * camera_losses['loss_cam_smooth']
        + args.coef_diff_depth_power * camera_losses['loss_diff_depth_power']
    )
    all_losses = {
        'loss_v': physics_losses['loss_v'],
        'loss_d_acc': physics_losses['loss_d_acc'],
        'loss_d_jerk': physics_losses['loss_d_jerk'],
        'loss_obj_avoidance': physics_losses['loss_avoid'],
        'loss_collide': physics_losses['loss_collide'],
        'loss_cam_smooth': camera_losses['loss_cam_smooth'],
        'loss_diff_depth_power': camera_losses['loss_diff_depth_power'],
        'loss_diff_depth_blur': camera_losses['loss_diff_depth_blur'],
        'loss_diff_depth_noise': camera_losses['loss_diff_depth_noise'],
        'loss_diff_depth_fill': camera_losses['loss_diff_depth_fill'],
    }
    return loss, all_losses

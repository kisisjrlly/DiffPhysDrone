"""Losses for task-driven grayscale active sensing."""

import torch
import torch.nn.functional as F


def velocity_tracking_loss(v_hist, tv_hist, win=12):
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


def barrier(x, v_to_pt):
    return (v_to_pt * (1 - x).relu().clamp(max=5.0).pow(2)).mean()


def compute_physics_losses(
    v_chunk,
    tv_chunk,
    act_chunk,
    vec_chunk,
    p_chunk,
    margin,
    prev_act_tail,
    win=12,
):
    _ = p_chunk
    loss_v = velocity_tracking_loss(v_chunk, tv_chunk, win=win)
    act_for_smooth = torch.cat([prev_act_tail[None], act_chunk], 0)
    jerk = act_for_smooth.diff(1, 0).mul(15)
    loss_d_acc = act_chunk.pow(2).sum(-1).mean()
    loss_d_jerk = jerk.pow(2).sum(-1).mean()

    dist = torch.norm(vec_chunk + 1e-6, 2, -1) - margin
    with torch.no_grad():
        v_to = (-torch.diff(dist, 1, 1) * 135).clamp_min(1)
    dist_next = dist[:, 1:]
    loss_avoid = barrier(dist_next, v_to)
    loss_collide = F.softplus(dist_next.clamp(min=-3.0).mul(-32)).mul(v_to).mean()
    return {
        "loss_v": loss_v,
        "loss_d_acc": loss_d_acc,
        "loss_d_jerk": loss_d_jerk,
        "loss_avoid": loss_avoid,
        "loss_collide": loss_collide,
    }


def compute_camera_losses(cam_hist, cam_initial=None):
    if cam_hist is None:
        device = cam_initial.device if isinstance(cam_initial, torch.Tensor) else torch.device("cpu")
        return {"loss_cam_smooth": torch.zeros((), device=device)}
    seq = cam_hist
    if cam_initial is not None:
        init = cam_initial.to(device=seq.device, dtype=seq.dtype)
        if init.ndim == seq.ndim - 1:
            init = init.unsqueeze(0)
        seq = torch.cat([init.detach(), seq], dim=0)
    smooth = (
        seq.diff(1, 0).pow(2).mean()
        if seq.shape[0] > 1
        else torch.zeros((), device=seq.device, dtype=seq.dtype)
    )
    return {"loss_cam_smooth": smooth}


def aggregate_loss(physics_losses, camera_losses, args):
    loss = (
        args.coef_v * physics_losses["loss_v"]
        + args.coef_obj_avoidance * physics_losses["loss_avoid"]
        + args.coef_d_acc * physics_losses["loss_d_acc"]
        + args.coef_d_jerk * physics_losses["loss_d_jerk"]
        + args.coef_collide * physics_losses["loss_collide"]
        + args.coef_cam_smooth * camera_losses["loss_cam_smooth"]
    )
    terms = {
        "loss_v": physics_losses["loss_v"],
        "loss_d_acc": physics_losses["loss_d_acc"],
        "loss_d_jerk": physics_losses["loss_d_jerk"],
        "loss_obj_avoidance": physics_losses["loss_avoid"],
        "loss_collide": physics_losses["loss_collide"],
        "loss_cam_smooth": camera_losses["loss_cam_smooth"],
    }
    return loss, terms

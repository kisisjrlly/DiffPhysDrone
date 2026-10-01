"""Losses for task-driven grayscale active sensing."""

import torch
import torch.nn.functional as F


def velocity_tracking_loss(v_hist, tv_hist, win=12, valid_mask=None):
    if v_hist.shape[0] <= win:
        return torch.zeros((), device=v_hist.device, dtype=v_hist.dtype)
    v_cum = v_hist.cumsum(0)
    v_avg = (v_cum[win:] - v_cum[:-win]) / win
    tv_ref = tv_hist[win:]
    m = min(v_avg.shape[0], tv_ref.shape[0])
    if m <= 0:
        return torch.zeros((), device=v_hist.device, dtype=v_hist.dtype)
    delta_v = torch.norm(v_avg[:m] - tv_ref[:m], 2, -1)
    if valid_mask is None:
        return F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))
    valid = valid_mask[win:win + m].clone()
    valid[:-1] &= valid_mask[:m - 1]
    weights = valid.to(delta_v.dtype)
    denom = weights.sum().clamp_min(1.0)
    return F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v), reduction="none").mul(weights).sum() / denom


def barrier(x, v_to_pt, valid_mask=None):
    value = v_to_pt * (1 - x).relu().clamp(max=5.0).pow(2)
    if valid_mask is None:
        return value.mean()
    weights = valid_mask.to(value.dtype)
    return value.mul(weights).sum() / weights.sum().clamp_min(1.0)


def compute_physics_losses(
    v_chunk,
    tv_chunk,
    act_chunk,
    vec_chunk,
    p_chunk,
    margin,
    prev_act_tail,
    win=12,
    valid_mask=None,
    post_vec_chunk=None,
):
    _ = p_chunk
    loss_v = velocity_tracking_loss(v_chunk, tv_chunk, win=win, valid_mask=valid_mask)
    act_for_smooth = torch.cat([prev_act_tail[None], act_chunk], 0)
    jerk = act_for_smooth.diff(1, 0).mul(15)
    action_weights = valid_mask.to(act_chunk.dtype) if valid_mask is not None else None
    if action_weights is None:
        loss_d_acc = act_chunk.pow(2).sum(-1).mean()
        loss_d_jerk = jerk.pow(2).sum(-1).mean()
    else:
        denom = action_weights.sum().clamp_min(1.0)
        loss_d_acc = act_chunk.pow(2).sum(-1).mul(action_weights).sum() / denom
        jerk_valid = torch.cat([valid_mask[:1], valid_mask[1:] & valid_mask[:-1]], 0)
        jerk_weights = jerk_valid.to(jerk.dtype)
        loss_d_jerk = jerk.pow(2).sum(-1).mul(jerk_weights).sum() / jerk_weights.sum().clamp_min(1.0)

    if post_vec_chunk is None:
        pre_vec = vec_chunk[:-1]
        post_vec = vec_chunk[1:]
        transition_mask = None if valid_mask is None else valid_mask[:-1]
    else:
        pre_vec = vec_chunk
        post_vec = post_vec_chunk
        transition_mask = valid_mask
    dist_pre = torch.norm(pre_vec + 1e-6, 2, -1) - margin
    dist_post = torch.norm(post_vec + 1e-6, 2, -1) - margin
    with torch.no_grad():
        v_to = ((dist_pre - dist_post) * 135).clamp_min(1)
    loss_avoid = barrier(dist_post, v_to, valid_mask=transition_mask)
    collide = F.softplus(dist_post.clamp(min=-3.0).mul(-32)).mul(v_to)
    if valid_mask is not None:
        weights = transition_mask.to(collide.dtype)
        loss_collide = collide.mul(weights).sum() / weights.sum().clamp_min(1.0)
    else:
        loss_collide = collide.mean()
    return {
        "loss_v": loss_v,
        "loss_d_acc": loss_d_acc,
        "loss_d_jerk": loss_d_jerk,
        "loss_avoid": loss_avoid,
        "loss_collide": loss_collide,
    }


def compute_camera_losses(cam_hist, cam_initial=None, valid_mask=None):
    if cam_hist is None:
        device = cam_initial.device if isinstance(cam_initial, torch.Tensor) else torch.device("cpu")
        return {"loss_cam_smooth": torch.zeros((), device=device)}
    seq = cam_hist
    if cam_initial is not None:
        init = cam_initial.to(device=seq.device, dtype=seq.dtype)
        if init.ndim == seq.ndim - 1:
            init = init.unsqueeze(0)
        seq = torch.cat([init.detach(), seq], dim=0)
    values = seq.diff(1, 0).pow(2).mean(-1)
    if valid_mask is not None:
        weights = valid_mask.to(values.dtype)
        values = values.mul(weights)
        smooth = values.sum() / weights.sum().clamp_min(1.0)
    else:
        smooth = values.mean() if values.numel() else torch.zeros((), device=seq.device, dtype=seq.dtype)
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

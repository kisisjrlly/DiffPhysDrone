"""
统一的损失函数模块。
消除 Teacher 阶段、Student TBPTT chunk、Student 完整 BPTT 之间的重复代码。
"""
from typing import Optional

import torch
import torch.nn.functional as F

from camera_semantics import CameraSemantics


def velocity_tracking_loss(v_hist: torch.Tensor, tv_hist: torch.Tensor, win: int = 30):
    """速度主任务损失（平滑版）。

    先对真实速度做时间窗口平均，再和目标速度比较。
    """
    if v_hist.shape[0] <= win:
        return torch.zeros((), device=v_hist.device, dtype=v_hist.dtype)
    v_cum = v_hist.cumsum(0)
    v_avg = (v_cum[win:] - v_cum[:-win]) / win
    tv_ref = tv_hist[1:1 - win]
    m = min(v_avg.shape[0], tv_ref.shape[0])
    if m <= 0:
        return torch.zeros((), device=v_hist.device, dtype=v_hist.dtype)
    delta_v = torch.norm(v_avg[:m] - tv_ref[:m], 2, -1)
    return F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))


def barrier(x: torch.Tensor, v_to_pt):
    """障碍物避让屏障函数：当距离小于安全边距时产生巨大惩罚梯度。"""
    return (v_to_pt * (1 - x).relu().pow(2)).mean()


def compute_physics_losses(v_chunk, tv_chunk, act_chunk, vec_chunk, p_chunk,
                           margin, prev_act_tail,
                           v_roll=None, tv_roll=None, win=30):
    """计算所有与 物理/控制 相关的基础损失项。

    被 Teacher chunk、Student TBPTT chunk、Student 完整 BPTT 共用。

    Returns:
        dict 包含: loss_v, loss_d_acc, loss_d_jerk, loss_avoid, loss_collide, loss_ground
    """
    # 速度跟踪损失
    if v_roll is not None and len(v_roll) > 0:
        v_for_loss = torch.cat(v_roll + [v_chunk], 0)
    else:
        v_for_loss = v_chunk
    if tv_roll is not None and len(tv_roll) > 0:
        tv_for_loss = torch.cat(tv_roll + [tv_chunk], 0)
    else:
        tv_for_loss = tv_chunk
    loss_v = velocity_tracking_loss(v_for_loss, tv_for_loss, win=win)

    # 动作平滑度损失
    act_for_smooth = torch.cat([prev_act_tail[None], act_chunk], 0)
    jerk = act_for_smooth.diff(1, 0).mul(15)
    loss_d_acc = act_chunk.pow(2).sum(-1).mean()
    loss_d_jerk = jerk.pow(2).sum(-1).mean()

    # 避障与碰撞损失
    dist = torch.norm(vec_chunk, 2, -1) - margin
    with torch.no_grad():
        v_to = (-torch.diff(dist, 1, 1) * 135).clamp_min(1)
    loss_avoid = barrier(dist[:, 1:], v_to)
    loss_collide = F.softplus(dist[:, 1:].mul(-32)).mul(v_to).mean()

    # 地面亲和力损失
    loss_ground = p_chunk[..., 2].relu().pow(2).mean()

    return {
        'loss_v': loss_v,
        'loss_d_acc': loss_d_acc,
        'loss_d_jerk': loss_d_jerk,
        'loss_avoid': loss_avoid,
        'loss_collide': loss_collide,
        'loss_ground': loss_ground,
        # 用于跨 chunk 窗口缓冲
        'v_for_loss': v_for_loss,
        'tv_for_loss': tv_for_loss,
    }


def compute_camera_losses(cam_hist, cam_fov_seq, cam_exp_seq, cam_iso_seq, speed_seq,
                          use_active_depth, enable_camera_quality_loss,
                          cam_sem: Optional[CameraSemantics] = None):
    """计算所有与 相机控制/光学 相关的损失项。

    Returns:
        dict 包含: loss_cam_smooth, loss_fov_reg, loss_cam_range,
                   loss_blur, loss_noise, loss_active_depth_power, loss_active_depth_blur
    """
    device = speed_seq.device if isinstance(speed_seq, torch.Tensor) else cam_exp_seq.device
    result = {
        'loss_cam_smooth': torch.zeros((), device=device),
        'loss_fov_reg': torch.zeros((), device=device),
        'loss_cam_range': torch.zeros((), device=device),
        'loss_blur': torch.zeros((), device=device),
        'loss_noise': torch.zeros((), device=device),
        'loss_active_depth_power': torch.zeros((), device=device),
        'loss_active_depth_blur': torch.zeros((), device=device),
    }

    # 相机平滑度与正则化
    if cam_hist is not None and cam_hist.shape[0] > 1:
        cam_diff = cam_hist.diff(1, 0)
        result['loss_cam_smooth'] = cam_diff.pow(2).mean()
        result['loss_fov_reg'] = (cam_hist[:, :, 0] - 0.5).pow(2).mean()
        result['loss_cam_range'] = (cam_hist - 0.5).pow(2).mean()

    # 光学损失
    if cam_exp_seq is not None:
        if use_active_depth:
            result['loss_active_depth_power'] = cam_fov_seq.pow(2).mean()
            result['loss_active_depth_blur'] = (speed_seq * cam_exp_seq).mean()
        elif enable_camera_quality_loss:
            sem = cam_sem if cam_sem is not None else CameraSemantics()
            exp_phys = torch.as_tensor(
                sem.exposure_to_time(cam_exp_seq),
                device=cam_exp_seq.device,
                dtype=cam_exp_seq.dtype,
            )
            eff_f = 1.0 / cam_fov_seq.clamp(min=0.1)
            result['loss_blur'] = (speed_seq.pow(2) * exp_phys.pow(2) * eff_f.pow(2)).mean()
            iso_gain = torch.as_tensor(
                sem.iso_to_gain(cam_iso_seq),
                device=cam_iso_seq.device,
                dtype=cam_iso_seq.dtype,
            )
            noise_sigma = sem.shot_noise_base * iso_gain / exp_phys.clamp_min(1e-3)
            result['loss_noise'] = noise_sigma.pow(2).mean()

    return result


def compute_distill_loss(raw_act_history, raw_intent_history, raw_cam_history,
                         u_star, y_star, u_star_cam,
                         start_idx=None, end_idx=None):
    """计算蒸馏损失。

    如果提供了 start_idx/end_idx，则只使用对应的 slice（用于 TBPTT chunk）。
    如果不提供，则使用全部数据（用于完整 BPTT）。
    """
    device = None
    loss = torch.tensor(0.0)

    if y_star is not None and len(raw_intent_history) > 0:
        if start_idx is not None:
            student = torch.stack(raw_intent_history[start_idx:end_idx])
            teacher = torch.stack(y_star[start_idx:end_idx])
        else:
            student = torch.stack(raw_intent_history)
            teacher = torch.stack(y_star)
        device = student.device
        loss = loss.to(device)
        loss = loss + F.mse_loss(student, teacher)
    elif u_star is not None and len(raw_act_history) > 0:
        if start_idx is not None:
            student = torch.stack(raw_act_history[start_idx:end_idx])
            teacher = torch.stack(u_star[start_idx:end_idx])
        else:
            student = torch.stack(raw_act_history)
            teacher = torch.stack(u_star)
        device = student.device
        loss = loss.to(device)
        loss = loss + F.mse_loss(student, teacher)

    if u_star_cam is not None and len(raw_cam_history) > 0:
        if start_idx is not None:
            student_cam = torch.stack(raw_cam_history[start_idx:end_idx])
            teacher_cam = torch.stack(u_star_cam[start_idx:end_idx])
        else:
            student_cam = torch.stack(raw_cam_history)
            teacher_cam = torch.stack(u_star_cam)
        if device is None:
            device = student_cam.device
        loss = loss.to(device)
        loss = loss + F.mse_loss(student_cam, teacher_cam)

    if device is not None:
        loss = loss.to(device)
    return loss


def aggregate_loss(physics_losses, camera_losses, args,
                   loss_distill=None, distill_coef_iter=None,
                   loss_v_pred=None, loss_tilt=None,
                   chunk_count=None):
    """按系数加权汇总所有损失项，返回总损失与各分量 dict。

    Args:
        physics_losses: compute_physics_losses() 的返回值
        camera_losses: compute_camera_losses() 的返回值
        args: 命令行参数
        loss_distill: 蒸馏损失 (optional)
        distill_coef_iter: 当前迭代的蒸馏系数 (optional)
        loss_v_pred: 速度预测辅助损失 (optional, Teacher 阶段无此项)
        loss_tilt: 侧倾损失 (optional)
        chunk_count: 如果非 None，则按 chunk 数归一化 (Teacher TBPTT 需要)
    """
    device = physics_losses['loss_v'].device
    zero = torch.zeros((), device=device)
    if loss_v_pred is None:
        loss_v_pred = zero
    if loss_tilt is None:
        loss_tilt = zero
    if loss_distill is None:
        loss_distill = zero

    loss = (
        args.coef_v * physics_losses['loss_v']
        + args.coef_obj_avoidance * physics_losses['loss_avoid']
        + args.coef_d_acc * physics_losses['loss_d_acc']
        + args.coef_d_jerk * physics_losses['loss_d_jerk']
        + args.coef_collide * physics_losses['loss_collide']
        + args.coef_ground_affinity * physics_losses['loss_ground']
        + args.coef_v_pred * loss_v_pred
        + args.coef_cam_smooth * camera_losses['loss_cam_smooth']
        + args.coef_fov_reg * camera_losses['loss_fov_reg']
        + args.coef_cam_range * camera_losses['loss_cam_range']
        + args.coef_tilt * loss_tilt
        + args.coef_blur * camera_losses['loss_blur']
        + args.coef_noise * camera_losses['loss_noise']
        + args.coef_active_depth_power * camera_losses['loss_active_depth_power']
        + args.coef_active_depth_blur * camera_losses['loss_active_depth_blur']
    )

    if args.enable_teacher_student_training and distill_coef_iter is not None:
        loss = distill_coef_iter * loss_distill + args.student_physics_coef * loss

    if chunk_count is not None:
        loss = loss / chunk_count

    # 汇总所有分量（用于日志）
    all_losses = {
        'loss_v': physics_losses['loss_v'],
        'loss_d_acc': physics_losses['loss_d_acc'],
        'loss_d_jerk': physics_losses['loss_d_jerk'],
        'loss_obj_avoidance': physics_losses['loss_avoid'],
        'loss_collide': physics_losses['loss_collide'],
        'loss_ground_affinity': physics_losses['loss_ground'],
        'loss_v_pred': loss_v_pred,
        'loss_cam_smooth': camera_losses['loss_cam_smooth'],
        'loss_fov_reg': camera_losses['loss_fov_reg'],
        'loss_cam_range': camera_losses['loss_cam_range'],
        'loss_tilt': loss_tilt,
        'loss_blur': camera_losses['loss_blur'],
        'loss_noise': camera_losses['loss_noise'],
        'loss_active_depth_power': camera_losses['loss_active_depth_power'],
        'loss_active_depth_blur': camera_losses['loss_active_depth_blur'],
        'loss_distill': loss_distill,
    }

    return loss, all_losses

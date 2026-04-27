"""
统一的损失函数模块。
消除 Teacher 阶段、Student TBPTT chunk、Student 完整 BPTT 之间的重复代码。
"""
import torch
import torch.nn.functional as F

from rollout_ops import diff_depth_exposure_to_time


def velocity_tracking_loss(v_hist: torch.Tensor, tv_hist: torch.Tensor, win: int = 30):
    """速度主任务损失（平滑版）。

    先对真实速度做时间窗口平均，再和目标速度比较。
    v_avg[i] = mean(v_hist[i:i+win])，对应目标速度 tv_hist[i+win]。
    """
    if v_hist.shape[0] <= win:
        return torch.zeros((), device=v_hist.device, dtype=v_hist.dtype)
    v_cum = v_hist.cumsum(0)
    v_avg = (v_cum[win:] - v_cum[:-win]) / win   # shape: [T-win, B, 3]
    tv_ref = tv_hist[win:]                         # 对齐：v_avg[i] 对应 tv_hist[i+win]
    m = min(v_avg.shape[0], tv_ref.shape[0])
    if m <= 0:
        return torch.zeros((), device=v_hist.device, dtype=v_hist.dtype)
    delta_v = torch.norm(v_avg[:m] - tv_ref[:m], 2, -1)
    return F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))


def barrier(x: torch.Tensor, v_to_pt):
    """障碍物避让屏障函数：当距离小于安全边距时产生巨大惩罚梯度。
    clamp(max=5.0) 防止深度穿透时 (1-x) 过大导致梯度爆炸。"""
    return (v_to_pt * (1 - x).relu().clamp(max=5.0).pow(2)).mean()


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
    # 使用 clamp(min=1e-6) 防止零向量导致 torch.norm 反向传播产生 NaN (0/0)
    # clamp(min=-3.0) 限制穿透深度对 softplus 输入的影响，防止 softplus(96+) 导致梯度爆炸
    dist = torch.norm(vec_chunk + 1e-6, 2, -1) - margin
    with torch.no_grad():
        v_to = (-torch.diff(dist, 1, 1) * 135).clamp_min(1)
    loss_avoid = barrier(dist[:, 1:], v_to)
    loss_collide = F.softplus(dist[:, 1:].clamp(min=-3.0).mul(-32)).mul(v_to).mean()

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


def _infer_loss_device(*items):
    for item in items:
        if isinstance(item, torch.Tensor):
            return item.device
    return torch.device('cpu')


def compute_camera_losses(cam_hist, power_seq, exposure_seq, gain_seq, speed_seq,
                          fill_rate_seq=None, min_fill_rate=0.18,
                          camera_semantics=None,
                          power_baseline: float = 0.55,
                          cam_initial=None):
    """计算 diff_depth-only 分支的相机控制损失。"""
    device = _infer_loss_device(
        cam_hist, power_seq, exposure_seq, gain_seq, speed_seq, fill_rate_seq, cam_initial)
    power_baseline = float(power_baseline)
    result = {
        'loss_cam_smooth': torch.zeros((), device=device),
        'loss_diff_depth_power': torch.zeros((), device=device),
        'loss_diff_depth_blur': torch.zeros((), device=device),
        'loss_diff_depth_noise': torch.zeros((), device=device),
        'loss_diff_depth_fill': torch.zeros((), device=device),
    }

    # 相机平滑度与正则化。把 rollout/chunk 起始相机状态拼进去，
    # 约束“初始状态 -> 第一个网络输出”的跳变。
    if cam_hist is not None:
        cam_for_smooth = cam_hist
        if cam_initial is not None:
            init = cam_initial.to(device=cam_hist.device, dtype=cam_hist.dtype)
            if init.ndim == cam_hist.ndim - 1:
                init = init.unsqueeze(0)
            if init.shape[1:] != cam_hist.shape[1:]:
                raise ValueError(
                    f"cam_initial shape {tuple(init.shape)} incompatible with cam_hist {tuple(cam_hist.shape)}"
                )
            cam_for_smooth = torch.cat([init.detach(), cam_hist], dim=0)
        if cam_for_smooth.shape[0] > 1:
            cam_diff = cam_for_smooth.diff(1, 0)
            result['loss_cam_smooth'] = cam_diff.pow(2).mean()

    # diff_depth 光学损失
    if exposure_seq is not None:
        exp_phys = diff_depth_exposure_to_time(
            exposure_seq,
            camera_semantics=camera_semantics,
        )
        # 单一 power 语义：baseline 是低功率常态，只有超过 baseline 的部分付能耗成本。
        result['loss_diff_depth_power'] = F.relu(power_seq - power_baseline).pow(2).mean()
        result['loss_diff_depth_blur'] = (speed_seq * exp_phys).pow(2).mean()
        result['loss_diff_depth_noise'] = gain_seq.pow(2).mean()
    if fill_rate_seq is not None:
        fill_gap = F.relu(float(min_fill_rate) - fill_rate_seq)
        result['loss_diff_depth_fill'] = fill_gap.pow(2).mean()

    return result


def compute_distill_loss(raw_act_history, raw_intent_history, raw_cam_history,
                         u_star, y_star, u_star_cam,
                         start_idx=None, end_idx=None, device=None):
    """计算蒸馏损失。

    如果提供了 start_idx/end_idx，则只使用对应的 slice（用于 TBPTT chunk）。
    如果不提供，则使用全部数据（用于完整 BPTT）。
    """
    loss = torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
    target_device = device

    if y_star is not None and len(raw_intent_history) > 0:
        if start_idx is not None:
            student = torch.stack(raw_intent_history[start_idx:end_idx])
            teacher = torch.stack(y_star[start_idx:end_idx])
        else:
            student = torch.stack(raw_intent_history)
            teacher = torch.stack(y_star)
        target_device = student.device
        loss = loss.to(target_device)
        loss = loss + F.mse_loss(student, teacher)
    elif u_star is not None and len(raw_act_history) > 0:
        if start_idx is not None:
            student = torch.stack(raw_act_history[start_idx:end_idx])
            teacher = torch.stack(u_star[start_idx:end_idx])
        else:
            student = torch.stack(raw_act_history)
            teacher = torch.stack(u_star)
        target_device = student.device
        loss = loss.to(target_device)
        loss = loss + F.mse_loss(student, teacher)

    if u_star_cam is not None and len(raw_cam_history) > 0:
        if start_idx is not None:
            student_cam = torch.stack(raw_cam_history[start_idx:end_idx])
            teacher_cam = torch.stack(u_star_cam[start_idx:end_idx])
        else:
            student_cam = torch.stack(raw_cam_history)
            teacher_cam = torch.stack(u_star_cam)
        if target_device is None:
            target_device = student_cam.device
        loss = loss.to(target_device)
        loss = loss + F.mse_loss(student_cam, teacher_cam)

    if target_device is not None:
        loss = loss.to(target_device)
    return loss


def aggregate_loss(physics_losses, camera_losses, args,
                   loss_distill=None, distill_coef_iter=None,
                   chunk_count=None):
    """按系数加权汇总所有损失项，返回总损失与各分量 dict。

    Args:
        physics_losses: compute_physics_losses() 的返回值
        camera_losses: compute_camera_losses() 的返回值
        args: 命令行参数
        loss_distill: 蒸馏损失 (optional)
        distill_coef_iter: 当前迭代的蒸馏系数 (optional)
        chunk_count: 如果非 None，则按 chunk 数归一化 (Teacher TBPTT 需要)
    """
    device = physics_losses['loss_v'].device
    zero = torch.zeros((), device=device)
    if loss_distill is None:
        loss_distill = zero

    loss = (
        args.coef_v * physics_losses['loss_v']
        + args.coef_obj_avoidance * physics_losses['loss_avoid']
        + args.coef_d_acc * physics_losses['loss_d_acc']
        + args.coef_d_jerk * physics_losses['loss_d_jerk']
        + args.coef_collide * physics_losses['loss_collide']
        + args.coef_cam_smooth * camera_losses['loss_cam_smooth']
        + args.coef_diff_depth_power * camera_losses['loss_diff_depth_power']
        + args.coef_diff_depth_blur * camera_losses['loss_diff_depth_blur']
        + args.coef_diff_depth_noise * camera_losses['loss_diff_depth_noise']
        + args.coef_diff_depth_fill * camera_losses['loss_diff_depth_fill']
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
        'loss_cam_smooth': camera_losses['loss_cam_smooth'],
        'loss_diff_depth_power': camera_losses['loss_diff_depth_power'],
        'loss_diff_depth_blur': camera_losses['loss_diff_depth_blur'],
        'loss_diff_depth_noise': camera_losses['loss_diff_depth_noise'],
        'loss_diff_depth_fill': camera_losses['loss_diff_depth_fill'],
        'loss_distill': loss_distill,
    }

    return loss, all_losses

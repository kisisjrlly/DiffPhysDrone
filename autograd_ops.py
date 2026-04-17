import torch
import quadsim_cuda
import os


# Phase 1 Optimization: Remove Python-layer redundant synchronization
# C++ layer still maintains synchronize() at function boundaries for stability
_SYNC_BACKWARD = os.getenv("DIFFPHYS_SYNC_BACKWARD", "0") == "1"


def _maybe_sync_backward():
    # Disabled: C++ layer already handles synchronization
    pass


class RunFunction(torch.autograd.Function):
    """
    无人机物理动力学的前向与反向传播封装。
    调用底层的 CUDA C++ 扩展 (quadsim_cuda) 来加速物理仿真，并支持可微物理 (Differentiable Physics)。
    """
    @staticmethod
    def forward(ctx, R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, grad_decay, ctl_dt, airmode):
        act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, airmode)
        ctx.save_for_backward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next)
        ctx.grad_decay = grad_decay
        ctx.ctl_dt = ctl_dt
        return act_next, p_next, v_next, a_next

    @staticmethod
    def backward(ctx, d_act_next, d_p_next, d_v_next, d_a_next):
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next = ctx.saved_tensors
        d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, d_act_next, d_p_next, d_v_next, d_a_next,
            ctx.grad_decay, ctx.ctl_dt)
        _maybe_sync_backward()
        return None, None, None, None, None, d_act_pred, d_act, d_p, d_v, None, d_a, None, None, None


run = RunFunction.apply


class DiffDepthFunction(torch.autograd.Function):
    """
    可微深度传感器模型 (Differentiable Depth Sensor Model)。
    对 power/exposure/gain 三个传感器参数可微。
    fov_x_half_tan 为固定标量，不参与梯度计算。
    """
    @staticmethod
    def forward(ctx, fov_x_half_tan, power, exposure, gain, v,
                R_cam, pos, balls, cyl, cyl_h, voxels,
                n_drones_per_group, height, width, max_range):
        out = quadsim_cuda.render_diff_depth_forward(
            float(fov_x_half_tan),
            power.contiguous(),
            exposure.contiguous(),
            gain.contiguous(),
            v.contiguous(),
            R_cam.contiguous(),
            pos.contiguous(),
            balls, cyl, cyl_h, voxels,
            int(n_drones_per_group), int(height), int(width), float(max_range),
        )
        noisy_depth, quality = out[0], out[1]
        ctx.save_for_backward(
            noisy_depth, quality,
            power, exposure, gain, v,
            R_cam, pos, balls, cyl, cyl_h, voxels)
        ctx.fov_x_half_tan = float(fov_x_half_tan)
        ctx.n_drones_per_group = int(n_drones_per_group)
        ctx.height = int(height)
        ctx.width = int(width)
        ctx.max_range = float(max_range)
        return noisy_depth, quality

    @staticmethod
    def backward(ctx, grad_noisy_depth, grad_quality):
        (noisy_depth, quality,
         power, exposure, gain, v,
         R_cam, pos, balls, cyl, cyl_h, voxels) = ctx.saved_tensors

        grad_noisy_depth = grad_noisy_depth.contiguous()
        grad_quality = grad_quality.contiguous()

        grad_power, grad_exposure, grad_gain = quadsim_cuda.render_diff_depth_backward(
            grad_noisy_depth, grad_quality, noisy_depth, quality,
            ctx.fov_x_half_tan, power, exposure, gain, v,
            R_cam, pos, balls, cyl, cyl_h, voxels,
            int(ctx.n_drones_per_group), int(ctx.height), int(ctx.width), float(ctx.max_range),
        )
        _maybe_sync_backward()
        return (
            None, grad_power, grad_exposure, grad_gain,
            None, None, None, None, None, None, None, None, None, None, None,
        )


diff_depth = DiffDepthFunction.apply

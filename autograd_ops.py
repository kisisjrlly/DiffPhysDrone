import torch
import os
import sys

try:
    import quadsim_cuda
except ModuleNotFoundError:
    _src_dir = os.path.join(os.path.dirname(__file__), 'src')
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    import quadsim_cuda


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


class ActiveSensingSensorFunction(torch.autograd.Function):
    """Fused CUDA core for the minimal active-sensing sensor model.

    Inputs are a pre-rendered raw depth map and local scene mask.  Backward only
    returns gradients for power/exposure/gain; geometry depth is intentionally
    treated as non-differentiable in this branch.
    """

    @staticmethod
    def forward(ctx, depth, mask, power, exposure, gain, speed,
                regime_id: int, min_valid: float, max_range: float,
                exposure_t_min: float, exposure_t_span: float,
                iso_gain_base: float, iso_gain_scale: float, iso_gain_gamma: float,
                shot_noise_base: float):
        out = quadsim_cuda.active_sensing_sensor_forward(
            depth.contiguous(),
            mask.contiguous(),
            power.contiguous(),
            exposure.contiguous(),
            gain.contiguous(),
            speed.contiguous(),
            int(regime_id),
            float(min_valid),
            float(max_range),
            float(exposure_t_min),
            float(exposure_t_span),
            float(iso_gain_base),
            float(iso_gain_scale),
            float(iso_gain_gamma),
            float(shot_noise_base),
        )
        depth_obs, quality_obs, quality, valid_prob, hard_valid, effect = out
        raw = depth.clamp(float(min_valid), float(max_range)).contiguous()
        ctx.save_for_backward(
            raw, mask.contiguous(), quality, valid_prob, hard_valid,
            power, exposure, gain, speed)
        ctx.regime_id = int(regime_id)
        ctx.min_valid = float(min_valid)
        ctx.max_range = float(max_range)
        ctx.exposure_t_min = float(exposure_t_min)
        ctx.exposure_t_span = float(exposure_t_span)
        ctx.iso_gain_base = float(iso_gain_base)
        ctx.iso_gain_scale = float(iso_gain_scale)
        ctx.iso_gain_gamma = float(iso_gain_gamma)
        ctx.shot_noise_base = float(shot_noise_base)
        return depth_obs, quality_obs, quality, valid_prob, hard_valid, effect

    @staticmethod
    def backward(ctx, grad_depth_obs, grad_quality_obs,
                 grad_quality, grad_valid_prob, grad_hard_valid, grad_effect):
        (raw, mask, quality, valid_prob, hard_valid,
         power, exposure, gain, speed) = ctx.saved_tensors

        zeros = torch.zeros_like(raw)
        g_depth_obs = grad_depth_obs if grad_depth_obs is not None else zeros
        g_quality_obs = grad_quality_obs if grad_quality_obs is not None else zeros
        g_quality_out = grad_quality if grad_quality is not None else zeros
        g_valid_prob = grad_valid_prob if grad_valid_prob is not None else zeros
        g_effect = grad_effect if grad_effect is not None else zeros

        # Reference graph:
        #   valid_st = hard.detach() - valid.detach() + valid
        #   depth_obs = raw * valid_st
        #   quality_obs = quality * valid_st
        #   valid = sigmoid((quality - 0.42) / 0.055)
        # Combine all differentiable output paths into dL/dquality before
        # entering the fused kernel.  hard_valid is intentionally non-diff.
        dvalid_dquality = valid_prob * (1.0 - valid_prob) / 0.055
        grad_quality_total = (
            g_quality_out
            + g_quality_obs * hard_valid
            + (g_depth_obs * raw + g_quality_obs * quality + g_valid_prob) * dvalid_dquality
        )

        grad_power, grad_exposure, grad_gain = quadsim_cuda.active_sensing_sensor_backward(
            grad_quality_total.contiguous(),
            g_effect.contiguous(),
            raw,
            mask,
            quality,
            power,
            exposure,
            gain,
            speed,
            int(ctx.regime_id),
            float(ctx.min_valid),
            float(ctx.max_range),
            float(ctx.exposure_t_min),
            float(ctx.exposure_t_span),
            float(ctx.iso_gain_base),
            float(ctx.iso_gain_scale),
            float(ctx.iso_gain_gamma),
            float(ctx.shot_noise_base),
        )
        _maybe_sync_backward()
        return None, None, grad_power, grad_exposure, grad_gain, None, None, None, None, None, None, None, None, None, None


active_sensing_sensor = ActiveSensingSensorFunction.apply

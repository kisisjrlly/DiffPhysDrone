import torch
import quadsim_cuda


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
        # 强制同步以防止反向传播期间 GPU 积压任务过长导致桌面卡死 (TDR)
        torch.cuda.synchronize()
        return None, None, None, None, None, d_act_pred, d_act, d_p, d_v, None, d_a, None, None, None


run = RunFunction.apply


class DiffRenderFunction(torch.autograd.Function):
    """
    可微渲染函数 (Differentiable Rendering)。
    支持对每个 batch 的 FOV (视场角) 张量进行求导。
    """
    @staticmethod
    def forward(ctx, fov_x_half_tan, R_cam, pos, balls, cyl, cyl_h, voxels,
                n_drones_per_group, height, width):
        B = pos.shape[0]
        fov_x_half_tan = fov_x_half_tan.contiguous()
        R_cam = R_cam.contiguous()
        pos = pos.contiguous()
        canvas = torch.empty((B, height, width), device=pos.device)
        quadsim_cuda.render_diff_fov(canvas, balls, cyl, cyl_h, voxels,
                                     R_cam, pos, n_drones_per_group, fov_x_half_tan)
        ctx.save_for_backward(fov_x_half_tan, canvas, R_cam, pos, balls, cyl, cyl_h, voxels)
        ctx.n_drones_per_group = n_drones_per_group
        return canvas

    @staticmethod
    def backward(ctx, grad_output):
        fov, canvas, R_cam, pos, balls, cyl, cyl_h, voxels = ctx.saved_tensors
        grad_fov = torch.zeros_like(fov)
        quadsim_cuda.render_backward_fov(grad_fov, grad_output.contiguous(), canvas,
                                         balls, cyl, cyl_h, voxels, R_cam, pos,
                                         ctx.n_drones_per_group, fov)
        torch.cuda.synchronize()
        return grad_fov, None, None, None, None, None, None, None, None, None


diff_render = DiffRenderFunction.apply


class DiffRenderYuvYFunction(torch.autograd.Function):
    """
    Y 通道可微渲染函数。
    """
    @staticmethod
    def forward(ctx, fov_x_half_tan, exposure, iso,
                R_cam, pos, balls, cyl, cyl_h, voxels,
                n_drones_per_group, height, width):
        fov_x_half_tan = fov_x_half_tan.contiguous()
        exposure = exposure.contiguous()
        iso = iso.contiguous()
        R_cam = R_cam.contiguous()
        pos = pos.contiguous()
        y, depth_raw = quadsim_cuda.render_diff_yuv_y_forward(
            fov_x_half_tan, exposure, iso,
            R_cam, pos, balls, cyl, cyl_h, voxels,
            n_drones_per_group, height, width)
        ctx.save_for_backward(
            depth_raw, fov_x_half_tan, exposure, iso,
            R_cam, pos, balls, cyl, cyl_h, voxels)
        ctx.n_drones_per_group = n_drones_per_group
        return y

    @staticmethod
    def backward(ctx, grad_output):
        depth_raw, fov, exposure, iso, R_cam, pos, balls, cyl, cyl_h, voxels = ctx.saved_tensors
        grad_fov, grad_exposure, grad_iso = quadsim_cuda.render_diff_yuv_y_backward(
            grad_output.contiguous(),
            depth_raw,
            fov,
            exposure,
            iso,
            R_cam,
            pos,
            balls,
            cyl,
            cyl_h,
            voxels,
            ctx.n_drones_per_group)
        torch.cuda.synchronize()
        return grad_fov, grad_exposure, grad_iso, None, None, None, None, None, None, None, None, None


diff_render_yuv_y = DiffRenderYuvYFunction.apply


class DiffRenderActiveTofFunction(torch.autograd.Function):
    """
    Active ToF 可微渲染（CUDA 路径）。
    """
    @staticmethod
    def forward(ctx, fov_x_half_tan, power, exposure, gain, v,
                R_cam, pos, balls, cyl, cyl_h, voxels,
                n_drones_per_group, height, width, max_range):
        out = quadsim_cuda.render_active_tof_forward(
            fov_x_half_tan.contiguous(),
            power.contiguous(),
            exposure.contiguous(),
            gain.contiguous(),
            v.contiguous(),
            R_cam.contiguous(),
            pos.contiguous(),
            balls, cyl, cyl_h, voxels,
            int(n_drones_per_group), int(height), int(width), float(max_range),
        )
        noisy_depth, conf = out[0], out[1]
        ctx.save_for_backward(
            noisy_depth, conf,
            fov_x_half_tan, power, exposure, gain, v,
            R_cam, pos, balls, cyl, cyl_h, voxels)
        ctx.n_drones_per_group = int(n_drones_per_group)
        ctx.height = int(height)
        ctx.width = int(width)
        ctx.max_range = float(max_range)
        return noisy_depth, conf

    @staticmethod
    def backward(ctx, grad_noisy_depth, grad_conf):
        (noisy_depth, conf,
         fov_x_half_tan, power, exposure, gain, v,
         R_cam, pos, balls, cyl, cyl_h, voxels) = ctx.saved_tensors

        grad_noisy_depth = grad_noisy_depth.contiguous()
        grad_conf = grad_conf.contiguous()

        grad_fov, grad_power, grad_exposure, grad_gain = quadsim_cuda.render_active_tof_backward(
            grad_noisy_depth, grad_conf, noisy_depth, conf,
            fov_x_half_tan, power, exposure, gain, v,
            R_cam, pos, balls, cyl, cyl_h, voxels,
            int(ctx.n_drones_per_group), int(ctx.height), int(ctx.width), float(ctx.max_range),
        )
        # 强制同步以防止反向传播期间 GPU 积压任务过长导致桌面卡死 (TDR)
        torch.cuda.synchronize()
        return (
            grad_fov, grad_power, grad_exposure, grad_gain,
            None, None, None, None, None, None, None, None, None, None, None,
        )


diff_render_active_tof = DiffRenderActiveTofFunction.apply

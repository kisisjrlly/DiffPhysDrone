import torch
import quadsim_cuda
import os


_SYNC_BACKWARD = os.getenv("DIFFPHYS_SYNC_BACKWARD", "0") == "1"


def _maybe_sync_backward():
    if _SYNC_BACKWARD:
        torch.cuda.synchronize()


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
        normals = torch.empty((B, 3, height, width), device=pos.device)
        quadsim_cuda.render_diff_fov_with_normal(canvas, normals, balls, cyl, cyl_h, voxels,
                             R_cam, pos, n_drones_per_group, fov_x_half_tan)
        ctx.save_for_backward(fov_x_half_tan, canvas, normals, R_cam)
        ctx.n_drones_per_group = n_drones_per_group
        return canvas

    @staticmethod
    def backward(ctx, grad_output):
        fov, canvas, normals, R_cam = ctx.saved_tensors
        grad_fov = torch.zeros_like(fov)
        quadsim_cuda.render_backward_fov_from_normal(
            grad_fov,
            grad_output.contiguous(),
            canvas,
            normals,
            R_cam,
            fov,
        )
        _maybe_sync_backward()
        return grad_fov, None, None, None, None, None, None, None, None, None


diff_render = DiffRenderFunction.apply


class DiffRenderYuvYFunction(torch.autograd.Function):
    """
    Y 通道可微渲染函数。
    """
    @staticmethod
    def forward(ctx, fov_x_half_tan, exposure, iso,
                R_cam, pos, balls, cyl, cyl_h, voxels,
                n_drones_per_group, height, width,
                cam_light_dir, cam_ambient, cam_dir_intensity,
                cam_fog_beta, cam_airlight,
                cam_mat_ground, cam_mat_obstacle, cam_mat_spec,
                cam_dist_k1, cam_dist_k2, cam_flare_strength,
                cam_gamma, cam_prnu, cam_dsnu,
                cam_prev_y, cam_use_rolling, v, cam_ae_log_t,
                cam_profile_mask,
                cam_vignette_a, cam_vignette_b,
                cam_black_level, cam_sharpen_amount, cam_base_gain, cam_motion_blur_gain,
                cam_exposure_t_min, cam_exposure_t_span,
                cam_exposure_eff_min, cam_exposure_eff_max,
                cam_iso_gain_base, cam_iso_gain_scale, cam_iso_gain_gamma):
        fov_x_half_tan = fov_x_half_tan.contiguous()
        exposure = exposure.contiguous()
        iso = iso.contiguous()
        R_cam = R_cam.contiguous()
        pos = pos.contiguous()
        out = quadsim_cuda.render_diff_yuv_y_forward(
            fov_x_half_tan, exposure, iso,
            R_cam, pos, balls, cyl, cyl_h, voxels,
            n_drones_per_group, height, width,
            cam_light_dir, cam_ambient, cam_dir_intensity,
            cam_fog_beta, cam_airlight,
            cam_mat_ground, cam_mat_obstacle, cam_mat_spec,
            cam_dist_k1, cam_dist_k2, cam_flare_strength,
            cam_gamma, cam_prnu, cam_dsnu,
            cam_prev_y, cam_use_rolling, v, cam_ae_log_t,
            int(cam_profile_mask),
            cam_vignette_a, cam_vignette_b,
            cam_black_level, cam_sharpen_amount, cam_base_gain, cam_motion_blur_gain,
            cam_exposure_t_min, cam_exposure_t_span,
            cam_exposure_eff_min, cam_exposure_eff_max,
            cam_iso_gain_base, cam_iso_gain_scale, cam_iso_gain_gamma)
        if isinstance(out, (list, tuple)) and len(out) >= 3:
            y, depth_raw, normals = out[0], out[1], out[2]
        elif isinstance(out, (list, tuple)) and len(out) >= 2:
            y, depth_raw = out[0], out[1]
            normals = None
        else:
            raise RuntimeError("render_diff_yuv_y_forward 返回值异常，期望至少包含 (y, depth_raw)")
        if normals is None:
            normals = torch.zeros((depth_raw.shape[0], 3, depth_raw.shape[1], depth_raw.shape[2]), device=depth_raw.device, dtype=depth_raw.dtype)
        ctx.save_for_backward(
            depth_raw, fov_x_half_tan, exposure, iso,
            normals,
            R_cam, pos,
            balls, cyl, cyl_h, voxels,
            cam_light_dir, cam_ambient, cam_dir_intensity,
            cam_fog_beta, cam_airlight,
            cam_mat_ground, cam_mat_obstacle, cam_mat_spec,
            cam_dist_k1, cam_dist_k2, cam_flare_strength,
            cam_gamma, cam_prnu, cam_dsnu,
            cam_prev_y, cam_use_rolling, v, cam_ae_log_t)
        ctx.n_drones_per_group = int(n_drones_per_group)
        ctx.height = int(height)
        ctx.width = int(width)
        ctx.cam_profile_mask = int(cam_profile_mask)
        ctx.cam_vignette_a = float(cam_vignette_a)
        ctx.cam_vignette_b = float(cam_vignette_b)
        ctx.cam_black_level = float(cam_black_level)
        ctx.cam_sharpen_amount = float(cam_sharpen_amount)
        ctx.cam_base_gain = float(cam_base_gain)
        ctx.cam_motion_blur_gain = float(cam_motion_blur_gain)
        ctx.cam_exposure_t_min = float(cam_exposure_t_min)
        ctx.cam_exposure_t_span = float(cam_exposure_t_span)
        ctx.cam_exposure_eff_min = float(cam_exposure_eff_min)
        ctx.cam_exposure_eff_max = float(cam_exposure_eff_max)
        ctx.cam_iso_gain_base = float(cam_iso_gain_base)
        ctx.cam_iso_gain_scale = float(cam_iso_gain_scale)
        ctx.cam_iso_gain_gamma = float(cam_iso_gain_gamma)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (depth_raw, fov, exposure, iso, normals,
         R_cam, pos,
         balls, cyl, cyl_h, voxels,
         cam_light_dir, cam_ambient, cam_dir_intensity,
         cam_fog_beta, cam_airlight,
         cam_mat_ground, cam_mat_obstacle, cam_mat_spec,
         cam_dist_k1, cam_dist_k2, cam_flare_strength,
         cam_gamma, cam_prnu, cam_dsnu,
         cam_prev_y, cam_use_rolling, v, cam_ae_log_t) = ctx.saved_tensors

        need_grad_fov = bool(ctx.needs_input_grad[0])
        need_grad_exposure = bool(ctx.needs_input_grad[1])
        need_grad_iso = bool(ctx.needs_input_grad[2])

        grad_fov, grad_exposure, grad_iso = quadsim_cuda.render_diff_yuv_y_backward(
            grad_output.contiguous(),
            depth_raw,
            fov,
            exposure,
            iso,
            normals,
            R_cam, pos,
            balls, cyl, cyl_h, voxels,
            ctx.n_drones_per_group, ctx.height, ctx.width,
            cam_light_dir, cam_ambient, cam_dir_intensity,
            cam_fog_beta, cam_airlight,
            cam_mat_ground, cam_mat_obstacle, cam_mat_spec,
            cam_dist_k1, cam_dist_k2, cam_flare_strength,
            cam_gamma, cam_prnu, cam_dsnu,
            cam_prev_y, cam_use_rolling, v, cam_ae_log_t,
            ctx.cam_profile_mask,
            ctx.cam_vignette_a, ctx.cam_vignette_b,
            ctx.cam_black_level, ctx.cam_sharpen_amount, ctx.cam_base_gain, ctx.cam_motion_blur_gain,
            ctx.cam_exposure_t_min, ctx.cam_exposure_t_span,
            ctx.cam_exposure_eff_min, ctx.cam_exposure_eff_max,
            ctx.cam_iso_gain_base, ctx.cam_iso_gain_scale, ctx.cam_iso_gain_gamma,
            need_grad_fov, need_grad_exposure, need_grad_iso)
        _maybe_sync_backward()
        return (
            grad_fov if need_grad_fov else None,
            grad_exposure if need_grad_exposure else None,
            grad_iso if need_grad_iso else None,
            *([None] * 41),
        )


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
        _maybe_sync_backward()
        return (
            grad_fov, grad_power, grad_exposure, grad_gain,
            None, None, None, None, None, None, None, None, None, None, None,
        )


diff_render_active_tof = DiffRenderActiveTofFunction.apply

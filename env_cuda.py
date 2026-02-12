import math
import random
import time
import torch
import torch.nn.functional as F
import quadsim_cuda


class GDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None

g_decay = GDecay.apply


class RunFunction(torch.autograd.Function):
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
        return None, None, None, None, None, d_act_pred, d_act, d_p, d_v, None, d_a, None, None, None

run = RunFunction.apply


class DiffRenderFunction(torch.autograd.Function):
    """Differentiable rendering w.r.t. per-batch FOV tensor via CUDA forward/backward."""
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
        return grad_fov, None, None, None, None, None, None, None, None, None

diff_render = DiffRenderFunction.apply


def apply_camera_effects(depth, exposure, iso, focus_dist):
    """Apply differentiable camera sensor effects to depth image.

    Args:
        depth: (B, H, W) raw depth from renderer
        exposure: (B,) in [0, 1] from sigmoid
        iso: (B,) in [0, 1] from sigmoid
        focus_dist: (B,) in [0, 1] from sigmoid
    Returns:
        (B, H, W) processed depth with sensor effects
    """
    # Map [0,1] to physical ranges
    exposure_phys = exposure * 10 + 0.5       # [0.5, 10.5] ms
    iso_phys = iso * 6400 + 100               # [100, 6500]
    focus_phys = focus_dist * 20 + 0.5        # [0.5, 20.5] m

    # 1. Effective sensing range: max depth the sensor can detect
    #    Higher exposure / ISO -> longer range
    max_range = 2.0 + 1.5 * exposure_phys + 0.001 * iso_phys  # ~[3, 24] meters
    max_range = max_range[:, None, None]  # (B, 1, 1)
    # Smooth clamp: depth beyond max_range mapped to max_range (differentiable)
    depth = max_range - F.softplus(max_range - depth, beta=2.0)

    # 2. Depth noise: higher ISO increases noise, higher exposure decreases it
    noise_sigma = 0.03 * (1.0 + 2.0 * iso) / (exposure + 0.3)  # (B,)
    depth_dist_scale = depth.detach().clamp(0.3, 20) / 5.0  # noise scales with distance
    depth = depth + torch.randn_like(depth) * noise_sigma[:, None, None] * depth_dist_scale

    # 3. Focus distance: out-of-focus regions get degraded depth readings
    focus_phys = focus_phys[:, None, None]  # (B, 1, 1)
    dof_sigma = 4.0
    focus_weight = torch.exp(-((depth.detach() - focus_phys) ** 2) / (2 * dof_sigma ** 2))
    # In-focus: keep depth; out-of-focus: blend with detached (stops gradient → encourages focus)
    depth = depth * focus_weight + depth.detach() * (1 - focus_weight)

    return depth


class Env:
    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 single=False, gate=False, ground_voxels=False, scaffold=False, speed_mtp=1,
                 random_rotation=False, cam_angle=10,
                 wall_slit=False, ellipsoid_a=0.0, ellipsoid_c=0.0) -> None:
        self.device = device
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.grad_decay = grad_decay
        self.wall_slit = wall_slit
        self.ellipsoid_a = ellipsoid_a
        self.ellipsoid_c = ellipsoid_c
        self.use_ellipsoid = ellipsoid_a > 0 and ellipsoid_c > 0
        self.ball_w = torch.tensor([8., 18, 6, 0.2], device=device)
        self.ball_b = torch.tensor([0., -9, -1, 0.4], device=device)
        self.voxel_w = torch.tensor([8., 18, 6, 0.1, 0.1, 0.1], device=device)
        self.voxel_b = torch.tensor([0., -9, -1, 0.2, 0.2, 0.2], device=device)
        self.ground_voxel_w = torch.tensor([8., 18,  0, 2.9, 2.9, 1.9], device=device)
        self.ground_voxel_b = torch.tensor([0., -9, -1, 0.1, 0.1, 0.1], device=device)
        self.cyl_w = torch.tensor([8., 18, 0.35], device=device)
        self.cyl_b = torch.tensor([0., -9, 0.05], device=device)
        self.cyl_h_w = torch.tensor([8., 6, 0.1], device=device)
        self.cyl_h_b = torch.tensor([0., 0, 0.05], device=device)
        self.gate_w = torch.tensor([2.,  2,  1.0, 0.5], device=device)
        self.gate_b = torch.tensor([3., -1,  0.0, 0.5], device=device)
        self.v_wind_w = torch.tensor([1,  1,  0.2], device=device)
        self.g_std = torch.tensor([0., 0, -9.80665], device=device)
        self.roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5], device=device)
        self.sub_div = torch.linspace(0, 1. / 15, 10, device=device).reshape(-1, 1, 1)
        self.p_init = torch.as_tensor([
            [-1.5, -3.,  1],
            [ 9.5, -3.,  1],
            [-0.5,  1.,  1],
            [ 8.5,  1.,  1],
            [ 0.0,  3.,  1],
            [ 8.0,  3.,  1],
            [-1.0, -1.,  1],
            [ 9.0, -1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        self.p_end = torch.as_tensor([
            [8.,  3.,  1],
            [0.,  3.,  1],
            [8., -1.,  1],
            [0., -1.,  1],
            [8., -3.,  1],
            [0., -3.,  1],
            [8.,  1.,  1],
            [0.,  1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        self.flow = torch.empty((batch_size, 0, height, width), device=device)
        self.single = single
        self.gate = gate
        self.ground_voxels = ground_voxels
        self.scaffold = scaffold
        self.speed_mtp = speed_mtp
        self.random_rotation = random_rotation
        self.cam_angle = cam_angle
        self.fov_x_half_tan = fov_x_half_tan
        if wall_slit:
            self.single = True  # wall_slit forces single drone mode
        self.reset()
        # self.obj_avoid_grad_mtp = torch.tensor([0.5, 2., 1.], device=device)

    def reset(self):
        B = self.batch_size
        device = self.device

        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
            zeros, ones, zeros,
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),
        ], -1).reshape(B, 3, 3)

        # env
        self.balls = torch.rand((B, 30, 4), device=device) * self.ball_w + self.ball_b
        self.voxels = torch.rand((B, 30, 6), device=device) * self.voxel_w + self.voxel_b
        self.cyl = torch.rand((B, 30, 3), device=device) * self.cyl_w + self.cyl_b
        self.cyl_h = torch.rand((B, 2, 3), device=device) * self.cyl_h_w + self.cyl_h_b

        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan
        self.n_drones_per_group = random.choice([4, 8])
        self.drone_radius = random.uniform(0.1, 0.15)
        if self.single:
            self.n_drones_per_group = 1

        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        self.max_speed = (0.75 + 2.5 * rd) * self.speed_mtp
        scale = (self.max_speed - 0.5).clamp_min(1)

        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        roof = torch.rand((B,)) < 0.5
        self.balls[~roof, :15, :2] = self.cyl[~roof, :15, :2]
        self.voxels[~roof, :15, :2] = self.cyl[~roof, 15:, :2]
        self.balls[~roof, :15] = self.balls[~roof, :15] + self.roof_add[:4]
        self.voxels[~roof, :15] = self.voxels[~roof, :15] + self.roof_add
        self.balls[..., 0] = torch.minimum(torch.maximum(self.balls[..., 0], self.balls[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.balls[..., 3])
        self.voxels[..., 0] = torch.minimum(torch.maximum(self.voxels[..., 0], self.voxels[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.voxels[..., 3])
        self.cyl[..., 0] = torch.minimum(torch.maximum(self.cyl[..., 0], self.cyl[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl[..., 2])
        self.cyl_h[..., 0] = torch.minimum(torch.maximum(self.cyl_h[..., 0], self.cyl_h[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl_h[..., 2])
        self.voxels[roof, 0, 2] = self.voxels[roof, 0, 2] * 0.5 + 201
        self.voxels[roof, 0, 3:] = 200

        if self.ground_voxels:
            ground_balls_r = 8 + torch.rand((B, 2), device=device) * 6
            ground_balls_r_ground = 2 + torch.rand((B, 2), device=device) * 4
            ground_balls_h = ground_balls_r - (ground_balls_r.pow(2) - ground_balls_r_ground.pow(2)).sqrt()
            # |   ground_balls_h
            # ----- ground_balls_r_ground
            # |  /
            # | / ground_balls_r
            # |/
            self.balls[:, :2, 3] = ground_balls_r
            self.balls[:, :2, 2] = ground_balls_h - ground_balls_r - 1

            # planner shape in (0.1-2.0) times (0.1-2.0)
            ground_voxels = torch.rand((B, 10, 6), device=device) * self.ground_voxel_w + self.ground_voxel_b
            ground_voxels[:, :, 2] = ground_voxels[:, :, 5] - 1
            self.voxels = torch.cat([self.voxels, ground_voxels], 1)

        self.voxels[:, :, 1] *= (self.max_speed + 4) / scale
        self.balls[:, :, 1] *= (self.max_speed + 4) / scale
        self.cyl[:, :, 1] *= (self.max_speed + 4) / scale

        # gates
        if self.gate:
            gate = torch.rand((B, 4), device=device) * self.gate_w + self.gate_b
            p = gate[None, :, :3]
            nearest_pt = torch.empty_like(p)
            quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, 1)
            gate_x, gate_y, gate_z, gate_r = gate.unbind(-1)
            gate_x[(nearest_pt - p).norm(2, -1)[0] < 0.5] = -50
            ones = torch.ones_like(gate_x)
            gate = torch.stack([
                torch.stack([gate_x, gate_y + gate_r + 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z + gate_r + 5, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y - gate_r - 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z - gate_r - 5, ones * 0.05, ones * 5, ones * 5], -1),
            ], 1)

            self.voxels = torch.cat([self.voxels, gate], 1)
        self.voxels[..., 0] *= scale
        self.balls[..., 0] *= scale
        self.cyl[..., 0] *= scale
        self.cyl_h[..., 0] *= scale
        if self.ground_voxels:
            self.balls[:, :2, 0] = torch.minimum(torch.maximum(self.balls[:, :2, 0], ground_balls_r_ground + 0.3), scale * 8 - 0.3 - ground_balls_r_ground)

        # drone
        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device)
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)

        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        scale = torch.cat([
            scale,
            rd + 0.5,
            torch.rand_like(scale) - 0.5], -1)
        self.p = self.p_init * scale + torch.randn_like(scale) * 0.1
        self.p_target = self.p_end * scale + torch.randn_like(scale) * 0.1

        if self.random_rotation:
            yaw_bias = torch.rand(B//self.n_drones_per_group, device=device).repeat_interleave(self.n_drones_per_group, 0) * 1.5 - 0.75
            c = torch.cos(yaw_bias)
            s = torch.sin(yaw_bias)
            l = torch.ones_like(yaw_bias)
            o = torch.zeros_like(yaw_bias)
            R = torch.stack([c,-s, o, s, c, o, o, o, l], -1).reshape(B, 3, 3)
            self.p = torch.squeeze(R @ self.p[..., None], -1)
            self.p_target = torch.squeeze(R @ self.p_target[..., None], -1)
            self.voxels[..., :3] = (R @ self.voxels[..., :3].transpose(1, 2)).transpose(1, 2)
            self.balls[..., :3] = (R @ self.balls[..., :3].transpose(1, 2)).transpose(1, 2)
            self.cyl[..., :3] = (R @ self.cyl[..., :3].transpose(1, 2)).transpose(1, 2)

        # scaffold
        if self.scaffold and random.random() < 0.5:
            x = torch.arange(1, 6, dtype=torch.float, device=device)
            y = torch.arange(-3, 4, dtype=torch.float, device=device)
            z = torch.arange(1, 4, dtype=torch.float, device=device)
            _x, _y = torch.meshgrid(x, y)
            # + torch.rand_like(self.max_speed) * self.max_speed
            # + torch.randn_like(self.max_speed)
            scaf_v = torch.stack([_x, _y, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            x_bias = torch.rand_like(self.max_speed) * self.max_speed
            scale = 1 + torch.rand((B, 1, 1), device=device)
            scaf_v = scaf_v * scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed),
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            self.cyl = torch.cat([self.cyl, scaf_v], 1)
            _x, _z = torch.meshgrid(x, z)
            scaf_h = torch.stack([_x, _z, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            scaf_h = scaf_h * scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed) * 0.1,
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            self.cyl_h = torch.cat([self.cyl_h, scaf_h], 1)

        self.v = torch.randn((B, 3), device=device) * 0.2
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w
        self.act = torch.randn_like(self.v) * 0.1
        self.a = self.act
        self.dg = torch.randn((B, 3), device=device) * 0.2

        R = torch.zeros((B, 3, 3), device=device)
        self.R = quadsim_cuda.update_state_vec(R, self.act, torch.randn((B, 3), device=device) * 0.2 + F.normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay), 5)
        self.R_old = self.R.clone()
        self.p_old = self.p
        self.margin = torch.rand((B,), device=device) * 0.2 + 0.1

        # ==================== Wall-Slit Environment ====================
        if self.wall_slit:
            self._reset_wall_slit(B, device)

        # drag coef
        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3
        self.drag_2[:, 0] = 0
        self.z_drag_coef = torch.ones((B, 1), device=device)

    def _reset_wall_slit(self, B, device):
        """Override obstacle and drone placement for wall-slit scenario.

        Creates a thin wall (along YZ plane) with a narrow vertical slit.
        The slit is taller than it is wide, so the drone must roll/tilt
        sideways to pass through. Drone starts on one side of the wall,
        target is on the other side.
        """
        # Wall parameters (randomized per reset, shared across batch)
        wall_x = 2.0 + random.random() * 4.0      # wall x-position in [2, 6]
        slit_y_center = random.uniform(-1.0, 1.0)  # slit lateral center
        slit_z_center = random.uniform(0.0, 1.5)   # slit vertical center (above ground z=-1)
        slit_half_w = random.uniform(0.10, 0.18)    # half-width of slit (narrow, ~0.20-0.36m total)
        slit_half_h = random.uniform(0.35, 0.60)    # half-height of slit (tall, ~0.70-1.20m total)
        wall_thickness = 0.15                        # wall half-thickness in X

        # Store wall params for evaluation / logging
        self.wall_x = wall_x
        self.slit_y_center = slit_y_center
        self.slit_z_center = slit_z_center
        self.slit_half_w = slit_half_w
        self.slit_half_h = slit_half_h

        # Build 4 voxels forming a wall with a rectangular slit opening:
        #   Left wall:  everything to the left of the slit opening
        #   Right wall: everything to the right of the slit opening
        #   Top wall:   above the slit opening (same Y span as slit)
        #   Bottom wall: below the slit opening (same Y span as slit)
        big = 10.0  # large half-extent to cover scene

        wall_voxels = torch.zeros((B, 4, 6), device=device)
        # Left: center_y = slit_y - slit_half_w - big, ry = big
        wall_voxels[:, 0, 0] = wall_x
        wall_voxels[:, 0, 1] = slit_y_center - slit_half_w - big
        wall_voxels[:, 0, 2] = slit_z_center
        wall_voxels[:, 0, 3] = wall_thickness
        wall_voxels[:, 0, 4] = big
        wall_voxels[:, 0, 5] = big

        # Right: center_y = slit_y + slit_half_w + big, ry = big
        wall_voxels[:, 1, 0] = wall_x
        wall_voxels[:, 1, 1] = slit_y_center + slit_half_w + big
        wall_voxels[:, 1, 2] = slit_z_center
        wall_voxels[:, 1, 3] = wall_thickness
        wall_voxels[:, 1, 4] = big
        wall_voxels[:, 1, 5] = big

        # Top: center_z = slit_z + slit_half_h + big, rz = big, ry = slit_half_w
        wall_voxels[:, 2, 0] = wall_x
        wall_voxels[:, 2, 1] = slit_y_center
        wall_voxels[:, 2, 2] = slit_z_center + slit_half_h + big
        wall_voxels[:, 2, 3] = wall_thickness
        wall_voxels[:, 2, 4] = slit_half_w
        wall_voxels[:, 2, 5] = big

        # Bottom: center_z = slit_z - slit_half_h - big, rz = big, ry = slit_half_w
        wall_voxels[:, 3, 0] = wall_x
        wall_voxels[:, 3, 1] = slit_y_center
        wall_voxels[:, 3, 2] = slit_z_center - slit_half_h - big
        wall_voxels[:, 3, 3] = wall_thickness
        wall_voxels[:, 3, 4] = slit_half_w
        wall_voxels[:, 3, 5] = big

        # Replace all obstacles with just the wall voxels
        # Push existing random obstacles out of scene
        self.balls[:, :, 2] = -200  # move all balls far below ground
        self.cyl[:, :, 2] = 0.001   # shrink cylinders to negligible
        self.cyl_h[:, :, 2] = 0.001
        self.voxels = wall_voxels

        # Drone placement: start before wall, target after wall
        dist_from_wall = 1.5 + random.random() * 1.5  # 1.5-3.0m from wall
        noise_y = torch.randn(B, device=device) * 0.3
        noise_z = torch.randn(B, device=device) * 0.2
        self.p = torch.stack([
            torch.full((B,), wall_x - dist_from_wall, device=device),
            torch.full((B,), slit_y_center, device=device) + noise_y,
            torch.full((B,), slit_z_center, device=device) + noise_z,
        ], -1)
        self.p_target = torch.stack([
            torch.full((B,), wall_x + dist_from_wall, device=device),
            torch.full((B,), slit_y_center, device=device) + noise_y * 0.5,
            torch.full((B,), slit_z_center, device=device) + noise_z * 0.5,
        ], -1)

        # Force single drone mode
        self.n_drones_per_group = 1
        self.drone_radius = 0.15

        # Lower max speed for precise maneuvering
        self.max_speed = torch.full((B, 1), 0.5 + random.random() * 1.0, device=device) * self.speed_mtp

        # Smaller margin for slit passage (the slit is tight!)
        self.margin = torch.full((B,), 0.02, device=device)

        # Re-initialize v, R, etc. with updated positions
        self.v = torch.randn((B, 3), device=device) * 0.1
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w * 0.3  # less wind
        self.act = torch.randn_like(self.v) * 0.05
        self.a = self.act
        self.dg = torch.randn((B, 3), device=device) * 0.1

        R = torch.zeros((B, 3, 3), device=device)
        self.R = quadsim_cuda.update_state_vec(
            R, self.act,
            torch.randn((B, 3), device=device) * 0.2 + F.normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay), 5)
        self.R_old = self.R.clone()
        self.p_old = self.p

    @staticmethod
    @torch.no_grad()
    def update_state_vec(R, a_thr, v_pred, alpha, yaw_inertia=5):
        self_forward_vec = R[..., 0]
        g_std = torch.tensor([0, 0, -9.80665], device=R.device)
        a_thr = a_thr - g_std
        thrust = torch.norm(a_thr, 2, -1, True)
        self_up_vec = a_thr / thrust
        forward_vec = self_forward_vec * yaw_inertia + v_pred
        forward_vec = self_forward_vec * alpha + F.normalize(forward_vec, 2, -1) * (1 - alpha)
        forward_vec[:, 2] = (forward_vec[:, 0] * self_up_vec[:, 0] + forward_vec[:, 1] * self_up_vec[:, 1]) / -self_up_vec[2]
        self_forward_vec = F.normalize(forward_vec, 2, -1)
        self_left_vec = torch.cross(self_up_vec, self_forward_vec)
        return torch.stack([
            self_forward_vec,
            self_left_vec,
            self_up_vec,
        ], -1)

    def render(self, ctl_dt):
        canvas = torch.empty((self.batch_size, self.height, self.width), device=self.device)
        # assert canvas.is_contiguous()
        # assert nearest_pt.is_contiguous()
        # assert self.balls.is_contiguous()
        # assert self.cyl.is_contiguous()
        # assert self.voxels.is_contiguous()
        # assert Rt.is_contiguous()
        quadsim_cuda.render(canvas, self.flow, self.balls, self.cyl, self.cyl_h,
                            self.voxels, self.R @ self.R_cam, self.R_old, self.p,
                            self.p_old, self.drone_radius, self.n_drones_per_group,
                            self._fov_x_half_tan)
        return canvas, None

    def render_diff(self, fov_tensor):
        """Render with differentiable per-batch FOV tensor. Gradients flow through fov_tensor."""
        canvas = diff_render(fov_tensor, self.R @ self.R_cam, self.p,
                             self.balls, self.cyl, self.cyl_h, self.voxels,
                             self.n_drones_per_group, self.height, self.width)
        return canvas

    def find_vec_to_nearest_pt(self):
        p = self.p + self.v * self.sub_div
        nearest_pt = torch.empty_like(p)
        if self.use_ellipsoid:
            quadsim_cuda.find_nearest_pt_ellipsoid(
                nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p,
                self.R.contiguous(), self.drone_radius, self.n_drones_per_group,
                self.ellipsoid_a, self.ellipsoid_c)
        else:
            quadsim_cuda.find_nearest_pt(
                nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p,
                self.drone_radius, self.n_drones_per_group)
        return nearest_pt - p

    def run(self, act_pred, ctl_dt=1/15, v_pred=None):
        self.dg = self.dg * math.sqrt(1 - ctl_dt / 4) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        self.p_old = self.p
        self.act, self.p, self.v, self.a = run(
            self.R, self.dg, self.z_drag_coef, self.drag_2, self.pitch_ctl_delay,
            act_pred, self.act, self.p, self.v, self.v_wind, self.a,
            self.grad_decay, ctl_dt, 0.5)
        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)

    def save_state(self):
        """Save a snapshot of the full environment state (for G-DAC inner loop replay)."""
        return {
            'p': self.p.clone(),
            'v': self.v.clone(),
            'a': self.a.clone(),
            'act': self.act.clone(),
            'R': self.R.clone(),
            'R_old': self.R_old.clone(),
            'p_old': self.p_old.clone() if isinstance(self.p_old, torch.Tensor) else self.p_old,
            'dg': self.dg.clone(),
            'v_wind': self.v_wind.clone(),
        }

    def restore_state(self, snapshot):
        """Restore environment state from a snapshot (for G-DAC inner loop replay)."""
        self.p = snapshot['p'].clone()
        self.v = snapshot['v'].clone()
        self.a = snapshot['a'].clone()
        self.act = snapshot['act'].clone()
        self.R = snapshot['R'].clone()
        self.R_old = snapshot['R_old'].clone()
        self.p_old = snapshot['p_old'].clone() if isinstance(snapshot['p_old'], torch.Tensor) else snapshot['p_old']
        self.dg = snapshot['dg'].clone()
        self.v_wind = snapshot['v_wind'].clone()

    def _run(self, act_pred, ctl_dt=1/15, v_pred=None):
        alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
        self.act = act_pred * (1 - alpha) + self.act * alpha
        self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
        z_drag = 0
        if self.z_drag_coef is not None:
            v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
            v_prep = self.v - v_up
            motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
            z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
        drag = self.drag_2 * self.v * self.v.norm(2, -1, True)
        a_next = self.act + self.dg - z_drag - drag
        self.p_old = self.p
        self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
        self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
        self.a = a_next

        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)


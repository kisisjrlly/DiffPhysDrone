"""CUDA-backed quadrotor environment for grayscale active sensing.

The environment retains generic geometric ray depth only as an internal
ray-intersection primitive. All legacy D455/active-stereo degradation logic has
been removed from this branch.
"""

import math
import random
import os
import sys

import torch
import torch.nn.functional as F

try:
    import quadsim_cuda
except ModuleNotFoundError:
    _src_dir = os.path.join(os.path.dirname(__file__), "src")
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    import quadsim_cuda

from autograd_ops import run
from render.ideal_gray import render_ideal_grayscale
from utils import g_decay


class Env:
    supported_scenarios = (
        "nominal",
        "dark",
        "bright",
        "dark_to_bright",
        "bright_to_dark",
    )

    def __init__(
        self,
        batch_size,
        width,
        height,
        grad_decay,
        device="cpu",
        fov_x_half_tan=0.82,
        eval_mode=False,
        cam_angle=5,
        ellipsoid_a=0.0,
        ellipsoid_c=0.0,
        camera_control_mode="learned",
        sensor_grad_mode="full",
        camera_ema_alpha=0.7,
        fixed_camera_exposure=0.35,
        fixed_camera_gain=0.15,
        fixed_random_exposure_min=0.10,
        fixed_random_exposure_max=0.90,
        fixed_random_gain_min=0.02,
        fixed_random_gain_max=0.90,
        scenarios=None,
        random_rotation=False,
        random_rotation_max_deg=45.0,
        simple_start_x=-1.5,
        simple_goal_x=1.5,
        simple_wall_x=0.0,
        simple_slit_center_y_min=-0.75,
        simple_slit_center_y_max=0.75,
        simple_slit_half_y=0.15,
        simple_slit_half_y_min=None,
        simple_slit_half_y_max=None,
        simple_slit_center_z=1.50,
        simple_back_wall_x_min=2.0,
        simple_back_wall_x_max=2.85,
        gray_nominal_ambient=0.18,
        gray_nominal_diffuse=0.72,
        gray_dark_scale=0.12,
        gray_bright_scale=2.2,
        gray_background_intensity=0.04,
        gray_transition_x=0.0,
        gray_transition_width=0.25,
        gray_light_jitter=0.08,
        gray_texture_strength=0.35,
        gray_texture_scale=5.0,
    ):
        self.device = device
        self.batch_size = int(batch_size)
        self.width = int(width)
        self.height = int(height)
        self.grad_decay = float(grad_decay)
        self.fov_x_half_tan = float(fov_x_half_tan)
        self._fov_x_half_tan = float(fov_x_half_tan)
        self.cam_angle = float(cam_angle)
        self.eval_mode = bool(eval_mode)

        self.ellipsoid_a = float(ellipsoid_a)
        self.ellipsoid_c = float(ellipsoid_c)
        self.use_ellipsoid = self.ellipsoid_a > 0 and self.ellipsoid_c > 0

        self.camera_control_mode = str(camera_control_mode).lower()
        self.sensor_grad_mode = str(sensor_grad_mode).lower()
        self.camera_ema_alpha = float(camera_ema_alpha)
        self.fixed_camera_exposure = float(fixed_camera_exposure)
        self.fixed_camera_gain = float(fixed_camera_gain)
        self.fixed_random_exposure_range = (
            float(fixed_random_exposure_min),
            float(fixed_random_exposure_max),
        )
        self.fixed_random_gain_range = (
            float(fixed_random_gain_min),
            float(fixed_random_gain_max),
        )

        self.scenarios = self._normalize_scenarios(scenarios)
        self.random_rotation = bool(random_rotation)
        self.random_rotation_max_rad = max(float(random_rotation_max_deg), 0.0) * math.pi / 180.0

        self.simple_start_x = float(simple_start_x)
        self.simple_goal_x = float(simple_goal_x)
        self.simple_wall_x = float(simple_wall_x)
        self.simple_slit_center_y_min = float(simple_slit_center_y_min)
        self.simple_slit_center_y_max = float(simple_slit_center_y_max)
        self.simple_slit_half_y = float(simple_slit_half_y)
        self.simple_slit_half_y_min = (
            self.simple_slit_half_y if simple_slit_half_y_min is None else float(simple_slit_half_y_min)
        )
        self.simple_slit_half_y_max = (
            self.simple_slit_half_y if simple_slit_half_y_max is None else float(simple_slit_half_y_max)
        )
        if self.simple_slit_half_y_max < self.simple_slit_half_y_min:
            self.simple_slit_half_y_min, self.simple_slit_half_y_max = (
                self.simple_slit_half_y_max,
                self.simple_slit_half_y_min,
            )
        self.simple_slit_center_z = float(simple_slit_center_z)
        self.simple_back_wall_x_min = float(simple_back_wall_x_min)
        self.simple_back_wall_x_max = float(simple_back_wall_x_max)
        self.simple_wall_half_x = 0.10
        self.simple_wall_half_z = 1.0

        self.gray_nominal_ambient = float(gray_nominal_ambient)
        self.gray_nominal_diffuse = float(gray_nominal_diffuse)
        self.gray_dark_scale = float(gray_dark_scale)
        self.gray_bright_scale = float(gray_bright_scale)
        self.gray_background_intensity = float(gray_background_intensity)
        self.gray_transition_x = float(gray_transition_x)
        self.gray_transition_width = float(gray_transition_width)
        self.gray_light_jitter = float(gray_light_jitter)
        self.gray_texture_strength = float(gray_texture_strength)
        self.gray_texture_scale = float(gray_texture_scale)

        self.g_std = torch.tensor([0.0, 0.0, -9.80665], device=device)
        self.v_wind_w = torch.tensor([1.0, 1.0, 0.2], device=device)
        self.sub_div = torch.linspace(0, 1.0 / 15.0, 10, device=device).reshape(-1, 1, 1)
        self.flow = torch.empty((self.batch_size, 0, self.height, self.width), device=device)

        self.fixed_max_speed = 1.15
        self.fixed_drone_radius = 0.12
        self.fixed_margin = 0.00
        self.fixed_pitch_ctl_delay = 12.0
        self.fixed_yaw_ctl_delay = 6.0
        self.fixed_drag_linear = 0.35
        self.fixed_wind_scale = 0.03

    def _normalize_scenarios(self, scenarios):
        if scenarios is None:
            return ["nominal"]
        out = []
        for raw in scenarios:
            name = str(raw).strip().lower().replace("-", "_")
            if name not in self.supported_scenarios:
                raise ValueError(f"unsupported grayscale scenario {raw!r}")
            if name not in out:
                out.append(name)
        return out or ["nominal"]

    def _choose_scene_names(self, B, scene_name=None):
        if scene_name is not None:
            name = str(scene_name).strip().lower().replace("-", "_")
            if name not in self.supported_scenarios:
                raise ValueError(f"unsupported grayscale scenario {scene_name!r}")
            return [name] * int(B)
        if self.eval_mode or len(self.scenarios) <= 1:
            return [random.choice(self.scenarios)] * int(B)
        names = [self.scenarios[i % len(self.scenarios)] for i in range(int(B))]
        random.shuffle(names)
        return names

    def _build_voxels(self, rows):
        if not rows:
            return torch.empty((0, 6), device=self.device)
        return torch.tensor(rows, device=self.device, dtype=torch.float32)

    def _choose_slit_centers(self, B):
        lo = min(self.simple_slit_center_y_min, self.simple_slit_center_y_max)
        hi = max(self.simple_slit_center_y_min, self.simple_slit_center_y_max)
        return torch.empty((B,), device=self.device).uniform_(lo, hi)

    def _choose_slit_half_widths(self, B):
        lo = min(self.simple_slit_half_y_min, self.simple_slit_half_y_max)
        hi = max(self.simple_slit_half_y_min, self.simple_slit_half_y_max)
        if abs(hi - lo) < 1e-9:
            return torch.full((B,), lo, device=self.device)
        return torch.empty((B,), device=self.device).uniform_(lo, hi)

    def _choose_back_wall_xs(self, B):
        lo = min(self.simple_back_wall_x_min, self.simple_back_wall_x_max)
        hi = max(self.simple_back_wall_x_min, self.simple_back_wall_x_max)
        if abs(hi - lo) < 1e-9:
            return torch.full((B,), lo, device=self.device)
        return torch.empty((B,), device=self.device).uniform_(lo, hi)

    def _build_wall_slit_voxel_layout(self, slit_center_y, slit_half_y, back_wall_x):
        wall_half_y = 1.0
        wall_half_z = self.simple_wall_half_z * 2
        wall_thickness = self.simple_wall_half_x
        back_wall_half_y = 3.0
        return self._build_voxels([
            [
                self.simple_wall_x,
                float(slit_center_y) - float(slit_half_y) - wall_half_y,
                self.simple_slit_center_z,
                wall_thickness,
                wall_half_y,
                wall_half_z,
            ],
            [
                self.simple_wall_x,
                float(slit_center_y) + float(slit_half_y) + wall_half_y,
                self.simple_slit_center_z,
                wall_thickness,
                wall_half_y,
                wall_half_z,
            ],
            [
                float(back_wall_x),
                0.0,
                self.simple_slit_center_z,
                wall_thickness,
                back_wall_half_y,
                wall_half_z,
            ],
        ])

    @staticmethod
    def _rotation_z(yaw):
        c, s = torch.cos(yaw), torch.sin(yaw)
        z, o = torch.zeros_like(yaw), torch.ones_like(yaw)
        return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(-1, 3, 3)

    def reset(self, scene_name=None):
        B, device = self.batch_size, self.device
        self.current_scene_names = self._choose_scene_names(B, scene_name=scene_name)
        self.current_scene_name = (
            self.current_scene_names[0]
            if all(x == self.current_scene_names[0] for x in self.current_scene_names)
            else "mixed"
        )

        cam_angle = torch.full((B,), self.cam_angle * math.pi / 180.0, device=device)
        zeros, ones = torch.zeros_like(cam_angle), torch.ones_like(cam_angle)
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
            zeros, ones, zeros,
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),
        ], -1).reshape(B, 3, 3)

        if self.random_rotation and self.random_rotation_max_rad > 0:
            yaw = (torch.rand(B, device=device) * 2.0 - 1.0) * self.random_rotation_max_rad
        else:
            yaw = torch.zeros(B, device=device)
        self.scene_yaw = yaw
        self.R_scene = self._rotation_z(yaw)
        self.R_scene_T = self.R_scene.transpose(1, 2).contiguous()

        self.slit_center_y = self._choose_slit_centers(B)
        self.slit_half_y = self._choose_slit_half_widths(B)
        self.back_wall_x = self._choose_back_wall_xs(B)
        self.voxels = torch.stack([
            self._build_wall_slit_voxel_layout(y, w, bx)
            for y, w, bx in zip(self.slit_center_y, self.slit_half_y, self.back_wall_x)
        ])

        start_local = torch.stack([
            torch.full((B,), self.simple_start_x, device=device),
            torch.zeros(B, device=device),
            torch.full((B,), self.simple_slit_center_z, device=device),
        ], -1)
        goal_local = torch.tensor(
            [self.simple_goal_x, 0.0, self.simple_slit_center_z],
            device=device,
        ).expand(B, 3).clone()
        self.p = torch.bmm(self.R_scene, start_local[:, :, None])[:, :, 0]
        self.p_target = torch.bmm(self.R_scene, goal_local[:, :, None])[:, :, 0]

        jitter = self.gray_light_jitter
        if jitter > 0:
            self.gray_light_factor = 1.0 + (torch.rand(B, device=device) * 2.0 - 1.0) * jitter
        else:
            self.gray_light_factor = torch.ones(B, device=device)

        self.n_drones_per_group = 1
        self.drone_radius = self.fixed_drone_radius
        self.max_speed = torch.full((B, 1), self.fixed_max_speed, device=device)
        self.margin = torch.full((B,), self.fixed_margin, device=device)
        self.pitch_ctl_delay = torch.full((B, 1), self.fixed_pitch_ctl_delay, device=device)
        self.yaw_ctl_delay = torch.full((B, 1), self.fixed_yaw_ctl_delay, device=device)
        self.drag_2 = torch.zeros((B, 2), device=device)
        self.drag_2[:, 1] = self.fixed_drag_linear
        self.z_drag_coef = torch.ones((B, 1), device=device)
        self.thr_est_error = torch.ones((B,), device=device)

        self.balls = torch.empty((B, 0, 4), device=device)
        self.cyl = torch.empty((B, 0, 3), device=device)
        self.cyl_h = torch.empty((B, 0, 3), device=device)
        self.v = torch.zeros((B, 3), device=device)
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w * self.fixed_wind_scale
        self.act = torch.zeros((B, 3), device=device)
        self.a = torch.zeros((B, 3), device=device)
        self.dg = torch.randn((B, 3), device=device) * 0.03

        R0 = torch.zeros((B, 3, 3), device=device)
        v_dir = F.normalize(self.p_target - self.p, 2, -1)
        self.R = quadsim_cuda.update_state_vec(
            R0, self.act, v_dir, torch.zeros_like(self.yaw_ctl_delay), 5
        )
        self.R_old = self.R.clone()
        self.p_old = self.p.clone()

    def _scenario_light_scale(self, local_x):
        scales = torch.ones_like(local_x)
        blend = torch.sigmoid(
            (local_x - self.gray_transition_x) / max(self.gray_transition_width, 1e-4)
        )
        for idx, name in enumerate(self.current_scene_names):
            if name == "dark":
                scales[idx] = self.gray_dark_scale
            elif name == "bright":
                scales[idx] = self.gray_bright_scale
            elif name == "dark_to_bright":
                scales[idx] = self.gray_dark_scale + (
                    self.gray_bright_scale - self.gray_dark_scale
                ) * blend[idx]
            elif name == "bright_to_dark":
                scales[idx] = self.gray_bright_scale + (
                    self.gray_dark_scale - self.gray_bright_scale
                ) * blend[idx]
        return scales * self.gray_light_factor

    def get_scene_effects_for_env(self, env_idx=0):
        idx = int(min(max(env_idx, 0), self.batch_size - 1))
        p_local = torch.bmm(self.R_scene_T, self.p[:, :, None])[:, :, 0]
        scale = self._scenario_light_scale(p_local[:, 0])
        return {
            "geometry_kind": "single_wall_slit",
            "illumination_scenario": self.current_scene_names[idx],
            "slit_center_y": float(self.slit_center_y[idx].detach().cpu()),
            "slit_half_y": float(self.slit_half_y[idx].detach().cpu()),
            "back_wall_x": float(self.back_wall_x[idx].detach().cpu()),
            "light_scale": float(scale[idx].detach().cpu()),
        }

    def render_gray_ideal(self, *, return_aux=False, **renderer_kwargs):
        B = self.batch_size
        render_R = torch.bmm(self.R_scene_T, self.R @ self.R_cam).contiguous()
        render_p = torch.bmm(self.R_scene_T, self.p[:, :, None])[:, :, 0].contiguous()
        depth = torch.empty(
            (B, self.height, self.width),
            device=self.p.device,
            dtype=self.p.dtype,
        )
        quadsim_cuda.render_depth(
            depth,
            self.balls,
            self.cyl,
            self.cyl_h,
            self.voxels,
            render_R,
            render_p,
            self.n_drones_per_group,
            self._fov_x_half_tan,
        )

        light_scale = self._scenario_light_scale(render_p[:, 0])
        gray, aux = render_ideal_grayscale(
            depth,
            render_R,
            render_p,
            fov_x_half_tan=self._fov_x_half_tan,
            ambient=self.gray_nominal_ambient * light_scale,
            diffuse=self.gray_nominal_diffuse * light_scale,
            background_intensity=self.gray_background_intensity * light_scale,
            texture_strength=self.gray_texture_strength,
            texture_scale=self.gray_texture_scale,
            return_aux=return_aux,
            **renderer_kwargs,
        )
        if return_aux:
            aux = dict(aux or {})
            aux["geometry_depth"] = depth.detach()
            aux["light_scale"] = light_scale.detach()
        return gray, aux

    def find_vec_to_nearest_pt(self):
        p_world = self.p + self.v * self.sub_div
        p = torch.matmul(
            self.R_scene_T.unsqueeze(0),
            p_world.unsqueeze(-1),
        ).squeeze(-1).contiguous()
        nearest_pt = torch.empty_like(p)
        if self.use_ellipsoid:
            R_local = torch.bmm(self.R_scene_T, self.R).contiguous()
            quadsim_cuda.find_nearest_pt_ellipsoid(
                nearest_pt,
                self.balls,
                self.cyl,
                self.cyl_h,
                self.voxels,
                p,
                R_local,
                self.drone_radius,
                self.n_drones_per_group,
                self.ellipsoid_a,
                self.ellipsoid_c,
            )
        else:
            quadsim_cuda.find_nearest_pt(
                nearest_pt,
                self.balls,
                self.cyl,
                self.cyl_h,
                self.voxels,
                p,
                self.drone_radius,
                self.n_drones_per_group,
            )
        vec_local = nearest_pt - p
        return torch.matmul(
            self.R_scene.unsqueeze(0),
            vec_local.unsqueeze(-1),
        ).squeeze(-1)

    def get_scene_yaw_for_env(self, env_idx=0):
        idx = int(min(max(env_idx, 0), self.batch_size - 1))
        return float(self.scene_yaw[idx].detach().cpu())

    def get_world_voxels_for_env(self, env_idx=0):
        idx = int(min(max(env_idx, 0), self.batch_size - 1))
        return self.voxels[idx].detach().cpu().numpy()

    def get_world_balls_for_env(self, env_idx=0):
        idx = int(min(max(env_idx, 0), self.batch_size - 1))
        balls = self.balls[idx].detach()
        if balls.numel() == 0:
            return balls.cpu().numpy()
        centers = torch.matmul(self.R_scene[idx], balls[:, :3].T).T
        out = balls.clone()
        out[:, :3] = centers
        return out.cpu().numpy()

    def get_world_cyl_for_env(self, env_idx=0):
        idx = int(min(max(env_idx, 0), self.batch_size - 1))
        cyl = self.cyl[idx].detach()
        if cyl.numel() == 0:
            return cyl.cpu().numpy()
        centers = torch.cat(
            [cyl[:, :2], torch.zeros((cyl.shape[0], 1), device=cyl.device)],
            dim=-1,
        )
        centers = torch.matmul(self.R_scene[idx], centers.T).T
        out = cyl.clone()
        out[:, :2] = centers[:, :2]
        return out.cpu().numpy()

    def get_world_cyl_h_for_env(self, env_idx=0):
        idx = int(min(max(env_idx, 0), self.batch_size - 1))
        cyl_h = self.cyl_h[idx].detach()
        if cyl_h.numel() == 0:
            return cyl_h.cpu().numpy()
        centers = torch.stack(
            [cyl_h[:, 0], torch.zeros_like(cyl_h[:, 0]), cyl_h[:, 1]],
            dim=-1,
        )
        centers = torch.matmul(self.R_scene[idx], centers.T).T
        out = cyl_h.clone()
        out[:, 0] = centers[:, 0]
        out[:, 1] = centers[:, 2]
        return out.cpu().numpy()

    def run(self, act_pred, ctl_dt=1 / 15, v_pred=None):
        self.dg = (
            self.dg * math.sqrt(max(1 - ctl_dt / 4, 0.0))
            + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        )
        self.p_old = self.p
        self.act, self.p, self.v, self.a = run(
            self.R,
            self.dg,
            self.z_drag_coef,
            self.drag_2,
            self.pitch_ctl_delay,
            act_pred,
            self.act,
            self.p,
            self.v,
            self.v_wind,
            self.a,
            self.grad_decay,
            ctl_dt,
            0.5,
        )
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        if v_pred is None:
            v_pred = self.p_target - self.p
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)

    def save_state(self):
        return {
            "p": self.p.clone(),
            "v": self.v.clone(),
            "a": self.a.clone(),
            "act": self.act.clone(),
            "R": self.R.clone(),
            "R_old": self.R_old.clone(),
            "p_old": self.p_old.clone(),
            "dg": self.dg.clone(),
            "v_wind": self.v_wind.clone(),
        }

    def restore_state(self, snapshot):
        for key in ("p", "v", "a", "act", "R", "R_old", "p_old", "dg", "v_wind"):
            setattr(self, key, snapshot[key].clone())

    def _run(self, act_pred, ctl_dt=1 / 15, v_pred=None):
        alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
        self.act = act_pred * (1 - alpha) + self.act * alpha
        self.dg = (
            self.dg * math.sqrt(max(1 - ctl_dt, 0.0))
            + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
        )
        z_drag = 0
        if self.z_drag_coef is not None:
            v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
            v_prep = self.v - v_up
            motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
            z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
        drag = self.drag_2 * self.v * self.v.norm(2, -1, True)
        a_next = self.act + self.dg - z_drag - drag
        self.p_old = self.p
        self.p = (
            g_decay(self.p, self.grad_decay ** ctl_dt)
            + self.v * ctl_dt
            + 0.5 * self.a * ctl_dt ** 2
        )
        self.v = (
            g_decay(self.v, self.grad_decay ** ctl_dt)
            + (self.a + a_next) / 2 * ctl_dt
        )
        self.a = a_next
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        if v_pred is None:
            v_pred = self.p_target - self.p
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)

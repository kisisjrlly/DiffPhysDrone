import math
import random
import os
import sys

import torch
import torch.nn.functional as F
try:
    import quadsim_cuda
except ModuleNotFoundError:
    _src_dir = os.path.join(os.path.dirname(__file__), 'src')
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    import quadsim_cuda

from autograd_ops import run, active_sensing_sensor
from camera_semantics import CameraSemantics
from utils import g_decay


class Env:
    """Minimal shared-gate active-sensing environment.

    Geometry is fixed across scenarios: start -> occluder -> gate wall with one
    slot -> goal.  Scenarios only change the local sensor degradation around
    the visible opening/gate material.
    """

    supported_scenarios = ('glare', 'specular', 'dark')
    supported_slots = ('far_left', 'left', 'right', 'far_right')
    slot_y = {
        'far_left': -1.12,
        'left': -0.56,
        'right': 0.56,
        'far_right': 1.12,
    }

    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.82,
                 eval_mode=False, cam_angle=5, ellipsoid_a=0.0, ellipsoid_c=0.0,
                 cam_power_baseline=0.5, camera_control_mode='learned', sensor_grad_mode='full',
                 fixed_camera_power=0.4, fixed_camera_exposure=0.9, fixed_camera_gain=0.65,
                 fixed_random_power_min=0.1, fixed_random_power_max=0.95,
                 fixed_random_exposure_min=0.1, fixed_random_exposure_max=0.92,
                 fixed_random_gain_min=0.02, fixed_random_gain_max=0.9,
                 cam_exposure_t_min=0.25, cam_exposure_t_span=2.75,
                 cam_exposure_eff_min=0.25, cam_exposure_eff_max=3.0,
                 cam_iso_gain_base=1.0, cam_iso_gain_scale=0.8, cam_iso_gain_gamma=0.6,
                 cam_shot_noise_base=0.01, depth_min_valid=0.3, depth_max_range=6.0,
                 scenarios=None, sun_glare_eval_slot=None, diff_sensor_impl=None,
                 random_rotation=False, random_rotation_max_deg=45.0,
                 **_ignored) -> None:
        self.device = device
        self.batch_size = int(batch_size)
        self.width = int(width)
        self.height = int(height)
        self.grad_decay = float(grad_decay)
        self.depth_min_valid = max(float(depth_min_valid), 1e-3)
        self.depth_max_range = max(float(depth_max_range), self.depth_min_valid + 1e-3)
        self.fov_x_half_tan = float(fov_x_half_tan)
        self._fov_x_half_tan = float(fov_x_half_tan)
        self.cam_angle = float(cam_angle)
        self.eval_mode = bool(eval_mode)
        self.ellipsoid_a = float(ellipsoid_a)
        self.ellipsoid_c = float(ellipsoid_c)
        self.use_ellipsoid = self.ellipsoid_a > 0 and self.ellipsoid_c > 0

        self.g_std = torch.tensor([0.0, 0.0, -9.80665], device=device)
        self.v_wind_w = torch.tensor([1.0, 1.0, 0.2], device=device)
        self.sub_div = torch.linspace(0, 1. / 15, 10, device=device).reshape(-1, 1, 1)
        self.flow = torch.empty((self.batch_size, 0, self.height, self.width), device=device)

        self.cam_power_baseline = float(cam_power_baseline)
        self.camera_control_mode = str(camera_control_mode).lower()
        self.sensor_grad_mode = str(sensor_grad_mode).lower()
        self.fixed_camera_power = float(fixed_camera_power)
        self.fixed_camera_exposure = float(fixed_camera_exposure)
        self.fixed_camera_gain = float(fixed_camera_gain)
        self.fixed_random_power_range = (float(fixed_random_power_min), float(fixed_random_power_max))
        self.fixed_random_exposure_range = (float(fixed_random_exposure_min), float(fixed_random_exposure_max))
        self.fixed_random_gain_range = (float(fixed_random_gain_min), float(fixed_random_gain_max))
        self.cam_sem = CameraSemantics(
            exposure_t_min=float(cam_exposure_t_min),
            exposure_t_span=float(cam_exposure_t_span),
            exposure_eff_min=float(cam_exposure_eff_min),
            exposure_eff_max=float(cam_exposure_eff_max),
            iso_gain_base=float(cam_iso_gain_base),
            iso_gain_scale=float(cam_iso_gain_scale),
            iso_gain_gamma=float(cam_iso_gain_gamma),
            shot_noise_base=float(cam_shot_noise_base),
        )

        self.scenarios = self._normalize_scenarios(scenarios)
        self.scene_name_to_id = {name: idx for idx, name in enumerate(self.supported_scenarios)}
        self.current_scene_name = self.scenarios[0]
        self.current_scene_id = self.scene_name_to_id[self.current_scene_name]
        self.sun_glare_eval_slot = self._canonical_slot(sun_glare_eval_slot)
        self.random_rotation = bool(random_rotation)
        self.random_rotation_max_rad = max(float(random_rotation_max_deg), 0.0) * math.pi / 180.0
        self.current_scene_effects = {}
        self.last_diff_depth_debug = None
        self.last_diff_depth_train_aux = None

        impl = {'diff_depth': 'cuda'}
        if diff_sensor_impl is not None:
            impl.update({str(k): str(v).lower() for k, v in dict(diff_sensor_impl).items()})
        self.diff_sensor_impl = impl

        self.fixed_max_speed = 1.15
        self.fixed_drone_radius = 0.12
        self.fixed_margin = 0.05
        self.fixed_pitch_ctl_delay = 12.0
        self.fixed_yaw_ctl_delay = 6.0
        self.fixed_drag_linear = 0.35
        self.fixed_wind_scale = 0.03

    def _build_voxels(self, rows):
        if not rows:
            return torch.empty((0, 6), device=self.device)
        return torch.tensor(rows, device=self.device, dtype=torch.float32)

    def _normalize_scenarios(self, scenarios):
        if scenarios is None:
            return list(self.supported_scenarios)
        out = []
        for raw in scenarios:
            name = str(raw).strip().lower().replace('-', '_')
            if name not in self.supported_scenarios:
                raise ValueError(f'unsupported scene {raw!r}')
            if name not in out:
                out.append(name)
        return out or list(self.supported_scenarios)

    def _canonical_slot(self, slot):
        if slot is None:
            return None
        name = str(slot).strip().lower().replace('-', '_')
        aliases = {'fl': 'far_left', 'farleft': 'far_left', 'l': 'left', 'r': 'right', 'fr': 'far_right', 'farright': 'far_right'}
        name = aliases.get(name, name)
        if name not in self.supported_slots:
            raise ValueError(f'unsupported slot {slot!r}')
        return name

    def _choose_scene_name(self, scene_name=None):
        if scene_name is not None:
            name = str(scene_name).strip().lower().replace('-', '_')
            if name not in self.supported_scenarios:
                raise ValueError(f'unsupported scene {scene_name!r}')
            return name
        return random.choice(self.scenarios)

    def _choose_slots(self, B):
        if self.eval_mode and self.sun_glare_eval_slot is not None:
            return [self.sun_glare_eval_slot] * B
        slots = list(self.supported_slots)
        return [slots[i % len(slots)] for i in range(B)]

    def _build_sun_glare_voxel_layout(self, gap_y_center, *, occluder_x=0.88,
                                      occluder_half_y=0.48, divider_x=1.58,
                                      gate_x=1.82, gap_half_w=0.18):
        """
        Probe-then-commit shared gate map.

        This restores the original geometry: a central occluder, three thin
        divider fins, a four-slot gate wall, and a back wall after the opening.
        The public scene name only changes the local sensor degradation near
        the opening; geometry stays identical for glare/specular/dark.
        """
        voxel_half_w = 0.25
        voxel_half_h = 1.5

        guide = self._build_voxels([
            [-1.65, -1.48, 1.5, voxel_half_w, voxel_half_w, voxel_half_h],
            [-1.65,  1.48, 1.5, voxel_half_w, voxel_half_w, voxel_half_h],
        ])
        occluder = self._build_voxels([
            [float(occluder_x), 0.00, 1.5, 0.10, float(occluder_half_y), voxel_half_h],
        ])
        lane_dividers = self._build_voxels([
            [float(divider_x), -0.84, 1.5, 0.22, 0.05, voxel_half_h],
            [float(divider_x),  0.00, 1.5, 0.22, 0.05, voxel_half_h],
            [float(divider_x),  0.84, 1.5, 0.22, 0.05, voxel_half_h],
        ])
        big = 4.0
        wall_thickness = 0.15
        gap_z_center = 1.50
        gap_half_h = 1.05
        gate_wall = self._build_voxels([
            [float(gate_x), gap_y_center - float(gap_half_w) - big, gap_z_center, wall_thickness, big, big],
            [float(gate_x), gap_y_center + float(gap_half_w) + big, gap_z_center, wall_thickness, big, big],
            [float(gate_x), gap_y_center, gap_z_center + gap_half_h + big, wall_thickness, float(gap_half_w), big],
            [float(gate_x), gap_y_center, gap_z_center - gap_half_h - big, wall_thickness, float(gap_half_w), big],
        ])
        back_wall = self._build_voxels([
            [float(gate_x) + 1.83, 0.00, 1.5, 0.10, 1.30, voxel_half_h],
        ])
        return torch.cat([guide, occluder, lane_dividers, gate_wall, back_wall], dim=0)

    def _rotation_z(self, yaw):
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        z = torch.zeros_like(yaw)
        o = torch.ones_like(yaw)
        return torch.stack([
            c, -s, z,
            s,  c, z,
            z,  z, o,
        ], -1).reshape(-1, 3, 3)

    def _scene_effects(self, scene_name, slot_name, gap_y):
        regime_id = float(self.scene_name_to_id[scene_name])
        return {
            'sensor_regime_name': scene_name,
            'sensor_regime_id': regime_id,
            'decision_open_slot_name': slot_name,
            'decision_open_slot_id': float(self.supported_slots.index(slot_name)),
            'decision_open_slot_y': float(gap_y),
            'hazard_center': [1.82, float(gap_y), 1.5],
            'hazard_half_y': 0.18,
            'hazard_half_z': 1.20,
            'hazard_softness': 0.045,
            'geometry_occluder_x': 0.88,
            'geometry_divider_x': 1.58,
            'geometry_gate_x': 1.82,
        }

    def _merge_batch_effects(self, effects_list):
        keys = effects_list[0].keys()
        out = {}
        for key in keys:
            vals = [fx[key] for fx in effects_list]
            if all(isinstance(v, (int, float)) for v in vals):
                out[key] = torch.tensor(vals, device=self.device, dtype=torch.float32)
            elif key in {'hazard_center'}:
                out[key] = torch.tensor(vals, device=self.device, dtype=torch.float32)
            elif all(v == vals[0] for v in vals):
                out[key] = vals[0]
            else:
                out[key] = vals
        return out

    def _set_scene_name(self, scene_name):
        self.current_scene_name = str(scene_name)
        self.current_scene_id = self.scene_name_to_id[self.current_scene_name]

    def reset(self, scene_name=None):
        B, device = self.batch_size, self.device
        scene_name = self._choose_scene_name(scene_name)
        self._set_scene_name(scene_name)
        self.last_diff_depth_debug = None
        self.last_diff_depth_train_aux = None

        cam_angle = torch.full((B,), float(self.cam_angle) * math.pi / 180.0, device=device)
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
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

        slots = self._choose_slots(B)
        gap_y = torch.tensor([self.slot_y[s] for s in slots], device=device, dtype=torch.float32)
        voxels = torch.stack([self._build_sun_glare_voxel_layout(float(y)) for y in gap_y], dim=0)
        start_y = torch.zeros(B, device=device)
        start_local = torch.stack([torch.full((B,), -2.8, device=device), start_y, torch.full((B,), 1.5, device=device)], -1)
        goal_local = torch.tensor([3.0, 0.0, 1.5], device=device).expand(B, 3).clone()
        start = torch.bmm(self.R_scene, start_local[:, :, None])[:, :, 0]
        goal = torch.bmm(self.R_scene, goal_local[:, :, None])[:, :, 0]
        effects = self._merge_batch_effects([self._scene_effects(scene_name, slots[i], float(gap_y[i])) for i in range(B)])
        local_hazard = effects['hazard_center'].to(device=device, dtype=torch.float32)
        effects['hazard_center_local'] = local_hazard.clone()
        effects['hazard_center'] = torch.bmm(self.R_scene, local_hazard[:, :, None])[:, :, 0]
        effects['scene_yaw'] = yaw
        effects['geometry_start_local'] = start_local
        effects['geometry_goal_local'] = goal_local
        effects['geometry_start'] = start
        effects['geometry_goal'] = goal
        self.current_scene_effects = effects

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
        self.voxels = voxels
        self.p = start
        self.p_target = goal
        self.v = torch.zeros((B, 3), device=device)
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w * self.fixed_wind_scale
        self.act = torch.zeros((B, 3), device=device)
        self.a = torch.zeros((B, 3), device=device)
        self.dg = torch.randn((B, 3), device=device) * 0.03

        R0 = torch.zeros((B, 3, 3), device=device)
        v_dir = F.normalize(self.p_target - self.p, 2, -1)
        self.R = quadsim_cuda.update_state_vec(R0, self.act, v_dir, torch.zeros_like(self.yaw_ctl_delay), 5)
        self.R_old = self.R.clone()
        self.p_old = self.p.clone()

    def get_scene_effects_for_env(self, env_idx=0):
        idx = int(min(max(env_idx, 0), self.batch_size - 1))
        out = {}
        for key, value in (self.current_scene_effects or {}).items():
            if torch.is_tensor(value):
                v = value.detach()
                if v.ndim >= 2 and v.shape[0] == self.batch_size:
                    v = v[idx]
                elif v.ndim == 1 and v.shape[0] == self.batch_size:
                    v = v[idx]
                out[key] = float(v.cpu().item()) if v.ndim == 0 else v.cpu().tolist()
            elif isinstance(value, list) and len(value) == self.batch_size:
                out[key] = value[idx]
            else:
                out[key] = value
        return out

    def _store_last_diff_depth_debug(self, debug):
        stored = {}
        for key, value in (debug or {}).items():
            if isinstance(value, dict):
                stored[key] = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in value.items()}
            else:
                stored[key] = value.detach() if torch.is_tensor(value) else value
        self.last_diff_depth_debug = stored

    def _store_last_diff_depth_train_aux(self, aux):
        self.last_diff_depth_train_aux = aux

    def get_last_diff_depth_train_aux(self):
        return self.last_diff_depth_train_aux or {}

    def export_last_diff_depth_debug(self, env_idx=0):
        debug = self.last_diff_depth_debug or {}
        out = {'scene_name': str(debug.get('scene_name', self.current_scene_name)), 'images': {}, 'scalars': {}}
        for key in ('raw_depth_map', 'quality_map', 'valid_prob_map', 'hard_valid_map', 'invalid_mask', 'scene_effect_map', 'scene_mask'):
            value = debug.get(key, None)
            if torch.is_tensor(value) and value.ndim >= 3 and value.shape[0] > 0:
                idx = int(min(max(env_idx, 0), value.shape[0] - 1))
                out['images'][key] = value[idx].detach().cpu().numpy()
        for key, value in (debug.get('scalars', {}) or {}).items():
            if torch.is_tensor(value):
                if value.ndim == 0:
                    out['scalars'][key] = float(value.detach().cpu().item())
                elif value.shape[0] > 0:
                    idx = int(min(max(env_idx, 0), value.shape[0] - 1))
                    out['scalars'][key] = float(value[idx].detach().cpu().item())
            elif isinstance(value, (int, float)):
                out['scalars'][key] = float(value)
        return out

    def _scene_mask(self, depth):
        B, H, W = depth.shape
        effects = self.current_scene_effects
        center = effects['hazard_center']
        half_y = effects['hazard_half_y']
        half_z = effects['hazard_half_z']
        softness = effects['hazard_softness']
        if not torch.is_tensor(half_y):
            half_y = torch.full((B,), float(half_y), device=depth.device)
        if not torch.is_tensor(half_z):
            half_z = torch.full((B,), float(half_z), device=depth.device)
        if not torch.is_tensor(softness):
            softness = torch.full((B,), float(softness), device=depth.device)
        center = center.to(depth.device, depth.dtype)

        ys = torch.linspace(-1.0, 1.0, W, device=depth.device, dtype=depth.dtype)
        zs = torch.linspace(-1.0, 1.0, H, device=depth.device, dtype=depth.dtype)
        yy, zz = torch.meshgrid(ys, zs, indexing='xy')
        yy = yy.unsqueeze(0)
        zz = zz.unsqueeze(0)

        # Project the local sensor-degradation region into the current camera.
        # This replaces the older slot_y -> image_x approximation.  The mask is
        # deliberately detached from pose so the sensor model only exposes
        # camera-parameter gradients; raw geometry remains non-differentiable.
        R_cam_world = (self.R @ self.R_cam).detach().to(depth.device, depth.dtype)
        pos = self.p.detach().to(depth.device, depth.dtype)
        rel = center.detach() - pos
        cam = torch.bmm(R_cam_world.transpose(1, 2), rel[:, :, None])[:, :, 0]
        x = cam[:, 0]
        x_safe = x.clamp_min(0.20)
        fov_x = torch.as_tensor(float(self._fov_x_half_tan), device=depth.device, dtype=depth.dtype)
        fov_y = fov_x * float(H) / float(max(W, 1))

        cy = ((-cam[:, 1] / x_safe) / fov_x).clamp(-1.5, 1.5)[:, None, None]
        cz = ((-cam[:, 2] / x_safe) / fov_y).clamp(-1.5, 1.5)[:, None, None]
        sy = (half_y.to(depth.device, depth.dtype) / x_safe / fov_x).clamp(0.04, 1.25)[:, None, None]
        sz = (half_z.to(depth.device, depth.dtype) / x_safe / fov_y).clamp(0.06, 1.25)[:, None, None]
        soft = (softness.to(depth.device, depth.dtype) / x_safe / fov_x).clamp(0.025, 0.18)[:, None, None]
        front_gate = torch.sigmoid((x - 0.08) / 0.04)[:, None, None]
        mask_y = torch.sigmoid((sy - (yy - cy).abs()) / soft)
        mask_z = torch.sigmoid((sz - (zz - cz).abs()) / soft)
        return (mask_y * mask_z * front_gate).clamp(0.0, 1.0)

    def _sensor_reference(self, depth, power, exposure, gain, max_range=None):
        max_range = float(self.depth_max_range if max_range is None else max_range)
        min_valid = float(self.depth_min_valid)
        raw = depth.clamp(min_valid, max_range)
        mask = self._scene_mask(raw)
        p = power.clamp(0, 1)[:, None, None]
        e01 = exposure.clamp(0, 1)[:, None, None]
        g01 = gain.clamp(0, 1)[:, None, None]
        exposure_t = self.cam_sem.exposure_to_time(exposure).clamp_min(1e-6)[:, None, None]
        gain_scale = self.cam_sem.iso_to_gain(gain).clamp_min(1e-6)[:, None, None]
        speed = self.v.norm(2, -1).detach().clamp_min(0.0)[:, None, None]
        regime = float(self.current_scene_id)

        d4 = raw[:, None]
        d_far = F.max_pool2d(d4, 3, stride=1, padding=1)[:, 0]
        d_near = -F.max_pool2d(-d4, 3, stride=1, padding=1)[:, 0]
        edge = ((d_far - d_near) / (raw + 0.18)).clamp(0.0, 1.0)

        dist = raw / max(max_range, 1e-6)
        active_signal = 1.70 * p * exposure_t / (raw.square() + 0.75)
        passive_signal = 0.10 * exposure_t * torch.sqrt(gain_scale)
        signal = active_signal + passive_signal
        ambient_ir = 0.18 + 0.55 * mask
        motion = (speed * exposure_t * 0.075).clamp(0.0, 1.6)
        washout = ambient_ir * exposure_t / (active_signal + 0.20)
        noise_proxy = float(self.cam_sem.shot_noise_base) * (0.45 + 0.18 * gain_scale) / (signal + 0.08)
        snr = signal / (0.18 + 0.55 * ambient_ir + 0.38 * noise_proxy + 0.45 * motion * (0.20 + edge))
        quality = torch.sigmoid(
            2.15 * snr
            - 0.95 * washout
            - 0.85 * edge
            - 1.45 * torch.relu(dist - 0.92)
        )
        effect = torch.zeros_like(raw)

        if regime == 0.0:  # glare: exposure overexposes local IR/sun patch; power helps recover.
            overexp = torch.sigmoid((e01 - 0.20) / 0.055)
            rescue = torch.sigmoid((p - 0.50) / 0.09)
            penalty = mask * overexp * (0.78 - 0.38 * rescue)
            bonus = mask * rescue * (1.0 - overexp) * 0.18
            quality = quality - penalty + bonus
            effect = penalty
        elif regime == 1.0:  # specular: high projector power washes out reflective gate material.
            wash = torch.sigmoid((p - 0.30) / 0.055) * (0.62 + 0.38 * torch.sigmoid((e01 - 0.22) / 0.07))
            safe = torch.sigmoid((0.40 - p) / 0.075)
            penalty = mask * wash * 1.08
            bonus = mask * safe * 0.30
            quality = quality - penalty + bonus
            effect = penalty
        else:  # dark: low-reflectance frame needs exposure/gain to lift the return signal.
            rescue = (
                torch.sigmoid((e01 - 0.36) / 0.08) * 0.55
                + torch.sigmoid((g01 - 0.32) / 0.09) * 0.45
            ).clamp(max=1.0)
            need = mask * 0.68
            penalty = need * (1.0 - rescue)
            quality = quality - penalty + mask * rescue * 0.24
            effect = penalty

        quality = quality.clamp(0.0, 1.0)
        quality_pre_valid = quality

        valid_prob = torch.sigmoid((quality - 0.42) / 0.055)
        hard_valid = (valid_prob > 0.5).to(raw.dtype)
        valid_st = hard_valid.detach() - valid_prob.detach() + valid_prob
        depth_obs = raw * valid_st
        quality_obs = quality * valid_st

        invalid = (1.0 - valid_st).clamp(0.0, 1.0)
        mask_mass = mask.sum(dim=(-2, -1)).clamp_min(1e-6)
        scalars = {
            'quality_mean': quality_obs.mean(dim=(-2, -1)),
            'invalid_rate': invalid.mean(dim=(-2, -1)),
            'scene_effect_mean': effect.mean(dim=(-2, -1)),
            'scene_mask_mean': mask.mean(dim=(-2, -1)),
            'glare_quality_mean': (quality_obs * mask).sum(dim=(-2, -1)) / mask_mass,
            'glare_invalid_rate': (invalid * mask).sum(dim=(-2, -1)) / mask_mass,
        }
        self._store_last_diff_depth_train_aux({
            'quality_pre_valid': quality_pre_valid,
            'valid_prob_map': valid_prob,
            'hard_valid_map': hard_valid,
            'valid_st_map': valid_st,
        })
        self._store_last_diff_depth_debug({
            'scene_name': self.current_scene_name,
            'raw_depth_map': raw,
            'quality_pre_valid': quality_pre_valid,
            'quality_map': quality_obs,
            'valid_prob_map': valid_prob,
            'hard_valid_map': hard_valid,
            'invalid_mask': invalid,
            'scene_effect_map': effect,
            'scene_mask': mask,
            'scalars': scalars,
        })
        return depth_obs, quality_obs

    def _sensor_cuda(self, depth, power, exposure, gain):
        max_range = float(self.depth_max_range)
        min_valid = float(self.depth_min_valid)
        raw = depth.clamp(min_valid, max_range)
        mask = self._scene_mask(raw)
        regime_id = int(self.current_scene_id)
        speed = self.v.norm(2, -1).detach().to(raw.dtype)
        depth_obs, quality_obs, quality, valid_prob, hard_valid, effect = active_sensing_sensor(
            raw, mask, power, exposure, gain, speed,
            regime_id, min_valid, max_range,
            self.cam_sem.exposure_t_min, self.cam_sem.exposure_t_span,
            self.cam_sem.iso_gain_base, self.cam_sem.iso_gain_scale,
            self.cam_sem.iso_gain_gamma, self.cam_sem.shot_noise_base)
        valid_st = hard_valid.detach() - valid_prob.detach() + valid_prob
        invalid = (1.0 - valid_st).clamp(0.0, 1.0)
        mask_mass = mask.sum(dim=(-2, -1)).clamp_min(1e-6)
        scalars = {
            'quality_mean': quality_obs.mean(dim=(-2, -1)),
            'invalid_rate': invalid.mean(dim=(-2, -1)),
            'scene_effect_mean': effect.mean(dim=(-2, -1)),
            'scene_mask_mean': mask.mean(dim=(-2, -1)),
            'glare_quality_mean': (quality_obs * mask).sum(dim=(-2, -1)) / mask_mass,
            'glare_invalid_rate': (invalid * mask).sum(dim=(-2, -1)) / mask_mass,
        }
        self._store_last_diff_depth_train_aux({
            'quality_pre_valid': quality,
            'valid_prob_map': valid_prob,
            'hard_valid_map': hard_valid,
            'valid_st_map': valid_st,
        })
        self._store_last_diff_depth_debug({
            'scene_name': self.current_scene_name,
            'raw_depth_map': raw,
            'quality_pre_valid': quality,
            'quality_map': quality_obs,
            'valid_prob_map': valid_prob,
            'hard_valid_map': hard_valid,
            'invalid_mask': invalid,
            'scene_effect_map': effect,
            'scene_mask': mask,
            'scalars': scalars,
        })
        return depth_obs, quality_obs

    def render_diff_depth(self, power, exposure, gain, max_range=None):
        _ = max_range
        B = power.shape[0]
        R_cam_world = (self.R @ self.R_cam).contiguous()
        render_R = torch.bmm(self.R_scene_T, R_cam_world).contiguous()
        render_p = torch.bmm(self.R_scene_T, self.p[:, :, None])[:, :, 0].contiguous()
        depth = torch.empty((B, self.height, self.width), device=power.device, dtype=power.dtype)
        quadsim_cuda.render_depth(
            depth,
            self.balls,
            self.cyl,
            self.cyl_h,
            self.voxels,
            render_R,
            render_p,
            self.n_drones_per_group,
            float(self._fov_x_half_tan),
        )
        if self.diff_sensor_impl.get('diff_depth', 'cuda') == 'cuda':
            return self._sensor_cuda(depth, power, exposure, gain)
        return self._sensor_reference(depth, power, exposure, gain, max_range=max_range)

    def render(self, ctl_dt):
        _ = ctl_dt
        canvas = torch.empty((self.batch_size, self.height, self.width), device=self.device)
        render_R = torch.bmm(self.R_scene_T, self.R @ self.R_cam).contiguous()
        render_R_old = torch.bmm(self.R_scene_T, self.R_old @ self.R_cam).contiguous()
        render_p = torch.bmm(self.R_scene_T, self.p[:, :, None])[:, :, 0].contiguous()
        render_p_old = torch.bmm(self.R_scene_T, self.p_old[:, :, None])[:, :, 0].contiguous()
        quadsim_cuda.render(canvas, self.flow, self.balls, self.cyl, self.cyl_h,
                            self.voxels, render_R, render_R_old, render_p,
                            render_p_old, self.drone_radius, self.n_drones_per_group,
                            self._fov_x_half_tan)
        return canvas, None

    def find_vec_to_nearest_pt(self):
        p_world = self.p + self.v * self.sub_div
        p = torch.matmul(self.R_scene_T.unsqueeze(0), p_world.unsqueeze(-1)).squeeze(-1).contiguous()
        nearest_pt = torch.empty_like(p)
        if self.use_ellipsoid:
            R_local = torch.bmm(self.R_scene_T, self.R).contiguous()
            quadsim_cuda.find_nearest_pt_ellipsoid(
                nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p,
                R_local, self.drone_radius, self.n_drones_per_group,
                self.ellipsoid_a, self.ellipsoid_c)
        else:
            quadsim_cuda.find_nearest_pt(
                nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p,
                self.drone_radius, self.n_drones_per_group)
        vec_local = nearest_pt - p
        return torch.matmul(self.R_scene.unsqueeze(0), vec_local.unsqueeze(-1)).squeeze(-1)

    def get_scene_yaw_for_env(self, env_idx=0):
        idx = int(min(max(env_idx, 0), self.batch_size - 1))
        return float(self.scene_yaw[idx].detach().cpu().item())

    def get_world_voxels_for_env(self, env_idx=0):
        idx = int(min(max(env_idx, 0), self.batch_size - 1))
        return self.voxels[idx].detach().cpu().numpy()

    def get_world_balls_for_env(self, env_idx=0):
        idx = int(min(max(env_idx, 0), self.batch_size - 1))
        balls = self.balls[idx].detach()
        if balls.numel() == 0:
            return balls.detach().cpu().numpy()
        centers = torch.matmul(self.R_scene[idx], balls[:, :3].T).T
        out = balls.clone()
        out[:, :3] = centers
        return out.detach().cpu().numpy()

    def get_world_cyl_for_env(self, env_idx=0):
        idx = int(min(max(env_idx, 0), self.batch_size - 1))
        cyl = self.cyl[idx].detach()
        if cyl.numel() == 0:
            return cyl.detach().cpu().numpy()
        centers = torch.cat([cyl[:, :2], torch.zeros((cyl.shape[0], 1), device=cyl.device, dtype=cyl.dtype)], dim=-1)
        centers = torch.matmul(self.R_scene[idx], centers.T).T
        out = cyl.clone()
        out[:, :2] = centers[:, :2]
        return out.detach().cpu().numpy()

    def get_world_cyl_h_for_env(self, env_idx=0):
        idx = int(min(max(env_idx, 0), self.batch_size - 1))
        cyl_h = self.cyl_h[idx].detach()
        if cyl_h.numel() == 0:
            return cyl_h.detach().cpu().numpy()
        centers = torch.stack([cyl_h[:, 0], torch.zeros_like(cyl_h[:, 0]), cyl_h[:, 1]], dim=-1)
        centers = torch.matmul(self.R_scene[idx], centers.T).T
        out = cyl_h.clone()
        out[:, 0] = centers[:, 0]
        out[:, 1] = centers[:, 2]
        return out.detach().cpu().numpy()

    def run(self, act_pred, ctl_dt=1/15, v_pred=None):
        self.dg = self.dg * math.sqrt(max(1 - ctl_dt / 4, 0.0)) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        self.p_old = self.p
        self.act, self.p, self.v, self.a = run(
            self.R, self.dg, self.z_drag_coef, self.drag_2, self.pitch_ctl_delay,
            act_pred, self.act, self.p, self.v, self.v_wind, self.a,
            self.grad_decay, ctl_dt, 0.5)
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        if v_pred is None:
            v_pred = self.p_target - self.p
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)

    def save_state(self):
        return {
            'p': self.p.clone(), 'v': self.v.clone(), 'a': self.a.clone(), 'act': self.act.clone(),
            'R': self.R.clone(), 'R_old': self.R_old.clone(), 'p_old': self.p_old.clone(),
            'dg': self.dg.clone(), 'v_wind': self.v_wind.clone(),
        }

    def restore_state(self, snapshot):
        self.p = snapshot['p'].clone()
        self.v = snapshot['v'].clone()
        self.a = snapshot['a'].clone()
        self.act = snapshot['act'].clone()
        self.R = snapshot['R'].clone()
        self.R_old = snapshot['R_old'].clone()
        self.p_old = snapshot['p_old'].clone()
        self.dg = snapshot['dg'].clone()
        self.v_wind = snapshot['v_wind'].clone()

    def _run(self, act_pred, ctl_dt=1/15, v_pred=None):
        alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
        self.act = act_pred * (1 - alpha) + self.act * alpha
        self.dg = self.dg * math.sqrt(max(1 - ctl_dt, 0.0)) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
        z_drag = 0
        if self.z_drag_coef is not None:
            v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
            v_prep = self.v - v_up
            motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
            z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
        drag = self.drag_2 * self.v * self.v.norm(2, -1, True)
        a_next = self.act + self.dg - z_drag - drag
        self.p_old = self.p
        self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt ** 2
        self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
        self.a = a_next
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        if v_pred is None:
            v_pred = self.p_target - self.p
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)

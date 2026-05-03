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
    """Minimal single-wall active-sensing environment.

    Geometry is deliberately small: one start, one goal, and one wall with a
    narrow gate at a random lateral position.  Scenarios only change the local
    sensor degradation around the gate material.
    """

    supported_scenarios = ('glare', 'specular', 'dark')
    supported_slots = ('far_left', 'left', 'right', 'far_right')
    slot_y = {
        'far_left': -0.54,
        'left': -0.18,
        'right': 0.18,
        'far_right': 0.54,
    }

    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.82,
                 eval_mode=False, cam_angle=5, ellipsoid_a=0.0, ellipsoid_c=0.0,
                 cam_power_baseline=0.5, camera_control_mode='learned', sensor_grad_mode='full',
                 cam_delta_max=0.02, cam_return_rate=0.05,
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
                 simple_start_x=-1.0, simple_goal_x=1.8, simple_wall_x=0.65,
                 simple_gate_y_min=-0.55, simple_gate_y_max=0.55,
                 simple_gate_half_y=0.20, simple_gate_half_y_min=None, simple_gate_half_y_max=None,
                 simple_gate_half_z=0.26,
                 simple_gate_z=1.50,
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
        self.cam_delta_max = float(cam_delta_max)
        self.cam_return_rate = float(cam_return_rate)
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
        self.current_scene_names = [self.current_scene_name] * self.batch_size
        self.current_scene_ids = torch.full(
            (self.batch_size,),
            int(self.current_scene_id),
            device=self.device,
            dtype=torch.long,
        )
        self.sun_glare_eval_slot = self._canonical_slot(sun_glare_eval_slot)
        self.random_rotation = bool(random_rotation)
        self.random_rotation_max_rad = max(float(random_rotation_max_deg), 0.0) * math.pi / 180.0
        self._eval_slot_cursor = 0
        self.current_scene_effects = {}
        self.last_diff_depth_debug = None
        self.last_diff_depth_train_aux = None
        self.simple_start_x = float(simple_start_x)
        self.simple_goal_x = float(simple_goal_x)
        self.simple_wall_x = float(simple_wall_x)
        self.simple_gate_y_min = float(simple_gate_y_min)
        self.simple_gate_y_max = float(simple_gate_y_max)
        self.simple_gate_half_y = float(simple_gate_half_y)
        self.simple_gate_half_y_min = self.simple_gate_half_y if simple_gate_half_y_min is None else float(simple_gate_half_y_min)
        self.simple_gate_half_y_max = self.simple_gate_half_y if simple_gate_half_y_max is None else float(simple_gate_half_y_max)
        if self.simple_gate_half_y_max < self.simple_gate_half_y_min:
            self.simple_gate_half_y_min, self.simple_gate_half_y_max = self.simple_gate_half_y_max, self.simple_gate_half_y_min
        self.simple_gate_half_z = float(simple_gate_half_z)
        self.simple_gate_z = float(simple_gate_z)
        self.simple_wall_half_z = 1.0

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

    def _canonical_scene_name(self, scene_name):
        name = str(scene_name).strip().lower().replace('-', '_')
        if name not in self.supported_scenarios:
            raise ValueError(f'unsupported scene {scene_name!r}')
        return name

    def _choose_scene_name(self, scene_name=None):
        if scene_name is not None:
            return self._canonical_scene_name(scene_name)
        return random.choice(self.scenarios)

    def _choose_scene_names(self, B, scene_name=None):
        if scene_name is not None:
            name = self._canonical_scene_name(scene_name)
            return [name] * int(B)
        if self.eval_mode or len(self.scenarios) <= 1:
            return [self._choose_scene_name(None)] * int(B)
        names = [self.scenarios[i % len(self.scenarios)] for i in range(int(B))]
        random.shuffle(names)
        return names

    def _choose_slots(self, B):
        if self.eval_mode and self.sun_glare_eval_slot is not None:
            return [self.sun_glare_eval_slot] * B
        slots = list(self.supported_slots)
        if self.eval_mode:
            start = int(self._eval_slot_cursor)
            self._eval_slot_cursor += int(B)
            return [slots[(start + i) % len(slots)] for i in range(B)]
        return [slots[i % len(slots)] for i in range(B)]

    def _slot_name_from_gate_y(self, y):
        vals = {name: abs(float(y) - float(v)) for name, v in self.slot_y.items()}
        return min(vals, key=vals.get)

    def _choose_gate_centers(self, B):
        if self.eval_mode and self.sun_glare_eval_slot is not None:
            slots = self._choose_slots(B)
            gate_y = torch.tensor([self.slot_y[s] for s in slots], device=self.device, dtype=torch.float32)
            return gate_y, slots
        lo = min(self.simple_gate_y_min, self.simple_gate_y_max)
        hi = max(self.simple_gate_y_min, self.simple_gate_y_max)
        gate_y = torch.empty((B,), device=self.device).uniform_(lo, hi)
        slots = [self._slot_name_from_gate_y(float(y)) for y in gate_y.detach().cpu().tolist()]
        return gate_y, slots

    def _choose_gate_half_widths(self, B):
        lo = min(self.simple_gate_half_y_min, self.simple_gate_half_y_max)
        hi = max(self.simple_gate_half_y_min, self.simple_gate_half_y_max)
        if abs(hi - lo) < 1e-9:
            return torch.full((B,), float(lo), device=self.device, dtype=torch.float32)
        return torch.empty((B,), device=self.device, dtype=torch.float32).uniform_(lo, hi)

    def _build_sun_glare_voxel_layout(self, gap_y_center, *, gate_x=None,
                                      gap_half_w=None, gap_half_h=None,
                                      gate_z=None):
        """Single wall with a narrow gate."""
        gate_x = self.simple_wall_x if gate_x is None else float(gate_x)
        gate_z = self.simple_gate_z if gate_z is None else float(gate_z)
        gap_half_w = self.simple_gate_half_y if gap_half_w is None else float(gap_half_w)
        gap_half_h = self.simple_gate_half_z if gap_half_h is None else float(gap_half_h)
        wall_half_y = 1.0
        wall_half_z = self.simple_wall_half_z * 2
        wall_thickness = 0.10
        back_wall_x = float(self.simple_goal_x) + 0.75
        back_wall_half_y = 3.0
        gate_wall = self._build_voxels([
            [float(gate_x), gap_y_center - float(gap_half_w) - wall_half_y, gate_z, wall_thickness, wall_half_y, wall_half_z],
            [float(gate_x), gap_y_center + float(gap_half_w) + wall_half_y, gate_z, wall_thickness, wall_half_y, wall_half_z],
            [back_wall_x, 0.0, gate_z, wall_thickness, back_wall_half_y, wall_half_z],
            # [float(gate_x), gap_y_center, gate_z + gap_half_h + wall_half_z, wall_thickness, float(gap_half_w), wall_half_z],
            # [float(gate_x), gap_y_center, gate_z - gap_half_h - wall_half_z, wall_thickness, float(gap_half_w), wall_half_z],
        ])
        return gate_wall

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

    def _scene_effects(self, scene_name, slot_name, gap_y, gap_half_w):
        regime_id = float(self.scene_name_to_id[scene_name])
        gap_half_w = float(gap_half_w)
        region_kind = 'opening' if str(scene_name) == 'glare' else 'vertical_edges'
        if region_kind == 'opening':
            hazard_half_y = gap_half_w
            hazard_half_z = float(self.simple_gate_half_z)
        else:
            hazard_half_y = float(gap_half_w + 0.20)
            hazard_half_z = float(self.simple_wall_half_z)
        return {
            'geometry_kind': 'single_wall_gate',
            'sensor_regime_name': scene_name,
            'sensor_regime_id': regime_id,
            'decision_open_slot_name': slot_name,
            'decision_open_slot_id': float(self.supported_slots.index(slot_name)),
            'decision_open_slot_y': float(gap_y),
            'hazard_center': [self.simple_wall_x, float(gap_y), self.simple_gate_z],
            # Glare is a backlight seen through the aperture. Specular/dark are
            # material effects on the gate frame edges.
            'hazard_region_kind': region_kind,
            'hazard_half_y': hazard_half_y,
            'hazard_half_z': hazard_half_z,
            'hazard_edge_half_y': 0.055,
            'hazard_softness': 0.045,
            'geometry_gate_x': float(self.simple_wall_x),
            'geometry_wall_x': float(self.simple_wall_x),
            'geometry_gap_half_w': gap_half_w,
            'geometry_gap_half_h': float(self.simple_gate_half_z),
            'geometry_gate_z': float(self.simple_gate_z),
            'geometry_wall_half_z': float(self.simple_wall_half_z),
            'geometry_back_wall_x': float(self.simple_goal_x + 0.75),
            'geometry_start_x': float(self.simple_start_x),
            'geometry_goal_x': float(self.simple_goal_x),
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

    def _set_scene_names(self, scene_names):
        names = [self._canonical_scene_name(name) for name in scene_names]
        ids = [int(self.scene_name_to_id[name]) for name in names]
        self.current_scene_names = names
        self.current_scene_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        if all(name == names[0] for name in names):
            self.current_scene_name = names[0]
            self.current_scene_id = ids[0]
        else:
            self.current_scene_name = 'mixed'
            self.current_scene_id = -1

    def reset(self, scene_name=None):
        B, device = self.batch_size, self.device
        scene_names = self._choose_scene_names(B, scene_name=scene_name)
        self._set_scene_names(scene_names)
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

        gap_y, slots = self._choose_gate_centers(B)
        gap_half_y = self._choose_gate_half_widths(B)
        voxels = torch.stack([
            self._build_sun_glare_voxel_layout(float(y), gap_half_w=float(w))
            for y, w in zip(gap_y, gap_half_y)
        ], dim=0)
        start_y = torch.zeros(B, device=device)
        start_local = torch.stack([
            torch.full((B,), float(self.simple_start_x), device=device),
            start_y,
            torch.full((B,), float(self.simple_gate_z), device=device),
        ], -1)
        goal_local = torch.tensor([float(self.simple_goal_x), 0.0, float(self.simple_gate_z)], device=device).expand(B, 3).clone()
        start = torch.bmm(self.R_scene, start_local[:, :, None])[:, :, 0]
        goal = torch.bmm(self.R_scene, goal_local[:, :, None])[:, :, 0]
        effects = self._merge_batch_effects([
            self._scene_effects(scene_names[i], slots[i], float(gap_y[i]), float(gap_half_y[i]))
            for i in range(B)
        ])
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
        idx = int(min(max(env_idx, 0), self.batch_size - 1))
        scene_names = debug.get('scene_names', None)
        if isinstance(scene_names, list) and len(scene_names) > idx:
            scene_name = scene_names[idx]
        else:
            scene_name = debug.get('scene_name', self.current_scene_name)
        out = {'scene_name': str(scene_name), 'images': {}, 'scalars': {}}
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
        def _batch_scalar(value, default):
            if value is None:
                value = default
            if torch.is_tensor(value):
                return value.to(depth.device, depth.dtype)
            return torch.full((B,), float(value), device=depth.device, dtype=depth.dtype)
        half_y = _batch_scalar(half_y, 0.25)
        half_z = _batch_scalar(half_z, 1.0)
        softness = _batch_scalar(softness, 0.045)
        center = center.to(depth.device, depth.dtype)

        ys = torch.linspace(-1.0, 1.0, W, device=depth.device, dtype=depth.dtype)
        zs = torch.linspace(-1.0, 1.0, H, device=depth.device, dtype=depth.dtype)
        yy, zz = torch.meshgrid(ys, zs, indexing='xy')

        # Project the local sensor-degradation region into the current camera.
        # This replaces the older slot_y -> image_x approximation.  The mask is
        # deliberately detached from pose so the sensor model only exposes
        # camera-parameter gradients; raw geometry remains non-differentiable.
        R_cam_world = (self.R @ self.R_cam).detach().to(depth.device, depth.dtype)
        pos = self.p.detach().to(depth.device, depth.dtype)
        fov_x = torch.as_tensor(float(self._fov_x_half_tan), device=depth.device, dtype=depth.dtype)
        fov_y = fov_x * float(H) / float(max(W, 1))
        region_kind = effects.get('hazard_region_kind', 'box')
        if isinstance(region_kind, list) and len(region_kind) == B:
            vertical_selector = torch.tensor(
                [str(kind) == 'vertical_edges' for kind in region_kind],
                device=depth.device,
                dtype=torch.bool,
            )
        else:
            vertical_selector = torch.full(
                (B,),
                str(region_kind) == 'vertical_edges',
                device=depth.device,
                dtype=torch.bool,
            )

        def _vertical_edges_mask():
            gap_half_w = _batch_scalar(effects.get('geometry_gap_half_w'), self.simple_gate_half_y)
            edge_half_y = _batch_scalar(effects.get('hazard_edge_half_y'), 0.055)
            edge_half_z = half_z
            local_y_axis_world = self.R_scene[:, :, 1].detach().to(depth.device, depth.dtype)
            edge_centers = torch.stack([
                center - gap_half_w[:, None] * local_y_axis_world,
                center + gap_half_w[:, None] * local_y_axis_world,
            ], dim=1)
            rel = edge_centers.detach() - pos[:, None, :]
            cam = torch.einsum('bij,bkj->bki', R_cam_world.transpose(1, 2), rel)
            x = cam[..., 0]
            x_safe = x.clamp_min(0.20)
            cy = ((-cam[..., 1] / x_safe) / fov_x).clamp(-1.5, 1.5)[:, :, None, None]
            cz = ((-cam[..., 2] / x_safe) / fov_y).clamp(-1.5, 1.5)[:, :, None, None]
            sy = (edge_half_y[:, None] / x_safe / fov_x).clamp(0.025, 0.50)[:, :, None, None]
            sz = (edge_half_z[:, None] / x_safe / fov_y).clamp(0.08, 1.25)[:, :, None, None]
            soft = (softness[:, None] / x_safe / fov_x).clamp(0.020, 0.18)[:, :, None, None]
            yy_e = yy[None, None]
            zz_e = zz[None, None]
            front_gate = torch.sigmoid((x - 0.08) / 0.04)[:, :, None, None]
            mask_y = torch.sigmoid((sy - (yy_e - cy).abs()) / soft)
            mask_z = torch.sigmoid((sz - (zz_e - cz).abs()) / soft)
            return (mask_y * mask_z * front_gate).amax(dim=1).clamp(0.0, 1.0)

        def _opening_mask():
            yy_o = yy.unsqueeze(0)
            zz_o = zz.unsqueeze(0)
            rel = center.detach() - pos
            cam = torch.bmm(R_cam_world.transpose(1, 2), rel[:, :, None])[:, :, 0]
            x = cam[:, 0]
            x_safe = x.clamp_min(0.20)

            cy = ((-cam[:, 1] / x_safe) / fov_x).clamp(-1.5, 1.5)[:, None, None]
            cz = ((-cam[:, 2] / x_safe) / fov_y).clamp(-1.5, 1.5)[:, None, None]
            sy = (half_y.to(depth.device, depth.dtype) / x_safe / fov_x).clamp(0.04, 1.25)[:, None, None]
            sz = (half_z.to(depth.device, depth.dtype) / x_safe / fov_y).clamp(0.06, 1.25)[:, None, None]
            soft = (softness.to(depth.device, depth.dtype) / x_safe / fov_x).clamp(0.025, 0.18)[:, None, None]
            front_gate = torch.sigmoid((x - 0.08) / 0.04)[:, None, None]
            mask_y = torch.sigmoid((sy - (yy_o - cy).abs()) / soft)
            mask_z = torch.sigmoid((sz - (zz_o - cz).abs()) / soft)
            return (mask_y * mask_z * front_gate).clamp(0.0, 1.0)

        if bool(vertical_selector.all().item()):
            return _vertical_edges_mask()
        if not bool(vertical_selector.any().item()):
            return _opening_mask()
        vertical_mask = _vertical_edges_mask()
        opening_mask = _opening_mask()
        return torch.where(vertical_selector[:, None, None], vertical_mask, opening_mask)

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
        quality_base = torch.sigmoid(
            2.15 * snr
            - 0.95 * washout
            - 0.85 * edge
            - 1.45 * torch.relu(dist - 0.92)
        )

        overexp = torch.sigmoid((e01 - 0.22) / 0.045)
        gain_sat = torch.sigmoid((g01 - 0.28) / 0.055)
        gain_exposure_sat = torch.sigmoid(((g01 + 0.85 * e01) - 0.52) / 0.070)
        rescue = torch.sigmoid((p - 0.50) / 0.09)
        rescue_window = torch.sigmoid((0.30 - e01) / 0.06)
        joint_sat = torch.sigmoid((p - 0.65) / 0.08) * torch.sigmoid((e01 - 0.32) / 0.06)
        under_power = torch.sigmoid((0.45 - p) / 0.08)
        glare_penalty = mask * (
            0.88 * overexp
            + 0.28 * joint_sat
            + 0.42 * under_power * rescue_window
            + 0.72 * gain_sat
            + 0.44 * gain_exposure_sat
        )
        low_gain_window = torch.sigmoid((0.26 - g01) / 0.06)
        glare_bonus = mask * rescue * rescue_window * low_gain_window * 0.34
        quality_glare = quality_base - glare_penalty + glare_bonus

        # Specular edge material blooms under active IR.  Use unsaturated
        # quadratic terms so high power keeps a clear negative gradient even
        # when the binary valid map is already poor near the wall.
        power_quad = p.square() * (0.78 + 0.22 * torch.sigmoid((e01 - 0.18) / 0.08))
        power_knee = torch.sigmoid((p - 0.22) / 0.060) * (0.35 + 0.65 * p)
        exposure_quad = e01.square() * (0.32 + 0.68 * torch.sigmoid((p - 0.18) / 0.08))
        exposure_bloom = torch.sigmoid((e01 - 0.42) / 0.075) * (0.45 + 0.55 * torch.sigmoid((g01 - 0.42) / 0.08))
        gain_quad = g01.square() * (0.30 + 0.70 * torch.sigmoid((e01 - 0.22) / 0.07))
        gain_bloom = torch.sigmoid((g01 - 0.32) / 0.060) * (0.40 + 0.60 * torch.sigmoid((e01 - 0.24) / 0.07))
        spec_safe = (
            torch.sigmoid((0.34 - p) / 0.060)
            * torch.sigmoid((0.42 - e01) / 0.08)
            * torch.sigmoid((0.32 - g01) / 0.07)
        )
        spec_very_safe = (
            torch.sigmoid((0.20 - p) / 0.050)
            * torch.sigmoid((0.26 - e01) / 0.060)
            * torch.sigmoid((0.18 - g01) / 0.060)
        )
        spec_penalty = mask * (
            1.25 * power_quad
            + 0.75 * power_knee
            + 0.50 * exposure_quad
            + 0.40 * exposure_bloom
            + 0.50 * gain_quad
            + 0.38 * gain_bloom
        )
        spec_bonus = mask * (0.42 * spec_safe + 0.22 * spec_very_safe)
        quality_specular = quality_base - spec_penalty + spec_bonus

        exposure_lift = torch.sigmoid((e01 - 0.62) / 0.070)
        gain_lift = torch.sigmoid((g01 - 0.52) / 0.075)
        projector_lift = torch.sigmoid((p - 0.45) / 0.10)
        dark_rescue = (
            exposure_lift * (
                0.10
                + 0.70 * gain_lift
                + 0.20 * projector_lift * gain_lift
            )
        ).clamp(max=1.0)
        dark_need = mask * 0.92
        dark_penalty = dark_need * (1.0 - dark_rescue)
        quality_dark = quality_base - dark_penalty + mask * dark_rescue * 0.24

        scene_ids = getattr(self, 'current_scene_ids', None)
        if scene_ids is None:
            scene_ids = torch.full((raw.shape[0],), int(self.current_scene_id), device=raw.device, dtype=torch.long)
        else:
            scene_ids = scene_ids.to(device=raw.device, dtype=torch.long)
        sid = scene_ids[:, None, None]
        quality = torch.where(
            sid == 0,
            quality_glare,
            torch.where(sid == 1, quality_specular, quality_dark),
        )
        effect = torch.where(
            sid == 0,
            glare_penalty,
            torch.where(sid == 1, spec_penalty, dark_penalty),
        )

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
        speed = self.v.norm(2, -1).detach().to(raw.dtype)
        scene_ids = getattr(self, 'current_scene_ids', None)
        if scene_ids is None:
            scene_ids = torch.full((raw.shape[0],), int(self.current_scene_id), device=raw.device, dtype=torch.long)
        else:
            scene_ids = scene_ids.to(device=raw.device, dtype=torch.long)

        def _sensor_call(idx, regime_id):
            return active_sensing_sensor(
                raw[idx].contiguous(),
                mask[idx].contiguous(),
                power[idx].contiguous(),
                exposure[idx].contiguous(),
                gain[idx].contiguous(),
                speed[idx].contiguous(),
                int(regime_id),
                min_valid,
                max_range,
                self.cam_sem.exposure_t_min,
                self.cam_sem.exposure_t_span,
                self.cam_sem.iso_gain_base,
                self.cam_sem.iso_gain_scale,
                self.cam_sem.iso_gain_gamma,
                self.cam_sem.shot_noise_base,
            )

        unique_ids = torch.unique(scene_ids, sorted=True)
        if int(unique_ids.numel()) == 1:
            depth_obs, quality_obs, quality, valid_prob, hard_valid, effect = _sensor_call(
                torch.arange(raw.shape[0], device=raw.device),
                int(unique_ids[0].item()),
            )
        else:
            depth_chunks = []
            quality_obs_chunks = []
            quality_chunks = []
            valid_prob_chunks = []
            hard_valid_chunks = []
            effect_chunks = []
            order_chunks = []
            for regime_id_t in unique_ids:
                idx = torch.nonzero(scene_ids == regime_id_t, as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                outs = _sensor_call(idx, int(regime_id_t.item()))
                depth_chunks.append(outs[0])
                quality_obs_chunks.append(outs[1])
                quality_chunks.append(outs[2])
                valid_prob_chunks.append(outs[3])
                hard_valid_chunks.append(outs[4])
                effect_chunks.append(outs[5])
                order_chunks.append(idx)
            order = torch.cat(order_chunks, dim=0)
            sort_idx = torch.argsort(order)
            depth_obs = torch.cat(depth_chunks, dim=0)[sort_idx]
            quality_obs = torch.cat(quality_obs_chunks, dim=0)[sort_idx]
            quality = torch.cat(quality_chunks, dim=0)[sort_idx]
            valid_prob = torch.cat(valid_prob_chunks, dim=0)[sort_idx]
            hard_valid = torch.cat(hard_valid_chunks, dim=0)[sort_idx]
            effect = torch.cat(effect_chunks, dim=0)[sort_idx]
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
            'scene_names': list(getattr(self, 'current_scene_names', [self.current_scene_name] * raw.shape[0])),
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

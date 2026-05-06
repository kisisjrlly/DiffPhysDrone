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
    """极简单墙细缝主动感知环境。

    几何结构刻意保持很小：一个起点、一个终点、一堵或多堵墙，以及墙上
    随机横向位置的竖直细缝。不同 scene 只改变细缝/墙边附近的局部传感器
    退化方式，不改变物理碰撞几何。
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
                 scene_layout='single_wall',
                 corridor_scene_sequence=None,
                 corridor_wall_xs=None,
                 corridor_wall_spacing=1.25,
                 corridor_stage_release_margin=0.18,
                 corridor_shuffle_scene_order=False,
                 random_rotation=False, random_rotation_max_deg=45.0,
                 simple_start_x=-1.0, simple_goal_x=1.8, simple_wall_x=0.65,
                 simple_slit_center_y_min=-0.55, simple_slit_center_y_max=0.55,
                 simple_slit_half_y=0.20, simple_slit_half_y_min=None, simple_slit_half_y_max=None,
                 simple_slit_effect_half_z=0.26,
                 simple_slit_center_z=1.50,
                 simple_slit_side_effect_width_y=0.20,
                 simple_slit_side_effect_half_z=1.00,
                 simple_glare_halo_width_y=0.18,
                 simple_glare_halo_extra_half_z=0.25,
                 simple_glare_halo_strength=0.45,
                 simple_back_wall_x_min=None,
                 simple_back_wall_x_max=None,
                 simple_slit_cue_halo_width_y=0.16,
                 simple_slit_cue_extra_half_z=0.28,
                 simple_key_cue_degrade_strength=0.90,
                 simple_specular_false_depth_strength=0.55) -> None:
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
        self.scene_layout = str(scene_layout).strip().lower()
        if self.scene_layout not in {'single_wall', 'three_wall_corridor'}:
            raise ValueError(f'unsupported scene_layout {scene_layout!r}')
        seq_raw = corridor_scene_sequence if corridor_scene_sequence is not None else ['dark', 'specular', 'glare']
        self.corridor_scene_sequence = self._normalize_scene_sequence(seq_raw)
        self.corridor_wall_spacing = max(float(corridor_wall_spacing), 1e-3)
        if corridor_wall_xs is None or len(corridor_wall_xs) == 0:
            self.corridor_wall_xs = None
        else:
            self.corridor_wall_xs = [float(x) for x in corridor_wall_xs]
        self.corridor_stage_release_margin = max(float(corridor_stage_release_margin), 0.0)
        self.corridor_shuffle_scene_order = bool(corridor_shuffle_scene_order)
        self._corridor_stage_effects = None
        self._corridor_stage_scene_names = None
        self._corridor_stage_slots = None
        self._corridor_stage_idx = None
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
        # 墙和细缝几何都先定义在局部 scene 坐标系中，之后才应用可选随机 yaw 旋转。
        #
        # simple_wall_x:
        #   带细缝前墙的局部 x 位置。
        #
        # simple_slit_center_y_min/max:
        #   细缝中心 y 的训练采样范围。每次 reset 会在这个区间内采样
        #   slit_center_y；eval 固定 slot 时则使用对应 slot 的 y。
        #
        # simple_slit_half_y:
        #   物理细缝在局部 y 方向的半宽。若配置了 min/max，则 min/max 的采样
        #   范围覆盖这个默认值。前墙由左右两块竖直墙体组成，中间空出
        #   2 * slit_half_y 的横向细缝。
        #
        # simple_slit_center_z:
        #   细缝中心高度，同时也是默认起点/终点飞行高度。
        #
        # simple_slit_effect_half_z:
        #   只用于 glare/opening 传感器效应 mask 的竖直半高，不代表真实物理
        #   碰撞边界。当前物理墙没有上/下门框，只有左右两侧竖直墙体。
        #
        # simple_slit_side_effect_width_y:
        #   specular/dark 材质区域从细缝边缘向左右墙面内部延伸的宽度。
        #   它不占用细缝空洞，只覆盖墙面 patch。
        #
        # simple_slit_side_effect_half_z:
        #   specular/dark 墙面材质 patch 的竖直半高。
        #
        # simple_glare_halo_width_y / extra_half_z / strength:
        #   glare 场景中，红外强光除了穿过细缝核心区域，还会在图像里污染
        #   周围墙缝轮廓。这里用一个更大的 halo 区域表示泛光/串扰，strength
        #   表示 halo 相对核心强光的强度。
        #
        # simple_back_wall_x_min/max:
        #   细缝后方背墙的局部 x 采样范围。背墙距离随机化后，policy 不能
        #   长期依赖一个固定的“透过细缝看到远墙”的深度模板。
        #
        # simple_slit_cue_halo_width_y / extra_half_z:
        #   真实 D455 中，坏曝光/反光/红外眩光会影响左右目图像中的细缝
        #   内部、边缘和后墙纹理。这里用一个覆盖细缝核心并稍微外扩的
        #   cue mask 表示这类关键观测线索。
        #
        # simple_key_cue_degrade_strength:
        #   scene effect 对关键 cue 的最大退化强度。
        #
        # simple_specular_false_depth_strength:
        #   specular 高 active IR 下，局部镜面反射造成错误近深度/边缘漂移
        #   的强度。
        self.simple_slit_center_y_min = float(simple_slit_center_y_min)
        self.simple_slit_center_y_max = float(simple_slit_center_y_max)
        self.simple_slit_half_y = float(simple_slit_half_y)
        self.simple_slit_half_y_min = self.simple_slit_half_y if simple_slit_half_y_min is None else float(simple_slit_half_y_min)
        self.simple_slit_half_y_max = self.simple_slit_half_y if simple_slit_half_y_max is None else float(simple_slit_half_y_max)
        if self.simple_slit_half_y_max < self.simple_slit_half_y_min:
            self.simple_slit_half_y_min, self.simple_slit_half_y_max = self.simple_slit_half_y_max, self.simple_slit_half_y_min
        self.simple_slit_effect_half_z = float(simple_slit_effect_half_z)
        self.simple_slit_center_z = float(simple_slit_center_z)
        self.simple_slit_side_effect_width_y = float(simple_slit_side_effect_width_y)
        self.simple_slit_side_effect_half_z = float(simple_slit_side_effect_half_z)
        self.simple_glare_halo_width_y = float(simple_glare_halo_width_y)
        self.simple_glare_halo_extra_half_z = float(simple_glare_halo_extra_half_z)
        self.simple_glare_halo_strength = float(simple_glare_halo_strength)
        default_back_wall_x = float(self.simple_goal_x) + 0.75
        self.simple_back_wall_x_min = default_back_wall_x if simple_back_wall_x_min is None else float(simple_back_wall_x_min)
        self.simple_back_wall_x_max = self.simple_back_wall_x_min if simple_back_wall_x_max is None else float(simple_back_wall_x_max)
        if self.simple_back_wall_x_max < self.simple_back_wall_x_min:
            self.simple_back_wall_x_min, self.simple_back_wall_x_max = self.simple_back_wall_x_max, self.simple_back_wall_x_min
        self.simple_slit_cue_halo_width_y = float(simple_slit_cue_halo_width_y)
        self.simple_slit_cue_extra_half_z = float(simple_slit_cue_extra_half_z)
        self.simple_key_cue_degrade_strength = float(simple_key_cue_degrade_strength)
        self.simple_specular_false_depth_strength = float(simple_specular_false_depth_strength)
        # 前墙左右墙体的物理半高。物理墙比材质 patch 更高，避免材质范围
        # 和碰撞几何被误认为同一个参数。
        self.simple_wall_half_z = 1.0

        impl = {'diff_depth': 'cuda'}
        if diff_sensor_impl is not None:
            impl.update({str(k): str(v).lower() for k, v in dict(diff_sensor_impl).items()})
        self.diff_sensor_impl = impl

        self.fixed_max_speed = 1.15
        self.fixed_drone_radius = 0.12
        self.fixed_margin = 0.00
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

    def _normalize_scene_sequence(self, items):
        out = []
        if items is None:
            items = ['dark', 'specular', 'glare']
        for raw in items:
            for token in str(raw).split(','):
                name = token.strip().lower().replace('-', '_')
                if not name:
                    continue
                if name not in self.supported_scenarios:
                    raise ValueError(f'unsupported corridor scene {raw!r}')
                out.append(name)
        return out or ['dark', 'specular', 'glare']

    def _corridor_wall_positions(self):
        if self.corridor_wall_xs is not None:
            xs = [float(x) for x in self.corridor_wall_xs]
        else:
            xs = [
                float(self.simple_wall_x) + i * float(self.corridor_wall_spacing)
                for i in range(len(self.corridor_scene_sequence))
            ]
        if any(xs[i + 1] <= xs[i] for i in range(len(xs) - 1)):
            raise ValueError('corridor wall positions must be strictly increasing')
        return xs

    def _choose_corridor_sequences(self, B, scene_name=None):
        base = list(self.corridor_scene_sequence)
        if scene_name is not None:
            # eval 时传入 scene_name 仍然支持：用同一种 effect 填满整条 corridor，
            # 方便做 ablation；正常三段 corridor eval 不需要传 scene_name。
            base = [self._canonical_scene_name(scene_name)] * len(base)
        seqs = []
        for _ in range(int(B)):
            seq = list(base)
            if self.corridor_shuffle_scene_order and not self.eval_mode and len(seq) > 1:
                random.shuffle(seq)
            seqs.append(seq)
        return seqs

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

    def _slot_name_from_slit_y(self, y):
        vals = {name: abs(float(y) - float(v)) for name, v in self.slot_y.items()}
        return min(vals, key=vals.get)

    def _choose_slit_centers(self, B):
        if self.eval_mode and self.sun_glare_eval_slot is not None:
            slots = self._choose_slots(B)
            slit_center_y = torch.tensor([self.slot_y[s] for s in slots], device=self.device, dtype=torch.float32)
            return slit_center_y, slots
        lo = min(self.simple_slit_center_y_min, self.simple_slit_center_y_max)
        hi = max(self.simple_slit_center_y_min, self.simple_slit_center_y_max)
        slit_center_y = torch.empty((B,), device=self.device).uniform_(lo, hi)
        slots = [self._slot_name_from_slit_y(float(y)) for y in slit_center_y.detach().cpu().tolist()]
        return slit_center_y, slots

    def _choose_slit_half_widths(self, B):
        lo = min(self.simple_slit_half_y_min, self.simple_slit_half_y_max)
        hi = max(self.simple_slit_half_y_min, self.simple_slit_half_y_max)
        if abs(hi - lo) < 1e-9:
            return torch.full((B,), float(lo), device=self.device, dtype=torch.float32)
        return torch.empty((B,), device=self.device, dtype=torch.float32).uniform_(lo, hi)

    def _choose_back_wall_xs(self, B):
        lo = min(self.simple_back_wall_x_min, self.simple_back_wall_x_max)
        hi = max(self.simple_back_wall_x_min, self.simple_back_wall_x_max)
        if abs(hi - lo) < 1e-9:
            return torch.full((B,), float(lo), device=self.device, dtype=torch.float32)
        return torch.empty((B,), device=self.device, dtype=torch.float32).uniform_(lo, hi)

    def _build_wall_slit_voxel_layout(self, slit_center_y, *, wall_x=None,
                                      slit_half_y=None,
                                      slit_center_z=None,
                                      back_wall_x=None,
                                      include_back_wall=True):
        """构造一堵带竖直细缝的墙。

        这里没有物理上/下门框；z 方向的 effect 半高只属于传感器退化 mask。
        """
        wall_x = self.simple_wall_x if wall_x is None else float(wall_x)
        slit_center_z = self.simple_slit_center_z if slit_center_z is None else float(slit_center_z)
        slit_half_y = self.simple_slit_half_y if slit_half_y is None else float(slit_half_y)
        wall_half_y = 1.0
        wall_half_z = self.simple_wall_half_z * 2
        wall_thickness = 0.10
        back_wall_x = self.simple_back_wall_x_min if back_wall_x is None else float(back_wall_x)
        back_wall_half_y = 3.0
        # 前墙由左右两块竖直墙体组成，中间空出的部分就是细缝。单墙模式下
        # 终点后方放一堵背墙；多墙 corridor 中，前两堵墙透过细缝看到的是
        # 下一堵墙，只有最后再放一个背墙。
        rows = [
            [float(wall_x), slit_center_y - float(slit_half_y) - wall_half_y, slit_center_z, wall_thickness, wall_half_y, wall_half_z],
            [float(wall_x), slit_center_y + float(slit_half_y) + wall_half_y, slit_center_z, wall_thickness, wall_half_y, wall_half_z],
        ]
        if include_back_wall:
            rows.append([back_wall_x, 0.0, slit_center_z, wall_thickness, back_wall_half_y, wall_half_z])
        wall_slit_voxels = self._build_voxels(rows)
        return wall_slit_voxels

    def _build_corridor_voxel_layout(self, slit_centers_y, slit_half_ys, wall_xs, back_wall_x):
        rows = []
        wall_half_y = 1.0
        wall_half_z = self.simple_wall_half_z * 2
        wall_thickness = 0.10
        for wall_x, slit_y, slit_half_y in zip(wall_xs, slit_centers_y, slit_half_ys):
            rows.append([
                float(wall_x),
                float(slit_y) - float(slit_half_y) - wall_half_y,
                float(self.simple_slit_center_z),
                wall_thickness,
                wall_half_y,
                wall_half_z,
            ])
            rows.append([
                float(wall_x),
                float(slit_y) + float(slit_half_y) + wall_half_y,
                float(self.simple_slit_center_z),
                wall_thickness,
                wall_half_y,
                wall_half_z,
            ])
        rows.append([
            float(back_wall_x),
            0.0,
            float(self.simple_slit_center_z),
            wall_thickness,
            3.0,
            wall_half_z,
        ])
        return self._build_voxels(rows)

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

    def _scene_effects(self, scene_name, slot_name, slit_center_y, slit_half_y, back_wall_x, *, wall_x=None):
        regime_id = float(self.scene_name_to_id[scene_name])
        wall_x = self.simple_wall_x if wall_x is None else float(wall_x)
        slit_half_y = float(slit_half_y)
        region_kind = 'opening' if str(scene_name) == 'glare' else 'side_wall_patches'
        if region_kind == 'opening':
            # glare 表示从细缝透进来的强背光，mask 中心在细缝内部，
            # 竖直范围由 simple_slit_effect_half_z 控制；halo 会额外污染
            # 细缝两侧墙面和更高/更低的局部图像区域。
            hazard_half_y = float(slit_half_y + self.simple_glare_halo_width_y)
            hazard_half_z = float(self.simple_slit_effect_half_z + self.simple_glare_halo_extra_half_z)
        else:
            # specular/dark 是细缝两侧墙面 patch 的材质效应。patch 从
            # 细缝边缘向墙体内部延伸，不覆盖细缝空洞本身。
            hazard_half_y = float(slit_half_y + 0.5 * self.simple_slit_side_effect_width_y)
            hazard_half_z = float(self.simple_slit_side_effect_half_z)
        return {
            'geometry_kind': 'single_wall_slit',
            'sensor_regime_name': scene_name,
            'sensor_regime_id': regime_id,
            'slit_slot_name': slot_name,
            'slit_slot_id': float(self.supported_slots.index(slot_name)),
            'slit_center_y': float(slit_center_y),
            'hazard_center': [wall_x, float(slit_center_y), self.simple_slit_center_z],
            # glare 是穿过细缝的背光；specular/dark 是细缝墙边材质效应。
            'hazard_region_kind': region_kind,
            'hazard_half_y': hazard_half_y,
            'hazard_half_z': hazard_half_z,
            'side_effect_width_y': float(self.simple_slit_side_effect_width_y),
            'side_effect_half_y': float(0.5 * self.simple_slit_side_effect_width_y),
            'side_effect_half_z': float(self.simple_slit_side_effect_half_z),
            'glare_core_half_y': float(slit_half_y),
            'glare_core_half_z': float(self.simple_slit_effect_half_z),
            'glare_halo_width_y': float(self.simple_glare_halo_width_y),
            'glare_halo_half_y': float(slit_half_y + self.simple_glare_halo_width_y),
            'glare_halo_half_z': float(self.simple_slit_effect_half_z + self.simple_glare_halo_extra_half_z),
            'glare_halo_strength': float(self.simple_glare_halo_strength),
            'hazard_softness': 0.045,
            'geometry_wall_x': float(wall_x),
            'slit_half_y': slit_half_y,
            'slit_effect_half_z': float(self.simple_slit_effect_half_z),
            'slit_center_z': float(self.simple_slit_center_z),
            'geometry_wall_half_z': float(self.simple_wall_half_z),
            'geometry_back_wall_x': float(back_wall_x),
            'geometry_start_x': float(self.simple_start_x),
            'geometry_goal_x': float(self.simple_goal_x),
            'slit_cue_halo_width_y': float(self.simple_slit_cue_halo_width_y),
            'slit_cue_half_y': float(slit_half_y + self.simple_slit_cue_halo_width_y),
            'slit_cue_half_z': float(self.simple_slit_effect_half_z + self.simple_slit_cue_extra_half_z),
            'slit_cue_extra_half_z': float(self.simple_slit_cue_extra_half_z),
            'key_cue_degrade_strength': float(self.simple_key_cue_degrade_strength),
            'specular_false_depth_strength': float(self.simple_specular_false_depth_strength),
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
        if self.scene_layout == 'three_wall_corridor':
            return self._reset_three_wall_corridor(scene_name=scene_name)
        return self._reset_single_wall(scene_name=scene_name)

    def _reset_single_wall(self, scene_name=None):
        B, device = self.batch_size, self.device
        scene_names = self._choose_scene_names(B, scene_name=scene_name)
        self._set_scene_names(scene_names)
        self._corridor_stage_effects = None
        self._corridor_stage_scene_names = None
        self._corridor_stage_slots = None
        self._corridor_stage_idx = None
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

        slit_center_y, slots = self._choose_slit_centers(B)
        slit_half_y = self._choose_slit_half_widths(B)
        back_wall_x = self._choose_back_wall_xs(B)
        voxels = torch.stack([
            self._build_wall_slit_voxel_layout(float(y), slit_half_y=float(w), back_wall_x=float(bx))
            for y, w, bx in zip(slit_center_y, slit_half_y, back_wall_x)
        ], dim=0)
        start_y = torch.zeros(B, device=device)
        start_local = torch.stack([
            torch.full((B,), float(self.simple_start_x), device=device),
            start_y,
            torch.full((B,), float(self.simple_slit_center_z), device=device),
        ], -1)
        goal_local = torch.tensor([float(self.simple_goal_x), 0.0, float(self.simple_slit_center_z)], device=device).expand(B, 3).clone()
        start = torch.bmm(self.R_scene, start_local[:, :, None])[:, :, 0]
        goal = torch.bmm(self.R_scene, goal_local[:, :, None])[:, :, 0]
        effects = self._merge_batch_effects([
            self._scene_effects(
                scene_names[i], slots[i], float(slit_center_y[i]), float(slit_half_y[i]), float(back_wall_x[i]),
                wall_x=float(self.simple_wall_x),
            )
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

    def _reset_three_wall_corridor(self, scene_name=None):
        B, device = self.batch_size, self.device
        self.last_diff_depth_debug = None
        self.last_diff_depth_train_aux = None

        stage_wall_xs = self._corridor_wall_positions()
        K = len(stage_wall_xs)
        corridor_sequences = self._choose_corridor_sequences(B, scene_name=scene_name)
        if any(len(seq) != K for seq in corridor_sequences):
            raise ValueError('corridor scene sequence length must match corridor wall count')

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

        slit_centers, slot_rows, slit_half_rows = [], [], []
        for _ in range(K):
            sy, slots = self._choose_slit_centers(B)
            hw = self._choose_slit_half_widths(B)
            slit_centers.append(sy)
            slot_rows.append(slots)
            slit_half_rows.append(hw)
        slit_center_y = torch.stack(slit_centers, dim=1)
        slit_half_y = torch.stack(slit_half_rows, dim=1)
        back_wall_x = self._choose_back_wall_xs(B)

        voxels = torch.stack([
            self._build_corridor_voxel_layout(
                [float(slit_center_y[i, k]) for k in range(K)],
                [float(slit_half_y[i, k]) for k in range(K)],
                stage_wall_xs,
                float(back_wall_x[i]),
            )
            for i in range(B)
        ], dim=0)

        start_y = torch.zeros(B, device=device)
        start_local = torch.stack([
            torch.full((B,), float(self.simple_start_x), device=device),
            start_y,
            torch.full((B,), float(self.simple_slit_center_z), device=device),
        ], -1)
        goal_local = torch.tensor([float(self.simple_goal_x), 0.0, float(self.simple_slit_center_z)], device=device).expand(B, 3).clone()
        start = torch.bmm(self.R_scene, start_local[:, :, None])[:, :, 0]
        goal = torch.bmm(self.R_scene, goal_local[:, :, None])[:, :, 0]

        stage_effects = []
        stage_names = []
        stage_slots = []
        for k, wall_x in enumerate(stage_wall_xs):
            scene_names_k = [corridor_sequences[i][k] for i in range(B)]
            slots_k = [slot_rows[k][i] for i in range(B)]
            effects_k = self._merge_batch_effects([
                self._scene_effects(
                    scene_names_k[i],
                    slots_k[i],
                    float(slit_center_y[i, k]),
                    float(slit_half_y[i, k]),
                    float(back_wall_x[i]),
                    wall_x=float(wall_x),
                )
                for i in range(B)
            ])
            local_hazard = effects_k['hazard_center'].to(device=device, dtype=torch.float32)
            effects_k['hazard_center_local'] = local_hazard.clone()
            effects_k['hazard_center'] = torch.bmm(self.R_scene, local_hazard[:, :, None])[:, :, 0]
            effects_k['scene_yaw'] = yaw
            effects_k['geometry_start_local'] = start_local
            effects_k['geometry_goal_local'] = goal_local
            effects_k['geometry_start'] = start
            effects_k['geometry_goal'] = goal
            effects_k['geometry_kind'] = 'three_wall_corridor'
            effects_k['corridor_stage_idx'] = torch.full((B,), float(k), device=device)
            effects_k['corridor_num_stages'] = float(K)
            effects_k['corridor_wall_xs'] = list(stage_wall_xs)
            effects_k['corridor_scene_sequence'] = [list(seq) for seq in corridor_sequences]
            stage_effects.append(effects_k)
            stage_names.append(scene_names_k)
            stage_slots.append(slots_k)

        self._corridor_stage_effects = stage_effects
        self._corridor_stage_scene_names = stage_names
        self._corridor_stage_slots = stage_slots
        self._corridor_stage_idx = torch.zeros((B,), dtype=torch.long, device=device)

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
        self._update_corridor_active_stage(force=True)

    def _select_batch_effects(self, stage_indices):
        stage_indices = stage_indices.to(device=self.device, dtype=torch.long)
        selected = {}
        keys = self._corridor_stage_effects[0].keys()
        for key in keys:
            vals = [fx[key] for fx in self._corridor_stage_effects]
            first = vals[0]
            if torch.is_tensor(first):
                pieces = []
                for k, value in enumerate(vals):
                    value = value.to(device=self.device)
                    mask = stage_indices == int(k)
                    if value.ndim >= 1 and value.shape[0] == self.batch_size:
                        pieces.append(torch.where(mask.reshape((self.batch_size,) + (1,) * (value.ndim - 1)), value, torch.zeros_like(value)))
                    else:
                        expanded = value.reshape((1,) * max(value.ndim, 1)).expand((self.batch_size,) + tuple(value.shape))
                        pieces.append(torch.where(mask.reshape((self.batch_size,) + (1,) * value.ndim), expanded, torch.zeros_like(expanded)))
                selected[key] = torch.stack(pieces, dim=0).sum(dim=0)
            elif key in {'corridor_wall_xs'}:
                selected[key] = first
            elif isinstance(first, list) and len(first) == self.batch_size:
                out = []
                for b in range(self.batch_size):
                    out.append(vals[int(stage_indices[b].item())][b])
                selected[key] = out
            elif isinstance(first, list) and vals and all(isinstance(v, list) for v in vals):
                out = []
                for b in range(self.batch_size):
                    out.append(vals[int(stage_indices[b].item())][b])
                selected[key] = out
            else:
                out_vals = [vals[int(stage_indices[b].item())] for b in range(self.batch_size)]
                if all(v == out_vals[0] for v in out_vals):
                    selected[key] = out_vals[0]
                else:
                    selected[key] = out_vals
        return selected

    def _update_corridor_active_stage(self, force=False):
        _ = force
        if self.scene_layout != 'three_wall_corridor' or not self._corridor_stage_effects:
            return
        wall_xs = torch.tensor(self._corridor_wall_positions(), device=self.device, dtype=self.p.dtype)
        p_local = torch.bmm(self.R_scene_T, self.p[:, :, None])[:, :, 0]
        x_local = p_local[:, 0]
        thresholds = wall_xs + float(self.corridor_stage_release_margin)
        stage_idx = (x_local[:, None] > thresholds[None, :]).sum(dim=1).clamp(max=len(wall_xs) - 1).long()
        self._corridor_stage_idx = stage_idx
        effects = self._select_batch_effects(stage_idx)
        effects['corridor_active_stage'] = stage_idx.float()
        effects['corridor_local_x'] = x_local.detach()
        effects['corridor_stage_wall_x'] = wall_xs[stage_idx].detach()
        self.current_scene_effects = effects
        names = []
        for b in range(self.batch_size):
            k = int(stage_idx[b].item())
            name = self._corridor_stage_scene_names[k][b]
            names.append(name)
        self._set_scene_names(names)

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
        for key in (
            'raw_depth_map',
            'quality_map',
            'valid_prob_map',
            'hard_valid_map',
            'invalid_mask',
            'scene_effect_map',
            'scene_mask',
            'slit_cue_mask',
            'key_cue_artifact_map',
        ):
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

    def _batch_scalar_for_sensor(self, value, default, B, device, dtype):
        if value is None:
            value = default
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        if isinstance(value, list) and len(value) == B:
            return torch.tensor(value, device=device, dtype=dtype)
        return torch.full((B,), float(value), device=device, dtype=dtype)

    def _slit_cue_mask(self, depth):
        """Project the slit opening/back-wall cue into the current image.

        This mask is deliberately different from the material mask used for
        specular/dark side-wall patches.  It covers the visual shortcut that a
        depth camera gets by seeing the far back wall through the slit.
        """
        B, H, W = depth.shape
        effects = self.current_scene_effects
        center = effects['hazard_center'].to(depth.device, depth.dtype)
        half_y = self._batch_scalar_for_sensor(
            effects.get('slit_cue_half_y'),
            self.simple_slit_half_y + self.simple_slit_cue_halo_width_y,
            B,
            depth.device,
            depth.dtype,
        )
        half_z = self._batch_scalar_for_sensor(
            effects.get('slit_cue_half_z'),
            self.simple_slit_effect_half_z + self.simple_slit_cue_extra_half_z,
            B,
            depth.device,
            depth.dtype,
        )
        softness = self._batch_scalar_for_sensor(
            effects.get('hazard_softness'),
            0.045,
            B,
            depth.device,
            depth.dtype,
        )

        ys = torch.linspace(-1.0, 1.0, W, device=depth.device, dtype=depth.dtype)
        zs = torch.linspace(-1.0, 1.0, H, device=depth.device, dtype=depth.dtype)
        yy, zz = torch.meshgrid(ys, zs, indexing='xy')
        yy = yy.unsqueeze(0)
        zz = zz.unsqueeze(0)

        R_cam_world = (self.R @ self.R_cam).detach().to(depth.device, depth.dtype)
        pos = self.p.detach().to(depth.device, depth.dtype)
        fov_x = torch.as_tensor(float(self._fov_x_half_tan), device=depth.device, dtype=depth.dtype)
        fov_y = fov_x * float(H) / float(max(W, 1))

        rel = center.detach() - pos
        cam = torch.bmm(R_cam_world.transpose(1, 2), rel[:, :, None])[:, :, 0]
        x = cam[:, 0]
        x_safe = x.clamp_min(0.20)
        cy = ((-cam[:, 1] / x_safe) / fov_x).clamp(-1.5, 1.5)[:, None, None]
        cz = ((-cam[:, 2] / x_safe) / fov_y).clamp(-1.5, 1.5)[:, None, None]
        sy = (half_y / x_safe / fov_x).clamp(0.04, 1.25)[:, None, None]
        sz = (half_z / x_safe / fov_y).clamp(0.06, 1.25)[:, None, None]
        soft = (softness * 1.35 / x_safe / fov_x).clamp(0.030, 0.24)[:, None, None]
        in_front = torch.sigmoid((x - 0.08) / 0.04)[:, None, None]
        mask_y = torch.sigmoid((sy - (yy - cy).abs()) / soft)
        mask_z = torch.sigmoid((sz - (zz - cz).abs()) / soft)
        return (mask_y * mask_z * in_front).clamp(0.0, 1.0)

    def _key_cue_artifacts(self, raw, power, exposure, gain, quality, valid_prob, hard_valid, depth_obs):
        """Apply scene-specific artifacts to the slit/back-wall depth cue.

        The fused CUDA/Python sensor core models local quality.  This wrapper
        adds the part that matters for the benchmark shortcut: whether the far
        back-wall depth seen through the slit remains a clean template under
        bad camera settings.
        """
        B = raw.shape[0]
        cue_mask = self._slit_cue_mask(raw)
        p = power.clamp(0, 1)[:, None, None]
        e01 = exposure.clamp(0, 1)[:, None, None]
        g01 = gain.clamp(0, 1)[:, None, None]
        strength = self._batch_scalar_for_sensor(
            self.current_scene_effects.get('key_cue_degrade_strength'),
            self.simple_key_cue_degrade_strength,
            B,
            raw.device,
            raw.dtype,
        ).clamp(0.0, 1.0)[:, None, None]
        spec_false_strength = self._batch_scalar_for_sensor(
            self.current_scene_effects.get('specular_false_depth_strength'),
            self.simple_specular_false_depth_strength,
            B,
            raw.device,
            raw.dtype,
        ).clamp(0.0, 1.0)[:, None, None]

        # dark 场景不能只污染 slit 两侧墙面；低曝光/低增益/弱投光时，
        # 透过 slit 看到的后墙深度 cue 也应该变成低 SNR / invalid。
        # 这里用“缺少任一关键量就会变差”的加性 knee，避免 fixed_mid
        # 仍然留下稳定的中间 valid 长条。
        dark_cue_bad = (
            0.50 * torch.sigmoid((0.68 - e01) / 0.075)
            + 0.38 * torch.sigmoid((0.58 - g01) / 0.075)
            + 0.22 * torch.sigmoid((0.54 - p) / 0.090)
        ).clamp(0.0, 1.0)
        glare_bad = (
            0.72 * torch.sigmoid((e01 - 0.26) / 0.055)
            + 0.50 * torch.sigmoid((g01 - 0.24) / 0.060)
            + 0.30 * torch.sigmoid((0.42 - p) / 0.09)
        ).clamp(0.0, 1.0)
        # Specular should not collapse every reasonable camera setting into the
        # same black invalid blob.  Low projector/exposure/gain should preserve
        # most of the slit cue; high active IR or high exposure/gain should
        # create local holes, false near depths, and edge drift.
        spec_power_hot = torch.sigmoid((p - 0.56) / 0.085)
        spec_exposure_hot = torch.sigmoid((e01 - 0.58) / 0.100)
        spec_gain_hot = torch.sigmoid((g01 - 0.48) / 0.090)
        spec_joint_hot = torch.maximum(spec_exposure_hot, spec_gain_hot)
        spec_bloom = (
            0.60 * spec_power_hot
            + 0.30 * spec_power_hot * spec_joint_hot
            + 0.20 * spec_exposure_hot * spec_gain_hot
        ).clamp(0.0, 1.0)
        spec_safe = (
            torch.sigmoid((0.48 - p) / 0.080)
            * torch.sigmoid((0.52 - e01) / 0.100)
            * torch.sigmoid((0.42 - g01) / 0.090)
        )

        scene_ids = getattr(self, 'current_scene_ids', None)
        if scene_ids is None:
            scene_ids = torch.full((B,), int(self.current_scene_id), device=raw.device, dtype=torch.long)
        else:
            scene_ids = scene_ids.to(device=raw.device, dtype=torch.long)
        sid = scene_ids[:, None, None]
        cue_bad = torch.where(
            sid == 0,
            glare_bad,
            torch.where(sid == 1, spec_bloom, dark_cue_bad),
        )

        raw4 = raw[:, None]
        raw_far = F.max_pool2d(raw4, 3, stride=1, padding=1)[:, 0]
        raw_near = -F.max_pool2d(-raw4, 3, stride=1, padding=1)[:, 0]
        local_edge = ((raw_far - raw_near) / (raw + 0.18)).clamp(0.0, 1.0)
        ys = torch.linspace(-1.0, 1.0, raw.shape[-1], device=raw.device, dtype=raw.dtype)
        zs = torch.linspace(-1.0, 1.0, raw.shape[-2], device=raw.device, dtype=raw.dtype)
        yy, zz = torch.meshgrid(ys, zs, indexing='xy')
        texture = (0.5 + 0.5 * torch.sin((11.0 * yy + 7.0 * zz) * math.pi)).unsqueeze(0)
        spec_hole_shape = (0.18 + 0.62 * local_edge + 0.20 * texture).clamp(0.0, 1.0)
        spec_artifact = (cue_mask * spec_bloom * strength * spec_hole_shape).clamp(0.0, 1.0)
        non_spec_artifact = (cue_mask * cue_bad * strength).clamp(0.0, 1.0)
        artifact = torch.where(sid == 1, spec_artifact, non_spec_artifact)

        cue_quality_other = quality - non_spec_artifact * (0.58 + 0.34 * cue_bad)
        spec_recovery = cue_mask * spec_safe * (1.0 - 0.55 * spec_bloom) * 0.46
        cue_quality_spec = quality + spec_recovery - spec_artifact * (0.28 + 0.46 * spec_bloom)
        cue_quality = torch.where(sid == 1, cue_quality_spec, cue_quality_other).clamp(0.0, 1.0)
        cue_valid_prob = torch.sigmoid((cue_quality - 0.42) / 0.055)
        cue_hard_valid = (cue_valid_prob > 0.5).to(raw.dtype)
        valid_st = cue_hard_valid.detach() - cue_valid_prob.detach() + cue_valid_prob

        spec_wrong = (
            cue_mask
            * spec_bloom
            * spec_false_strength
            * (0.30 + 0.70 * local_edge)
        ).clamp(0.0, 1.0)
        left_depth = torch.roll(raw, shifts=1, dims=-1)
        right_depth = torch.roll(raw, shifts=-1, dims=-1)
        edge_drift_depth = torch.minimum(torch.minimum(left_depth, right_depth), raw * 0.70)
        false_depth = torch.lerp(raw, edge_drift_depth, (0.62 * spec_wrong).clamp(0.0, 1.0))
        false_depth = false_depth.clamp_min(float(self.depth_min_valid))
        raw_with_false = torch.where((sid == 1) & (spec_wrong > 0.10), false_depth, raw)
        cue_depth_obs = raw_with_false * valid_st
        cue_quality_obs = cue_quality * valid_st

        return {
            'depth_obs': cue_depth_obs,
            'quality_obs': cue_quality_obs,
            'quality': cue_quality,
            'valid_prob': cue_valid_prob,
            'hard_valid': cue_hard_valid,
            'valid_st': valid_st,
            'invalid': (1.0 - valid_st).clamp(0.0, 1.0),
            'cue_mask': cue_mask,
            'artifact': artifact,
            'spec_bloom': spec_bloom.expand_as(raw),
            'spec_wrong': spec_wrong,
            'artifact_effect': artifact + spec_wrong * 0.35,
        }

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

        # 将局部传感器退化区域投影到当前相机视野。mask 对位姿 detach，
        # 这样传感器模型只暴露相机参数梯度，原始几何仍保持不可微。
        R_cam_world = (self.R @ self.R_cam).detach().to(depth.device, depth.dtype)
        pos = self.p.detach().to(depth.device, depth.dtype)
        fov_x = torch.as_tensor(float(self._fov_x_half_tan), device=depth.device, dtype=depth.dtype)
        fov_y = fov_x * float(H) / float(max(W, 1))
        region_kind = effects.get('hazard_region_kind', 'box')
        if isinstance(region_kind, list) and len(region_kind) == B:
            side_patch_selector = torch.tensor(
                [str(kind) == 'side_wall_patches' for kind in region_kind],
                device=depth.device,
                dtype=torch.bool,
            )
        else:
            side_patch_selector = torch.full(
                (B,),
                str(region_kind) == 'side_wall_patches',
                device=depth.device,
                dtype=torch.bool,
            )

        def _side_wall_patch_mask():
            slit_half_y = _batch_scalar(effects.get('slit_half_y'), self.simple_slit_half_y)
            patch_half_y = _batch_scalar(
                effects.get('side_effect_half_y'),
                0.5 * self.simple_slit_side_effect_width_y,
            )
            patch_half_z = _batch_scalar(
                effects.get('side_effect_half_z'),
                self.simple_slit_side_effect_half_z,
            )
            local_y_axis_world = self.R_scene[:, :, 1].detach().to(depth.device, depth.dtype)
            patch_offsets = slit_half_y + patch_half_y
            patch_centers = torch.stack([
                center - patch_offsets[:, None] * local_y_axis_world,
                center + patch_offsets[:, None] * local_y_axis_world,
            ], dim=1)
            rel = patch_centers.detach() - pos[:, None, :]
            cam = torch.einsum('bij,bkj->bki', R_cam_world.transpose(1, 2), rel)
            x = cam[..., 0]
            x_safe = x.clamp_min(0.20)
            cy = ((-cam[..., 1] / x_safe) / fov_x).clamp(-1.5, 1.5)[:, :, None, None]
            cz = ((-cam[..., 2] / x_safe) / fov_y).clamp(-1.5, 1.5)[:, :, None, None]
            sy = (patch_half_y[:, None] / x_safe / fov_x).clamp(0.025, 0.50)[:, :, None, None]
            sz = (patch_half_z[:, None] / x_safe / fov_y).clamp(0.08, 1.25)[:, :, None, None]
            soft = (softness[:, None] / x_safe / fov_x).clamp(0.020, 0.18)[:, :, None, None]
            yy_e = yy[None, None]
            zz_e = zz[None, None]
            in_front = torch.sigmoid((x - 0.08) / 0.04)[:, :, None, None]
            mask_y = torch.sigmoid((sy - (yy_e - cy).abs()) / soft)
            mask_z = torch.sigmoid((sz - (zz_e - cz).abs()) / soft)
            return (mask_y * mask_z * in_front).amax(dim=1).clamp(0.0, 1.0)

        def _opening_mask():
            yy_o = yy.unsqueeze(0)
            zz_o = zz.unsqueeze(0)
            rel = center.detach() - pos
            cam = torch.bmm(R_cam_world.transpose(1, 2), rel[:, :, None])[:, :, 0]
            x = cam[:, 0]
            x_safe = x.clamp_min(0.20)

            cy = ((-cam[:, 1] / x_safe) / fov_x).clamp(-1.5, 1.5)[:, None, None]
            cz = ((-cam[:, 2] / x_safe) / fov_y).clamp(-1.5, 1.5)[:, None, None]
            core_half_y = _batch_scalar(effects.get('glare_core_half_y'), self.simple_slit_half_y)
            core_half_z = _batch_scalar(effects.get('glare_core_half_z'), self.simple_slit_effect_half_z)
            sy = (core_half_y.to(depth.device, depth.dtype) / x_safe / fov_x).clamp(0.04, 1.25)[:, None, None]
            sz = (core_half_z.to(depth.device, depth.dtype) / x_safe / fov_y).clamp(0.06, 1.25)[:, None, None]
            soft = (softness.to(depth.device, depth.dtype) / x_safe / fov_x).clamp(0.025, 0.18)[:, None, None]
            in_front = torch.sigmoid((x - 0.08) / 0.04)[:, None, None]
            core_mask_y = torch.sigmoid((sy - (yy_o - cy).abs()) / soft)
            core_mask_z = torch.sigmoid((sz - (zz_o - cz).abs()) / soft)
            core = (core_mask_y * core_mask_z * in_front).clamp(0.0, 1.0)

            halo_half_y = _batch_scalar(effects.get('glare_halo_half_y'), self.simple_slit_half_y)
            halo_half_z = _batch_scalar(effects.get('glare_halo_half_z'), self.simple_slit_effect_half_z)
            halo_strength = _batch_scalar(effects.get('glare_halo_strength'), self.simple_glare_halo_strength).clamp(0.0, 1.0)
            halo_sy = (halo_half_y.to(depth.device, depth.dtype) / x_safe / fov_x).clamp(0.04, 1.25)[:, None, None]
            halo_sz = (halo_half_z.to(depth.device, depth.dtype) / x_safe / fov_y).clamp(0.06, 1.25)[:, None, None]
            halo_soft = (softness.to(depth.device, depth.dtype) * 1.7 / x_safe / fov_x).clamp(0.035, 0.28)[:, None, None]
            halo_mask_y = torch.sigmoid((halo_sy - (yy_o - cy).abs()) / halo_soft)
            halo_mask_z = torch.sigmoid((halo_sz - (zz_o - cz).abs()) / halo_soft)
            halo = (halo_mask_y * halo_mask_z * in_front * halo_strength[:, None, None]).clamp(0.0, 1.0)
            return torch.maximum(core, halo).clamp(0.0, 1.0)

        if bool(side_patch_selector.all().item()):
            return _side_wall_patch_mask()
        if not bool(side_patch_selector.any().item()):
            return _opening_mask()
        side_patch_mask = _side_wall_patch_mask()
        opening_mask = _opening_mask()
        return torch.where(side_patch_selector[:, None, None], side_patch_mask, opening_mask)

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
        power_quad = p.square() * (0.38 + 0.62 * torch.sigmoid((e01 - 0.50) / 0.10))
        power_knee = torch.sigmoid((p - 0.56) / 0.085) * (0.25 + 0.75 * p)
        exposure_quad = e01.square() * (0.20 + 0.80 * torch.sigmoid((p - 0.48) / 0.10))
        exposure_bloom = torch.sigmoid((e01 - 0.58) / 0.100) * (0.35 + 0.65 * torch.sigmoid((g01 - 0.48) / 0.09))
        gain_quad = g01.square() * (0.18 + 0.82 * torch.sigmoid((e01 - 0.48) / 0.10))
        gain_bloom = torch.sigmoid((g01 - 0.48) / 0.090) * (0.35 + 0.65 * torch.sigmoid((e01 - 0.44) / 0.10))
        spec_safe = (
            torch.sigmoid((0.48 - p) / 0.080)
            * torch.sigmoid((0.52 - e01) / 0.10)
            * torch.sigmoid((0.42 - g01) / 0.09)
        )
        spec_very_safe = (
            torch.sigmoid((0.28 - p) / 0.060)
            * torch.sigmoid((0.34 - e01) / 0.070)
            * torch.sigmoid((0.24 - g01) / 0.070)
        )
        spec_penalty = mask * (
            0.36 * power_quad
            + 0.74 * power_knee
            + 0.22 * exposure_quad
            + 0.26 * exposure_bloom
            + 0.22 * gain_quad
            + 0.26 * gain_bloom
        )
        spec_bonus = mask * (0.34 * spec_safe + 0.18 * spec_very_safe)
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
        valid_prob = torch.sigmoid((quality - 0.42) / 0.055)
        hard_valid = (valid_prob > 0.5).to(raw.dtype)
        valid_st = hard_valid.detach() - valid_prob.detach() + valid_prob
        depth_obs = raw * valid_st
        quality_obs = quality * valid_st

        artifact = self._key_cue_artifacts(
            raw, power, exposure, gain, quality, valid_prob, hard_valid, depth_obs)
        depth_obs = artifact['depth_obs']
        quality_obs = artifact['quality_obs']
        quality = artifact['quality']
        valid_prob = artifact['valid_prob']
        hard_valid = artifact['hard_valid']
        valid_st = artifact['valid_st']
        cue_mask = artifact['cue_mask']
        combined_mask = torch.maximum(mask, cue_mask).clamp(0.0, 1.0)
        effect = (effect + artifact['artifact_effect']).clamp(0.0, 1.0)
        quality_pre_valid = quality

        invalid = (1.0 - valid_st).clamp(0.0, 1.0)
        mask_mass = combined_mask.sum(dim=(-2, -1)).clamp_min(1e-6)
        scalars = {
            'quality_mean': quality_obs.mean(dim=(-2, -1)),
            'invalid_rate': invalid.mean(dim=(-2, -1)),
            'scene_effect_mean': effect.mean(dim=(-2, -1)),
            'scene_mask_mean': combined_mask.mean(dim=(-2, -1)),
            'slit_cue_mask_mean': cue_mask.mean(dim=(-2, -1)),
            'key_cue_artifact_mean': artifact['artifact'].mean(dim=(-2, -1)),
            'glare_quality_mean': (quality_obs * combined_mask).sum(dim=(-2, -1)) / mask_mass,
            'glare_invalid_rate': (invalid * combined_mask).sum(dim=(-2, -1)) / mask_mass,
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
            'scene_mask': combined_mask,
            'slit_cue_mask': cue_mask,
            'key_cue_artifact_map': artifact['artifact'],
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
        artifact = self._key_cue_artifacts(
            raw, power, exposure, gain, quality, valid_prob, hard_valid, depth_obs)
        depth_obs = artifact['depth_obs']
        quality_obs = artifact['quality_obs']
        quality = artifact['quality']
        valid_prob = artifact['valid_prob']
        hard_valid = artifact['hard_valid']
        valid_st = artifact['valid_st']
        cue_mask = artifact['cue_mask']
        combined_mask = torch.maximum(mask, cue_mask).clamp(0.0, 1.0)
        effect = (effect + artifact['artifact_effect']).clamp(0.0, 1.0)
        invalid = artifact['invalid']
        mask_mass = combined_mask.sum(dim=(-2, -1)).clamp_min(1e-6)
        scalars = {
            'quality_mean': quality_obs.mean(dim=(-2, -1)),
            'invalid_rate': invalid.mean(dim=(-2, -1)),
            'scene_effect_mean': effect.mean(dim=(-2, -1)),
            'scene_mask_mean': combined_mask.mean(dim=(-2, -1)),
            'slit_cue_mask_mean': cue_mask.mean(dim=(-2, -1)),
            'key_cue_artifact_mean': artifact['artifact'].mean(dim=(-2, -1)),
            'glare_quality_mean': (quality_obs * combined_mask).sum(dim=(-2, -1)) / mask_mass,
            'glare_invalid_rate': (invalid * combined_mask).sum(dim=(-2, -1)) / mask_mass,
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
            'scene_mask': combined_mask,
            'slit_cue_mask': cue_mask,
            'key_cue_artifact_map': artifact['artifact'],
            'scalars': scalars,
        })
        return depth_obs, quality_obs

    def render_diff_depth(self, power, exposure, gain, max_range=None):
        _ = max_range
        self._update_corridor_active_stage()
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
        self._update_corridor_active_stage()
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
        # 历史接口名保留为 world，但 voxel box 本身是局部场景坐标；
        # rerun_vis 会根据 scene_yaw 统一旋转到 world，避免双重旋转。
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

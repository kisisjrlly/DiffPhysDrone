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

    几何结构刻意保持很小：一个起点、一个终点、一堵墙，以及墙上随机横向位置
    的竖直细缝。不同 scene 只改变细缝/墙边附近的局部传感器退化方式。
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
                 simple_slit_center_y_min=-0.55, simple_slit_center_y_max=0.55,
                 simple_slit_half_y=0.20, simple_slit_half_y_min=None, simple_slit_half_y_max=None,
                 simple_slit_effect_half_z=0.26,
                 simple_slit_center_z=1.50,
                 simple_slit_side_effect_width_y=0.50,
                 simple_slit_side_effect_half_z=1.10,
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
        self.simple_wall_half_x = 0.10
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
                                      back_wall_x=None):
        """构造一堵带竖直细缝的墙。

        这里没有物理上/下门框；z 方向的 effect 半高只属于传感器退化 mask。
        """
        wall_x = self.simple_wall_x if wall_x is None else float(wall_x)
        slit_center_z = self.simple_slit_center_z if slit_center_z is None else float(slit_center_z)
        slit_half_y = self.simple_slit_half_y if slit_half_y is None else float(slit_half_y)
        wall_half_y = 1.0
        wall_half_z = self.simple_wall_half_z * 2
        wall_thickness = self.simple_wall_half_x
        back_wall_x = self.simple_back_wall_x_min if back_wall_x is None else float(back_wall_x)
        back_wall_half_y = 3.0
        # 前墙由左右两块竖直墙体组成，中间空出的部分就是细缝。终点后方放一堵
        # 背墙，让相机透过细缝时仍能获得深度返回，而不是整片无效深度。
        wall_slit_voxels = self._build_voxels([
            [float(wall_x), slit_center_y - float(slit_half_y) - wall_half_y, slit_center_z, wall_thickness, wall_half_y, wall_half_z],
            [float(wall_x), slit_center_y + float(slit_half_y) + wall_half_y, slit_center_z, wall_thickness, wall_half_y, wall_half_z],
            [back_wall_x, 0.0, slit_center_z, wall_thickness, back_wall_half_y, wall_half_z],
        ])
        return wall_slit_voxels

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

    def _scene_effects(self, scene_name, slot_name, slit_center_y, slit_half_y, back_wall_x):
        regime_id = float(self.scene_name_to_id[scene_name])
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
            'hazard_center': [self.simple_wall_x, float(slit_center_y), self.simple_slit_center_z],
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
            'geometry_wall_x': float(self.simple_wall_x),
            'geometry_wall_half_x': float(self.simple_wall_half_x),
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
            self._scene_effects(scene_names[i], slots[i], float(slit_center_y[i]), float(slit_half_y[i]), float(back_wall_x[i]))
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
            'aperture_artifact_map',
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
        raw = debug.get('raw_depth_map', None)
        scene_mask = debug.get('scene_mask', None)
        if torch.is_tensor(raw) and raw.ndim >= 3 and raw.shape[0] > 0:
            front_hit, back_hit = self._scene_hit_masks(raw)
            hit_idx = int(min(max(env_idx, 0), raw.shape[0] - 1))
            front_np = front_hit[hit_idx].detach().cpu().numpy()
            back_np = back_hit[hit_idx].detach().cpu().numpy()
            out['images']['front_wall_hit_mask'] = front_np
            out['images']['back_wall_hit_mask'] = back_np
            out['scalars']['front_wall_hit_mean'] = float(front_hit[hit_idx].mean().detach().cpu().item())
            out['scalars']['back_wall_hit_mean'] = float(back_hit[hit_idx].mean().detach().cpu().item())
            if torch.is_tensor(scene_mask) and scene_mask.ndim >= 3 and scene_mask.shape[0] > 0:
                mask_idx = int(min(max(env_idx, 0), scene_mask.shape[0] - 1))
                leak = scene_mask[mask_idx].to(back_hit.device, back_hit.dtype) * back_hit[hit_idx]
                out['scalars']['scene_mask_on_back_wall_mean'] = float(leak.mean().detach().cpu().item())
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

    def _key_cue_artifacts(self, raw, scene_mask, power, exposure, gain, quality, valid_prob, hard_valid, depth_obs):
        """Apply opening-only artifacts that are not part of side-wall material patches.

        Glare is an opening/back-light effect and can corrupt the slit/back-wall
        cue.  Dark and specular are modeled as side-wall material patches, so
        they must not directly corrupt the open slit/back-wall cue here.
        """
        B = raw.shape[0]
        cue_mask = self._slit_cue_mask(raw)
        side_mask = scene_mask.clamp(0.0, 1.0)
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

        glare_bad = (
            0.72 * torch.sigmoid((e01 - 0.26) / 0.055)
            + 0.50 * torch.sigmoid((g01 - 0.24) / 0.060)
            + 0.30 * torch.sigmoid((0.42 - p) / 0.09)
        ).clamp(0.0, 1.0)
        # Specular material artifacts belong to the side-wall patches, not the
        # open slit cue.  This term is kept for reporting/debug maps, but is
        # gated by side_mask below rather than cue_mask.
        spec_power_hot = torch.sigmoid((p - 0.47) / 0.075)
        spec_exposure_hot = torch.sigmoid((e01 - 0.58) / 0.100)
        spec_gain_hot = torch.sigmoid((g01 - 0.36) / 0.080)
        spec_joint_hot = torch.maximum(spec_exposure_hot, spec_gain_hot)
        spec_bloom = (
            0.52 * spec_power_hot
            + 0.46 * spec_gain_hot
            + 0.32 * spec_power_hot * spec_joint_hot
            + 0.22 * spec_exposure_hot * spec_gain_hot
        ).clamp(0.0, 1.0)
        spec_safe = (
            torch.sigmoid((0.48 - p) / 0.080)
            * torch.sigmoid((0.52 - e01) / 0.100)
            * torch.sigmoid((0.42 - g01) / 0.090)
        )
        dark_weak_return = (
            0.34 * torch.sigmoid((0.52 - p) / 0.085)
            + 0.60 * torch.sigmoid((0.60 - e01) / 0.070)
            + 0.58 * torch.sigmoid((0.50 - g01) / 0.070)
        ).clamp(0.0, 1.0)
        dark_recovery = (
            torch.sigmoid((e01 - 0.60) / 0.070)
            * torch.sigmoid((g01 - 0.50) / 0.075)
            * (0.55 + 0.45 * torch.sigmoid((p - 0.42) / 0.10))
        ).clamp(0.0, 1.0)

        scene_ids = getattr(self, 'current_scene_ids', None)
        if scene_ids is None:
            scene_ids = torch.full((B,), int(self.current_scene_id), device=raw.device, dtype=torch.long)
        else:
            scene_ids = scene_ids.to(device=raw.device, dtype=torch.long)
        sid = scene_ids[:, None, None]
        glare_cue_artifact = cue_mask * glare_bad * strength

        raw4 = raw[:, None]
        raw_far = F.max_pool2d(raw4, 3, stride=1, padding=1)[:, 0]
        raw_near = -F.max_pool2d(-raw4, 3, stride=1, padding=1)[:, 0]
        local_edge = ((raw_far - raw_near) / (raw + 0.18)).clamp(0.0, 1.0)
        back_wall_like = torch.sigmoid((raw - (raw_near + 0.42 * (raw_far - raw_near + 1e-6))) / 0.045)
        aperture_edge_gate = torch.sigmoid((local_edge - 0.045) / 0.030)
        H, W = raw.shape[-2:]
        ys = torch.linspace(-1.0, 1.0, W, device=raw.device, dtype=raw.dtype)
        zs = torch.linspace(-1.0, 1.0, H, device=raw.device, dtype=raw.dtype)
        yy, zz = torch.meshgrid(ys, zs, indexing='xy')

        def _hash_noise(h, w, salt):
            gy = torch.arange(h, device=raw.device, dtype=raw.dtype)
            gx = torch.arange(w, device=raw.device, dtype=raw.dtype)
            gy, gx = torch.meshgrid(gy, gx, indexing='ij')
            value = torch.sin(
                (gx + 17.31 * float(salt)) * 12.9898
                + (gy - 7.17 * float(salt)) * 78.233
                + float(salt) * 0.123
            ) * 43758.5453
            return value - torch.floor(value)

        def _correlated_noise(cell_h, cell_w, salt):
            coarse_h = max(2, int(math.ceil(float(H) / float(cell_h))))
            coarse_w = max(2, int(math.ceil(float(W) / float(cell_w))))
            base = _hash_noise(coarse_h, coarse_w, salt)[None, None]
            return F.interpolate(
                base,
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            )[0, 0]

        # D455-like failures should look like irregular correlation-window and
        # projected-pattern dropouts, not a clean sinusoidal texture.  The maps
        # below are deterministic for reproducibility but multi-scale and
        # aperiodic enough to avoid reviewer-visible checker/stripe artifacts.
        coarse_noise = _correlated_noise(13, 17, 1.0)
        mid_noise = _correlated_noise(7, 9, 2.0)
        fine_noise = _hash_noise(H, W, 3.0)
        material_texture = (
            0.48 * coarse_noise
            + 0.34 * mid_noise
            + 0.18 * fine_noise
        ).clamp(0.0, 1.0).unsqueeze(0)
        aperture_noise = (
            0.52 * _correlated_noise(11, 15, 4.0)
            + 0.30 * _correlated_noise(5, 7, 5.0)
            + 0.18 * _hash_noise(H, W, 6.0)
        ).clamp(0.0, 1.0).unsqueeze(0)
        edge_noise = (
            0.46 * _correlated_noise(5, 11, 7.0)
            + 0.34 * _correlated_noise(3, 5, 8.0)
            + 0.20 * _hash_noise(H, W, 9.0)
        ).clamp(0.0, 1.0).unsqueeze(0)
        row_noise = (
            0.58 * _correlated_noise(3, 96, 21.0)
            + 0.42 * _correlated_noise(5, 96, 22.0)
        ).clamp(0.0, 1.0).unsqueeze(0)
        col_noise = (
            0.55 * _correlated_noise(96, 4, 23.0)
            + 0.45 * _correlated_noise(96, 7, 24.0)
        ).clamp(0.0, 1.0).unsqueeze(0)
        aperture_edge_irregular = (
            0.36 * edge_noise
            + 0.34 * aperture_noise
            + 0.18 * row_noise
            + 0.12 * col_noise
        ).clamp(0.0, 1.0)
        # The material probes and paper figures are judged on observed depth,
        # not only on latent quality.  Low-reflectance and specular side-wall
        # patches therefore need patch-wide structured dropout/false depth,
        # with extra damage at depth discontinuities.  Keeping this gated by
        # side_mask prevents the material effect from leaking into the back wall
        # visible through the slit.
        spec_hole_shape = (0.50 + 0.24 * local_edge + 0.26 * material_texture).clamp(0.0, 1.0)
        dark_hole_shape = (0.66 + 0.10 * local_edge + 0.24 * material_texture).clamp(0.0, 1.0)
        spec_artifact = (side_mask * spec_bloom * strength * spec_hole_shape).clamp(0.0, 1.0)
        dark_artifact = (
            side_mask
            * dark_weak_return
            * (1.0 - 0.70 * dark_recovery)
            * strength
            * dark_hole_shape
        ).clamp(0.0, 1.0)
        artifact = torch.where(
            sid == 0,
            glare_cue_artifact,
            torch.where(sid == 1, spec_artifact, dark_artifact),
        )

        glare_quality = quality - glare_cue_artifact * (0.58 + 0.34 * glare_bad)
        material_artifact = torch.where(
            sid == 1,
            spec_artifact,
            torch.where(sid == 2, dark_artifact, torch.zeros_like(raw)),
        )
        material_mass = side_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        material_context = (material_artifact * side_mask).sum(dim=(-2, -1), keepdim=True) / material_mass
        material_support = F.max_pool2d(material_artifact[:, None], 17, stride=1, padding=8)[:, 0]
        material_support_wide = F.max_pool2d(material_artifact[:, None], 45, stride=1, padding=22)[:, 0]

        # Depth cameras do not observe a perfectly crisp front-wall/back-wall
        # discontinuity at a narrow aperture.  If the adjacent side-wall material
        # loses structured-light/stereo support, the aperture cue also becomes
        # mottled because the matching window straddles invalid or false-return
        # pixels.  This is deliberately driven by the side-wall material artifact
        # above; the raw back wall is not reclassified as dark/specular material.
        aperture_edge = (
            cue_mask
            * back_wall_like
            * aperture_edge_gate
        ).clamp(0.0, 1.0)
        aperture_body = (cue_mask * back_wall_like).clamp(0.0, 1.0)
        aperture_speckle = aperture_noise
        aperture_area = aperture_body.mean(dim=(-2, -1), keepdim=True)
        aperture_large = torch.sigmoid((aperture_area - 0.30) / 0.080)
        aperture_tiny = torch.sigmoid((0.018 - aperture_area) / 0.006)
        cue4 = cue_mask[:, None]
        cue_max = F.max_pool2d(cue4, 5, stride=1, padding=2)[:, 0]
        cue_min = -F.max_pool2d(-cue4, 5, stride=1, padding=2)[:, 0]
        cue_boundary = (cue_max - cue_min).clamp(0.0, 1.0)
        # A D455-like matching window can smear the aperture boundary into the
        # slit, but it should not erase a whole narrow opening.  Keep a narrow
        # boundary influence for hard invalidation, and use a wider/softer term
        # only for mild quality and false-depth effects.
        cue_edge_window = F.max_pool2d(cue_boundary[:, None], 13, stride=1, padding=6)[:, 0]
        cue_side_window = F.max_pool2d(cue_boundary[:, None], 23, stride=1, padding=11)[:, 0]
        cue_support_window = F.avg_pool2d(cue_boundary[:, None], 25, stride=1, padding=12)[:, 0].clamp(0.0, 1.0)
        material_response_bad = torch.where(
            sid == 1,
            (spec_bloom * (1.0 - 0.42 * spec_safe)).clamp(0.0, 1.0),
            torch.where(
                sid == 2,
                (dark_weak_return * (1.0 - 0.68 * dark_recovery)).clamp(0.0, 1.0),
                torch.zeros_like(raw),
            ),
        )
        # Aperture ambiguity should be strongest near the depth discontinuity
        # and only partially affect the slit interior.  The second wall seen
        # through the opening is ordinary material, so central back-wall pixels
        # should suffer only sparse matching failures rather than wholesale
        # invalidation.
        aperture_body_gate = (
            aperture_body
            * (0.18 + 0.82 * aperture_edge_gate)
            * (0.74 + 0.26 * aperture_speckle)
        ).clamp(0.0, 1.0)
        aperture_center_gate = (
            aperture_body
            * (1.0 - aperture_edge_gate).clamp(0.0, 1.0)
            * (0.76 + 0.24 * aperture_speckle)
        ).clamp(0.0, 1.0)
        aperture_window_loss = (
            aperture_body_gate
            * material_response_bad
            * strength
            * (0.50 + 0.34 * cue_edge_window + 0.14 * cue_support_window)
        ).clamp(0.0, 1.0)
        support_loss = (
            0.42 * material_support
            + 0.28 * material_context
            + 0.92 * aperture_window_loss
        ).clamp(0.0, 1.0)
        support_edge = (
            aperture_edge
            * support_loss
            * (0.36 + 0.64 * aperture_edge_irregular)
        ).clamp(0.0, 1.0)
        support_body = (
            aperture_body_gate
            * support_loss
            * (0.58 + 0.42 * aperture_speckle)
        ).clamp(0.0, 1.0)
        dark_selector = (sid == 2).to(raw.dtype)
        spec_selector = (sid == 1).to(raw.dtype)
        material_selector = ((sid == 1) | (sid == 2)).to(raw.dtype)
        # Bad dark/specular settings should not leave a perfectly vertical,
        # fixed-width aperture outline.  Real stereo/IR depth around a narrow
        # slit tends to have row-dependent edge erosion, mixed pixels and flying
        # depths because the matching support straddles invalid side material.
        edge_fray = (
            aperture_edge
            * material_selector
            * material_response_bad
            * strength
            * (0.42 + 0.58 * aperture_edge_irregular)
        ).clamp(0.0, 1.0)
        edge_fray = F.max_pool2d(edge_fray[:, None], (5, 7), stride=1, padding=(2, 3))[:, 0]
        edge_fray = (edge_fray * aperture_body * (0.74 + 0.26 * cue_edge_window)).clamp(0.0, 1.0)
        dark_aperture_artifact = (support_edge * dark_selector).clamp(0.0, 1.0)
        spec_aperture_artifact = (support_edge * spec_selector).clamp(0.0, 1.0)
        dark_body_artifact = (support_body * dark_selector).clamp(0.0, 1.0)
        spec_body_artifact = (support_body * spec_selector).clamp(0.0, 1.0)
        near_specular = torch.sigmoid((0.85 - raw) / 0.18)
        spec_quality = (
            quality
            - spec_artifact * (1.85 + 1.15 * spec_bloom + 0.95 * near_specular)
            - spec_aperture_artifact * (2.20 + 1.05 * spec_bloom)
            - spec_body_artifact * (0.28 + 0.18 * spec_bloom)
        )
        dark_quality = (
            quality
            - dark_artifact * (2.45 + 1.05 * dark_weak_return)
            - dark_aperture_artifact * (2.32 + 1.05 * dark_weak_return)
            - dark_body_artifact * (0.30 + 0.20 * dark_weak_return)
        )
        cue_quality = torch.where(
            sid == 0,
            glare_quality,
            torch.where(sid == 1, spec_quality, dark_quality),
        ).clamp(0.0, 1.0)
        material_aperture_center = (material_selector * aperture_center_gate).clamp(0.0, 1.0)
        # The second wall visible through the aperture is ordinary material.
        # Bad dark/specular side patches can reduce confidence in those pixels,
        # but should not make the aperture center disappear as a solid black
        # stripe.  Preserve a weak, speckled quality floor there; edge pixels
        # remain governed by the stronger aperture dropout below.
        center_quality_floor = (
            0.405
            + 0.050 * (1.0 - material_response_bad)
            + 0.050 * aperture_speckle
            - 0.015 * spec_selector * spec_bloom
        ).clamp(0.38, 0.52)
        cue_quality = torch.where(
            material_aperture_center > 0.20,
            torch.maximum(cue_quality, center_quality_floor),
            cue_quality,
        ).clamp(0.0, 1.0)
        cue_valid_prob = torch.sigmoid((cue_quality - 0.42) / 0.055)
        material_aperture_artifact = torch.where(
            sid == 1,
            torch.maximum(spec_aperture_artifact, spec_body_artifact),
            torch.where(sid == 2, torch.maximum(dark_aperture_artifact, dark_body_artifact), torch.zeros_like(raw)),
        )
        aperture_mismatch_field = (
            material_selector
            * aperture_body
            * strength
            * (
                0.18 * material_response_bad
                + 0.70 * material_response_bad * material_support_wide
                + 0.22 * material_support_wide
                + 0.22 * aperture_edge_gate
                + 0.28 * edge_fray
                + 0.12 * cue_support_window
                + 0.14 * aperture_speckle
            )
        ).clamp(0.0, 1.0)
        aperture_mismatch_threshold = (
            0.14
            + 0.18 * (
                0.58 * _correlated_noise(9, 13, 14.0)
                + 0.42 * _correlated_noise(4, 7, 15.0)
            ).unsqueeze(0)
        ).clamp(0.08, 0.40)
        aperture_mismatch_prob = torch.sigmoid(
            (aperture_mismatch_field - aperture_mismatch_threshold) / 0.060
        ).clamp(0.0, 1.0)
        aperture_mismatch_hard = (aperture_mismatch_prob > 0.5).to(raw.dtype)
        aperture_mismatch = (
            aperture_mismatch_hard.detach()
            - aperture_mismatch_prob.detach()
            + aperture_mismatch_prob
        )
        # Hard invalidation should be an aperture-edge phenomenon.  Earlier
        # versions let the wide support window erase the whole back-wall stripe
        # visible through the slit, which is not a good D455-like model: bad
        # side material should fray/mix the edge before it turns the entire
        # opening black.
        aperture_dropout_field = (
            material_selector
            * aperture_body
            * material_response_bad
            * strength
            * (0.05 + 0.72 * aperture_edge_gate + 0.14 * cue_support_window + 0.32 * edge_fray)
            * (0.55 + 0.45 * aperture_speckle)
            * (0.72 + 0.48 * edge_fray)
            * (1.02 + 0.14 * spec_selector * spec_bloom)
        ).clamp(0.0, 1.0)
        aperture_dropout_threshold = (
            0.27
            + 0.23 * (
                0.62 * _correlated_noise(8, 12, 10.0)
                + 0.38 * _correlated_noise(4, 6, 11.0)
            ).unsqueeze(0)
        ).clamp(0.15, 0.56)
        aperture_dropout_prob = (
            torch.sigmoid((aperture_dropout_field - aperture_dropout_threshold) / 0.070)
            * (0.30 + 0.42 * aperture_edge_gate + 0.28 * edge_fray)
        ).clamp(0.0, 1.0)
        edge_template_field = (
            aperture_edge
            * material_selector
            * material_response_bad
            * strength
            * (0.68 + 0.32 * cue_edge_window)
            * (0.52 + 0.48 * aperture_edge_irregular)
        ).clamp(0.0, 1.0)
        edge_template_threshold = (
            0.24
            + 0.22 * (
                0.54 * row_noise
                + 0.28 * edge_noise
                + 0.18 * aperture_noise
            )
        ).clamp(0.14, 0.50)
        edge_template_dropout_prob = (
            torch.sigmoid((edge_template_field - edge_template_threshold) / 0.055)
            * (0.22 + 0.52 * aperture_edge_irregular)
        ).clamp(0.0, 1.0)
        body_template_field = (
            aperture_body
            * material_selector
            * material_response_bad
            * strength
            * (
                0.10
                + 0.22 * aperture_speckle
                + 0.16 * row_noise
                + 0.28 * cue_edge_window
                + 0.18 * aperture_body_gate
            )
        ).clamp(0.0, 1.0)
        body_template_threshold = (
            0.36
            + 0.18 * (
                0.46 * aperture_noise
                + 0.34 * row_noise
                + 0.20 * col_noise
            )
        ).clamp(0.24, 0.58)
        body_template_dropout_prob = (
            torch.sigmoid((body_template_field - body_template_threshold) / 0.070)
            * (0.14 + 0.22 * aperture_body_gate + 0.14 * cue_edge_window)
        ).clamp(0.0, 1.0)
        body_flying_gate = (
            torch.sigmoid((aperture_speckle - 0.56) / 0.090)
            * torch.sigmoid((0.62 * row_noise + 0.38 * edge_noise - 0.46) / 0.100)
        ).clamp(0.0, 1.0)
        spec_body_extra_field = (
            aperture_body
            * spec_selector
            * spec_bloom
            * strength
            * (0.20 + 0.36 * cue_edge_window + 0.26 * aperture_speckle)
        ).clamp(0.0, 1.0)
        spec_body_extra_threshold = (
            0.42
            + 0.14 * (0.55 * row_noise + 0.45 * aperture_noise)
        ).clamp(0.30, 0.62)
        spec_body_extra_dropout_prob = (
            torch.sigmoid((spec_body_extra_field - spec_body_extra_threshold) / 0.070)
            * (0.32 + 0.26 * cue_edge_window)
        ).clamp(0.0, 1.0)
        aperture_dropout_hard = (aperture_dropout_prob > 0.5).to(raw.dtype)
        aperture_dropout = (
            aperture_dropout_hard.detach()
            - aperture_dropout_prob.detach()
            + aperture_dropout_prob
        )
        edge_template_dropout_hard = (edge_template_dropout_prob > 0.5).to(raw.dtype)
        edge_template_dropout = (
            edge_template_dropout_hard.detach()
            - edge_template_dropout_prob.detach()
            + edge_template_dropout_prob
        )
        # Body template damage is represented as mixed/flying valid depths below,
        # not as hard holes.  Keeping this out of the invalid mask prevents the
        # unrealistic "whole slit becomes black" failure mode.
        body_template_dropout_hard = (
            (body_template_dropout_prob > 0.38).to(raw.dtype)
            * aperture_body
            * (0.22 + 0.78 * torch.sigmoid((row_noise - 0.46) / 0.10))
            * (0.35 + 0.65 * (1.0 - body_flying_gate))
        ).clamp(0.0, 1.0)
        body_template_dropout = (
            body_template_dropout_hard.detach()
            - body_template_dropout_prob.detach()
            + body_template_dropout_prob
        )
        spec_body_extra_dropout_hard = torch.zeros_like(spec_body_extra_dropout_prob)
        spec_body_extra_dropout = (
            spec_body_extra_dropout_hard.detach()
            - spec_body_extra_dropout_prob.detach()
            + spec_body_extra_dropout_prob
        )
        aperture_dropout = torch.maximum(aperture_dropout, edge_template_dropout)
        aperture_dropout = torch.maximum(aperture_dropout, body_template_dropout)
        aperture_dropout_hard = torch.maximum(aperture_dropout_hard, edge_template_dropout_hard)
        aperture_dropout_hard = torch.maximum(aperture_dropout_hard, body_template_dropout_hard)
        spatial_valid_threshold = (
            0.47
            + material_selector
            * material_aperture_artifact
            * (0.08 + 0.18 * aperture_speckle)
        ).clamp(0.40, 0.72)
        cue_hard_valid = (cue_valid_prob > spatial_valid_threshold).to(raw.dtype)
        cue_hard_valid = cue_hard_valid * (1.0 - aperture_dropout)
        # A narrow aperture surrounded by poor IR-return material often yields
        # valid but wrong "flying" depths rather than a uniformly black hole.
        # Preserve those mismatched returns as valid so the observed slit is
        # degraded without erasing the ordinary back wall behind it.
        aperture_false_valid_prob = (
            torch.maximum(
                aperture_mismatch_prob,
                torch.maximum(
                    edge_template_dropout_prob
                    * (0.24 + 0.22 * aperture_edge_irregular)
                    * (1.0 - 0.66 * aperture_edge_gate),
                    body_template_dropout_prob * (0.06 + 0.20 * aperture_speckle) * body_flying_gate,
                ),
            )
            * aperture_body
            * material_selector
            * (0.58 + 0.42 * cue_edge_window)
            * (0.78 + 0.22 * aperture_center_gate)
            * (1.0 - 0.42 * aperture_dropout_hard)
            * (1.0 - 0.36 * edge_template_dropout_hard)
            * (1.0 - 0.42 * body_template_dropout_hard)
            * (1.0 - 0.42 * spec_body_extra_dropout_hard)
            * (1.0 - 0.42 * edge_fray)
        ).clamp(0.0, 1.0)
        aperture_false_valid_hard = (aperture_false_valid_prob > 0.50).to(raw.dtype)
        aperture_false_valid = (
            aperture_false_valid_hard.detach()
            - aperture_false_valid_prob.detach()
            + aperture_false_valid_prob
        )
        # Keep a weak center return through the aperture under bad dark/specular
        # settings.  It is intentionally represented as a false/mixed depth
        # later, not as a clean far-wall cue.  This avoids the unrealistic
        # failure mode where a visible physical opening becomes a solid black
        # column while still degrading fixed-camera observations.
        aperture_center_false_valid_field = (
            aperture_center_gate
            * material_selector
            * strength
            * (
                0.20 * material_response_bad
                + 0.14 * material_response_bad * material_support_wide
                + 0.07 * cue_support_window * material_response_bad
            )
            * (0.56 + 0.44 * aperture_speckle)
            * (1.0 - 0.72 * body_template_dropout_prob)
            * (0.46 + 0.54 * body_flying_gate)
        ).clamp(0.0, 1.0)
        aperture_center_false_valid_threshold = (
            0.22
            + 0.18 * (
                0.44 * row_noise
                + 0.34 * aperture_noise
                + 0.22 * col_noise
            )
        ).clamp(0.16, 0.48)
        aperture_center_false_valid_prob = torch.sigmoid(
            (aperture_center_false_valid_field - aperture_center_false_valid_threshold) / 0.065
        ).clamp(0.0, 1.0)
        aperture_center_false_valid_hard = (aperture_center_false_valid_prob > 0.54).to(raw.dtype)
        aperture_center_false_valid = (
            aperture_center_false_valid_hard.detach()
            - aperture_center_false_valid_prob.detach()
            + aperture_center_false_valid_prob
        )
        # When the visible aperture is only a few pixels wide, every back-wall
        # pixel can be classified as an edge by the local 3x3 depth test.  Real
        # D455 observations may be poor in this case, but the opening should
        # not become a perfectly black column for all weak-return settings.
        # Keep a sparse set of valid mixed returns for thin apertures.
        thin_aperture_false_valid_field = (
            aperture_body
            * aperture_tiny
            * material_selector
            * material_response_bad
            * strength
            * (0.34 + 0.66 * aperture_speckle)
        ).clamp(0.0, 1.0)
        thin_aperture_false_valid_threshold = (
            0.18
            + 0.18 * (
                0.50 * aperture_noise
                + 0.30 * row_noise
                + 0.20 * col_noise
            )
        ).clamp(0.12, 0.42)
        thin_aperture_false_valid_prob = torch.sigmoid(
            (thin_aperture_false_valid_field - thin_aperture_false_valid_threshold) / 0.070
        ).clamp(0.0, 1.0)
        thin_aperture_false_valid_hard = (thin_aperture_false_valid_prob > 0.45).to(raw.dtype)
        thin_aperture_false_valid = (
            thin_aperture_false_valid_hard.detach()
            - thin_aperture_false_valid_prob.detach()
            + thin_aperture_false_valid_prob
        )
        # Back-wall pixels exactly on the aperture edge should not become a
        # perfectly black ruler.  Some of them remain valid as mixed/flying
        # returns; because this is gated by aperture_edge rather than side_mask,
        # it breaks the clean edge without reviving the adjacent bad material.
        aperture_edge_false_valid_field = (
            aperture_edge
            * material_selector
            * material_response_bad
            * strength
            * (0.42 + 0.58 * aperture_edge_irregular)
            * torch.sigmoid((aperture_noise - 0.42) / 0.10)
            * (1.0 - 0.35 * aperture_dropout_hard)
        ).clamp(0.0, 1.0)
        aperture_edge_false_valid_threshold = (
            0.18
            + 0.20 * (
                0.50 * row_noise
                + 0.30 * edge_noise
                + 0.20 * col_noise
            )
        ).clamp(0.12, 0.42)
        aperture_edge_false_valid_prob = torch.sigmoid(
            (aperture_edge_false_valid_field - aperture_edge_false_valid_threshold) / 0.070
        ).clamp(0.0, 1.0)
        aperture_edge_false_valid_hard = (aperture_edge_false_valid_prob > 0.50).to(raw.dtype)
        aperture_edge_false_valid = (
            aperture_edge_false_valid_hard.detach()
            - aperture_edge_false_valid_prob.detach()
            + aperture_edge_false_valid_prob
        )
        # Never repair side-wall material into a continuous valid strip.  Earlier
        # versions used sparse-looking false-valid terms here to avoid a fully
        # black aperture, but those terms landed on the adjacent material patch
        # and created a stable valid separator between the invalid patch and the
        # slit.  Aperture/back-wall returns are handled below; side material is
        # allowed to fail or produce false depth only through the base material
        # model, not through an edge rescue.
        material_edge_false_valid_prob = torch.zeros_like(raw)
        material_edge_false_valid_hard = torch.zeros_like(raw)
        material_edge_false_valid = (
            material_edge_false_valid_hard.detach()
            - material_edge_false_valid_prob.detach()
            + material_edge_false_valid_prob
        )
        side_material_dropout_field = (
            side_mask
            * material_selector
            * material_response_bad
            * strength
            * (0.70 + 0.18 * cue_edge_window + 0.12 * local_edge)
            * (0.72 + 0.28 * material_texture)
        ).clamp(0.0, 1.0)
        side_material_dropout_threshold = (
            0.22
            + 0.18 * (
                0.46 * material_texture
                + 0.34 * edge_noise
                + 0.20 * row_noise
            )
        ).clamp(0.14, 0.48)
        side_material_dropout_prob = torch.sigmoid(
            (side_material_dropout_field - side_material_dropout_threshold) / 0.070
        ).clamp(0.0, 1.0)
        side_material_dropout_hard = (side_material_dropout_prob > 0.50).to(raw.dtype)
        side_material_dropout = (
            side_material_dropout_hard.detach()
            - side_material_dropout_prob.detach()
            + side_material_dropout_prob
        )
        body_gap_dropout_field = (
            aperture_body
            * material_selector
            * material_response_bad
            * strength
            * (0.44 + 0.56 * aperture_speckle)
            * (0.42 + 0.58 * row_noise)
            * (0.70 + 0.30 * (1.0 - aperture_edge_gate))
        ).clamp(0.0, 1.0)
        body_gap_dropout_threshold = (
            0.28
            + 0.18 * (
                0.40 * aperture_noise
                + 0.34 * col_noise
                + 0.26 * edge_noise
            )
        ).clamp(0.18, 0.54)
        body_gap_dropout_prob = (
            torch.sigmoid((body_gap_dropout_field - body_gap_dropout_threshold) / 0.075)
            * aperture_body
            * (0.14 + 0.56 * material_response_bad + 0.16 * dark_selector * dark_weak_return)
            * (1.0 - 0.55 * body_flying_gate)
            * (1.0 - 0.58 * dark_selector * dark_recovery)
        ).clamp(0.0, 1.0)
        body_gap_dropout_hard = (body_gap_dropout_prob > 0.13).to(raw.dtype)
        body_gap_dropout = (
            body_gap_dropout_hard.detach()
            - body_gap_dropout_prob.detach()
            + body_gap_dropout_prob
        )
        cue_hard_valid = torch.maximum(cue_hard_valid, aperture_false_valid)
        cue_hard_valid = torch.maximum(cue_hard_valid, aperture_edge_false_valid)
        cue_hard_valid = torch.maximum(cue_hard_valid, aperture_center_false_valid)
        cue_hard_valid = torch.maximum(cue_hard_valid, thin_aperture_false_valid)
        cue_hard_valid = torch.maximum(cue_hard_valid, material_edge_false_valid)
        cue_hard_valid = cue_hard_valid * (1.0 - side_material_dropout)
        cue_hard_valid = cue_hard_valid * (1.0 - body_gap_dropout)
        valid_st = cue_hard_valid.detach() - cue_valid_prob.detach() + cue_valid_prob

        spec_wrong = (
            side_mask
            * spec_bloom
            * spec_false_strength
            * (0.66 + 0.24 * local_edge + 0.10 * material_texture)
            * (0.75 + 0.45 * near_specular)
        ).clamp(0.0, 1.0)
        left_depth = torch.roll(raw, shifts=1, dims=-1)
        right_depth = torch.roll(raw, shifts=-1, dims=-1)
        edge_drift_depth = torch.minimum(torch.minimum(left_depth, right_depth), raw * 0.70)
        false_depth = torch.minimum(edge_drift_depth, raw * (0.20 + 0.36 * (1.0 - spec_wrong)))
        false_depth = torch.lerp(raw, false_depth, (0.96 * spec_wrong).clamp(0.0, 1.0))
        false_depth = false_depth.clamp_min(float(self.depth_min_valid))
        raw_with_false = torch.where((sid == 1) & (spec_wrong > 0.05), false_depth, raw)
        mixed_edge_depth = torch.lerp(
            raw_near,
            raw_far,
            (0.32 + 0.38 * aperture_speckle + 0.14 * aperture_edge_irregular).clamp(0.0, 1.0),
        )
        dark_edge_target = torch.lerp(
            raw,
            mixed_edge_depth,
            (0.58 + 0.24 * dark_weak_return).clamp(0.0, 0.86),
        ).clamp_min(float(self.depth_min_valid))
        spec_edge_target = torch.lerp(
            raw,
            mixed_edge_depth,
            (0.66 + 0.20 * spec_bloom).clamp(0.0, 0.90),
        ).clamp_min(float(self.depth_min_valid))
        material_edge_target = torch.where(sid == 1, spec_edge_target, dark_edge_target)
        raw_with_false = torch.where(
            material_edge_false_valid > 0.08,
            torch.lerp(raw_with_false, material_edge_target, (0.45 * material_edge_false_valid).clamp(0.0, 0.55)),
            raw_with_false,
        )
        raw_with_false = torch.where(
            aperture_edge_false_valid > 0.08,
            torch.lerp(raw_with_false, material_edge_target, (0.58 * aperture_edge_false_valid).clamp(0.0, 0.70)),
            raw_with_false,
        )
        spec_aperture_wrong = (spec_body_artifact * spec_false_strength * 0.58).clamp(0.0, 1.0)
        aperture_false_depth = torch.lerp(
            raw,
            torch.minimum(edge_drift_depth, raw * 0.58).clamp_min(float(self.depth_min_valid)),
            spec_aperture_wrong,
        )
        raw_with_false = torch.where((sid == 1) & (spec_aperture_wrong > 0.08), aperture_false_depth, raw_with_false)
        dark_aperture_target = torch.minimum(
            edge_drift_depth,
            raw * (0.56 + 0.12 * aperture_speckle),
        ).clamp_min(float(self.depth_min_valid))
        spec_aperture_target = torch.minimum(
            edge_drift_depth,
            raw * (0.50 + 0.12 * aperture_speckle),
        ).clamp_min(float(self.depth_min_valid))
        aperture_target = torch.where(sid == 1, spec_aperture_target, dark_aperture_target)
        aperture_mix = (
            (
                0.62 * aperture_mismatch_prob
                + 0.14 * aperture_mismatch
                + 0.20 * edge_fray
                + 0.10 * body_template_dropout_prob * body_flying_gate
                + 0.22 * aperture_center_false_valid_prob * material_response_bad
                + 0.20 * thin_aperture_false_valid_prob * material_response_bad
                + 0.10 * cue_support_window * material_response_bad
            )
            * material_selector
            * (0.78 + 0.18 * material_response_bad + 0.10 * aperture_edge_gate)
        ).clamp(0.0, 0.90)
        raw_with_false = torch.lerp(raw_with_false, aperture_target, aperture_mix)
        cue_depth_obs = raw_with_false * valid_st
        cue_quality_obs = cue_quality * valid_st
        aperture_artifact = torch.maximum(
            torch.maximum(
                material_aperture_artifact,
                torch.maximum(
                    torch.maximum(aperture_dropout_prob, spec_body_extra_dropout_prob),
                    torch.maximum(
                        torch.maximum(edge_template_dropout_prob, body_template_dropout_prob),
                        torch.maximum(
                            torch.maximum(material_edge_false_valid_prob, aperture_edge_false_valid_prob),
                            torch.maximum(
                                torch.maximum(aperture_center_false_valid_prob, thin_aperture_false_valid_prob),
                                torch.maximum(body_gap_dropout_prob, side_material_dropout_prob),
                            ),
                        ),
                    ),
                ),
            ),
            aperture_mismatch_prob,
        )

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
            'aperture_artifact': aperture_artifact,
            'spec_bloom': spec_bloom.expand_as(raw),
            'spec_wrong': spec_wrong,
            'artifact_effect': artifact + spec_wrong * 0.35 + aperture_artifact,
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
            center_local = effects.get('hazard_center_local', None)
            if center_local is None:
                center_local = torch.bmm(
                    self.R_scene_T.detach().to(depth.device, depth.dtype),
                    center.detach()[:, :, None],
                )[:, :, 0]
            else:
                center_local = center_local.to(depth.device, depth.dtype)
            wall_x = _batch_scalar(effects.get('geometry_wall_x'), self.simple_wall_x)
            wall_half_x = _batch_scalar(effects.get('geometry_wall_half_x'), self.simple_wall_half_x)
            slit_center_y = center_local[:, 1]
            slit_center_z = _batch_scalar(effects.get('slit_center_z'), self.simple_slit_center_z)

            # Reconstruct the raw hit point for each depth pixel in scene-local
            # coordinates.  Dark/specular are wall material properties, so they
            # may only affect rays whose raw geometry actually hit the front wall
            # patch beside the slit, never the back-wall cue visible through it.
            R_cam_scene = torch.bmm(
                self.R_scene_T.detach().to(depth.device, depth.dtype),
                R_cam_world,
            )
            pos_local = torch.bmm(
                self.R_scene_T.detach().to(depth.device, depth.dtype),
                pos[:, :, None],
            )[:, :, 0]
            u = torch.arange(H, device=depth.device, dtype=depth.dtype)
            v = torch.arange(W, device=depth.device, dtype=depth.dtype)
            fu = (2.0 * (u + 0.5) / float(max(H, 1)) - 1.0) * fov_y - 1e-5
            fv = (2.0 * (v + 0.5) / float(max(W, 1)) - 1.0) * fov_x - 1e-5
            fu_grid = fu[None, :, None]
            fv_grid = fv[None, None, :]
            ray = (
                R_cam_scene[:, :, 0][:, :, None, None]
                - fu_grid[:, None] * R_cam_scene[:, :, 2][:, :, None, None]
                - fv_grid[:, None] * R_cam_scene[:, :, 1][:, :, None, None]
            )
            hit = pos_local[:, :, None, None] + depth[:, None] * ray
            hit_x = hit[:, 0]
            hit_y = hit[:, 1]
            hit_z = hit[:, 2]

            wall_soft = torch.full((B,), 0.018, device=depth.device, dtype=depth.dtype)
            side_soft = softness.clamp_min(0.018)
            wall_gate = torch.sigmoid(
                ((wall_half_x[:, None, None] + 0.025) - (hit_x - wall_x[:, None, None]).abs())
                / wall_soft[:, None, None]
            )
            # Side-wall materials are defined from the physical slit edge
            # outward.  A symmetric box sigmoid centered in each patch leaves a
            # half-strength band exactly at the slit edge, which shows up in
            # observed-depth panels as an artificial valid wall-colored stripe
            # between the invalid side patch and the open slit.  Gate from the
            # slit edge outward instead: front-wall hits adjacent to the slit
            # receive full material strength, with softness only at the outer
            # patch boundary.  Back-wall returns through the slit are still
            # protected by wall_gate.
            patch_width = 2.0 * patch_half_y
            dy_from_slit_edge = (hit_y - slit_center_y[:, None, None]).abs() - slit_half_y[:, None, None]
            patch_y = torch.sigmoid(
                (patch_width[:, None, None] - dy_from_slit_edge)
                / side_soft[:, None, None]
            )
            patch_z = torch.sigmoid(
                (patch_half_z[:, None, None] - (hit_z - slit_center_z[:, None, None]).abs())
                / side_soft[:, None, None]
            )
            return (wall_gate * patch_y * patch_z).clamp(0.0, 1.0)

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

    def _scene_hit_masks(self, depth):
        """Return raw-geometry hit masks for front wall and back wall.

        These maps are diagnostic. They let probe scripts distinguish material
        effects on the front wall from accidental leakage into the back-wall
        return visible through the slit.
        """
        B, H, W = depth.shape
        effects = self.current_scene_effects
        device = depth.device
        dtype = depth.dtype
        R_cam_world = (self.R @ self.R_cam).detach().to(device, dtype)
        pos = self.p.detach().to(device, dtype)
        fov_x = torch.as_tensor(float(self._fov_x_half_tan), device=device, dtype=dtype)
        fov_y = fov_x * float(H) / float(max(W, 1))
        R_cam_scene = torch.bmm(self.R_scene_T.detach().to(device, dtype), R_cam_world)
        pos_local = torch.bmm(self.R_scene_T.detach().to(device, dtype), pos[:, :, None])[:, :, 0]

        u = torch.arange(H, device=device, dtype=dtype)
        v = torch.arange(W, device=device, dtype=dtype)
        fu = (2.0 * (u + 0.5) / float(max(H, 1)) - 1.0) * fov_y - 1e-5
        fv = (2.0 * (v + 0.5) / float(max(W, 1)) - 1.0) * fov_x - 1e-5
        ray = (
            R_cam_scene[:, :, 0][:, :, None, None]
            - fu[None, None, :, None] * R_cam_scene[:, :, 2][:, :, None, None]
            - fv[None, None, None, :] * R_cam_scene[:, :, 1][:, :, None, None]
        )
        hit = pos_local[:, :, None, None] + depth[:, None] * ray
        hit_x = hit[:, 0]

        wall_x = self._batch_scalar_for_sensor(
            effects.get('geometry_wall_x'),
            self.simple_wall_x,
            B,
            device,
            dtype,
        )
        wall_half_x = self._batch_scalar_for_sensor(
            effects.get('geometry_wall_half_x'),
            self.simple_wall_half_x,
            B,
            device,
            dtype,
        )
        back_wall_x = self._batch_scalar_for_sensor(
            effects.get('geometry_back_wall_x'),
            self.simple_back_wall_x_min,
            B,
            device,
            dtype,
        )
        front_hit = torch.sigmoid(((wall_half_x[:, None, None] + 0.020) - (hit_x - wall_x[:, None, None]).abs()) / 0.014)
        back_hit = torch.sigmoid((0.070 - (hit_x - back_wall_x[:, None, None]).abs()) / 0.016)
        return front_hit.clamp(0.0, 1.0), back_hit.clamp(0.0, 1.0)

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
        power_knee = torch.sigmoid((p - 0.47) / 0.075) * (0.25 + 0.75 * p)
        exposure_quad = e01.square() * (0.20 + 0.80 * torch.sigmoid((p - 0.48) / 0.10))
        exposure_bloom = torch.sigmoid((e01 - 0.58) / 0.100) * (0.35 + 0.65 * torch.sigmoid((g01 - 0.48) / 0.09))
        gain_quad = g01.square() * (0.18 + 0.82 * torch.sigmoid((e01 - 0.48) / 0.10))
        gain_bloom = torch.sigmoid((g01 - 0.36) / 0.080) * (0.35 + 0.65 * torch.sigmoid((e01 - 0.44) / 0.10))
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
            + 0.92 * power_knee
            + 0.22 * exposure_quad
            + 0.26 * exposure_bloom
            + 0.22 * gain_quad
            + 0.40 * gain_bloom
        )
        spec_bonus = mask * (0.34 * spec_safe + 0.18 * spec_very_safe)
        quality_specular = quality_base - spec_penalty + spec_bonus

        exposure_lift = torch.sigmoid((e01 - 0.62) / 0.070)
        gain_lift = torch.sigmoid((g01 - 0.52) / 0.075)
        projector_lift = torch.sigmoid((p - 0.45) / 0.10)
        low_reflectance_return = (
            exposure_lift * (
                0.10
                + 0.70 * gain_lift
                + 0.20 * projector_lift * gain_lift
            )
        ).clamp(max=1.0)
        dark_need = mask * 0.92
        dark_penalty = dark_need * (1.0 - low_reflectance_return)
        quality_dark = quality_base - dark_penalty + mask * low_reflectance_return * 0.24

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
            raw, mask, power, exposure, gain, quality, valid_prob, hard_valid, depth_obs)
        depth_obs = artifact['depth_obs']
        quality_obs = artifact['quality_obs']
        quality = artifact['quality']
        valid_prob = artifact['valid_prob']
        hard_valid = artifact['hard_valid']
        valid_st = artifact['valid_st']
        cue_mask = artifact['cue_mask']
        glare_selector = sid == 0
        combined_mask = torch.where(glare_selector, torch.maximum(mask, cue_mask), mask).clamp(0.0, 1.0)
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
            'aperture_artifact_map': artifact['aperture_artifact'],
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
            raw, mask, power, exposure, gain, quality, valid_prob, hard_valid, depth_obs)
        depth_obs = artifact['depth_obs']
        quality_obs = artifact['quality_obs']
        quality = artifact['quality']
        valid_prob = artifact['valid_prob']
        hard_valid = artifact['hard_valid']
        valid_st = artifact['valid_st']
        cue_mask = artifact['cue_mask']
        glare_selector = scene_ids[:, None, None] == 0
        combined_mask = torch.where(glare_selector, torch.maximum(mask, cue_mask), mask).clamp(0.0, 1.0)
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
            'aperture_artifact_map': artifact['aperture_artifact'],
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

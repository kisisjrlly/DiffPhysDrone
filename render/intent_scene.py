"""Two independently offset apertures in a closed, world-lit corridor.

The goal never discloses either aperture position. Surface illumination is
attached to world objects, not to the drone's x coordinate. The exact same
boxes are used by the renderer and collision checks.
"""

import torch
from torch.utils.checkpoint import checkpoint

from .box_scene import trace_boxes
from .ideal_gray import render_ideal_grayscale
from .pose import body_angular_velocity, rotation_exp


def build_intent_gates(env, *, centers=None, half_widths=None):
    batch, device = env.batch_size, env.device
    signs = torch.randint(0, 2, (batch, 2), device=device) * 2 - 1
    magnitudes = torch.empty((batch, 2), device=device).uniform_(0.50, 0.75)
    env.gate_centers_y = signs * magnitudes if centers is None else centers.to(device).clone()
    sampled_widths = torch.empty((batch, 2), device=device).uniform_(
        env.simple_slit_half_y_min, env.simple_slit_half_y_max
    )
    env.gate_half_widths = sampled_widths if half_widths is None else half_widths.to(device).clone()
    env.gate_x = torch.tensor([env.simple_wall_x, env.simple_wall_x + env.intent_gate_spacing], device=device)
    env.slit_center_y = env.gate_centers_y[:, 0]
    env.slit_half_y = env.gate_half_widths[:, 0]
    env.gate_lights = torch.ones((batch, 2), device=device)
    for idx, name in enumerate(env.current_scene_names):
        if name == "nominal":
            bits = torch.randint(0, 2, (2,), device=device)
            env.gate_lights[idx] = torch.where(bits.bool(), env.gray_bright_scale, env.gray_dark_scale)
        else:
            levels = {
                "dark": [env.gray_dark_scale, env.gray_dark_scale],
                "bright": [env.gray_bright_scale, env.gray_bright_scale],
                "dark_to_bright": [env.gray_dark_scale, env.gray_bright_scale],
                "bright_to_dark": [env.gray_bright_scale, env.gray_dark_scale],
            }[name]
            env.gate_lights[idx] = torch.tensor(levels, device=device)
    layouts, surface_lights = [], []
    zc, extent = env.simple_slit_center_z, 2.0
    floor, ceiling = zc - 1.15, zc + 1.15
    aperture_half_z, thickness = 0.55, env.simple_wall_half_x
    for b in range(batch):
        boxes, lights = [], []
        for g in range(2):
            x, center, half = float(env.gate_x[g]), float(env.gate_centers_y[b, g]), float(env.gate_half_widths[b, g])
            left, right = center - half, center + half
            boxes.extend([
                [x, (-extent + left) / 2, zc, thickness, (left + extent) / 2, (ceiling - floor) / 2],
                [x, (right + extent) / 2, zc, thickness, (extent - right) / 2, (ceiling - floor) / 2],
                [x, center, (floor + zc - aperture_half_z) / 2, thickness, half, (zc - aperture_half_z - floor) / 2],
                [x, center, (zc + aperture_half_z + ceiling) / 2, thickness, half, (ceiling - zc - aperture_half_z) / 2],
            ])
            lights.extend([float(env.gate_lights[b, g])] * 4)
        back = env.simple_goal_x + 1.0
        front = env.simple_start_x - 1.0
        mid, length = (back + front) / 2, (back - front) / 2
        boxes.extend([
            [mid, -extent - thickness, zc, length, thickness, (ceiling - floor) / 2],
            [mid, extent + thickness, zc, length, thickness, (ceiling - floor) / 2],
            [mid, 0, floor - thickness, length, extent, thickness],
            [mid, 0, ceiling + thickness, length, extent, thickness],
            [back, 0, zc, thickness, extent, (ceiling - floor) / 2],
        ])
        # Avoid an always-bright back wall acting as a free silhouette cue in
        # the dark task. Corridor and terminal background are world-lit too.
        lights.extend([float(env.gate_lights[b].min())] * 4 + [float(env.gate_lights[b, -1])])
        layouts.append(boxes)
        surface_lights.append(lights)
    env.voxels = torch.tensor(layouts, device=device, dtype=torch.float32)
    env.surface_lights = torch.tensor(surface_lights, device=device, dtype=torch.float32)


def render_boxes_irradiance(env, position, rotation):
    depth, normals, ids = trace_boxes(env.voxels, rotation, position, env.height, env.width, env._fov_x_half_tan)
    image, aux = render_ideal_grayscale(
        depth, rotation, position, normals,
        fov_x_half_tan=env._fov_x_half_tan,
        ambient=env.gray_nominal_ambient, diffuse=env.gray_nominal_diffuse,
        background_intensity=env.gray_background_intensity,
        texture_strength=env.gray_texture_strength, texture_scale=env.gray_texture_scale,
        detach_geometry=False, return_aux=True,
    )
    light = env.surface_lights.gather(1, ids.flatten(1)).reshape_as(depth)
    image = image * light[:, None] * env.gray_light_factor[:, None, None, None]
    aux.update(geometry_depth=depth.detach(), geometry_normals=normals.detach(),
               light_scale=light.mean(dim=(1, 2)) * env.gray_light_factor)
    return image, aux


def render_exposure(env, exposure01, *, return_aux=False):
    """Uniform shutter integration over the past exposure, ending at now.

    Constant velocity and angular rate during <=8ms are local approximations.
    There is no access to future simulator states. Timing/sample count are
    explicit; no unmeasured multiplier amplifies motion blur.
    """
    position = (env.R_scene_T @ env.p[:, :, None]).squeeze(-1)
    velocity = (env.R_scene_T @ env.v[:, :, None]).squeeze(-1)
    rotation = env.R_scene_T @ env.R
    omega = body_angular_velocity(env.R_old, env.R, env.last_control_dt)
    if env.observation_grad_mode == "detached":
        position, velocity, rotation, omega = (value.detach() for value in (position, velocity, rotation, omega))
    camera_rotation = rotation @ env.R_cam
    with torch.no_grad():
        sharp, aux = render_boxes_irradiance(env, position, camera_rotation)
    samples = env.gray_temporal_samples
    if samples > 1:
        duration = env.gray_camera.calibration.exposure_to_us(exposure01) * 1e-6
        accumulated = torch.zeros_like(sharp)
        for fraction in (torch.arange(samples, device=position.device, dtype=position.dtype) + 0.5) / samples:
            offset = -duration * fraction
            sample_p = position + velocity * offset[:, None]
            sample_R = rotation @ rotation_exp(omega * offset[:, None]) @ env.R_cam
            def image_at(p, r):
                return render_boxes_irradiance(env, p, r)[0]
            if torch.is_grad_enabled() and (sample_p.requires_grad or sample_R.requires_grad):
                sample = checkpoint(image_at, sample_p, sample_R, use_reentrant=False)
            else:
                sample = image_at(sample_p, sample_R)
            accumulated = accumulated + sample / samples
        image = accumulated
    else:
        image, aux = render_boxes_irradiance(env, position, camera_rotation)
    aux["temporal_blur_mae"] = (image.detach() - sharp).abs().mean(dim=(1, 2, 3))
    aux["angular_speed_rad_s"] = omega.norm(dim=-1)
    aux["temporal_samples"] = samples
    if return_aux:
        return image, aux
    return image, None

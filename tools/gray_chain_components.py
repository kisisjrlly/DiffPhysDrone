"""Audit-only composition of existing slit components, never a new renderer.

Production scene_layout dispatches multiple systems together. Here rendering,
lighting, shutter, dynamics and scoring are independently selectable; M11 is
required to match production joint_slit before any attribution is reported.
"""
import contextlib
from dataclasses import dataclass, fields, replace
import math
import torch
import quadsim_cuda
from render.box_scene import trace_boxes, box_clearance
from render.ideal_gray import render_ideal_grayscale
from render.slit_scene import build_joint_slit, slit_surface_light
from render.intent_scene import render_exposure
from render.pose import update_attitude


@dataclass(frozen=True)
class Variant:
    zero_jitter: bool=False
    extend_horizon: bool=False
    long_approach: bool=False
    transition_shift: bool=False
    hard_saturation: bool=False
    corridor: bool=False
    box_rays: bool=False
    torch_dynamics: bool=False
    swept_collision: bool=False
    surface_light: bool=False
    temporal_exposure: bool=False

    @classmethod
    def stage(cls,index):
        if not 0<=index<=11:raise ValueError('Migration stage must be 0..11')
        return cls(**{f.name:i<index for i,f in enumerate(fields(cls))})

    def args(self,base):
        import copy
        args=copy.deepcopy(base)
        args.scene_layout='single_wall_slit'  # geometry and components are patched separately
        args.control_dt_jitter=0. if self.zero_jitter else .1
        args.timesteps=90 if self.extend_horizon else 60
        args.simple_start_x=-2.5 if self.long_approach else -1.5
        args.gray_transition_x=-.65 if self.transition_shift else 0.
        args.gray_saturation_mode='ste' if self.hard_saturation else 'soft'
        args.gray_temporal_samples=9 if self.temporal_exposure else 1
        args.observation_grad_mode='detached'
        return args


@contextlib.contextmanager
def scene_mode(env,mode):
    old=env.scene_layout;env.scene_layout=mode
    try:yield
    finally:env.scene_layout=old


class Components:
    def __init__(self,variant):self.variant=variant

    def install(self,env):
        """Reset changes no random draw; only declared geometry/engine changes."""
        old_reset=env.reset
        def reset(*a,**kw):
            old_reset(*a,**kw)
            if self.variant.corridor:build_joint_slit(env)
            if self.variant.torch_dynamics:
                env.R=update_attitude(torch.zeros_like(env.R),env.act,torch.nn.functional.normalize(env.p_target-env.p,2,-1),
                    torch.zeros_like(env.yaw_ctl_delay),5)
                env.R_old=env.R.clone()
        env.reset=reset
        old_run=env.run
        def run(command,dt,v_pred=None):
            with scene_mode(env,'joint_slit' if self.variant.torch_dynamics else 'single_wall_slit'):
                return old_run(command,dt,v_pred)
        env.run=run
        if self.variant.swept_collision:
            def nearest():
                # Legacy interface used by run_one_episode's pre-step checks.
                # It is read-only; transition scoring is separately selected.
                clearance=box_clearance((env.R_scene_T @ env.p[...,None])[...,0][None],env.voxels,env.drone_radius)[0]
                v=torch.zeros_like(env.p);v[:,0]=clearance.clamp_min(0)
                return v
            env.find_vec_to_nearest_pt=nearest

    def render(self,env,exposure,gain,differentiable=False):
        v=self.variant
        if v.temporal_exposure:
            # At M11 and its declared reverse variants, reuse production shutter
            # integration rather than implement another integration law.
            # Non-world lighting/ray reversals are supplied through a temporary
            # irradiance provider, restored on exit (single-threaded audit).
            import render.intent_scene as scene
            old=scene.render_scene_irradiance
            def irradiance(environment,position,rotation):
                return self.irradiance(environment,position,rotation)
            scene.render_scene_irradiance=irradiance
            try:ideal,aux=render_exposure(env,exposure,return_aux=True)
            finally:scene.render_scene_irradiance=old
            gray,camera=env.gray_camera(ideal,exposure,gain,motion=None,enable_noise=env.gray_enable_noise)
            aux.update(camera);aux['blur_strength']=aux['temporal_blur_mae'][:,None,None,None]
            return gray,aux
        position=(env.R_scene_T @ env.p[...,None])[...,0]
        rotation=env.R_scene_T @ env.R @ env.R_cam
        if not v.box_rays and not v.surface_light:
            # Preserve exact legacy arithmetic for M0..M6.
            with scene_mode(env,'single_wall_slit'):
                ideal,aux=env.render_gray_ideal(return_aux=True)
        else:ideal,aux=self.irradiance(env,position,rotation)
        depth=aux['geometry_depth']
        valid=torch.where(depth<99,depth,torch.full_like(depth,float('nan')))
        characteristic=torch.nanmedian(valid.flatten(1),dim=1).values
        characteristic=torch.where(torch.isfinite(characteristic),characteristic,torch.full_like(characteristic,3.)).clamp_min(env.gray_motion_depth_floor)
        motion=env.v.norm(dim=-1).detach()/characteristic
        gray,camera=env.gray_camera(ideal,exposure,gain,motion=motion,enable_noise=env.gray_enable_noise)
        aux.update(camera);aux.update(characteristic_depth=characteristic,motion_proxy=motion)
        return gray,aux

    def irradiance(self,env,position,rotation):
        v=self.variant
        if v.box_rays:
            depth,normals,_=trace_boxes(env.voxels,rotation,position,env.height,env.width,env._fov_x_half_tan)
        else:
            depth=position.new_empty((env.batch_size,env.height,env.width))
            normals=position.new_empty((env.batch_size,env.height,env.width,3))
            quadsim_cuda.render_geometry(depth,normals,env.balls,env.cyl,env.cyl_h,env.voxels,
                rotation.contiguous(),position.contiguous(),env.n_drones_per_group,env._fov_x_half_tan)
        kwargs=dict(fov_x_half_tan=env._fov_x_half_tan,texture_strength=env.gray_texture_strength,
            texture_scale=env.gray_texture_scale,detach_geometry=False,return_aux=True)
        if v.surface_light:
            image,aux=render_ideal_grayscale(depth,rotation,position,normals,
                ambient=env.gray_nominal_ambient,diffuse=env.gray_nominal_diffuse,
                background_intensity=env.gray_background_intensity,**kwargs)
            light=slit_surface_light(env,aux['points'])
            image=image*light[:,None]*env.gray_light_factor[:,None,None,None]
            light_scale=light.mean(dim=(1,2))*env.gray_light_factor
        else:
            light_scale=env._scenario_light_scale(position[:,0])
            image,aux=render_ideal_grayscale(depth,rotation,position,normals,
                ambient=env.gray_nominal_ambient*light_scale,
                diffuse=env.gray_nominal_diffuse*light_scale,
                background_intensity=env.gray_background_intensity*light_scale,**kwargs)
        aux.update(geometry_depth=depth.detach(),geometry_normals=normals.detach(),light_scale=light_scale.detach())
        return image,aux

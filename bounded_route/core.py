"""One causal renderer/reader/controller/objective for all bounded-route arms.

Simulator truth is confined to the episode factory, teacher collection, loss and
metrics. Policy input dataclasses contain no scene objects or labels.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import hashlib
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from tools.intent_common import load_intent_args
from train_utils import build_env
from rollout_ops import render_gray_sensor, update_camera_params
from render.exposure_history import record_history_step
from render.pose import body_angular_velocity

SCENARIOS = ('dark', 'bright', 'dark_to_bright', 'bright_to_dark')
DT = 1 / 15
STEPS = 60
MOTION_LOW = (0.6, 0., 0., 0., .2, .5)
MOTION_HIGH = (1.15, 3., 3., 3., 1., 1.)
MOTION_INITIAL = (1., .5, .5, .5, .6, 1.)
CONFIG = 'configs/gray_bounded_joint.args'


def initial_motion(device='cpu'):
    lo, hi = torch.tensor(MOTION_LOW, device=device), torch.tensor(MOTION_HIGH, device=device)
    return (torch.tensor(MOTION_INITIAL, device=device) - lo) / (hi - lo)


def physical_motion(theta):
    lo, hi = theta.new_tensor(MOTION_LOW), theta.new_tensor(MOTION_HIGH)
    return lo + theta.clamp(0, 1) * (hi - lo)


def named_seed(namespace, episode, step=0, stream='layout'):
    key = f'bounded-route-v1:{namespace}:{episode}:{step}:{stream}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % (2**63-1)


def randn_for(namespace, ids, step, stream, shape, device):
    # Independent of batch composition, call count, recording and method.
    rows = []
    for i in ids:
        gen = torch.Generator(device='cpu').manual_seed(named_seed(namespace, i, step, stream))
        rows.append(torch.randn(shape, generator=gen))
    return torch.stack(rows).to(device)


class EpisodeFactory:
    def __init__(self, config=CONFIG, device='cuda'):
        self.config, self.device = str(config), torch.device(device)
        self.args = load_intent_args(config)
        self.cache = {}

    def one(self, namespace, episode):
        key = (namespace, episode)
        if key not in self.cache:
            # Reset RNG is isolated; all per-episode initial values are stored.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(named_seed(namespace, episode))
                env = build_env(1, self.args, torch.device('cpu'), eval_mode=True)
                env.reset(scene_name=SCENARIOS[episode % 4])
                # Stratified signs, not success-selected layouts.
                mag = env.slit_center_y.abs()
                env.slit_center_y = mag * (1 if (episode // 4) % 2 else -1)
                env.voxels = torch.stack([env._build_wall_slit_voxel_layout(
                    env.slit_center_y[0], env.slit_half_y[0], env.back_wall_x[0])])
                self.cache[key] = {k: v.detach().clone() for k, v in vars(env).items()
                                   if isinstance(v, torch.Tensor)}
        return self.cache[key]

    def make(self, namespace, ids):
        # Env constructor may draw randomness; do not let it touch user/global streams.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(191117)
            env = build_env(len(ids), self.args, torch.device('cpu'), eval_mode=True)
            env.reset(scene_name='dark')
        rows = [self.one(namespace, int(i)) for i in ids]
        for k, first in rows[0].items():
            if first.ndim and first.shape[0] == 1:
                value = torch.cat([r[k] for r in rows], dim=0)
            else:
                value = first.clone()
            setattr(env, k, value.to(self.device))
        env.device = self.device
        env.gray_camera = env.gray_camera.to(self.device)
        env.current_scene_names = [SCENARIOS[int(i) % 4] for i in ids]
        env.current_scene_name = 'mixed'
        # Standalone explicit feedback controller; no change to legacy validator.
        env.joint_backward = 'consistent'
        env.exposure_history.prime_from_env(env)
        return env


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(2, 8, 3, 2, 1), nn.SiLU(),
                                  nn.Conv2d(8, 16, 3, 2, 1), nn.SiLU(),
                                  nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU())
        self.hidden = nn.Linear(32*5*6 + 9, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, images, rotation):
        features = self.conv(images).flatten(1)
        return self.out(F.silu(self.hidden(torch.cat([features, rotation.flatten(1)], 1)))).squeeze(1)


def preprocess(current, previous=None, zero=False):
    if previous is None:
        previous = current
    images = torch.cat([current, previous], 1)
    images = F.adaptive_avg_pool2d(images, (36, 48)).clamp(0, 1)*2-1
    return torch.zeros_like(images) if zero else images


@dataclass
class FlightObservation:
    position: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    rotation: torch.Tensor
    angular_velocity: torch.Tensor
    estimated_offset: torch.Tensor
    previous_center: torch.Tensor
    filtered_center: torch.Tensor
    locked_center: torch.Tensor
    locked: torch.Tensor
    crossed: torch.Tensor
    goal: torch.Tensor


def flight_rule(obs, theta):
    v0, ke, kd, km, alpha, lateral = physical_motion(theta).unbind()
    p, v, a = obs.position, obs.velocity, obs.acceleration
    center = p[:, 1] + obs.estimated_offset
    new_filter = obs.filtered_center + alpha * (center - obs.filtered_center)
    lock = obs.locked | (p[:, 0] >= -.32)
    locked_center = torch.where(obs.locked, obs.locked_center, new_filter)
    center_ref = torch.where(lock, locked_center, new_filter)
    e = (obs.estimated_offset.abs()/.3).clamp(0, 2)
    inconsistent = ((center - obs.previous_center).abs()/.1).clamp(0, 2)
    moving = (v[:, 1].abs()/.6 + obs.angular_velocity.norm(dim=-1)).clamp(0, 2)
    speed = v0 / (1 + ke*e + kd*inconsistent + km*moving)
    yref = p[:, 1] + lateral*(center_ref - p[:, 1])
    yref = torch.where(obs.crossed, obs.goal[:, 1], yref)
    speed = torch.minimum(speed, (2*(obs.goal[:, 0]-p[:, 0])).clamp_min(0))
    horizon = 2*DT + 1/12
    predicted_p = p + horizon*v + .5*horizon*horizon*a
    predicted_v = v + horizon*a
    refp = torch.stack([p[:, 0]+horizon*speed, yref, obs.goal[:, 2]], 1)
    refv = torch.stack([speed, speed*0, speed*0], 1)
    command = (6*(refp-predicted_p) + 4*(refv-predicted_v) + .35*v).clamp(-2.2, 2.2)
    return command, center, new_filter, locked_center, lock, speed


def nominal_prediction(p, v, a, rotation, applied, queue, proposed):
    """One-period own-state/known-queue forecast, without wind/scene/future state."""
    from render.pose import dynamics_step, update_attitude
    command = queue[0] if queue else proposed
    b = p.shape[0]
    newact, pn, vn, an = dynamics_step(rotation, torch.zeros_like(p), p.new_ones(b, 1),
        p.new_tensor([0., .35]).expand(b, 2), p.new_full((b, 1), 12.), command,
        applied, p, v, torch.zeros_like(p), a, 1., DT, .5)
    rn = update_attitude(rotation, newact, p.new_tensor([1.5, 0., 1.5])[None]-pn,
                         p.new_full((b, 1), math.exp(-6*DT)), 5)
    w = body_angular_velocity(rotation, rn, DT)
    return vn, w


def flow_speed(previous, current):
    """Fixed 5x5 dense Lucas-Kanade, regularization 1e-4, px/s at policy scale."""
    if previous is None:
        return current.new_zeros(current.shape[0])
    images = F.adaptive_avg_pool2d(torch.cat([previous, current], 1), (36, 48))
    a, b = images[:, :1], images[:, 1:]
    mid = (a+b)*.5
    gx = F.pad((mid[..., 2:]-mid[..., :-2])*.5, (1,1,0,0))
    gy = F.pad((mid[..., 2:,:]-mid[..., :-2,:])*.5, (0,0,1,1))
    avg = lambda x: F.avg_pool2d(x,5,1,2)
    xx, yy, xy = avg(gx*gx)+1e-4, avg(gy*gy)+1e-4, avg(gx*gy)
    xt, yt = avg(gx*(b-a)), avg(gy*(b-a))
    det = (xx*yy-xy.square()).clamp_min(1e-10)
    u, v = (xy*yt-yy*xt)/det, (xy*xt-xx*yt)/det
    texture = avg(gx.square()+gy.square()) > 1e-4
    speed = (u.square()+v.square()+1e-12).sqrt()/DT
    rows = []
    for i in range(len(a)):
        values = speed[i][texture[i]]
        rows.append(torch.quantile(values, .9) if values.numel() >= 20 else speed.new_tensor(0.))
    return torch.stack(rows)


def mean_ae(gray, exposure_us, gain):
    ratio = (.35/(gray.mean((1,2,3))+1e-6)).clamp(.5, 2.)
    requested = exposure_us*ratio
    t = requested.clamp(100., 50000.)
    g = gain*(requested/t)
    return t, g.clamp(1., 8.)


def camera_rule(mode, gray, previous, exposure_us, gain, position, pred_v, pred_w, theta):
    t, g = mean_ae(gray, exposure_us, gain)
    if mode == 'gradient':
        from controllers.auto_exposure import _gradient_score
        scales = gray.new_tensor([.5,.75,1.,1.5,2.])
        scores = torch.stack([_gradient_score((gray*s).clamp(0,1)) for s in scales],1)
        ratio = scales[scores.argmax(1)]
        raw = exposure_us*ratio
        t = raw.clamp(100.,50000.)
        g = (gain*raw/t).clamp(1.,8.)
    elif mode == 'motion':
        focal = 48/(2*.82)
        pixels_per_second = torch.maximum(flow_speed(previous, gray), focal*pred_w.norm(dim=1))
        cap = (1e6/pixels_per_second.clamp_min(1e-6)).clamp(100.,50000.)
        limited = torch.minimum(t, cap)
        g = (g*t/limited).clamp(1.,8.)
        t = limited
    elif mode == 'learned':
        progress = ((position[:, 0]+1.5)/3).clamp(0,1)*2-1
        motion = (pred_v[:, 1].abs()/.6 + pred_w.norm(dim=1)).clamp(0,2)
        features = torch.stack([torch.ones_like(progress), progress, motion],1)
        coeff = theta.clamp(-1,1)*2
        t = t * torch.exp(math.log(2)*torch.tanh(features@coeff[:3]))
        g = g * torch.exp(math.log(2)*torch.tanh(features@coeff[3:]))
    elif mode != 'mean':
        raise ValueError(mode)
    t = torch.maximum(torch.minimum(t, exposure_us*2), exposure_us*.5).clamp(100.,50000.)
    g = torch.maximum(torch.minimum(g, gain*2), gain*.5).clamp(1.,8.)
    return t, g


def objective(final_distance, closest_clearance, remaining, effort):
    terminal = ((final_distance-.35).clamp_min(0)/3).square()
    risk = F.softplus((.0011-closest_clearance)/.02).square()
    return 4*terminal + 10*risk + remaining + .01*effort


class Runner:
    def __init__(self, factory, reader):
        self.factory, self.reader = factory, reader
        self.device = factory.device

    def rollout(self, namespace, ids, motion, camera, mode='mean', **kwargs):
        # Hard rasterization can amplify even tiny batch-dependent GEMM differences.
        # Canonical single-episode arithmetic is shared by trial and gradient arms.
        if len(ids) == 1:
            return self._rollout_single(namespace, ids, motion, camera, mode, **kwargs)
        items = [self._rollout_single(namespace, [i], motion, camera, mode, **kwargs) for i in ids]
        loss = torch.cat([x[0] for x in items])
        episodes = [ep for x in items for ep in x[1]]
        traces, data = [], []
        if items[0][2]:
            for k in range(len(items[0][2])):
                row = {}
                for key, value in items[0][2][k].items():
                    values = [x[2][k][key] for x in items]
                    row[key] = (torch.cat(values) if isinstance(value, torch.Tensor) else
                                [v for values_i in values for v in values_i] if isinstance(value,list) else value)
                traces.append(row)
        if items[0][3]:
            for k in range(len(items[0][3])):
                data.append(tuple(torch.cat([x[3][k][j] for x in items]) for j in range(6)))
        return loss, episodes, traces, data

    def _rollout_single(self, namespace, ids, motion, camera, mode='mean', *, teacher=False,
                collect=False, record=False, zero=False, fixed_camera=None, teacher_speed=None,
                teacher_bias=0., steps=STEPS):
        env = self.factory.make(namespace, ids)
        b = len(ids)
        e = env.p.new_full((b,), (890.-100)/49900)
        g = env.p.new_zeros(b)
        env._camera_command_queue = []
        env._camera_effective_delay_frames = 0
        alive = torch.ones(b,dtype=torch.bool,device=self.device)
        success, collision, crossed, locked = [torch.zeros_like(alive) for _ in range(4)]
        prevcenter = env.p[:,1].clone()
        filtered, lockcenter = prevcenter.clone(), prevcenter.clone()
        queue = [env.act.clone(), env.act.clone()]
        previous = None
        closest = env.p.new_full((b,),100.)
        remaining, effort = env.p.new_zeros(b), env.p.new_zeros(b)
        endtime = env.p.new_full((b,),4.)
        traces, data = [], []
        for k in range(steps):
            shot = randn_for(namespace, ids,k,'shot',(1,72,96),self.device)
            read = randn_for(namespace, ids,k,'read',(1,72,96),self.device)
            gray, aux = render_gray_sensor(env,e,g,True,shot_noise=shot,read_noise=read)
            inputs = preprocess(gray, previous, zero=zero)
            if collect:
                data.append((inputs.detach().cpu(),env.R.detach().cpu(),
                             (env.slit_center_y-env.p[:,1]).detach().cpu(),alive.cpu().clone(),
                             (env.p[:,0].detach().cpu()),aux['exposure_us'].detach().cpu().reshape(b)))
            offset = env.slit_center_y-env.p[:,1] if teacher else self.reader(inputs,env.R)
            if teacher:
                offset = offset + teacher_bias*math.sin(math.pi*k/60)
            obs = FlightObservation(env.p,env.v,env.a,env.R,
                body_angular_velocity(env.R_old,env.R,DT),offset,prevcenter,filtered,lockcenter,
                locked,crossed,env.p_target)
            m = motion
            if teacher_speed is not None:
                values = motion.new_tensor([teacher_speed,0.,0.,0.,.6,1.])
                lo,hi = values.new_tensor(MOTION_LOW),values.new_tensor(MOTION_HIGH)
                # Collection speeds include .3; bypass only v0, keep common controller.
                m = (values-lo)/(hi-lo)
                command,center,filtered,lockcenter,locked,speed = flight_rule(obs,m)
                if teacher_speed < .6:
                    # Exactly same velocity/position reference with lower requested speed.
                    h=2*DT+1/12
                    correction=(6*h+4)*(teacher_speed-speed)
                    command = torch.stack([(command[:,0]+correction).clamp(-2.2,2.2),command[:,1],command[:,2]],1)
            else:
                command,center,filtered,lockcenter,locked,speed = flight_rule(obs,m)
            prevcenter = center
            pv,pw = nominal_prediction(env.p,env.v,env.a,env.R,env.act,queue,command)
            tus=env.gray_camera.calibration.exposure_to_us(e)
            gf=env.gray_camera.calibration.gain_to_factor(g)
            nt,ng=camera_rule(mode,gray,previous,tus,gf,env.p,pv,pw,camera)
            if fixed_camera is not None:
                ft,fg=fixed_camera
                nt=torch.maximum(torch.minimum(tus.new_full((b,),ft),tus*2),tus*.5).clamp(100,50000)
                ng=torch.maximum(torch.minimum(gf.new_full((b,),fg),gf*2),gf*.5).clamp(1,8)
            request=torch.stack([(nt-100)/49900,ng.log()/math.log(8)],1)
            applied=torch.stack([e,g],1)
            e,g,_=update_camera_params(request,e,g,env)
            executed=queue.pop(0)
            queue.append(command)
            before_p,before_v,before_a,before_act,before_r,before_rold=(env.p,env.v,env.a,env.act,env.R,env.R_old)
            if record:
                row=dict(step=k,time_s=k*DT,position=env.p.detach().cpu().tolist(),
                    velocity=env.v.detach().cpu().tolist(),rotation=env.R.detach().cpu().tolist(),
                    estimate_offset=offset.detach().cpu().tolist(),
                    perception_error=(offset-(env.slit_center_y-env.p[:,1])).detach().cpu().tolist(),
                    actual_offset=(env.slit_center_y-env.p[:,1]).detach().cpu().tolist(),
                    applied_eg=applied.detach().cpu().tolist(),exposure_us=tus.detach().cpu().tolist(),
                    requested_exposure_us=nt.detach().cpu().tolist(),gain=gf.detach().cpu().tolist(),
                    requested_gain=ng.detach().cpu().tolist(),submitted=command.detach().cpu().tolist(),
                    executed=executed.detach().cpu().tolist(),alive=alive.detach().cpu().tolist(),
                    frame=F.interpolate(gray.detach(),(36,48),mode='area').cpu())
            previous=gray
            record_history_step(env,DT)
            noise=randn_for(namespace,ids,k,'dynamics',(3,),self.device)
            env.run(executed,DT,env.p_target-env.p,disturbance_noise=noise)
            for name,old in [('p',before_p),('v',before_v),('a',before_a),('act',before_act),('R',before_r),('R_old',before_rold)]:
                value=getattr(env,name)
                mask=alive.reshape(b,*([1]*(value.ndim-1)))
                setattr(env,name,torch.where(mask,value,old))
            gap=env.transition_clearance()
            closest=torch.minimum(closest,torch.where(alive,gap,closest))
            hit=alive & (env.collision_since_last_step() | (gap<=.0011))
            crossed = crossed | (alive & ~hit & (env.p[:,0]>=.1+float(env.drone_radius)))
            goal=alive & ~hit & crossed & ((env.p-env.p_target).norm(dim=1)<.35)
            done=hit|goal
            endtime=torch.where(done,endtime.new_full((b,),(k+1)*DT),endtime)
            success=success|goal
            collision=collision|hit
            dist=(env.p-env.p_target).norm(dim=1)
            residual=((dist-.35).clamp_min(0)/3).square()
            remaining=remaining+torch.where(success,torch.zeros_like(residual),residual)/STEPS
            effort=effort+torch.where(alive,command.square().sum(1)/(3*2.2**2),torch.zeros_like(dist))/STEPS
            alive=alive & ~done
            if record:
                row.update(clearance=gap.detach().cpu().tolist(),success=success.detach().cpu().tolist(),collision=collision.detach().cpu().tolist())
                traces.append(row)
        final_distance=(env.p-env.p_target).norm(dim=1)
        loss=objective(final_distance,closest,remaining,effort)
        episodes=[dict(id=int(i),namespace=namespace,scenario=SCENARIOS[int(i)%4],
            success=bool(success[j]),collision=bool(collision[j]),timeout=bool(alive[j]),
            crossed=bool(crossed[j]),time_s=float(endtime[j]) if success[j] else 4.,
            terminal_time_s=float(endtime[j]),control=float(effort[j]),loss=float(loss[j].detach()),
            min_clearance=float(closest[j].detach()),final_distance=float(final_distance[j].detach()),
            true_center=float(env.slit_center_y[j])) for j,i in enumerate(ids)]
        return loss, episodes, traces, data

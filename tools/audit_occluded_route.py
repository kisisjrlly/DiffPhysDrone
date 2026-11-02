#!/usr/bin/env python3
"""Bounded, non-learning route diagnostic. Oracle image templates, not a policy."""
import argparse
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from config import set_global_seed
from tools.intent_common import load_intent_args
from train_utils import build_env, write_run_provenance
from render.occluded_route import build_occluded_route
from render.intent_scene import render_exposure, render_scene_irradiance
from rollout_ops import init_camera_params, update_camera_params, render_gray_sensor, _camera_delay_frames
from controllers.auto_exposure import mean_auto_exposure_target
from motion_prediction import predict_exposure_motion

MOTIONS=['hold','left','right','left_settle','right_settle']
CAMERAS=[(e,g) for e in (.03,.30,.85) for g in (.02,.15)]+['ae']


def reference(a,b,time,duration):
    u=min(max(time/duration,0),1)
    return (a+(10*u**3-15*u**4+6*u**5)*(b-a),
            (30*u**2-60*u**3+30*u**4)/duration*(b-a),
            (60*u-180*u**2+120*u**3)/duration**2*(b-a))


def control(env, position, velocity, acceleration):
    horizon=2/15+1/env.fixed_pitch_ctl_delay
    pp=env.p+env.v*horizon+.5*env.a*horizon**2
    pv=env.v+env.a*horizon
    return (acceleration+6*(position-pp)+4*(velocity-pv)+env.fixed_drag_linear*env.v).clamp(-2.2,2.2)


def setup(split, device):
    args=load_intent_args('configs/gray_occluded_route.args',batch_size=8)
    seed=1102100 if split=='development' else 1102200
    set_global_seed(seed)
    env=build_env(8,args,torch.device(device));env.reset('dark')
    env.current_scene_names=['dark','dark','bright','bright']*2
    build_occluded_route(env,[-1,1,-1,1]*2)
    return args,env,seed


@torch.no_grad()
def run_arm(split,motion,camera,device,record=False):
    args,env,seed=setup(split,device)
    initial=env.p.clone();side=env.route_blocked_side.clone()
    e,g=init_camera_params(env,8,env.device)
    queue=[env.act.clone(),env.act.clone()]
    alive=torch.ones(8,dtype=torch.bool,device=env.device)
    collision=~alive;success=~alive;chosen=torch.ones_like(side)
    times=env.p.new_full((8,),8.);energy=env.p.new_zeros(8)
    motion_sign=0 if motion=='hold' else (1 if motion.startswith('left') else -1)
    destination=initial.clone();destination[:,1]=motion_sign*.4
    duration=.9 if motion.endswith('settle') else 1.6
    decision_p=None;decision=None;frames=[];trajectory=[];settings=[];prediction_errors=[];timeline=[]
    for t in range(120):
        time=t/15
        image=None
        torch.manual_seed(seed+1000+t)
        torch.cuda.manual_seed_all(seed+1000+t)
        if t<=24 or record or camera=="ae":
            image,_=render_gray_sensor(env,e,g,differentiable=False)
        if t==24:
            templates=[]
            for alt in (-1,1):
                build_occluded_route(env,[alt]*8)
                irr,_=render_exposure(env,e)
                templates.append(env.gray_camera(irr,e,g,enable_noise=False)[0])
            build_occluded_route(env,side)
            decoder_image=F.interpolate(image,size=(args.gray_nn_height,args.gray_nn_width),mode="area")
            templates=[F.interpolate(im,size=(args.gray_nn_height,args.gray_nn_width),mode="area") for im in templates]
            delta=templates[1]-templates[0]
            score=((decoder_image-(templates[0]+templates[1])*.5)*delta).flatten(1).sum(1)
            # Truly uninformative/tied views use a deterministic, balanced tie break.
            ties=score.abs()<1e-12
            guessed=torch.where(score>0,torch.ones_like(side),-torch.ones_like(side))
            fallback=torch.tensor([-1,-1,1,1,-1,-1,1,1],device=env.device)
            guessed=torch.where(ties,fallback,guessed)
            chosen=-guessed
            decision={'correct':(guessed==side).cpu().tolist(),'scores':score.cpu().tolist(),
                      'template_cue_rms':delta.square().flatten(1).mean(1).sqrt().cpu().tolist(),
                      'ties':ties.cpu().tolist(),'chosen_route':chosen.cpu().tolist(),
                      'decision_time_s':time,'privileged_templates':True}
            decision_p=env.p.clone()
        if t<24:
            p,v,a=reference(initial,destination,time+2/15+1/env.fixed_pitch_ctl_delay,duration)
        else:
            points=[decision_p,decision_p.clone(),decision_p.clone(),env.p_target]
            points[1][:,0]=-.25;points[1][:,1]=chosen*.975;points[1][:,2]=1.5
            points[2][:,0]=2.5;points[2][:,1]=chosen*.975;points[2][:,2]=1.5
            elapsed=time-1.6+2/15+1/env.fixed_pitch_ctl_delay
            durations=[2.,2.7,1.7]
            idx=0
            while idx<2 and elapsed>durations[idx]:
                elapsed-=durations[idx];idx+=1
            p,v,a=reference(points[idx],points[idx+1],elapsed,durations[idx])
        command=control(env,p,v,a)
        prediction=predict_exposure_motion(env,queue,command,_camera_delay_frames(env),
            float(env.gray_camera.calibration.exposure_to_us(e).max()))
        if camera=='ae':
            target=mean_auto_exposure_target(image,e,g) if image is not None else torch.stack([e,g],-1)
        else: target=e.new_tensor(camera)[None].expand(8,-1)
        if record:
            frames.append(image[0,0].cpu().numpy());trajectory.append(env.p[0].cpu().numpy())
            settings.append([float(e[0]),float(g[0])])
        exposure_s=env.gray_camera.calibration.exposure_to_us(e)*1e-6
        timeline.append({'step':t,'issue_s':time,'effective_s':time+prediction['effective_time_s'],
            'current_frame_end_s':time,'current_exposure_start_s':(time-exposure_s).cpu().tolist(),
            'current_frame_velocity':env.v.cpu().tolist(),
            'predicted_next_position':prediction['position'].cpu().tolist(),
            'requested':target.cpu().tolist(),'applied_before':torch.stack([e,g],-1).cpu().tolist(),
            'predicted_velocity':prediction['velocity'].cpu().tolist()})
        e,g,_=update_camera_params(target,e,g,env)
        queue.append(command)
        torch.manual_seed(seed+2000+t)
        torch.cuda.manual_seed_all(seed+2000+t)
        env.run(queue.pop(0),1/15,env.p_target-env.p)
        timeline[-1]['actual_next_position']=env.p.cpu().tolist()
        timeline[-1]['actual_next_velocity']=env.v.cpu().tolist()
        prediction_errors.append((prediction['position']-env.p).norm(dim=-1).cpu().tolist())
        hit=env.collision_since_last_step() | (env.transition_clearance()<=args.collision_clearance)
        reached=(env.p-env.p_target).norm(dim=-1)<.35
        collision|=alive&hit;success|=alive&~hit&reached
        terminal=alive&(hit|reached)
        times=torch.where(terminal,times.new_full((8,),(t+1)/15),times)
        energy+=alive*command.square().sum(1)/15
        alive&=~terminal
    result={'split':split,'motion':motion,'camera':camera,'seed':seed,'success':success.cpu().tolist(),
        'collision':collision.cpu().tolist(),'timeout':alive.cpu().tolist(),'time_s':times.cpu().tolist(),
        'energy':energy.cpu().tolist(),'decision':decision,'prediction_position_error_mean_m':float(np.mean(prediction_errors)),
        'timeline':timeline,'claim':'Privileged template/controller diagnostic; not learned navigation.'}
    return result,dict(frames=np.array(frames),trajectory=np.array(trajectory),settings=np.array(settings))


def visibility(device):
    args,e,_=setup('development',device)
    result=[]
    for y in (0,.4,-.4):
        e.p[:]=e.p.new_tensor([-1,y,1.5]);e.R[:]=torch.eye(3,device=e.device)
        images=[]
        for side in (-1,1):
            build_occluded_route(e,[side]*8)
            images.append(render_scene_irradiance(e,e.p,e.R@e.R_cam)[0])
        result.append({'y_m':y,'max_image_difference':float((images[0]-images[1]).abs().max())})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out_dir',required=True);p.add_argument('--device',default='cuda')
    cli=p.parse_args();torch.set_num_threads(4)
    out=Path(cli.out_dir);out.mkdir(parents=True,exist_ok=False)
    development=[]
    for motion in MOTIONS:
        for camera in CAMERAS:
            r,_=run_arm('development',motion,camera,cli.device)
            (out/f'dev_{motion}_{str(camera)}.json').write_text(json.dumps(r,indent=2)+'\n')
            development.append(r)
            print(motion,camera,sum(r['decision']['correct']),sum(r['success']),flush=True)
    def rank(r):return (-sum(r['success']),-sum(r['decision']['correct']),sum(r['time_s']),sum(r['energy']))
    fixed=[r for r in development if r['camera']!='ae']
    best=min(fixed,key=rank)
    ae=min([r for r in development if r['camera']=='ae'],key=rank)
    selected=[best,ae,min([r for r in fixed if r['motion']=='hold'],key=rank),
              min([r for r in fixed if r['camera']==(.30,.02)],key=rank)]
    locked=[{'motion':r['motion'],'camera':r['camera']} for r in selected]
    (out/'locked_candidates.json').write_text(json.dumps(locked,indent=2)+'\n')
    confirmed=[]
    for i,s in enumerate(locked):
        r,media=run_arm('confirmation',s['motion'],s['camera'],cli.device,record=i==0)
        (out/f'confirmation_{i}.json').write_text(json.dumps(r,indent=2)+'\n');confirmed.append(r)
        if i==0:np.savez_compressed(out/'demo.npz',**media)
    summary={'visibility':visibility(cli.device),'calibrated':False,'policy_training':False,
             'development':[ {k:v for k,v in r.items() if k!='timeline'} for r in development],
             'locked_candidates':locked,'confirmation':[{k:v for k,v in r.items() if k!='timeline'} for r in confirmed],
             'limitations':['8+8 screening episodes only','oracle templates and waypoint controller',
              'image templates use simulator pose, settings and lighting; optimistic decoder',
              'not learning algorithm comparison','warp_ste cannot model disocclusion']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    args,_,_=setup('development',cli.device);write_run_provenance(out,args)
    print('CONFIRMATION',[(r['motion'],r['camera'],sum(r['success']),sum(r['decision']['correct'])) for r in confirmed],flush=True)

if __name__=='__main__':main()

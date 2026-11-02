#!/usr/bin/env python3
"""Compare actual action perturbations with the current local warp gradient."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tools.audit_occluded_route import setup, reference, control
from render.occluded_route import build_occluded_route
from render.intent_scene import render_exposure


def cue(scale_value, device, grad=False):
    _,env,seed=setup('development',device)
    scale=env.p.new_tensor(scale_value,requires_grad=grad)
    start=env.p.clone();end=start+torch.stack([scale*0,scale*.4,scale*0])[None]
    q=[env.act]*2
    for t in range(24):
        p,v,a=reference(start,end,t/15+2/15+1/env.fixed_pitch_ctl_delay,1.6)
        cmd=control(env,p,v,a);q.append(cmd)
        torch.manual_seed(seed+2000+t);torch.cuda.manual_seed_all(seed+2000+t)
        env.run(q.pop(0),1/15,env.p_target-env.p)
    images=[]
    for side in (-1,1):
        build_occluded_route(env,[side]*8)
        images.append(render_exposure(env,env.p.new_full((8,),.3))[0])
    metric=(images[0]-images[1]).square().mean()
    derivative=float(torch.autograd.grad(metric,scale)[0]) if grad else None
    return float(metric.detach()),derivative


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');c=p.parse_args()
    torch.set_num_threads(4);rows=[]
    for scale in (0.,.5,1.):
        value,derivative=cue(scale,c.device,True)
        lo,_=cue(scale-.05,c.device);hi,_=cue(scale+.05,c.device)
        rows.append(dict(action_scale=scale,cue_mse=value,autograd=derivative,
            finite_action_difference=(hi-lo)/.1,finite_action_step=.05))
    Path(c.out).write_text(json.dumps(dict(rows=rows,metric='noiseless irradiance template separation after 24 real dynamics steps',
        limitation='diagnostic geometry used only in scoring; warp_ste has no disocclusion derivative; not navigation objective or a trained policy'),indent=2)+'\n')
    print(rows)
if __name__=='__main__':main()

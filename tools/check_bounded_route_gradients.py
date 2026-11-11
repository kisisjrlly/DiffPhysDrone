#!/usr/bin/env python
"""Fixed-state derivatives for the common renderer; diagnosis, no admission gate.
Does not select policies, physical parameters, scenes, or evaluation episodes.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from bounded_route.core import EpisodeFactory, Reader, randn_for, preprocess
from rollout_ops import render_gray_sensor
from render.exposure_history import record_history_step


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out_dir',required=True);p.add_argument('--device',default='cuda');args=p.parse_args()
    out=Path(args.out_dir);torch.set_num_threads(4);device=torch.device(args.device)
    start=time.monotonic();f=EpisodeFactory(device=device)
    reader=Reader().to(device);reader.load_state_dict(torch.load(out/'reader.pth',weights_only=False,map_location=device)['model']);reader.eval().requires_grad_(False)
    rows=[]
    for tag,command in [('stationary',[0,0,0]),('lateral',[0,.8,0]),('turning',[.4,1.6,0])]:
        env=f.make('gradient_state',[0])
        for k in range(8):
            record_history_step(env,1/15)
            env.run(torch.tensor([command],device=device),1/15,env.p_target-env.p,disturbance_noise=torch.zeros(1,3,device=device))
        for tus in (890.,8000.,50000.):
            for variable in ('exposure','gain','position_y'):
                # All perturbations reuse the SAME raw shot/read draws.
                shot=randn_for('gradient_state',[0],8,'shot',(1,72,96),device)
                read=randn_for('gradient_state',[0],8,'read',(1,72,96),device)
                basep=env.p.clone();e0=(tus-100)/49900;g0=.2
                initial={'exposure':e0,'gain':g0,'position_y':0.}[variable]
                h={'exposure':1e-4,'gain':1e-4,'position_y':1e-4}[variable]
                def score(x):
                    e=x.reshape(1) if variable=='exposure' else env.p.new_tensor([e0])
                    g=x.reshape(1) if variable=='gain' else env.p.new_tensor([g0])
                    env.p=basep + torch.stack([x*0,x,x*0]).reshape(1,3) if variable=='position_y' else basep
                    image,_=render_gray_sensor(env,e,g,shot_noise=shot,read_noise=read)
                    predicted=reader(preprocess(image),env.R)
                    return predicted.square().mean(),image
                x=env.p.new_tensor(initial,requires_grad=True)
                value,image=score(x);gradient=torch.autograd.grad(value,x,allow_unused=True)[0]
                grad=float(gradient) if gradient is not None else 0.
                plus,_=score(x.detach()+h);minus,_=score(x.detach()-h)
                finite=float((plus-minus)/(2*h))
                detached,_=score(x.detach())
                rows.append(dict(state=tag,exposure_us=tus,variable=variable,step=h,
                    autograd=grad,finite_difference=finite,same_direction=grad*finite>0,
                    forward_value=float(value.detach()),detached_value=float(detached.detach()),
                    actual_position=basep.detach().cpu().tolist(),actual_velocity=env.v.detach().cpu().tolist(),
                    actual_rotation=env.R.detach().cpu().tolist(),
                    limitation='Local warp depth/visibility derivative approximate; max-exposure finite difference crosses bound' if tus==50000 else 'Local image-warp approximation; not exact hard-render derivative'))
                env.p=basep
    torch.cuda.synchronize() if device.type=='cuda' else None
    elapsed=time.monotonic()-start
    (out/'fixed_state_gradient_audit.json').write_text(json.dumps(dict(rows=rows,
        interpretation='Reports actual motion and raw direction agreement; not a task-benefit or training gate.'),indent=2)+'\n')
    with (out/'ledger.jsonl').open('a') as file:file.write(json.dumps(dict(kind='gradient_fixed_states',seconds=elapsed,arm=None,time=time.time()))+'\n')
    print('fixed-state gradient audit saved',len(rows),'conditions',elapsed,'seconds')


if __name__=='__main__':main()

"""Diagnostics only: sensor-view cancellation and external action removal."""
import argparse,copy,json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from config import build_parser,parse_scenarios,validate_args
from eval import run_one_episode
from intent_model import build_policy
from train_utils import build_env
from rollout_ops import render_gray_sensor
from tools.evaluate_pole_coordination import frozen_view_renderer
from tools.run_intent_pilot import bootstrap_difference


def cancel_initial_lateral_view(steps=24):
    t=0
    def render(env,e,g,differentiable=False):
        nonlocal t
        t+=1
        if t>steps: return render_gray_sensor(env,e,g,differentiable=differentiable)
        proxy=copy.copy(env)
        p=(env.R_scene_T@env.p[:,:,None]).squeeze(-1).clone()
        v=(env.R_scene_T@env.v[:,:,None]).squeeze(-1).clone()
        p[:,1]=0;v[:,1]=0
        proxy.p=(env.R_scene@p[:,:,None]).squeeze(-1)
        proxy.v=(env.R_scene@v[:,:,None]).squeeze(-1)
        return render_gray_sensor(proxy,e,g,differentiable=differentiable)
    return render


def remove_initial_lateral_action(steps=12):
    def override(t,env,a):
        if t>=steps:return a,False
        actual=a.clone();actual[:,1]=0
        return actual,True
    return override


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out_dir',required=True)
    p.add_argument('--seed',type=int,default=788000);p.add_argument('--episodes',type=int,default=40)
    p.add_argument('--device',default='cpu');cli=p.parse_args();torch.set_num_threads(1)
    checkpoint=Path(cli.checkpoint);out=Path(cli.out_dir);out.mkdir(parents=True,exist_ok=False)
    args=build_parser().parse_args([]);vars(args).update(json.loads((checkpoint.parent/'args.json').read_text()))
    args.batch_size=1;args.vis_enable=False;args.vis_episode_idx=-1;args.scenarios=parse_scenarios(args.scenarios)
    args.imx900_calibration=str(checkpoint.parent/'imx900_calibration.json');validate_args(args)
    model=build_policy(args).to(cli.device).eval();model.load_state_dict(torch.load(checkpoint,map_location=cli.device))
    record={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'seed_range':[cli.seed,cli.seed+cli.episodes-1],'device':cli.device,'calibrated':False,'claim':'Counterfactual diagnostics, not physical baselines; action-removal changes both sensing and obstacle avoidance. Success is necessary but not sufficient evidence of task-driven measurement.','conditions':{}}
    for condition in ('nominal','frozen_view','no_initial_lateral_view_change','no_initial_lateral_command','fixed_camera'):
        args.camera_control_mode='fixed' if condition=='fixed_camera' else 'learned'
        env=build_env(1,args,torch.device(cli.device),eval_mode=True);rows=[];traces=[]
        with torch.no_grad():
            for i in range(cli.episodes):
                sensor=frozen_view_renderer() if condition=='frozen_view' else cancel_initial_lateral_view() if condition=='no_initial_lateral_view_change' else None
                action=remove_initial_lateral_action() if condition=='no_initial_lateral_command' else None
                row,trace=run_one_episode(i,args.scenarios[i%len(args.scenarios)],args,model,env,None,torch.device(cli.device),collect_trace=True,episode_seed=cli.seed+i,sensor_renderer=sensor,action_override=action)
                rows.append(row);traces.extend(trace)
        groups={}
        for row in traces:groups.setdefault(row['episode_idx'],[]).append(row)
        value={'episodes':rows,'success':float(np.mean([r['success_rate'] for r in rows])),'collision':float(np.mean([r['collision_rate'] for r in rows])),'median_early_lateral_extent':float(np.median([max(abs(x['local_y']) for x in v if x['step']<24) for v in groups.values()])),'median_early_forward_progress':float(np.median([max(x['local_x']-args.simple_start_x for x in v if x['step']<24) for v in groups.values()])),'any_external_action_override':any(r['external_action_override'] for r in traces)}
        record['conditions'][condition]=value
        if condition!='nominal':value['success_minus_nominal']=bootstrap_difference([r['success_rate'] for r in rows],[r['success_rate'] for r in record['conditions']['nominal']['episodes']])
        (out/(condition+'_traces.json')).write_text(json.dumps(traces)+'\n')
        (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        print(condition,{k:v for k,v in value.items() if k!='episodes'},flush=True)
if __name__=='__main__':main()

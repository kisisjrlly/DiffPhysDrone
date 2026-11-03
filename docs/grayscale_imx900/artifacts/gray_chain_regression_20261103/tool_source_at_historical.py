#!/usr/bin/env python3
"""Reproduce archived grayscale evaluations before attributing scene-transfer failures.

No optimizer, training, model conversion, or new episode selection. Workers use
separate interpreters so archived/current Python modules cannot mix.
"""
import argparse
import contextlib
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = Path('/home/zhaoguodong/work/code/DiffPhysDrone')
ARCHIVE = PRIMARY/'logs/imx900_formal_20260924_103343'
MEAN = PRIMARY/'logs/imx900_mean_ae_gate_20260924'
MODELS = {
    'N': ('2026-09-24-10-33-44', '748d5e11fcf4ed66af462e80c2787d7aa798ae25e37227aa31b955fcdb621b7a'),
    'A': ('2026-09-24-11-35-14', '87875d118803503ab66bab855f07f542de4fc8460d4c2918fb38f27c6d344168'),
}
ARMS = ('N_gray','N_zero','A_ae')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def checkpoints(key):
    return PRIMARY/'checkpoint'/MODELS[key][0]/'final.pth'


def configuration(arm):
    """Resolve historical values explicitly; never inherit a newer experiment."""
    if arm.startswith('N'):
        data=copy.deepcopy(json.loads((ARCHIVE/'evaluation/manifest.json').read_text())['args'])
    else:
        # evaluation.log is the authoritative eval command, including AMP's
        # archived default True. Parse with the archived parser in the worker.
        data=json.loads(checkpoints('A').with_name('args.json').read_text())
        data.update(seed=30000,batch_size=1,deterministic=True,amp=True,
                    vis_enable=False,vis_episode_idx=-1,wandb_disabled=True)
    data.update(scene_layout='single_wall_slit',policy_arch='legacy',joint_backward='legacy',
        observation_grad_mode='detached',gray_temporal_samples=1,control_dt_jitter=.1,
        flight_max_speed=1.15,camera_queue_aligned_intent=False,vis_enable=False,
        vis_episode_idx=-1,policy_gray_mode='zero' if arm=='N_zero' else 'gray')
    data['imx900_calibration']=str(checkpoints(arm[0]).with_name('imx900_calibration.json'))
    return data


def reference_rows(arm):
    source=ARCHIVE/'evaluation/episodes.csv' if arm.startswith('N') else MEAN/'evaluation/episodes.csv'
    rows=list(csv.DictReader(source.open()))
    if arm.startswith('N'):
        name='gray' if arm=='N_gray' else 'gray_zeroed'
        rows=[r for r in rows if r['method']==name]
    assert len(rows)==100
    return rows


def preflight(out):
    manifest=json.loads((ARCHIVE/'manifest.json').read_text())
    bad=[name for name,h in manifest['source_sha256'].items() if sha(ARCHIVE/'source'/name)!=h]
    if bad:raise RuntimeError(f'Archived source mismatch: {bad}')
    import torch
    import quadsim_cuda
    for key,(_,expected) in MODELS.items():
        if sha(checkpoints(key))!=expected:raise RuntimeError(f'Frozen checkpoint {key} mismatch')
        old=json.loads(checkpoints(key).with_name('imx900_calibration.json').read_text())
        if old!=json.loads((ARCHIVE/'source/configs/calibration/imx900_provisional.json').read_text()):
            raise RuntimeError('Calibration mismatch')
        if old['calibrated']:raise RuntimeError('Unexpected measured profile')
    for name in ('src/quadsim.cpp','src/quadsim_kernel.cu','src/dynamics_kernel.cu'):
        if sha(ARCHIVE/'source'/name)!=sha(ROOT/name):raise RuntimeError('CUDA sources differ; isolated build required')
    # Confirm the mixed-light eval command, not merely its training command.
    command=next(s for s in (MEAN/'evaluation.log').read_text().splitlines() if s.startswith('command'))
    if '--seed 30000' not in command or '--eval_episodes 100' not in command or '--no-amp' in command:
        raise RuntimeError('Historical mixed-light command differs from the locked protocol')
    out.mkdir(parents=True,exist_ok=True)
    fingerprints={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'tools/audit_policy_sensing_decision.py',ROOT/'eval.py',ROOT/'model.py',ROOT/'env_cuda.py',ROOT/'rollout_ops.py']}
    fingerprint=hashlib.sha256(json.dumps(fingerprints,sort_keys=True).encode()).hexdigest()
    existing=out/'protocol.json'
    if existing.exists() and json.loads(existing.read_text())['source_fingerprint']!=fingerprint:
        raise RuntimeError('Audit source changed: use a new output directory, never silently reuse results')
    if not existing.exists():
        write_json(existing,dict(source_fingerprint=fingerprint,source_sha256=fingerprints,
            git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            archived_source_sha256=manifest['source_sha256'],checkpoints={k:dict(path=str(checkpoints(k)),sha256=v[1]) for k,v in MODELS.items()},
            configurations={a:configuration(a) for a in ARMS},
            source_archive=str(ARCHIVE/'source'),mixed_eval_command=command,
            torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),
            extension_path=quadsim_cuda.__file__,extension_sha256=sha(quadsim_cuda.__file__),
            calibrated=False,no_training=True))


class Recorder:
    """Read-only hooks; no extra rendering, preprocessing or random draws."""
    def __init__(self,env,model,evaluation):
        import numpy as np
        self.np=np;self.env=env;self.model=model;self.rows=[];self.pending={};self.originals=[]
        self.initial={name:getattr(env,name).detach().cpu().numpy().copy() for name in
            ('p','v','a','R','act','dg','v_wind','voxels','slit_center_y','back_wall_x','gray_light_factor')}
        def wrap_preprocess(original):
            def wrapped(image,*args,**kwargs):
                result=original(image,*args,**kwargs)
                self.capture('network_pixels',result)
                return result
            return wrapped
        self.replace(model,'preprocess_gray_input',wrap_preprocess(model.preprocess_gray_input))
        self.handles=[model.register_forward_pre_hook(self.before,with_kwargs=True),
                      model.register_forward_hook(self.after,with_kwargs=True)]
        original_run=env.run
        def run(command,dt,v_pred=None):
            self.capture('executed_command',command)
            self.pending['dt']=np.asarray(dt)
            self.pending['rng_cpu_before_dynamics']=__import__('torch').get_rng_state().numpy().copy()
            self.pending['rng_cuda_before_dynamics']=__import__('torch').cuda.get_rng_state().cpu().numpy().copy()
            result=original_run(command,dt,v_pred)
            for name in ('p','v','a','R','act','dg'):
                self.capture('post_'+name,getattr(env,name))
            self.rows.append(self.pending);self.pending={}
            return result
        self.replace(env,'run',run)
        original_update=evaluation.update_camera_params
        def update(target,e,g,env_arg):
            result=original_update(target,e,g,env_arg)
            self.capture('requested_eg',env_arg._camera_requested_state)
            self.capture('next_eg',__import__('torch').stack(result[:2],-1))
            return result
        self.replace(evaluation,'update_camera_params',update)

    def capture(self,name,value):
        if value is not None:self.pending[name]=value.detach().cpu().numpy().copy()

    def replace(self,obj,name,value):
        self.originals.append((obj,name,getattr(obj,name)));setattr(obj,name,value)

    def before(self,module,args,kwargs):
        import torch
        self.pending['rng_cpu_before_network']=torch.get_rng_state().numpy().copy()
        self.pending['rng_cuda_before_network']=torch.cuda.get_rng_state().cpu().numpy().copy()
        for name in ('p','v','a','R','act','dg'):
            self.capture('pre_'+name,getattr(self.env,name))
        self.capture('state',args[0]);self.capture('hidden_before',args[1] if len(args)>1 else None)
        self.capture('camera_hidden_before',kwargs.get('cam_hx'))
        self.capture('raw_pixels',kwargs.get('gray_obs'))
        self.capture('camera_state',kwargs.get('camera_state'))

    def after(self,module,args,kwargs,result):
        for name,value in zip(('raw_action','camera_output','hidden_after','camera_hidden_after'),result):
            self.capture(name,value)

    def close(self,path):
        for h in self.handles:h.remove()
        for obj,name,value in reversed(self.originals):setattr(obj,name,value)
        arrays={f'initial_{k}':v for k,v in self.initial.items()}
        for key in sorted(set().union(*(r.keys() for r in self.rows))):
            present=[i for i,r in enumerate(self.rows) if key in r]
            arrays[key]=self.np.stack([self.rows[i][key] for i in present])
            arrays[key+'_steps']=self.np.asarray(present)
        self.np.savez_compressed(path,**arrays)


def worker(out,runner,arm):
    import torch
    torch.set_num_threads(4)
    code=ARCHIVE/'source' if runner=='archived' else ROOT
    sys.path.insert(0,str(code))
    # No current project module is imported before choosing the source root.
    import config
    import eval as evaluation
    from model import Model
    from train_utils import build_env
    args=config.build_parser().parse_args([])
    for name,value in configuration(arm).items():setattr(args,name,value)
    args.scenarios=config.parse_scenarios(args.scenarios)
    config.validate_args(args)
    device=torch.device('cuda')
    config.set_global_seed(args.seed,True)
    env=build_env(1,args,device,eval_mode=True)
    model=Model(10,3,include_camera_state_in_obs=False,gray_nn_width=48,gray_nn_height=36).to(device)
    model.load_state_dict(torch.load(checkpoints(arm[0]),map_location=device),strict=True);model.eval()
    before={k:sha(checkpoints(k)) for k in MODELS}
    target=out/runner/arm;target.mkdir(parents=True,exist_ok=True)
    rows=[]
    source_modules={m.__name__:str(m.__file__) for m in (config,evaluation,sys.modules['model'],sys.modules['train_utils'])}
    write_json(target/'runtime.json',dict(modules=source_modules,configuration=vars(args),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        parameter_count=sum(p.numel() for p in model.parameters()),checkpoint_hash=before[arm[0]]))
    refs=reference_rows(arm)
    with torch.no_grad():
        for index,reference in enumerate(refs):
            path=target/f'ep_{index:03d}.json'
            if path.exists():
                rows.append(json.loads(path.read_text()));continue
            scenario=reference['scenario'];seed=int(reference['episode_seed'])
            # Recorder attaches after the exact episode reset, before any image.
            original_reset=env.reset;record=[]
            def reset(*a,**kw):
                result=original_reset(*a,**kw)
                record.append(Recorder(env,model,evaluation))
                return result
            env.reset=reset
            try:
                if runner=='diagnostic':
                    from tools.audit_policy_sensing_decision import run_configured_episode
                    row=run_configured_episode(index,scenario,args,model,env,device,evaluation=evaluation)
                else:
                    row,_=evaluation.run_one_episode(index,scenario,args,model,env,None,device)
            finally:
                env.reset=original_reset
                if record:record[0].close(target/f'ep_{index:03d}.npz')
            write_json(path,row);rows.append(row)
            if (index+1)%20==0:print(runner,arm,index+1,flush=True)
    for k in MODELS:
        if sha(checkpoints(k))!=before[k]:raise RuntimeError('Frozen checkpoint modified')
    write_json(target/'result.json',dict(rows=rows,successes=sum(r['success_rate'] for r in rows),
        collisions=sum(r['collision_rate'] for r in rows),checkpoint_unchanged=True))
    print(runner,arm,'complete',sum(r['success_rate'] for r in rows),flush=True)


def compare_historical(out):
    errors=[];summary={}
    for arm in ARMS:
        result=json.loads((out/'archived'/arm/'result.json').read_text())
        for i,(old,new) in enumerate(zip(reference_rows(arm),result['rows'])):
            for key in ('success_rate','collision_rate','stop_reason','steps'):
                a=float(old[key]) if key in ('success_rate','collision_rate','steps') else old[key]
                if a!=new[key]:errors.append(dict(arm=arm,episode=i,key=key,old=a,new=new[key]))
            for key in ('final_goal_dist','min_clearance','exposure_mean','gain_mean'):
                import math
                if not math.isclose(float(old[key]),float(new[key]),rel_tol=1e-5,abs_tol=1e-6):
                    errors.append(dict(arm=arm,episode=i,key=key,old=float(old[key]),new=new[key]))
        summary[arm]=dict(successes=result['successes'],collisions=result['collisions'])
    value=dict(passed=not errors,summary=summary,differences=errors)
    write_json(out/'historical_comparison.json',value)
    return value


def run_jobs(out,runner):
    for arm in ARMS:
        target=out/runner/arm/'result.json'
        if target.exists():continue
        log=out/f'{runner}_{arm}.log'
        with log.open('a') as handle:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',runner,
                '--arm',arm,'--out_dir',str(out)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Worker failed: {log}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir',type=Path,required=True)
    parser.add_argument('--stage',choices=['historical','parity','migration','summary'])
    parser.add_argument('--worker',choices=['archived','standard','diagnostic'])
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args();out=args.out_dir.resolve()
    if args.worker:
        worker(out,args.worker,args.arm);return
    preflight(out)
    if args.stage=='historical':
        run_jobs(out,'archived');print(json.dumps(compare_historical(out),ensure_ascii=False,indent=2))
    elif args.stage=='parity':
        if not json.loads((out/'historical_comparison.json').read_text())['passed']:
            raise RuntimeError('Archived reproduction failed; migration is not authorized by the protocol')
        run_jobs(out,'standard');run_jobs(out,'diagnostic')
        parity(out)
    elif args.stage=='migration':
        migration(out)
    elif args.stage=='summary':
        summarize(out)
    else:parser.error('--stage is required')

if __name__=='__main__':main()

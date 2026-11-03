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
    if existing.exists():
        locked=json.loads(existing.read_text())
        if locked['configurations']!={a:configuration(a) for a in ARMS}:
            raise RuntimeError('Locked configuration changed')
        if locked['archived_source_sha256']!=manifest['source_sha256']:
            raise RuntimeError('Archived inputs changed')
        # Per-worker numerical dependency fingerprints guard every cached job.
        versions=out/'source_versions.json'
        history=json.loads(versions.read_text()) if versions.exists() else []
        if not history or history[-1]['source_fingerprint']!=fingerprint:
            history.append(dict(source_fingerprint=fingerprint,source_sha256=fingerprints))
            write_json(versions,history)
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



def compare_traces(left,right,amp=False):
    import numpy as np
    order=['dt','pre_p','pre_v','pre_a','pre_R','pre_act','pre_dg',
        'raw_pixels','network_pixels','state','hidden_before','camera_hidden_before',
        'camera_state','rng_cpu_before_network','rng_cuda_before_network',
        'raw_action','camera_output','hidden_after','camera_hidden_after',
        'requested_eg','next_eg','executed_command','rng_cpu_before_dynamics',
        'rng_cuda_before_dynamics','post_p','post_v','post_a','post_R','post_act','post_dg']
    with np.load(left) as az,np.load(right) as bz:
        a={k:az[k] for k in az.files};b={k:bz[k] for k in bz.files}
        for key in sorted(k for k in a if k.startswith('initial_')):
            if key not in b or a[key].shape!=b[key].shape or not np.allclose(a[key],b[key],atol=1e-6,rtol=1e-5):
                return dict(step=-1,variable=key,reason='initial_state_mismatch')
        for step in range(max(len(a['dt']),len(b['dt']))):
            for key in order:
                if key not in a or key not in b:
                    if (key in a)!=(key in b):return dict(step=step,variable=key,reason='field_missing')
                    continue
                ia=np.flatnonzero(a[key+'_steps']==step);ib=np.flatnonzero(b[key+'_steps']==step)
                if not len(ia) or not len(ib):
                    if len(ia)!=len(ib):return dict(step=step,variable=key,reason='different_termination')
                    continue
                x,y=a[key][ia[0]],b[key][ib[0]]
                if x.shape!=y.shape:return dict(step=step,variable=key,reason='shape_mismatch')
                integer=np.issubdtype(x.dtype,np.integer)
                neural=key in ('network_pixels','state','hidden_before','camera_hidden_before',
                               'raw_action','camera_output','hidden_after','camera_hidden_after')
                atol,rtol=(1e-3,1e-3) if amp and neural else (1e-6,1e-5)
                ok=np.array_equal(x,y) if integer else np.allclose(x,y,atol=atol,rtol=rtol)
                if not ok:
                    return dict(step=step,variable=key,reason='value_mismatch',
                        max_absolute_difference=float(np.abs(x.astype('float64')-y.astype('float64')).max()),
                        atol=atol,rtol=rtol)
    return None


def parity(out):
    differences=[];counts={}
    for runner in ('standard','diagnostic'):
        counts[runner]={}
        for arm in ARMS:
            base=out/'archived'/arm;other=out/runner/arm
            a=json.loads((base/'result.json').read_text());b=json.loads((other/'result.json').read_text())
            counts[runner][arm]=b['successes']
            for index,(left,right) in enumerate(zip(a['rows'],b['rows'])):
                discrete=[k for k in ('episode_seed','scenario','success_rate','collision_rate','stop_reason','steps') if left[k]!=right[k]]
                first=compare_traces(base/f'ep_{index:03d}.npz',other/f'ep_{index:03d}.npz',arm=='A_ae')
                if first or discrete:
                    differences.append(dict(runner=runner,arm=arm,episode_index=index,
                        episode_seed=left['episode_seed'],first_difference=first,discrete_differences=discrete))
    result=dict(passed=not differences,successes=counts,differences=differences)
    write_json(out/'parity_comparison.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def job_fingerprint(runner,arm):
    import ast
    tree=ast.parse(Path(__file__).read_text())
    names={'Recorder','worker','configuration','reference_rows','checkpoints','write_json','sha'}
    functions={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}
    root=ARCHIVE/'source' if runner=='archived' else ROOT
    files=[*root.glob('*.py')]
    for folder in ('render','sensors','controllers'):
        files.extend((root/folder).glob('*.py'))
    if runner=='diagnostic':files.append(ROOT/'tools/audit_policy_sensing_decision.py')
    value=dict(functions=functions,files={str(p.relative_to(root)):sha(p) for p in sorted(files)},
        configuration=configuration(arm),checkpoint_sha256=sha(checkpoints(arm[0])),
        references=reference_rows(arm))
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def run_jobs(out,runner):
    for arm in ARMS:
        target=out/runner/arm/'result.json'
        fingerprint=job_fingerprint(runner,arm)
        guard=target.parent/'input_fingerprint.json'
        if guard.exists() and json.loads(guard.read_text())['fingerprint']!=fingerprint:
            raise RuntimeError(f'Worker inputs changed; refusing stale cache: {target}')
        if not guard.exists():
            if target.exists():
                # The archived jobs ran while other runner implementations were
                # being added. Their actual numerical worker code must match.
                import ast
                old=ast.parse((out/'tool_source_at_historical.py').read_text())
                new=ast.parse(Path(__file__).read_text())
                names={'Recorder','worker','configuration','reference_rows','checkpoints','write_json','sha'}
                def numerical(t):
                    return {n.name:ast.dump(n,include_attributes=False) for n in t.body
                        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}
                if runner!='archived' or numerical(old)!=numerical(new):
                    raise RuntimeError('Cannot establish cache compatibility')
            write_json(guard,dict(fingerprint=fingerprint,source_sha256=sha(__file__)))
        if target.exists():continue
        log=out/f'{runner}_{arm}.log'
        with log.open('a') as handle:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',runner,
                '--arm',arm,'--out_dir',str(out)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Worker failed: {log}')


def resolved_args(arm='A_ae'):
    import config
    args=config.build_parser().parse_args([])
    for k,v in configuration(arm).items():setattr(args,k,v)
    args.scenarios=config.parse_scenarios(args.scenarios)
    config.validate_args(args)
    return args


def make_model(device):
    import torch
    from model import Model
    model=Model(10,3,include_camera_state_in_obs=False,gray_nn_width=48,gray_nn_height=36).to(device)
    model.load_state_dict(torch.load(checkpoints('A'),map_location=device),strict=True)
    return model.eval()


def run_migration_job(out,name,variant,production=False):
    """Same indexed random draws; independent dispatch only in this audit."""
    import torch
    import eval as evaluation
    import config
    from train_utils import build_env
    from tools.gray_chain_components import Components
    from tools.audit_policy_sensing_decision import run_configured_episode
    torch.set_num_threads(4)
    args=variant.args(resolved_args())
    if production:args.scene_layout='joint_slit'
    device=torch.device('cuda')
    config.set_global_seed(args.seed,True)
    env=build_env(1,args,device,eval_mode=True)
    component=Components(variant)
    if not production:component.install(env)
    model=make_model(device)
    target=out/'migration'/name;target.mkdir(parents=True,exist_ok=True)
    inputs=dict(variant=vars(variant),production=production,args=vars(args),checkpoint=sha(checkpoints('A')),
        dependencies={p:sha(ROOT/p) for p in ('tools/gray_chain_components.py','tools/audit_policy_sensing_decision.py','env_cuda.py','eval.py','rollout_ops.py')},
        numerical_source=__import__('inspect').getsource(run_migration_job),references=reference_rows('A_ae')[:16])
    guard=target/'inputs.json'
    if guard.exists() and json.loads(guard.read_text())!=inputs:raise RuntimeError(f'Migration cache changed: {name}')
    write_json(guard,inputs)
    rows=[]
    for i,ref in enumerate(reference_rows('A_ae')[:16]):
        path=target/f'ep_{i:03d}.json'
        if path.exists():rows.append(json.loads(path.read_text()));continue
        original=env.reset;records=[]
        def reset(*a,**kw):
            result=original(*a,**kw);records.append(Recorder(env,model,evaluation));return result
        env.reset=reset
        try:
            if production:
                row,_=evaluation.run_one_episode(i,ref['scenario'],args,model,env,None,device)
            else:
                row=run_configured_episode(i,ref['scenario'],args,model,env,device,
                    evaluation=evaluation,renderer=component.render,
                    collision_mode='swept' if variant.swept_collision else 'legacy')
        finally:
            env.reset=original
            if records:records[0].close(target/f'ep_{i:03d}.npz')
        write_json(path,row);rows.append(row)
    result=dict(name=name,variant=vars(variant),production=production,rows=rows,
        successes=sum(r['success_rate'] for r in rows),collisions=sum(r['collision_rate'] for r in rows),
        checkpoint_unchanged=sha(checkpoints('A'))==MODELS['A'][1])
    if not result['checkpoint_unchanged']:raise RuntimeError('Checkpoint changed')
    write_json(target/'result.json',result)
    print(name,result['successes'],result['collisions'],flush=True)
    return result


def compare_migration_jobs(out,left,right):
    a=out/'migration'/left;b=out/'migration'/right
    errors=[]
    aa=json.loads((a/'result.json').read_text())['rows'];bb=json.loads((b/'result.json').read_text())['rows']
    for i,(x,y) in enumerate(zip(aa,bb)):
        discrete=[k for k in ('success_rate','collision_rate','steps','stop_reason') if x[k]!=y[k]]
        first=compare_traces(a/f'ep_{i:03d}.npz',b/f'ep_{i:03d}.npz',True)
        if discrete or first:errors.append(dict(episode=i,discrete=discrete,first_difference=first))
    return dict(passed=not errors,differences=errors)


def action_difference(a,b):
    import numpy as np
    with np.load(a) as x,np.load(b) as y:
        aa=x['raw_action'].astype(float);bb=y['raw_action'].astype(float)
        n=min(len(aa),len(bb))
        return float(np.abs(aa[:n]-bb[:n]).mean()) if n else 0.


def migration(out):
    from tools.gray_chain_components import Variant
    from dataclasses import replace,fields
    if not json.loads((out/'parity_comparison.json').read_text())['passed']:
        raise RuntimeError('Unresolved execution parity; stop before migration')
    results=[run_migration_job(out,f'M{i:02d}',Variant.stage(i)) for i in range(12)]
    run_migration_job(out,'production',Variant.stage(11),production=True)
    production=compare_migration_jobs(out,'M11','production')
    # M0 has to agree with the independently evaluated old protocol, not just
    # the composed runner itself.
    errors=[]
    for i in range(16):
        diff=compare_traces(out/'diagnostic/A_ae'/f'ep_{i:03d}.npz',out/'migration/M00'/f'ep_{i:03d}.npz',True)
        if diff:errors.append(dict(episode=i,first_difference=diff))
    validation=dict(m0=dict(passed=not errors,differences=errors),m11=production)
    write_json(out/'migration_validation.json',validation)
    if errors or not production['passed']:raise RuntimeError('Composition disagrees with independent runner; stop attribution')
    rankings=[]
    for i in range(1,12):
        prev,cur=results[i-1],results[i]
        failures=[k for k,(a,b) in enumerate(zip(prev['rows'],cur['rows'])) if a['success_rate'] and not b['success_rate']]
        change=sum(action_difference(out/'migration'/f'M{i-1:02d}'/f'ep_{k:03d}.npz',out/'migration'/f'M{i:02d}'/f'ep_{k:03d}.npz') for k in range(16))/16
        rankings.append(dict(step=i,factor=fields(Variant)[i-1].name,new_failures=failures,action_change=change))
    rankings.sort(key=lambda r:(-len(r['new_failures']),-r['action_change'],r['step']))
    final_new=[i for i,(a,b) in enumerate(zip(results[0]['rows'],results[-1]['rows'])) if a['success_rate'] and not b['success_rate']]
    selected=[r['factor'] for r in rankings[:2]] if final_new else []
    lock=dict(rankings=rankings,final_new_failures=final_new,selected=selected,max_reverse_checks=3)
    lockpath=out/'reverse_lock.json'
    if lockpath.exists() and json.loads(lockpath.read_text())!=lock:raise RuntimeError('Reverse lock changed')
    write_json(lockpath,lock)
    reverse=[]
    if selected:
        for factor in selected:
            reverse.append(run_migration_job(out,'reverse_'+factor,replace(Variant.stage(11),**{factor:False})))
        reverse.append(run_migration_job(out,'reverse_both',replace(Variant.stage(11),**{f:False for f in selected})))
    write_json(out/'migration_results.json',dict(stages=results,validation=validation,reverse_lock=lock,reverse=reverse))
    endpoint(out)


def endpoint(out):
    """Reconnect to the existing 32-batch diagnostic, score only locked eight."""
    import torch
    import eval as evaluation
    from tools.audit_slit_information_deadline import HistoricalBatch,SPLITS,SCENARIOS
    from tools.audit_policy_sensing_decision import NetworkRun,flight_args
    from render.intent_scene import render_exposure
    from render.pose import body_angular_velocity
    torch.set_num_threads(4)
    history=HistoricalBatch('cuda');model=make_model(torch.device('cuda'))
    output={}
    for name,mode,zero in [('ae_gray','mean_ae',False),('ae_zero','mean_ae',True),('fixed_gray','fixed',False)]:
        rows=[];traces={}
        for scenario in SCENARIOS:
            env=history.new_env(scenario);env.camera_control_mode=mode
            env.fixed_camera_exposure=.10;env.fixed_camera_gain=0.
            args=flight_args();args.camera_control_mode=mode;args.amp=False
            run=None
            def renderer(environment,e,g,differentiable=False):
                ideal,aux=render_exposure(environment,e,return_aux=True)
                shot,read=history.noise(run.step,range(32))
                gray,camera=environment.gray_camera(ideal,e,g,motion=None,shot_noise=shot,read_noise=read,enable_noise=True)
                aux.update(camera);return gray,aux
            def stepper(environment,command,dt,target,step):history.run_step(environment,command,step)
            run=NetworkRun(args=args,env=env,renderer=renderer,stepper=stepper,
                dt_provider=lambda step:1/15,evaluation=evaluation,collision_mode='swept')
            timeline=[]
            for _ in range(90):
                pre=dict(position=env.p.cpu().tolist(),velocity=env.v.cpu().tolist(),rotation=env.R.cpu().tolist(),
                    angular_velocity=body_angular_velocity(env.R_old,env.R,env.last_control_dt).cpu().tolist())
                row=run.advance(None,model,zero=zero)
                row.update(pre,success=run.success.cpu().tolist(),collision=run.collision.cpu().tolist(),
                    requested_eg=env._camera_requested_state.cpu().tolist(),
                    next_position=env.p.cpu().tolist())
                timeline.append(row)
                if not run.alive[list(SPLITS['development'])].any():break
            rows.extend(run.episodes(SPLITS['development'],scenario));traces[scenario]=timeline
        output[name]=dict(episodes=rows,timelines=traces,successes=sum(r['success'] for r in rows),
            collisions=sum(r['collision'] for r in rows),zero_only_actor=zero,camera_mode=mode)
        print('endpoint',name,output[name]['successes'],flush=True)
    write_json(out/'endpoint_results.json',dict(checkpoint_sha256=sha(checkpoints('A')),arms=output,
        numerical_source=__import__('inspect').getsource(endpoint),
        source_sha256={p:sha(ROOT/p) for p in ('tools/audit_slit_information_deadline.py','tools/audit_policy_sensing_decision.py','tools/audit_sensing_scale.py','configs/gray_joint_slit.args','env_cuda.py','model.py','rollout_ops.py','sensors/imx900_camera.py','render/intent_scene.py')},
        limitation='Existing historical tracking-success-selected layouts; no independent generalization evidence.'))



def decoded_submissions(trace,max_acc_cmd):
    """Recover world commands from logged local outputs; no simulator call/RNG.

    The independent check against the executed queue two steps later prevents
    this derived audit field from silently disagreeing with actual execution.
    """
    import torch
    from rollout_ops import build_local_frame,decode_action_direct
    from types import SimpleNamespace
    shape=trace['pre_R'].shape
    rotations=torch.from_numpy(trace['pre_R'].copy()).reshape(-1,3,3)
    env=SimpleNamespace(R=rotations)
    frame=build_local_frame(env)
    raw=torch.from_numpy(trace['raw_action'].copy()).float().reshape(-1,3)
    commands=decode_action_direct(raw,frame,env,len(raw),max_acc_cmd)
    return commands.numpy().reshape(shape[0],shape[1],3)



def summarize(out):
    """Export compact evidence and one figure; never launch another experiment."""
    import csv
    import platform
    import shutil
    import numpy as np
    import torch
    for runner in ('archived','standard','diagnostic'):
        for arm in ARMS:
            guard=json.loads((out/runner/arm/'input_fingerprint.json').read_text())
            if guard['fingerprint']!=job_fingerprint(runner,arm):raise RuntimeError('Stale evaluation cache')
    import inspect
    for guardpath in (out/'migration').glob('*/inputs.json'):
        guard=json.loads(guardpath.read_text())
        if guard['checkpoint']!=sha(checkpoints('A')) or guard['numerical_source']!=inspect.getsource(run_migration_job):
            raise RuntimeError('Stale migration worker')
        for path,expected in guard['dependencies'].items():
            if sha(ROOT/path)!=expected:raise RuntimeError(f'Stale migration source: {path}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    historical=json.loads((out/'historical_comparison.json').read_text())
    parity_result=json.loads((out/'parity_comparison.json').read_text())
    migration_result=json.loads((out/'migration_results.json').read_text())
    validation=migration_result['validation']
    if not historical['passed'] or not parity_result['passed'] or not all(v['passed'] for v in validation.values()):
        raise RuntimeError('Unresolved reproduction/composition error; no attribution summary')
    artifact=ROOT/'docs/grayscale_imx900/artifacts/gray_chain_regression_20261103'
    artifact.mkdir(parents=True,exist_ok=True)
    comparisons=[]
    for i in range(1,12):
        for k in range(16):
            a=out/'migration'/f'M{i-1:02d}'/f'ep_{k:03d}.npz'
            b=out/'migration'/f'M{i:02d}'/f'ep_{k:03d}.npz'
            comparisons.append(dict(step=i,episode_seed=30000+k,
                first_difference=compare_traces(a,b,True),action_change=action_difference(a,b)))
    write_json(out/'migration_first_differences.json',comparisons)
    # Verify the matching random draws from recorded streams (not matching
    # realized images, states or dt, which may intentionally change).
    random_errors=[]
    for i in range(1,12):
        for k in range(16):
            with np.load(out/'migration/M00'/f'ep_{k:03d}.npz') as az,np.load(out/'migration'/f'M{i:02d}'/f'ep_{k:03d}.npz') as bz:
                for field in ('initial_dg','initial_v_wind','initial_slit_center_y','initial_gray_light_factor',
                              'rng_cpu_before_network','rng_cuda_before_network','rng_cpu_before_dynamics','rng_cuda_before_dynamics'):
                    x,y=az[field],bz[field]
                    if not field.startswith('initial_'):x,y=x[:min(len(x),len(y))],y[:min(len(x),len(y))]
                    if not np.array_equal(x,y):random_errors.append(dict(stage=i,episode=k,variable=field))
    write_json(out/'matched_random_process.json',dict(passed=not random_errors,errors=random_errors,
        rule='Episode seed reset; image phase seed+1 and dynamics phase seed+2 each step; same draw shapes.'))
    if random_errors:raise RuntimeError('Random process mismatched')
    rows=[]
    for runner in ('archived','standard','diagnostic'):
        for arm in ARMS:
            result=json.loads((out/runner/arm/'result.json').read_text())
            rows.extend(dict(group=runner,condition=arm,**r) for r in result['rows'])
    for stage in migration_result['stages']+migration_result['reverse']:
        rows.extend(dict(group='migration',condition=stage['name'],**r) for r in stage['rows'])
    production=json.loads((out/'migration/production/result.json').read_text())
    rows.extend(dict(group='migration',condition='production',**r) for r in production['rows'])
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (artifact/'episodes.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    endpoint_result=json.loads((out/'endpoint_results.json').read_text())
    if endpoint_result['numerical_source']!=inspect.getsource(endpoint) or endpoint_result['checkpoint_sha256']!=sha(checkpoints('A')):
        raise RuntimeError('Stale endpoint result')
    for path,expected in endpoint_result['source_sha256'].items():
        if sha(ROOT/path)!=expected:raise RuntimeError(f'Stale endpoint source: {path}')
    compact=copy.deepcopy(endpoint_result)
    for arm in compact['arms'].values():arm.pop('timelines')
    write_json(artifact/'endpoint_results.json',compact)
    for name in ('protocol.json','source_versions.json','historical_comparison.json','parity_comparison.json',
                 'migration_results.json','migration_validation.json','migration_first_differences.json',
                 'reverse_lock.json','matched_random_process.json'):
        shutil.copyfile(out/name,artifact/name)
    shutil.copyfile(out/'tool_source_at_historical.py',artifact/'tool_source_at_historical.py')
    for runner in ('archived','standard','diagnostic'):
        for arm in ARMS:
            shutil.copyfile(out/runner/arm/'runtime.json',artifact/f'{runner}_{arm}_runtime.json')
    # A read-only view of the first old-success/new-failure episode, not a
    # newly selected evaluation set or a new intervention.
    representative=next(i for i,(a,b) in enumerate(zip(migration_result['stages'][2]['rows'],migration_result['stages'][3]['rows']))
        if a['success_rate'] and not b['success_rate'])
    traces={}
    queue_errors=[]
    for path in sorted(out.glob('*/*/ep_*.npz')):
        with np.load(path) as z:
            data={k:z[k] for k in ('pre_R','raw_action','executed_command')}
        commands=decoded_submissions(data,resolved_args().max_acc_cmd)
        if not np.allclose(commands[:-2],data['executed_command'][2:],atol=1e-6,rtol=1e-5):queue_errors.append(str(path))
    write_json(artifact/'submission_queue_validation.json',dict(passed=not queue_errors,errors=queue_errors,
        provenance='World submissions decoded from logged local raw_action and pre_R; checked against execution two steps later. No extra policy/render call.'))
    if queue_errors:raise RuntimeError('Decoded submissions disagree with recorded execution')
    for name in ('M02','M03','M11','reverse_long_approach'):
        source=out/'migration'/name/f'ep_{representative:03d}.npz'
        shutil.copyfile(source,artifact/f'{name}_representative.npz')
        with np.load(source) as data:traces[name]={k:data[k] for k in data.files}
        traces[name]['submitted_command']=decoded_submissions(traces[name],resolved_args().max_acc_cmd)
        np.savez_compressed(artifact/f'{name}_representative.npz',**traces[name])
    fig=plt.figure(figsize=(13,8),layout='constrained')
    grid=fig.add_gridspec(3,4,height_ratios=[1.,1.2,1.])
    ax=fig.add_subplot(grid[0,:])
    values=[s['successes'] for s in migration_result['stages']]
    colors=['#315b7a']*3+['#b2604b']*9
    ax.bar(range(12),values,color=colors)
    ax.set(xticks=range(12),xticklabels=[f'M{i}' for i in range(12)],ylim=(0,17),ylabel='Successes / 16',
        title='Same mixed-light model + mean AE | fixed historical episodes')
    for i,v in enumerate(values):ax.text(i,v+.3,str(int(v)),ha='center',fontsize=9)
    ax.text(3.5,12,'M2 → M3: change start only\n−1.5 m → −2.5 m; 8 → 0 successes',color='#943e2e')
    ax.text(7.6,12,'Full new pipeline: 0/16\nRestore old start: 1/16\nRestore start + jitter: 1/16',fontsize=10)
    for col,name in enumerate(('M02','M03')):
        t=traces[name];index=int(np.argmin(np.abs(t['pre_p'][:,0,0]+1.)))
        ax=fig.add_subplot(grid[1,col]);ax.imshow(t['raw_pixels'][index,0,0],cmap='gray',vmin=0,vmax=1,aspect='equal')
        ax.set_title(f'{name}: actual frame near x=−1 m\nstep {index}, mean {t["raw_pixels"][index,0,0].mean():.3f}',fontsize=10)
        ax.axis('off')
    ax=fig.add_subplot(grid[1,2:])
    for name,color in [('M02','#315b7a'),('M03','#b2604b'),('M11','#777777')]:
        t=traces[name];p=np.concatenate([t['pre_p'][:1,0],t['post_p'][:,0]])
        ax.plot(p[:,0],p[:,1],label=name,color=color)
    center=float(traces['M02']['initial_slit_center_y'][0])
    ax.plot([0,0],[-1.2,center-.15],color='black',lw=3);ax.plot([0,0],[center+.15,1.2],color='black',lw=3)
    ax.set(xlabel='x (m)',ylabel='y (m)',title=f'Episode {30000+representative}: top view',ylim=(-.6,.6));ax.legend(fontsize=8)
    ax=fig.add_subplot(grid[2,:])
    for name,color in [('M02','#315b7a'),('M03','#b2604b')]:
        t=traces[name];time=np.r_[0,np.cumsum(t['dt'])][:-1]
        ax.plot(time,t['raw_action'][:,0,1],label=f'{name}: lateral command',color=color)
    ax.set(xlabel='Elapsed simulation time (s)',ylabel='Lateral acceleration command',title='Actual network output; two-step actuator queue retained')
    ax.legend(fontsize=9)
    for ax in fig.axes:
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(artifact/'chain_regression.png',dpi=180)
    fig.savefig(artifact/'chain_regression.pdf')
    plt.close(fig)
    # Save a small preview for inspection, without loading the large figure.
    from PIL import Image
    image=Image.open(artifact/'chain_regression.png');image.thumbnail((1040,640));image.convert('RGB').save(out/'figure_preview.jpg',quality=82)
    write_json(artifact/'migration_inputs.json',{p.parent.name:json.loads(p.read_text()) for p in sorted((out/'migration').glob('*/inputs.json'))})
    write_json(artifact/'trace_inventory.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.glob('*/*/ep_*.npz'))})
    sources=[ROOT/p for p in ('tools/audit_gray_chain_regression.py','tools/gray_chain_components.py','tools/audit_policy_sensing_decision.py','tests/test_gray_chain_regression.py')]
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    calibration={k:dict(path=str(checkpoints(k).with_name('imx900_calibration.json')),sha256=sha(checkpoints(k).with_name('imx900_calibration.json'))) for k in MODELS}
    write_json(artifact/'delivery_manifest.json',dict(source_sha256=source_hashes,calibration=calibration,
        args_sha256={k:sha(checkpoints(k).with_name('args.json')) for k in MODELS},
        historical_reference_sha256={str(p):sha(p) for p in (ARCHIVE/'evaluation/manifest.json',ARCHIVE/'evaluation/episodes.csv',MEAN/'evaluation/episodes.csv',MEAN/'evaluation.log')},
        checkpoint_unchanged={k:sha(checkpoints(k))==MODELS[k][1] for k in MODELS},
        python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        complete_timelines=str(out),representative_episode_seed=30000+representative,
        no_training=True,calibrated=False,
        result='multiple_transfer_effects_unresolved',
        next_only='At the original start, isolate the remaining M4–M11 transfer combination; no learning-method change yet.'))
    print('Evidence exported:',artifact,flush=True)



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
        sys.path.insert(0,str(ROOT))
        migration(out)
    elif args.stage=='summary':
        sys.path.insert(0,str(ROOT))
        summarize(out)
    else:parser.error('--stage is required')

if __name__=='__main__':main()

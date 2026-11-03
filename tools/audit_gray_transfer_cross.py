#!/usr/bin/env python3
"""Bounded 2x2 transfer audit: two new conditions, no training or sample search."""
import argparse
import copy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_gray_chain_regression import (MODELS,Recorder,checkpoints,sha,write_json,
    resolved_args,reference_rows,run_migration_job,compare_traces,make_model)
from tools.gray_chain_components import Variant,Components

PREVIOUS=ROOT/'logs/gray_chain_regression_20261103'
PREVIOUS_EVIDENCE=ROOT/'docs/grayscale_imx900/artifacts/gray_chain_regression_20261103'
ARTIFACT=ROOT/'docs/grayscale_imx900/artifacts/gray_transfer_cross_20261103'


def variants():
    base=Variant.stage(2)
    return dict(OO=base,WO=replace(Variant.stage(11),long_approach=False,
        torch_dynamics=False,swept_collision=False),OD=replace(base,torch_dynamics=True,
        swept_collision=True),WD=replace(Variant.stage(11),long_approach=False))


def numerical_sources():
    paths=[*ROOT.glob('*.py')]
    for folder in ('render','sensors','controllers'):
        paths.extend((ROOT/folder).glob('*.py'))
    paths.extend(ROOT/name for name in ('tools/audit_gray_chain_regression.py',
        'tools/gray_chain_components.py','tools/audit_policy_sensing_decision.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def lock_protocol(out):
    import torch
    import quadsim_cuda
    if sha(checkpoints('A'))!=MODELS['A'][1]:raise RuntimeError('Frozen A checkpoint changed')
    previous=json.loads((PREVIOUS_EVIDENCE/'delivery_manifest.json').read_text())
    for name in ('tools/audit_gray_chain_regression.py','tools/gray_chain_components.py','tools/audit_policy_sensing_decision.py'):
        if sha(ROOT/name)!=previous['source_sha256'][name]:raise RuntimeError(f'Previous numerical source changed: {name}')
    if sha(checkpoints('A').with_name('imx900_calibration.json'))!=previous['calibration']['A']['sha256']:
        raise RuntimeError('Calibration changed')
    inputs=dict(variants={k:vars(v) for k,v in variants().items()},
        args={k:vars(v.args(resolved_args())) for k,v in variants().items()},
        checkpoint_sha256=sha(checkpoints('A')),calibration_sha256=previous['calibration']['A']['sha256'],
        references=reference_rows('A_ae')[:16],sources=numerical_sources(),
        variant_definition=inspect.getsource(variants),worker_source=inspect.getsource(run_migration_job),
        previous_root=str(PREVIOUS),torch=torch.__version__,cuda=torch.version.cuda,
        extension_path=quadsim_cuda.__file__,extension_sha256=sha(quadsim_cuda.__file__),
        new_closed_loop_conditions=['WO','OD'],new_episodes=32,no_training=True)
    inputs=json.loads(json.dumps(inputs))
    path=out/'protocol.json';out.mkdir(parents=True,exist_ok=True)
    if path.exists() and json.loads(path.read_text())!=inputs:raise RuntimeError('Locked inputs changed; refusing cached cross results')
    write_json(path,inputs)
    versions=out/'tool_versions.json'
    history=json.loads(versions.read_text()) if versions.exists() else []
    if not history or history[-1]['sha256']!=sha(__file__):
        history.append(dict(sha256=sha(__file__)));write_json(versions,history)
    return inputs


def source_cell(out,cell):
    if cell=='OO':return PREVIOUS/'migration/M02'
    if cell=='WD':return PREVIOUS/'migration/reverse_long_approach'
    return out/'migration'/cell


def verify_cell(path,variant,previous=False):
    guard=json.loads((path/'inputs.json').read_text())
    expected=json.loads(json.dumps(vars(variant.args(resolved_args()))))
    if guard['args']!=expected or guard['variant']!=vars(variant) or guard['production']:
        raise RuntimeError(f'Incorrect cell configuration: {path}')
    if guard['checkpoint']!=MODELS['A'][1] or guard['numerical_source']!=inspect.getsource(run_migration_job):
        raise RuntimeError('Numerical worker/model mismatch')
    if guard['references']!=reference_rows('A_ae')[:16]:raise RuntimeError('Episode reference mismatch')
    for name,digest in guard['dependencies'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError(f'Cell source mismatch: {name}')
    result=json.loads((path/'result.json').read_text())
    if len(result['rows'])!=16:raise RuntimeError('Incomplete cell')
    inventory=json.loads((PREVIOUS_EVIDENCE/'trace_inventory.json').read_text()) if previous else None
    for i,row in enumerate(result['rows']):
        if row!=json.loads((path/f'ep_{i:03d}.json').read_text()):raise RuntimeError('Episode/result mismatch')
        if row['episode_seed']!=30000+i:raise RuntimeError('Episode order changed')
        trace=path/f'ep_{i:03d}.npz'
        if inventory and sha(trace)!=inventory[str(trace.relative_to(PREVIOUS))]:
            raise RuntimeError('Frozen trace changed')
    return result


def paired_results(results):
    rows=[]
    for i in range(16):
        values={name:bool(result['rows'][i]['success_rate']) for name,result in results.items()}
        rows.append(dict(seed=30000+i,scenario=results['OO']['rows'][i]['scenario'],**values))
    summary={}
    for cell,result in results.items():
        summary[cell]=dict(successes=int(sum(r['success_rate'] for r in result['rows'])),
            collisions=int(sum(r['collision_rate'] for r in result['rows'])),
            timeouts=sum(r['stop_reason']=='timeout' for r in result['rows']),
            lost=[r['seed'] for r in rows if r['OO'] and not r[cell]],
            recovered=[r['seed'] for r in rows if not r['OO'] and r[cell]])
    world_loss=summary['OO']['successes']-summary['WO']['successes']
    dynamics_loss=summary['OO']['successes']-summary['OD']['successes']
    if world_loss>dynamics_loss:decision='world_first'
    elif dynamics_loss>world_loss:decision='motion_first'
    else:decision='interaction_or_unresolved'
    return dict(cells=summary,paired=rows,priority=decision,
        caution='Engineering prioritization on existing historical episodes; no significance/generalization or joint-learning claim.')


def validate_matching(out):
    import numpy as np
    errors=[];differences=[]
    for cell in ('WO','OD','WD'):
        for i in range(16):
            a=source_cell(out,'OO')/f'ep_{i:03d}.npz';b=source_cell(out,cell)/f'ep_{i:03d}.npz'
            with np.load(a) as az,np.load(b) as bz:
                for key in ('initial_p','initial_v','initial_a','initial_R','initial_act','initial_dg','initial_v_wind',
                    'initial_slit_center_y','initial_gray_light_factor','rng_cpu_before_network','rng_cuda_before_network',
                    'rng_cpu_before_dynamics','rng_cuda_before_dynamics'):
                    if key not in az or key not in bz:raise RuntimeError(f'Missing {key}')
                    x,y=az[key],bz[key]
                    if not key.startswith('initial_'):n=min(len(x),len(y));x,y=x[:n],y[:n]
                    if not np.array_equal(x,y):errors.append(dict(cell=cell,seed=30000+i,key=key))
                for key in ('raw_pixels','raw_action','post_p'):
                    x,y=az[key],bz[key];n=min(len(x),len(y))
                    axes=tuple(range(1,x.ndim))
                    atol,rtol=(1e-3,1e-3) if key=='raw_action' else (1e-6,1e-5)
                    bad=np.flatnonzero(np.any(np.abs(x[:n].astype(float)-y[:n].astype(float))>atol+rtol*np.abs(y[:n]),axis=axes))
                    differences.append(dict(cell=cell,seed=30000+i,variable=key,first_step=int(bad[0]) if len(bad) else None,
                        maximum_absolute_difference=float(np.abs(x[:n].astype(float)-y[:n].astype(float)).max())))
    value=dict(passed=not errors,errors=errors,first_differences=differences)
    write_json(out/'matching.json',value)
    if errors:raise RuntimeError('Cross-cell initial state/random process differs')
    return value


def cross(out):
    lock_protocol(out)
    results={}
    for cell,variant in variants().items():
        path=source_cell(out,cell)
        if cell in ('WO','OD'):run_migration_job(out,cell,variant)
        results[cell]=verify_cell(path,variant,cell in ('OO','WD'))
    matching=validate_matching(out)
    result=paired_results(results)
    result['source_paths']={cell:str(source_cell(out,cell)) for cell in results}
    result['matching_passed']=matching['passed']
    write_json(out/'cross_results.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def load_trace(path):
    import numpy as np
    with np.load(path) as z:return {key:z[key] for key in z.files}


def sparse_value(trace,key,step,device):
    import numpy as np
    import torch
    if key not in trace:return None
    ids=np.flatnonzero(trace[key+'_steps']==step)
    return torch.from_numpy(trace[key][ids[0]].copy()).to(device) if len(ids) else None


def restore_pose(env,trace,step):
    import torch
    for name in ('p','v','a','R','act','dg'):
        setattr(env,name,torch.from_numpy(trace['pre_'+name][step].copy()).to(env.device))
    env.R_old=torch.from_numpy(trace['pre_R'][max(0,step-1)].copy()).to(env.device)
    env.p_old=torch.from_numpy(trace['pre_p'][max(0,step-1)].copy()).to(env.device)
    env.last_control_dt=float(trace['dt'][step-1]) if step else 1/15


def recorded_settings(trace,step,env):
    import torch
    if step:
        values=torch.from_numpy(trace['next_eg'][step-1].copy()).to(env.device)
        return values[:,0],values[:,1]
    return (env.p.new_full((env.batch_size,),.10),env.p.new_zeros(env.batch_size))


def observation_steps(trace):
    import numpy as np
    result=[('initial',0)]
    # Scene yaw is zero in these fixed historical evaluations. Reject a future
    # protocol change instead of treating world x as scene x implicitly.
    for label,x in [('at_minus_1m',-1.),('at_minus_half_m',-.5)]:
        ids=np.flatnonzero(trace['pre_p'][:,0,0]>=x)
        if len(ids):result.append((label,int(ids[0])))
        else:result.append((label,None))
    return result


def offline(out):
    """Matched-state image intervention; NOT another flight or trained policy."""
    lock_protocol(out)
    cross_result=json.loads((out/'cross_results.json').read_text())
    if cross_result['priority']!='world_first':
        write_json(out/'offline_results.json',dict(executed=False,reason='Scene/image block is not the larger measured loss; no image intervention added.'))
        return
    import numpy as np
    import torch
    from torch.cuda.amp import autocast
    from config import set_global_seed
    from train_utils import build_env
    from rollout_ops import build_local_frame,build_state_vector,compute_target_velocity
    torch.set_num_threads(4);device=torch.device('cuda')
    model=make_model(device)
    environments={};components={}
    for cell in ('OO','WO'):
        args=variants()[cell].args(resolved_args())
        environments[cell]=build_env(1,args,device,eval_mode=True)
        components[cell]=Components(variants()[cell]);components[cell].install(environments[cell])
    target=out/'offline';target.mkdir(exist_ok=True)
    rows=[];errors=[]
    with torch.no_grad():
        for i,ref in enumerate(reference_rows('A_ae')[:16]):
            trace=load_trace(source_cell(out,'OO')/f'ep_{i:03d}.npz')
            seed=30000+i
            for env in environments.values():
                set_global_seed(seed,True);env.reset(ref['scenario'])
                if not torch.equal(env.R_scene,torch.eye(3,device=device)[None]):raise RuntimeError('Nonzero scene yaw requires scene-frame sampling')
                for key in ('p','v','R','dg','v_wind','slit_center_y','gray_light_factor'):
                    actual=getattr(env,key).cpu().numpy()
                    if not np.array_equal(actual,trace['initial_'+key]):raise RuntimeError(f'Offline reset mismatch {key}')
            for label,step in observation_steps(trace):
                if step is None:
                    rows.append(dict(seed=seed,scenario=ref['scenario'],label=label,missing=True));continue
                outputs={};arrays={}
                for cell in ('OO','WO'):
                    env=environments[cell]
                    frames=[]
                    frame_steps=[step] if step==0 else [step,step-1]
                    for t in frame_steps:
                        restore_pose(env,trace,t)
                        e,g=recorded_settings(trace,t,env)
                        set_global_seed((seed*1000003+3*t+1)%(2**32),True)
                        image,_=components[cell].render(env,e,g,False)
                        frames.append(image)
                    if step==0:frames.append(frames[0])
                    pixels=torch.cat(frames,1)
                    restore_pose(env,trace,step)
                    e,g=recorded_settings(trace,step,env)
                    state,_,camera_state,motion_state=build_state_vector(env,
                        compute_target_velocity(env.p_target-env.p,env),build_local_frame(env),e,g,False,False)
                    original_state=sparse_value(trace,'state',step,device)
                    torch.testing.assert_close(state,original_state,atol=1e-6,rtol=1e-5)
                    torch.testing.assert_close(camera_state,sparse_value(trace,'camera_state',step,device),atol=0,rtol=0)
                    recorded=[];original_preprocess=model.preprocess_gray_input
                    def preprocess(*a,**kw):
                        result=original_preprocess(*a,**kw);recorded.append(result);return result
                    model.preprocess_gray_input=preprocess
                    try:
                        with autocast(enabled=True):
                            action,_,_,_=model(original_state,sparse_value(trace,'hidden_before',step,device),
                                gray_obs=pixels,cam_hx=sparse_value(trace,'camera_hidden_before',step,device),
                                camera_state=camera_state,camera_motion_state=motion_state)
                    finally:model.preprocess_gray_input=original_preprocess
                    if len(recorded)!=1:raise RuntimeError('Preprocessing repeated')
                    outputs[cell]=dict(raw=pixels.cpu().numpy(),network=recorded[0].cpu().numpy(),action=action.float().cpu().numpy())
                    arrays.update({cell+'_'+k:v for k,v in outputs[cell].items()})
                old,new=outputs['OO'],outputs['WO']
                raw_error=float(np.abs(old['raw']-trace['raw_pixels'][step]).max())
                network_error=float(np.abs(old['network']-trace['network_pixels'][step]).max())
                action_error=float(np.abs(old['action']-trace['raw_action'][step].astype(float)).max())
                passed=bool(np.allclose(old['raw'],trace['raw_pixels'][step],atol=1e-6,rtol=1e-5)
                    and np.allclose(old['network'],trace['network_pixels'][step],atol=1e-6,rtol=1e-5)
                    and np.allclose(old['action'],trace['raw_action'][step],atol=1e-3,rtol=1e-3))
                item=dict(seed=seed,scenario=ref['scenario'],label=label,step=step,missing=False,
                    reconstruction_passed=passed,raw_reconstruction_max=raw_error,
                    network_reconstruction_max=network_error,action_reconstruction_max=action_error,
                    image_mae=float(np.abs(old['raw']-new['raw']).mean()),
                    network_input_mae=float(np.abs(old['network']-new['network']).mean()),
                    action_max_difference=float(np.abs(old['action']-new['action']).max()),
                    old_action=old['action'].tolist(),new_action=new['action'].tolist(),
                    position=trace['pre_p'][step].tolist(),applied_eg=[e.cpu().tolist(),g.cpu().tolist()],
                    old_flight_success=cross_result['paired'][i]['OO'],new_world_flight_success=cross_result['paired'][i]['WO'])
                rows.append(item)
                if not passed:errors.append(dict(seed=seed,label=label,step=step,raw=raw_error,network=network_error,action=action_error))
                arrays['position']=trace['pre_p'][step]
                np.savez_compressed(target/f'{seed}_{label}.npz',**arrays)
    value=dict(executed=True,rows=rows,reconstruction_passed=not errors,errors=errors,
        source_sha256=sha(__file__),numerical_source=inspect.getsource(offline),
        helper_sources={f.__name__:inspect.getsource(f) for f in (load_trace,sparse_value,restore_pose,recorded_settings,observation_steps)},
        limitation='Both frames rendered on the same recorded trajectory with fixed E/G, state and recurrent memories. Immediate action response, not a new closed-loop success result.')
    write_json(out/'offline_results.json',value)
    if errors:raise RuntimeError('Original frame/action replay failed; stop interpretation')
    print('Matched-state image replay:',len(rows),'samples; original reconstruction passed',flush=True)



def sphere_segment_certificate(start,end,boxes,radius):
    """Conservative lower bound for continuous spherical clearance.

    Sphere/box signed distance is 1-Lipschitz. Every point on a straight
    segment lies within length/(2*(17-1)) of a grid sample. Subtract that
    amount plus a numerical guard from the minimum sampled clearance.
    This audits existing trajectories; it never changes collision scoring.
    """
    import torch
    from render.box_scene import box_clearance
    start,end,boxes=(v.to(dtype=torch.float64,device='cpu') for v in (start,end,boxes))
    fractions=torch.linspace(0,1,17,dtype=torch.float64)[:,None,None]
    points=start[None]+fractions*(end-start)[None]
    minimum=box_clearance(points,boxes,radius).amin(0)
    lower=minimum-(end-start).norm(dim=-1)/32-1e-6
    return minimum,lower


def terminal_geometry(out):
    import torch
    from train_utils import build_env
    from render.box_scene import swept_box_collision
    args=variants()['OO'].args(resolved_args())
    env=build_env(1,args,torch.device('cuda'),eval_mode=True)
    env.reset('dark')
    radius=float(env.drone_radius);threshold=float(args.collision_clearance)
    rows=[]
    cross_result=json.loads((out/'cross_results.json').read_text())
    # The four cases lost by enabling the motion/collision block, conditional
    # on the scene block. These are prespecified by paired outcomes, not by
    # whichever clearance happens to support an explanation.
    for cell,reference in [('OD','OO'),('WD','WO')]:
        for i in range(16):
            before=cross_result['paired'][i][reference];after=cross_result['paired'][i][cell]
            if not before or after:continue
            trace=load_trace(source_cell(out,cell)/f'ep_{i:03d}.npz')
            start=torch.from_numpy(trace['pre_p'][-1].copy())
            end=torch.from_numpy(trace['post_p'][-1].copy())
            boxes=torch.from_numpy(trace['initial_voxels'].copy())
            sampled,lower=sphere_segment_certificate(start,end,boxes,radius)
            broad=bool(swept_box_collision(start,end,boxes,radius).item())
            ref=load_trace(source_cell(out,reference)/f'ep_{i:03d}.npz')
            all_start=torch.from_numpy(ref['pre_p'][:,0].copy())
            all_end=torch.from_numpy(ref['post_p'][:,0].copy())
            all_boxes=torch.from_numpy(ref['initial_voxels'].copy()).expand(len(all_start),-1,-1)
            _,reference_lower=sphere_segment_certificate(all_start,all_end,all_boxes,radius)
            rows.append(dict(cell=cell,reference_cell=reference,seed=30000+i,
                stop_step=len(trace['dt'])-1,expanded_box_sweep=broad,
                sampled_sphere_clearance_m=float(sampled.item()),
                certified_sphere_clearance_lower_m=float(lower.item()),
                certified_above_collision_threshold=bool((lower>threshold).item()),
                reference_full_path_lower_m=float(reference_lower.min()),
                reference_full_path_certified_clear=bool((reference_lower>threshold).all()),
                start=start.tolist(),end=end.tolist()))
    value=dict(radius_m=radius,collision_threshold_m=threshold,rows=rows,
        source=inspect.getsource(sphere_segment_certificate),
        limitation='Certificate applies to recorded piecewise-linear spherical motion. No continuation after a stopped episode and no revised flight-success count are inferred.')
    write_json(out/'terminal_geometry.json',value)
    return value



def summarize(out):
    """Export evidence only, with unchanged 32 newly executed episodes."""
    import csv
    import shutil
    import platform
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    lock_protocol(out)
    for cell,variant in variants().items():verify_cell(source_cell(out,cell),variant,cell in ('OO','WD'))
    cross_result=json.loads((out/'cross_results.json').read_text())
    offline_result=json.loads((out/'offline_results.json').read_text())
    geometry=json.loads((out/'terminal_geometry.json').read_text())
    if offline_result.get('executed'):
        if not offline_result['reconstruction_passed'] or offline_result['numerical_source']!=inspect.getsource(offline):
            raise RuntimeError('Offline replay mismatch or changed code')
        for name,source in offline_result['helper_sources'].items():
            if inspect.getsource(globals()[name])!=source:raise RuntimeError('Offline helper changed')
    if geometry['source']!=inspect.getsource(sphere_segment_certificate):raise RuntimeError('Certificate source changed')
    if not json.loads((out/'matching.json').read_text())['passed']:raise RuntimeError('Matching check failed')
    safe_stops=[r for r in geometry['rows'] if r['expanded_box_sweep'] and r['certified_above_collision_threshold']]
    priority='collision_geometry_first' if safe_stops else cross_result['priority']
    decision=dict(priority=priority,
        reason='Declared sphere is clear at newly lost cases stopped by its conservative expanded-box enclosure.' if safe_stops else cross_result['priority'],
        safe_terminal_cases=[dict(cell=r['cell'],seed=r['seed'],clearance_lower_mm=r['certified_sphere_clearance_lower_m']*1000) for r in safe_stops],
        no_revised_success_claim=True,no_training=True,
        next_only='Make the continuous collision criterion consistent with the declared spherical body, then reevaluate the same locked episodes; do not change camera, training or collision margin.')
    write_json(out/'decision.json',decision)
    ARTIFACT.mkdir(parents=True,exist_ok=True)
    for name in ('protocol.json','cross_results.json','matching.json','offline_results.json','terminal_geometry.json','decision.json','tool_versions.json'):
        shutil.copyfile(out/name,ARTIFACT/name)
    records=[];cell_inputs={}
    for cell in variants():
        path=source_cell(out,cell)
        result=json.loads((path/'result.json').read_text())
        records.extend(dict(cell=cell,reused=cell in ('OO','WD'),**r) for r in result['rows'])
        cell_inputs[cell]=json.loads((path/'inputs.json').read_text())
    write_json(ARTIFACT/'cell_inputs.json',cell_inputs)
    keys=sorted(set().union(*(r.keys() for r in records)))
    with (ARTIFACT/'episodes.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows(records)
    with (ARTIFACT/'paired.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=['seed','scenario','OO','WO','OD','WD'],lineterminator='\n')
        writer.writeheader();writer.writerows(cross_result['paired'])
    representative=cross_result['cells']['WO']['lost'][0]
    snapshots=[r for r in offline_result['rows'] if not r.get('missing') and r['seed']==representative]
    for row in snapshots:
        name=f'{representative}_{row["label"]}.npz'
        shutil.copyfile(out/'offline'/name,ARTIFACT/name)
    for cell in ('WO','OD'):
        index=cross_result['cells'][cell]['lost'][0]-30000
        source=source_cell(out,cell)/f'ep_{index:03d}.npz'
        shutil.copyfile(source,ARTIFACT/f'{cell}_{index+30000}_trace.npz')
    fig=plt.figure(figsize=(13,10),layout='constrained')
    grid=fig.add_gridspec(3,4,height_ratios=[1.1,1.25,.8])
    ax=fig.add_subplot(grid[0,:2]);names=['OO','WO','OD','WD']
    values=[cross_result['cells'][n]['successes'] for n in names]
    ax.bar(names,values,color=['#395c77','#738b9d','#6e8d83','#ad6c57'])
    for i,v in enumerate(values):ax.text(i,v+.25,f'{v}/16',ha='center')
    ax.set(ylim=(0,17),ylabel='Recorded successes / 16',title='Two new conditions; two frozen references')
    ax.text(.5,13,'OO old scene / old motion\nWO new scene / old motion\nOD old scene / new motion\nWD new scene / new motion',fontsize=9)
    ax=fig.add_subplot(grid[0,2:]);matrix=np.array([[r[n] for n in names] for r in cross_result['paired']],int)
    ax.imshow(matrix.T,cmap=ListedColormap(['#e7e9eb','#477d6b']),vmin=0,vmax=1,aspect='auto',interpolation='nearest')
    ax.set(yticks=range(4),yticklabels=names,xticks=range(16),xticklabels=[str(i) for i in range(16)],
        xlabel='Historical seed = 30000 + index',title='Paired outcomes: green = success')
    selected=next(r for r in snapshots if r['label']=='at_minus_1m')
    with np.load(out/'offline'/f'{representative}_at_minus_1m.npz') as z:
        for col,cell in enumerate(('OO','WO')):
            ax=fig.add_subplot(grid[1,col]);ax.imshow(np.clip(z[cell+'_raw'][0,0],0,1)**(1/2.2),cmap='gray',vmin=0,vmax=1)
            ax.set_title(('Old' if cell=='OO' else 'New')+' frame: same pose, E/G, history\nShared display gamma 1/2.2',fontsize=10);ax.axis('off')
    ax=fig.add_subplot(grid[1,2:])
    values=[r['certified_sphere_clearance_lower_m']*1000 for r in geometry['rows']]
    labels=[f'{r["cell"]}\n{r["seed"]}' for r in geometry['rows']]
    ax.bar(labels,values,color='#738b9d')
    ax.axhline(geometry['collision_threshold_m']*1000,color='#ad6c57',ls='--',label='Collision margin (1.1 mm)')
    for i,v in enumerate(values):ax.text(i,v+.25,f'≥{v:.1f}',ha='center',fontsize=9)
    ax.set(ylabel='Certified sphere clearance (mm)',title='Four early stops with a clear sphere');ax.legend(fontsize=8)
    ax=fig.add_subplot(grid[2,:])
    labels=['initial','at_minus_1m','at_minus_half_m'];groups=[]
    for label in labels:
        group=[r['action_max_difference'] for r in offline_result['rows'] if r['label']==label and not r['missing']]
        groups.append(group)
    means=[np.mean(g) for g in groups]
    ax.bar(range(3),means,width=.5,color='#395c77',label='Mean max-component action change')
    for i,g in enumerate(groups):
        ax.scatter(np.full(len(g),i)+np.linspace(-.17,.17,len(g)),g,s=15,color='#ad6c57',alpha=.65)
    ax.set(xticks=range(3),xticklabels=['Initial frame','First x ≥ −1.0 m','First x ≥ −0.5 m'],
        ylabel='Acceleration command change (m/s²)',title='48 matched-state image comparisons; original frame/action replay passed')
    ax.legend(fontsize=8)
    for ax in fig.axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(ARTIFACT/'cross_and_geometry.png',dpi=180)
    fig.savefig(ARTIFACT/'cross_and_geometry.pdf')
    plt.close(fig)
    from PIL import Image
    image=Image.open(ARTIFACT/'cross_and_geometry.png');image.thumbnail((1040,800));image.convert('RGB').save(out/'preview.jpg',quality=83)
    traces={str(p.relative_to(out)):sha(p) for p in sorted(out.glob('migration/*/ep_*.npz'))}
    traces.update({str(p.relative_to(out)):sha(p) for p in sorted((out/'offline').glob('*.npz'))})
    write_json(ARTIFACT/'trace_inventory.json',traces)
    write_json(ARTIFACT/'delivery_manifest.json',dict(
        source_sha256={p:sha(ROOT/p) for p in ('tools/audit_gray_transfer_cross.py','tests/test_gray_transfer_cross.py','docs/grayscale_imx900/GRAY_TRANSFER_CROSS_PLAN_2026_11_03.md')},
        checkpoint_sha256=sha(checkpoints('A')),calibration_sha256=sha(checkpoints('A').with_name('imx900_calibration.json')),
        python=sys.executable,python_version=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        full_timelines=str(out),previous_frozen_timelines=str(PREVIOUS),new_conditions=2,new_episodes=32,
        offline_snapshots=48,representative_image_seed=representative,priority=priority,
        production_numerical_sources_unchanged=True,no_training=True))
    print('Evidence exported:',ARTIFACT,flush=True)



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir',type=Path,default=ROOT/'logs/gray_transfer_cross_20261103')
    parser.add_argument('--stage',choices=['cross','offline','summary','all'],default='all')
    args=parser.parse_args();out=args.out_dir.resolve()
    if args.stage in ('cross','all'):
        cross(out)
        terminal_geometry(out)
    if args.stage in ('offline','all'):offline(out)
    if args.stage in ('summary','all'):summarize(out)

if __name__=='__main__':main()

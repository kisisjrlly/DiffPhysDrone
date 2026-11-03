"""Regression audit must isolate execution differences without changing physics."""
import copy
from dataclasses import fields,replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.audit_gray_chain_regression import (ARMS,MODELS,checkpoints,configuration,
    reference_rows,sha,compare_traces,resolved_args,make_model,Recorder)
from tools.gray_chain_components import Variant,Components


def test_reference_identity_amp_timing_and_camera_modes():
    n=configuration('N_gray');z=configuration('N_zero');a=configuration('A_ae')
    assert n['amp'] is False and a['amp'] is True
    assert n['camera_control_mode']=='fixed' and a['camera_control_mode']=='mean_ae'
    assert n['fixed_camera_exposure']==a['fixed_camera_exposure']==.1
    assert n['fixed_camera_gain']==a['fixed_camera_gain']==0
    assert {k:v for k,v in z.items() if k!='policy_gray_mode'}=={k:v for k,v in n.items() if k!='policy_gray_mode'}
    for arm,seed in [('N_gray',10000),('N_zero',10000),('A_ae',30000)]:
        rows=reference_rows(arm)
        assert [int(r['episode_seed']) for r in rows]==list(range(seed,seed+100))
        c=configuration(arm)
        assert c['timesteps']==60 and c['base_control_freq']==15 and c['simple_start_x']==-1.5
        assert c['control_dt_jitter']==.1 and c['gray_saturation_mode']=='soft'
        assert sha(checkpoints(arm[0]))==MODELS[arm[0]][1]
    assert [r['scenario'] for r in reference_rows('A_ae')[:16]]==['dark','bright','dark_to_bright','bright_to_dark']*4


def test_migration_changes_exactly_one_declared_factor():
    base=resolved_args()
    for i in range(1,12):
        left,right=Variant.stage(i-1),Variant.stage(i)
        assert [f.name for f in fields(left) if getattr(left,f.name)!=getattr(right,f.name)]==[fields(left)[i-1].name]
    assert Variant.stage(0).args(base).simple_start_x==-1.5
    final=Variant.stage(11).args(base)
    assert (final.simple_start_x,final.timesteps,final.gray_temporal_samples)==(-2.5,90,9)
    assert final.amp and final.camera_control_mode=='mean_ae'
    with pytest.raises(ValueError):Variant.stage(12)
    with pytest.raises(ValueError):Variant.stage(-1)


@pytest.mark.parametrize('key,step',[('raw_pixels',2),('executed_command',1),('dt',3)])
def test_trace_comparator_locates_injected_first_difference(tmp_path,key,step):
    data={'initial_p':np.zeros((1,3))}
    for name in ('dt','raw_pixels','executed_command'):
        data[name]=np.zeros((4,1));data[name+'_steps']=np.arange(4)
    a=tmp_path/'a.npz';b=tmp_path/'b.npz'
    np.savez_compressed(a,**data)
    data[key]=data[key].copy();data[key][step]+=1
    np.savez_compressed(b,**data)
    diff=compare_traces(a,b)
    assert (diff['step'],diff['variable'])==(step,key)


@pytest.fixture(scope='module')
def device():
    if not torch.cuda.is_available():pytest.skip('Archived CUDA replay required')
    torch.set_num_threads(4)
    return torch.device('cuda')


def environment(args,device):
    from train_utils import build_env
    return build_env(1,args,device,eval_mode=True)


@pytest.mark.parametrize('index',[0,11])
def test_composition_matches_independent_production_step(index,device):
    import eval as evaluation
    from config import set_global_seed
    from rollout_ops import init_camera_params,render_gray_sensor
    variant=Variant.stage(index);args=variant.args(resolved_args())
    production=copy.deepcopy(args)
    production.scene_layout='joint_slit' if index==11 else 'single_wall_slit'
    a=environment(args,device);b=environment(production,device)
    component=Components(variant);component.install(a)
    for env in (a,b):
        set_global_seed(30000,True);env.reset('dark')
    for name in ('p','R','voxels','dg','v_wind'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)
    ea,ga=init_camera_params(a,1,device);eb,gb=init_camera_params(b,1,device)
    for step in range(3):
        set_global_seed(891+step,True)
        ia,auxa=component.render(a,ea,ga)
        set_global_seed(891+step,True)
        ib,auxb=render_gray_sensor(b,eb,gb,False)
        torch.testing.assert_close(ia,ib,rtol=0,atol=0)
        command=a.p.new_tensor([[.3,.1,-.1]])
        for env in (a,b):
            set_global_seed(991+step,True);env.run(command,1/15,env.p_target-env.p)
        for name in ('p','v','a','R','dg'):
            torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)
        torch.testing.assert_close(a.find_vec_to_nearest_pt(),b.find_vec_to_nearest_pt(),rtol=0,atol=0)
        if index==11:torch.testing.assert_close(a.transition_clearance(),b.transition_clearance(),rtol=0,atol=0)


def test_recording_preserves_rng_trajectory_and_parameters(tmp_path,device):
    import eval as evaluation
    from tools.audit_policy_sensing_decision import run_configured_episode
    args=resolved_args();args.timesteps=4
    model=make_model(device)
    before={k:v.clone() for k,v in model.state_dict().items()}
    a=environment(args,device);b=environment(args,device)
    plain=run_configured_episode(0,'dark',args,model,a,device,evaluation=evaluation)
    reset=b.reset;records=[]
    def instrumented(*aa,**kk):
        result=reset(*aa,**kk);records.append(Recorder(b,model,evaluation));return result
    b.reset=instrumented
    try:recorded=run_configured_episode(0,'dark',args,model,b,device,evaluation=evaluation)
    finally:records[0].close(tmp_path/'trace.npz')
    assert plain==recorded
    for name in ('p','v','a','R','dg'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)
    for name,value in model.state_dict().items():torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    with np.load(tmp_path/'trace.npz') as trace:
        assert len(trace['dt'])==4
        # First two commands are already queued, rather than current output.
        np.testing.assert_equal(trace['executed_command'][:2],0)
        assert trace['raw_pixels'].shape[2:]==(2,72,96)
        assert trace['network_pixels'].shape[2:]==(2,36,48)


def test_zero_network_still_drives_ae_from_real_frame(device):
    import eval as evaluation
    from config import set_global_seed
    from tools.audit_policy_sensing_decision import NetworkRun
    args=resolved_args();env=environment(args,device)
    set_global_seed(30000,True);env.reset('dark')
    model=make_model(device);inputs=[];calls=[]
    hook=model.register_forward_pre_hook(lambda m,a,k:inputs.append(k['gray_obs'].clone()),with_kwargs=True)
    def renderer(*a,**k):
        calls.append(1)
        image,aux=evaluation.render_gray_sensor(*a,**k)
        return image,aux
    run=NetworkRun(args=args,env=env,renderer=renderer,dt_provider=lambda i:1/15,evaluation=evaluation)
    e=run.e.clone()
    try:row=run.advance(None,model,zero=True)
    finally:hook.remove()
    assert len(calls)==1 and inputs[0].count_nonzero()==0
    assert run.previous.count_nonzero()>0 and (run.e>e).any()
    assert torch.equal(run.queue[0],torch.zeros_like(run.queue[0]))


def test_trace_comparator_handles_changed_geometry_shape(tmp_path):
    a=tmp_path/'a.npz';b=tmp_path/'b.npz'
    np.savez_compressed(a,initial_voxels=np.zeros((1,3,6)),dt=np.zeros(1))
    np.savez_compressed(b,initial_voxels=np.zeros((1,7,6)),dt=np.zeros(1))
    diff=compare_traces(a,b)
    assert diff==dict(step=-1,variable='initial_voxels',reason='initial_state_mismatch')


def test_single_sensor_noise_and_single_preprocess(device,monkeypatch):
    import eval as evaluation
    from tools.audit_policy_sensing_decision import NetworkRun
    args=Variant.stage(11).args(resolved_args());env=environment(args,device)
    components=Components(Variant.stage(11));components.install(env);env.reset('dark')
    model=make_model(device);draws=[];preprocess_calls=[]
    randn=torch.randn_like;preprocess=model.preprocess_gray_input
    def count_random(value,*a,**kw):
        if value.shape==(1,1,72,96):draws.append(value.shape)
        return randn(value,*a,**kw)
    def count_preprocess(image,*a,**kw):
        preprocess_calls.append(kw.get('add_noise',a[0] if a else False))
        return preprocess(image,*a,**kw)
    monkeypatch.setattr(torch,'randn_like',count_random)
    monkeypatch.setattr(model,'preprocess_gray_input',count_preprocess)
    run=NetworkRun(args=args,env=env,renderer=components.render,
        dt_provider=lambda i:1/15,evaluation=evaluation,collision_mode='swept')
    run.advance(None,model)
    assert len(draws)==2  # one shot draw + one read draw, NOT per shutter sample
    assert preprocess_calls==[False]

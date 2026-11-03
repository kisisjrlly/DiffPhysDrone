import copy

import pytest
import torch

from controllers.auto_exposure import mean_auto_exposure_target
from model import Model
from rollout_ops import init_camera_params, update_camera_params
from tools.audit_slit_information_deadline import (
    HistoricalBatch, MOTIONS, SPLITS, BIASES_M, RELEASE_STEPS, command_for,
    release_estimate, rollout, choose_template, lock_candidates, decision,
    preprocess, observe, template_bank, set_candidate_centers, select_env,
)


@pytest.fixture(scope='module')
def history():
    if not torch.cuda.is_available():
        pytest.skip('Historical initial states and random streams were produced on CUDA')
    torch.set_num_threads(4)
    return HistoricalBatch('cuda')


def test_locked_budget_and_reference_times():
    assert SPLITS == {'development': (1,2,3,12), 'confirmation': (4,5,13,16)}
    assert set(SPLITS['development']).isdisjoint(SPLITS['confirmation'])
    assert RELEASE_STEPS == tuple(range(0,56,5))
    assert BIASES_M == (-.04,-.02,-.01,0.,.01,.02,.04)
    assert all(sum(m['durations']) == pytest.approx(5.6) for m in MOTIONS)


def test_unreleased_hidden_center_does_not_change_commands(history):
    a=history.new_env(); b=copy.deepcopy(a)
    b.slit_center_y = -a.slit_center_y
    b.voxels[:, :2, 1] += (b.slit_center_y-a.slit_center_y)[:,None]
    for step in (0,4):
        ea=release_estimate(a.slit_center_y,step,5,0.)
        eb=release_estimate(b.slit_center_y,step,5,0.)
        torch.testing.assert_close(command_for(a,ea,MOTIONS[0],step,history.args),
            command_for(b,eb,MOTIONS[0],step,history.args),rtol=0,atol=0)


def test_release_changes_only_estimate_not_state_or_pending_queue(history):
    env=history.new_env(); before=env.save_state()
    queue=[env.act.clone(), env.act.clone()+.1]
    old=[v.clone() for v in queue]
    value=release_estimate(env.slit_center_y,5,5,.01)
    command_for(env,value,MOTIONS[0],5,history.args)
    for name,tensor in before.items():
        torch.testing.assert_close(getattr(env,name),tensor,rtol=0,atol=0)
    for a,b in zip(queue,old):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(value,env.slit_center_y+.01)


def test_dynamics_replay_independent_of_render_and_global_rng(history):
    a=history.new_env();b=copy.deepcopy(a)
    command=a.act.clone()+.1
    for step in range(3):
        history.run_step(a,command,step)
        torch.randn(9999,device=a.device)
        history.noise(step,[1,2,3,12])
        history.run_step(b,command,step)
        for name in ('p','v','a','dg','R'):
            torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)


def test_observation_noise_once_original_preprocessing_and_common_draws(history,monkeypatch):
    env=history.new_env();e,g=init_camera_params(env,32,env.device)
    camera=env.gray_camera; original=camera.forward; calls=[]
    def spy(*args,**kwargs):
        calls.append(kwargs)
        return original(*args,**kwargs)
    monkeypatch.setattr(camera,'forward',spy)
    image,noisy=observe(history,env,[1],e,g,0)
    assert len(calls)==1 and calls[0]['enable_noise']
    torch.testing.assert_close(image,preprocess(noisy),rtol=0,atol=0)
    # Additional calls cannot change the indexed shot/read samples.
    a=history.noise(0,[1]);torch.randn(500,device=env.device);b=history.noise(0,[1])
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_templates_do_not_change_original_geometry_or_retain_true_label(history):
    env=history.new_env(); e,g=init_camera_params(env,32,env.device)
    before=env.voxels.clone(); grid=e.new_tensor([-.1,0.,.1])
    a=template_bank(env,[1],e,g,grid,chunk_size=2)
    other=copy.deepcopy(env)
    set_candidate_centers(other,other.slit_center_y+.2)
    b=template_bank(other,[1],e,g,grid,chunk_size=3)
    torch.testing.assert_close(a,b,rtol=0,atol=2e-5)
    torch.testing.assert_close(env.voxels,before,rtol=0,atol=0)


def test_decoder_two_frames_and_tie_keep_previous():
    grid=torch.tensor([-.01,0.,.01]); old=torch.tensor([.009])
    same=torch.zeros(1,3,1,2,2); image=torch.zeros(1,1,2,2)
    assert choose_template(image,same,old,grid).item()==pytest.approx(.01)
    templates=same.clone();templates[:,0]=.1;templates[:,1]=.2;templates[:,2]=.3
    previous=(torch.full_like(image,.3),templates*10)
    # Current frame alone favors +.01; previous frame strongly favors -.01.
    assert choose_template(torch.full_like(image,.3),templates,old,grid).item()==pytest.approx(.01)
    assert choose_template(torch.full_like(image,.3),templates,old,grid,previous).item()==pytest.approx(-.01)


def test_camera_and_flight_queue_order(history):
    env=history.new_env();env.camera_control_mode='learned'
    env._camera_effective_delay_frames=1
    e,g=init_camera_params(env,32,env.device)
    env._camera_effective_delay_frames=1
    target=e.new_full((32,2),.6)
    e1,g1,_=update_camera_params(target,e,g,env)
    torch.testing.assert_close(e1,e)
    e2,g2,_=update_camera_params(target*.5,e1,g1,env)
    torch.testing.assert_close(e2,target[:,0])
    queue=[env.act.clone(),env.act.clone()]
    new=env.act.clone()+.2
    original=env.act.clone()
    for step in range(3):
        executed=queue.pop(0);queue.append(new)
        torch.testing.assert_close(executed, original if step<2 else new)


def test_rollout_known_center_reference_and_failure_cost(history):
    ids=(*SPLITS['development'],*SPLITS['confirmation'])
    oracle,_=rollout(history,MOTIONS[0],mode='oracle',indices=ids,record=False)
    assert all(e['success'] and e['crossed'] and e['min_clearance_m']>.0011 for e in oracle['episodes'])
    late,_=rollout(history,MOTIONS[0],mode='release',indices=ids,release_step=55,bias=0.,record=False)
    assert any(not e['success'] for e in late['episodes'])
    assert all(e['scored_time_s']==6. for e in late['episodes'] if not e['success'])


def row(motion,camera,count,time=5.):
    return dict(motion=motion,camera=camera,episodes=[dict(success=i<count,
        scored_time_s=time if i<count else 6.,control_cost=1.) for i in range(8)])


def test_selection_uses_development_only_is_bounded_and_locks_strong_ae():
    cams=[dict(id=0,name='fixed',kind='fixed'),dict(id=5,name='mean_ae',kind='ae'),
          dict(id=6,name='gradient_ae',kind='ae')]
    rows=[row(m['name'],c,6 if c['id']==6 else 4) for m in MOTIONS for c in cams]
    lock=lock_candidates(rows,[m['name'] for m in MOTIONS])
    assert lock['status']=='locked' and len(lock['arms'])<=6
    assert lock['ae_family']=='gradient_ae'
    assert lock==lock_candidates(rows,[m['name'] for m in MOTIONS])
    assert lock_candidates(rows,['settle_early'])['status']=='control_confounded'


def test_decision_requires_both_margins_interaction_and_control():
    names=['base_ae','best_ae','base_fixed','joint','cross_fixed','joint_ae','cross_ae']
    counts=[3,4,4,7,4,4,3]
    lock=dict(status='locked',role_ids=dict(zip(names,range(7))),development_counts=dict(zip(names,counts)))
    arms=[row('original',dict(id=0),n) for n in counts]
    assert decision(lock,arms,{'original':True})['category']=='joint_candidate'
    assert decision(lock,arms,{'original':False})['category']=='control_confounded'
    arms[3]=row('original',dict(id=0),5)
    assert decision(lock,arms,{'original':True})['category']=='no_joint_value'

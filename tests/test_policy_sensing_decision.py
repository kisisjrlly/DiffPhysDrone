"""Real-network gate: history, memories and physical queues must be paired."""
import copy
from pathlib import Path
import pytest
import torch
from config import build_parser, parse_scenarios, validate_args
from tools.audit_policy_sensing_decision import (NetworkRun, flight_args, load_model,
    CHECKPOINT, CHECKPOINT_SHA, sha, baseline_gate, parameter_hash, adaptation_tokens)
from tools.audit_slit_information_deadline import HistoricalBatch, observe

@pytest.fixture(scope='module')
def history():
    if not torch.cuda.is_available():pytest.skip('Historical CUDA replay required')
    torch.set_num_threads(4)
    return HistoricalBatch('cuda')

@pytest.fixture(scope='module')
def model():
    if not CHECKPOINT.exists() or not torch.cuda.is_available():pytest.skip('Frozen local checkpoint / CUDA required')
    return load_model(CHECKPOINT)


def test_gate_stops_without_usable_vision_or_at_ceiling():
    assert baseline_gate(8,0)=='ceiling_stop'
    assert baseline_gate(3,0)=='needs_flight_baseline'
    assert baseline_gate(5,4)=='needs_flight_baseline'
    assert baseline_gate(4,2)=='pass'


def test_one_adaptation_budget_and_unchanged_legacy_contract(tmp_path):
    args=build_parser().parse_args(adaptation_tokens(tmp_path))
    args.scenarios=parse_scenarios(args.scenarios);validate_args(args)
    assert (args.num_iters,args.batch_size,args.lr,args.seed)==(600,8,1e-5,42)
    assert args.policy_arch=='legacy' and args.joint_backward=='legacy'
    assert args.train_flight_only and not args.train_camera_only and not args.amp
    assert args.observation_grad_mode=='detached' and args.sensor_grad_mode=='detached'
    assert args.camera_control_mode=='fixed' and args.fixed_camera_gain==0
    assert set(args.scenarios)=={'dark','bright','dark_to_bright','bright_to_dark'}
    assert args.coef_obj_avoidance==0 and args.coef_terminal_goal==0
    assert (args.coef_v,args.coef_collide,args.coef_d_acc,args.coef_d_jerk)==(10.5,6,.05,.05)


def test_strict_checkpoint_load_and_camera_freeze(model):
    assert sha(CHECKPOINT)==CHECKPOINT_SHA
    assert sum(p.numel() for p in model.parameters())==1640714
    saved=parameter_hash(model,True)
    model.freeze_camera_for_flight_only()
    assert all(not p.requires_grad for n,p in model.named_parameters() if model._is_camera_parameter(n))
    assert parameter_hash(model,True)==saved


def assert_runs_equal(a,b):
    for name in ('p','v','a','R','R_old','p_old','act','dg','v_wind'):
        torch.testing.assert_close(getattr(a.env,name),getattr(b.env,name),rtol=0,atol=0)
    for name in ('h','cam_h','previous','e','g','success','collision','effort','minimum'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)
    for x,y in zip(a.queue,b.queue):torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert a.step==b.step
    for x,y in zip(a.env._camera_command_queue,b.env._camera_command_queue):
        torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_noop_restore_preserves_image_memory_and_pending_commands(history,model):
    run=NetworkRun(history,'dark')
    for _ in range(3):run.advance(history,model)
    clone=run.snapshot()
    for _ in range(3):
        rowa=run.advance(history,model);rowb=clone.advance(history,model)
        assert rowa==rowb
        assert_runs_equal(run,clone)


def test_camera_command_changes_only_subsequent_observation(history,model):
    a=NetworkRun(history,'dark');b=a.snapshot()
    ra=a.advance(history,model,camera=(.10,0.));rb=b.advance(history,model,camera=(.85,0.))
    # Issuing a camera command cannot change the current frame or decision.
    torch.testing.assert_close(a.previous,b.previous,rtol=0,atol=0)
    assert ra['submitted_command']==rb['submitted_command']
    assert ra['command_effective_frame']==1
    _,ia=observe(history,a.env,range(32),a.e,a.g,1)
    _,ib=observe(history,b.env,range(32),b.e,b.g,1)
    assert not torch.equal(ia,ib)


def test_motion_offset_obeys_queue_and_clamp(history,model):
    a=NetworkRun(history,'dark');b=a.snapshot()
    ra=a.advance(history,model);rb=b.advance(history,model,offset=(0.,.4,0.))
    assert ra['executed_command']==rb['executed_command']
    for name in ('p','v','R'):
        torch.testing.assert_close(getattr(a.env,name),getattr(b.env,name),rtol=0,atol=0)
    local=(b.queue[-1][:,None] @ __import__('rollout_ops').build_local_frame(b.env)).squeeze(1)
    assert local.abs().max()<=flight_args().max_acc_cmd+1e-5
    assert not torch.equal(a.queue[-1],b.queue[-1])


def test_hidden_center_is_not_a_direct_network_input(history,model,monkeypatch):
    import tools.audit_policy_sensing_decision as audit
    a=NetworkRun(history,'dark');b=a.snapshot()
    b.env.slit_center_y=-a.env.slit_center_y
    # Hold the actual image fixed to isolate a forbidden hidden-label path.
    _,image=observe(history,a.env,range(32),a.e,a.g,0)
    monkeypatch.setattr(audit,'observe',lambda *args:(None,image))
    ra=a.advance(history,model);rb=b.advance(history,model)
    assert ra['submitted_command']==rb['submitted_command']


def test_snapshot_is_independent_and_preserves_camera_transaction(history,model):
    a=NetworkRun(history,'dark')
    a.env._camera_effective_delay_frames=2
    a.advance(history,model,camera=(.60,0.))
    b=a.snapshot()
    torch.testing.assert_close(a.env._camera_requested_state,b.env._camera_requested_state,rtol=0,atol=0)
    assert len(b.env._camera_command_queue)==1
    b.env._camera_command_queue[0].add_(.1)
    assert not torch.equal(a.env._camera_command_queue[0],b.env._camera_command_queue[0])
    b.h.add_(1)
    assert not torch.equal(a.h,b.h)
    b.previous.zero_()
    assert a.previous.abs().sum()>0

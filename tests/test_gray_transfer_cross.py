"""Cross audit: matched factors, old-data identity and conservative geometry bounds."""
import copy
from dataclasses import fields
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from render.box_scene import expanded_box_sweep, swept_box_collision
from tools.audit_gray_transfer_cross import (variants,paired_results,observation_steps,
    restore_pose,verify_cell,PREVIOUS,sphere_segment_certificate)


def test_cross_cells_switch_only_declared_blocks():
    v=variants()
    world={'transition_shift','hard_saturation','corridor','box_rays','surface_light','temporal_exposure'}
    motion={'torch_dynamics','swept_collision'}
    def changed(a,b):return {f.name for f in fields(a) if getattr(a,f.name)!=getattr(b,f.name)}
    assert changed(v['OO'],v['WO'])==changed(v['OD'],v['WD'])==world
    assert changed(v['OO'],v['OD'])==changed(v['WO'],v['WD'])==motion
    for cell in v.values():
        assert cell.zero_jitter and cell.extend_horizon and not cell.long_approach


def test_reused_corner_rejects_wrong_physics_configuration():
    with pytest.raises(RuntimeError,match='configuration'):
        verify_cell(PREVIOUS/'migration/M02',variants()['OD'],True)


def test_paired_outcomes_keep_lost_and_recovered_even_when_counts_match():
    old=[1,0]+[0]*14;swapped=[0,1]+[0]*14
    def result(values):return dict(rows=[dict(success_rate=v,collision_rate=1-v,
        stop_reason='goal' if v else 'collision',scenario='dark') for v in values])
    data=paired_results(dict(OO=result(old),WO=result(swapped),OD=result(old),WD=result(old)))
    assert data['cells']['OO']['successes']==data['cells']['WO']['successes']==1
    assert data['cells']['WO']['lost']==[30000]
    assert data['cells']['WO']['recovered']==[30001]


def test_observation_steps_use_first_recorded_crossing_and_report_missing():
    trace={'pre_p':np.array([[[-1.5,0,1.5]],[[-.9,0,1.5]],[[-.7,0,1.5]]])}
    assert observation_steps(trace)==[('initial',0),('at_minus_1m',1),('at_minus_half_m',None)]


def test_pose_restore_uses_previous_attitude_and_period_without_future_access():
    env=SimpleNamespace(device=torch.device('cpu'))
    trace={'pre_'+k:np.arange(6,dtype=np.float32).reshape(2,1,3) for k in ('p','v','a','R','act','dg')}
    trace['dt']=np.array([.07,.08])
    restore_pose(env,trace,1)
    torch.testing.assert_close(env.p,torch.tensor([[3.,4.,5.]]))
    torch.testing.assert_close(env.R_old,torch.tensor([[0.,1.,2.]]))
    assert env.last_control_dt==.07


def test_expanded_box_can_stop_a_safely_clear_spherical_corner_pass():
    boxes=torch.tensor([[[0.,0.,0.,1.,1.,1.]]])
    start=torch.tensor([[1.10,1.10,-.10]]);end=torch.tensor([[1.10,1.10,.10]])
    assert expanded_box_sweep(start,end,boxes,.12).item()
    assert not swept_box_collision(start,end,boxes,.12).item()
    sampled,lower=sphere_segment_certificate(start,end,boxes,.12)
    assert lower.item()>.0011
    assert lower.item()<sampled.item()


def test_clearance_certificate_rejects_a_segment_crossing_the_wall():
    boxes=torch.tensor([[[0.,0.,0.,.05,1.,1.]]])
    start=torch.tensor([[-1.,0.,0.]]);end=torch.tensor([[1.,0.,0.]])
    _,lower=sphere_segment_certificate(start,end,boxes,.12)
    assert lower.item()<0


def test_zero_length_certificate_preserves_positive_gap_with_numerical_guard():
    boxes=torch.tensor([[[0.,0.,0.,1.,1.,1.]]])
    point=torch.tensor([[2.,0.,0.]])
    sampled,lower=sphere_segment_certificate(point,point,boxes,.12)
    assert sampled.item()==pytest.approx(.88)
    assert lower.item()==pytest.approx(.879999)

"""Causal input-size branching and bounded diagnostic decision rules."""
import itertools
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from model import Model
from sensors.imx900_calibration import IMX900Calibration
from sensors.imx900_camera import IMX900DifferentiableCamera
from tools.audit_sensing_scale import (
    DISTANCES, EXPOSURES, SCENARIOS, image_environment, matched_gain, nuisance,
    project, resolution_inputs, screen_decision, sensor_branches,
)
from tools.intent_common import load_intent_args
from train_utils import build_env
from render.intent_scene import render_exposure
from render.pose import rotation_exp


def calibration():
    return IMX900Calibration.from_json('configs/calibration/imx900_provisional.json')


def test_two_sizes_share_exact_noisy_sensor_image_and_original_preprocess():
    image=torch.linspace(0,1,2*72*96).reshape(2,1,72,96)
    branches=resolution_inputs(image)
    torch.testing.assert_close(branches['96x72'],image*2-1,rtol=0,atol=0)
    expected=Model.preprocess_gray_input(SimpleNamespace(gray_nn_width=48,gray_nn_height=36),image)[:,:1]
    torch.testing.assert_close(branches['48x36'],expected,rtol=0,atol=0)
    torch.testing.assert_close(branches['48x36'],F.adaptive_avg_pool2d(image,(36,48))*2-1,rtol=0,atol=0)
    with pytest.raises(ValueError,match='96x72'):
        resolution_inputs(image[:,:,:36,:48])


def test_sensor_noise_applied_once_before_both_branches():
    cam=IMX900DifferentiableCamera(calibration(),saturation_mode='ste')
    watched=Mock(wraps=cam)
    irradiance=torch.full((2,1,72,96),.025)
    e,g=torch.tensor([.1,.6]),torch.full((2,),.02)
    generator=torch.Generator().manual_seed(7)
    shot=torch.randn(irradiance.shape,generator=generator)
    read=torch.randn(irradiance.shape,generator=generator)
    raw,branches,_=sensor_branches(watched,irradiance,e,g,shot,read)
    assert watched.call_count==1
    assert watched.call_args.kwargs['shot_noise'] is shot
    assert watched.call_args.kwargs['read_noise'] is read
    assert watched.call_args.kwargs['motion'] is None
    expected=cam(irradiance,e,g,shot_noise=shot,read_noise=read,enable_noise=True)[0]
    torch.testing.assert_close(raw,expected,rtol=0,atol=0)
    torch.testing.assert_close(branches['48x36'],F.adaptive_avg_pool2d(raw,(36,48))*2-1,rtol=0,atol=0)


def test_paired_nuisance_reused_but_fit_and_check_are_distinct():
    a=nuisance(391000,32,torch.device('cpu'))
    b=nuisance(391000,32,torch.device('cpu'))
    c=nuisance(491000,32,torch.device('cpu'))
    for key in a:
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        assert not torch.equal(a[key],c[key])


def test_static_counterfactual_preserves_pose_and_geometry_and_has_no_motion_blur():
    args=load_intent_args('configs/gray_joint_slit.args',batch_size=1,gray_enable_noise=False)
    reference=build_env(1,args,torch.device('cpu'));reference.reset('dark')
    state=dict(episode=0,center=float(reference.slit_center_y[0]),p=torch.tensor([-1.,.1,1.5]),
               v=torch.tensor([.8,.3,0.]),R=torch.eye(3),
               R_old=rotation_exp(torch.tensor([[0.,0.,-.02]]))[0])
    draws=nuisance(391000,4,torch.device('cpu'))
    moving=image_environment(args,reference,state,'dark',True,1,draws,4)
    static=image_environment(args,reference,state,'dark',False,1,draws,4)
    for key in ('p','R','voxels','gray_light_factor','gray_texture_phase'):
        torch.testing.assert_close(getattr(moving,key),getattr(static,key),rtol=0,atol=0)
    assert static.v.count_nonzero()==0
    torch.testing.assert_close(static.R_old,static.R,rtol=0,atol=0)
    _,aux=render_exposure(static,torch.full((4,),.85),return_aux=True)
    assert aux['temporal_blur_mae'].max()<1e-6
    _,moving_aux=render_exposure(moving,torch.full((4,),.85),return_aux=True)
    assert moving_aux['temporal_blur_mae'].max()>0


def baseline_rows(accuracy=.8):
    return [dict(requested_x=x,scenario=s,resolution=r,motion=m,exposure01=e,accuracy=accuracy)
            for x,s,r,m,e in itertools.product(DISTANCES,SCENARIOS,('48x36','96x72'),
                                              ('stationary','moving'),EXPOSURES)]


def test_decision_stops_on_ceiling_or_no_interaction():
    assert screen_decision(baseline_rows(.99))['category']=='probe_ceiling'
    assert screen_decision(baseline_rows())['category']=='no_tradeoff'
    rows=baseline_rows()
    # Exposure-only effect, same in both motions, is not coordination evidence.
    for r in rows:
        if r['exposure01']==.85:r['accuracy']=.95
    assert screen_decision(rows)['category']=='no_tradeoff'


def test_decision_distinguishes_native_only_from_existing_input_tradeoff():
    rows=baseline_rows()
    def add_candidate(resolution):
        for r in rows:
            if r['requested_x']==-1 and r['scenario']=='dark' and r['resolution']==resolution:
                r['accuracy']={('stationary',.1):.65,('stationary',.6):.85,('stationary',.85):.90,
                               ('moving',.1):.85,('moving',.6):.75,('moving',.85):.65}[r['motion'],r['exposure01']]
    add_candidate('96x72')
    assert screen_decision(rows)['category']=='native_only_tradeoff'
    add_candidate('48x36')
    assert screen_decision(rows)['category']=='current_input_tradeoff'


def test_exposure_units_gain_followup_and_projection_scale():
    c=calibration()
    torch.testing.assert_close(c.exposure_to_us(torch.tensor(EXPOSURES)),torch.tensor([890.,4840.,6815.]))
    match=matched_gain(c,.1,.85)
    assert match['available']
    short=c.exposure_to_us(torch.tensor(.1))*c.gain_to_factor(torch.tensor(match['gain01']))
    long=c.exposure_to_us(torch.tensor(.85))*c.gain_to_factor(torch.tensor(.02))
    torch.testing.assert_close(short,long)
    assert not matched_gain(c,0.,1.)['available']
    point=torch.tensor([[1.,0.,0.]])
    a,_=project(point,torch.zeros(3),torch.eye(3),96,72,.82)
    b,_=project(point,torch.zeros(3),torch.eye(3),48,36,.82)
    torch.testing.assert_close((a+.5)/2,b+.5)


def test_relative_exposure_effect_need_not_reverse_preference():
    rows=baseline_rows(.6)
    for r in rows:
        if r['resolution']=='96x72' and r['exposure01']==.85:
            r['accuracy']=.95 if r['motion']=='stationary' else .75
    assert screen_decision(rows)['category']=='native_only_tradeoff'

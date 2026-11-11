"""Behavior-level checks of the exact shared closed loop used by six arms."""
import copy
from dataclasses import replace
import torch
from bounded_route.core import EpisodeFactory, Reader, Runner, initial_motion
from rollout_ops import render_gray_sensor
from render.exposure_history import record_history_step


def make():
    torch.manual_seed(42)
    f=EpisodeFactory(device='cpu')
    reader=Reader().eval().requires_grad_(False)
    return f,reader,Runner(f,reader)


def test_two_step_flight_queue_and_next_frame_camera_effect():
    f,reader,runner=make()
    with torch.no_grad():
        _,_,rows,_=runner.rollout('timing',[0],initial_motion(),torch.zeros(6),'mean',record=True,steps=4)
    assert abs(rows[0]['exposure_us'][0]-890.0)<1e-3
    assert rows[0]['executed']==[[0.,0.,0.]]
    assert rows[1]['executed']==[[0.,0.,0.]]
    assert rows[2]['executed']==rows[0]['submitted']
    assert rows[3]['executed']==rows[1]['submitted']
    assert abs(rows[1]['exposure_us'][0]-rows[0]['requested_exposure_us'][0])<1e-3


def test_canonical_batch_grouping_and_logging_exact_trajectory():
    f,reader,runner=make()
    with torch.no_grad():
        a=runner.rollout('invariant',[0,1],initial_motion(),torch.zeros(6),record=True,steps=4)
        b=runner.rollout('invariant',[1],initial_motion(),torch.zeros(6),record=True,steps=4)
        c=runner.rollout('invariant',[0,1],initial_motion(),torch.zeros(6),record=False,steps=4)
    assert a[1][1]==b[1][0]
    assert a[1]==c[1]
    for x,y in zip(a[2],b[2]):
        assert x['position'][1]==y['position'][0]
        assert torch.equal(x['frame'][1],y['frame'][0])


def test_external_noise_and_disturbance_do_not_consume_global_rng():
    f,_,_=make();env=f.make('noise',[0])
    e=torch.tensor([.015831663]);g=torch.tensor([0.])
    torch.manual_seed(6543);before=torch.get_rng_state().clone()
    gray,_=render_gray_sensor(env,e,g,shot_noise=torch.zeros(1,1,72,96),read_noise=torch.zeros(1,1,72,96))
    record_history_step(env,1/15)
    env.run(torch.zeros(1,3),disturbance_noise=torch.zeros(1,3))
    assert torch.equal(before,torch.get_rng_state())
    assert torch.isfinite(gray).all()


def test_temporal_render_bypasses_box_blur_and_applies_sensor_once():
    f,_,_=make();a=f.make('camera',[1]);b=f.make('camera',[1])
    a.gray_camera.calibration=replace(a.gray_camera.calibration,blur_scale=1000)
    b.gray_camera.calibration=replace(b.gray_camera.calibration,blur_scale=0)
    e=torch.tensor([.4]);g=torch.tensor([0.])
    noise=torch.zeros(1,1,72,96)
    x,aux=render_gray_sensor(a,e,g,shot_noise=noise,read_noise=noise)
    y,_=render_gray_sensor(b,e,g,shot_noise=noise,read_noise=noise)
    assert torch.equal(x,y)
    assert aux['temporal_samples']==9
    from render.intent_scene import render_exposure
    irradiance,_=render_exposure(a,e,trajectory='history')
    once,_=a.gray_camera(irradiance,e,g,motion=None,shot_noise=noise,read_noise=noise)
    assert torch.equal(x,once)
    assert not aux.get('policy_input_normalized',False)


def test_saturation_physical_derivative_on_unsaturated_sensor_state():
    f,_,_=make();env=f.make('finite',[0]);cam=env.gray_camera
    # Probe the actual sensor with smooth radiance; no visibility discontinuity.
    light=torch.full((1,1,8,8),.05)
    t=torch.tensor([.015831663],requires_grad=True);g=torch.zeros(1)
    im,_=cam(light,t,g,motion=None,enable_noise=False)
    derivative=torch.autograd.grad(im.mean(),t)[0]
    delta=.00001
    plus=cam(light,t.detach()+delta,g,motion=None,enable_noise=False)[0].mean()
    minus=cam(light,t.detach()-delta,g,motion=None,enable_noise=False)[0].mean()
    torch.testing.assert_close(derivative,(plus-minus).reshape(1)/(2*delta),rtol=.002,atol=.001)
    saturated=cam(light,torch.ones(1,requires_grad=True),g,motion=None,enable_noise=False)[0]
    assert saturated.min()==1


def test_reader_weights_remain_frozen_after_closed_loop_camera_backward():
    f,reader,runner=make()
    original={k:v.clone() for k,v in reader.state_dict().items()}
    c=torch.zeros(6,requires_grad=True)
    loss,*_=runner.rollout('backprop',[0],initial_motion(),c,'learned',steps=6)
    loss.mean().backward()
    assert c.grad is not None and torch.isfinite(c.grad).all()
    assert all(torch.equal(v,original[k]) for k,v in reader.state_dict().items())
    assert all(p.grad is None for p in reader.parameters())

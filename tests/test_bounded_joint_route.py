"""Causal/timing/objective regressions for the standalone bounded experiment."""
import inspect
import torch
from bounded_route.core import (EpisodeFactory, FlightObservation, Runner, Reader, initial_motion,
    camera_rule, physical_motion, preprocess, objective, randn_for, flow_speed)


def test_policy_inputs_exclude_scene_truth():
    names=set(FlightObservation.__dataclass_fields__)
    assert not names & {'slit_center','depth','scenario','episode','light','voxels','wind'}
    assert 'env' not in inspect.signature(camera_rule).parameters


def test_random_samples_are_batch_and_record_independent():
    a=randn_for('test',[3,7],2,'shot',(1,3,4),'cpu')
    b=randn_for('test',[7],2,'shot',(1,3,4),'cpu')
    assert torch.equal(a[1],b[0])
    assert not torch.equal(a,randn_for('test',[3,7],2,'read',(1,3,4),'cpu'))


def test_shared_initial_parameters_and_zero_camera_residual():
    m=physical_motion(initial_motion())
    torch.testing.assert_close(m,torch.tensor([1.,.5,.5,.5,.6,1.]))
    gray=torch.full((2,1,72,96),.1)
    t,g=torch.full((2,),890.),torch.ones(2)
    p,v,w=torch.zeros(2,3),torch.zeros(2,3),torch.zeros(2,3)
    a=camera_rule('mean',gray,None,t,g,p,v,w,torch.zeros(6))
    b=camera_rule('learned',gray,None,t,g,p,v,w,torch.zeros(6))
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert torch.all(a[0]<=2*t) and torch.all(a[0]>=t*.5)


def test_shared_objective_penalizes_collision_and_unfinished_flight():
    d=torch.tensor([0.,3.,3.]);gap=torch.tensor([.1,.1,-.02])
    j=objective(d,gap,torch.tensor([0.,1.,1.]),torch.zeros(3))
    assert j[0]<j[1]<j[2]


def test_reader_frozen_weights_retain_image_derivative():
    torch.manual_seed(42)
    model=Reader().requires_grad_(False)
    image=torch.rand(2,2,36,48,requires_grad=True)
    output=model(image,torch.eye(3).repeat(2,1,1)).sum()
    output.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert image.grad.abs().sum()>0
    assert all(p.grad is None for p in model.parameters())


def test_preprocess_once_and_no_extra_noise():
    gray=torch.full((1,1,72,96),.25)
    x=preprocess(gray)
    assert x.shape==(1,2,36,48)
    torch.testing.assert_close(x,torch.full_like(x,-.5))
    assert torch.equal(preprocess(gray,zero=True),torch.zeros_like(x))


def test_episode_geometry_and_random_initial_state_batch_invariant():
    f=EpisodeFactory(device='cpu')
    a=f.make('unit',[0,1]);b=f.make('unit',[1])
    assert a.drone_radius==.12
    assert float(a.slit_half_y[0])==torch.tensor(.15).item()
    torch.testing.assert_close(a.slit_center_y[1:],b.slit_center_y,rtol=0,atol=0)
    torch.testing.assert_close(a.dg[1:],b.dg,rtol=0,atol=0)


def test_lucas_kanade_static_and_camera_command_gradient():
    image=torch.linspace(0,1,48).reshape(1,1,1,48).expand(1,1,36,48)
    assert float(flow_speed(image,image))<1e-3
    c=torch.zeros(6,requires_grad=True)
    t,g=camera_rule('learned',image,None,torch.tensor([890.]),torch.tensor([1.]),
        torch.tensor([[-1.,0.,1.5]]),torch.zeros(1,3),torch.zeros(1,3),c)
    t.sum().backward()
    assert torch.isfinite(c.grad).all() and c.grad.abs().sum()>0

import copy
import torch
import pytest
from config import set_global_seed
from tools.intent_common import load_intent_args
from train_utils import build_env
from render.occluded_route import build_occluded_route
from render.intent_scene import render_scene_irradiance
from render.box_scene import swept_box_collision
from motion_prediction import predict_exposure_motion


def make_env():
    args = load_intent_args('configs/gray_occluded_route.args', batch_size=1,
        gray_width=48, gray_height=36, gray_enable_noise=False)
    set_global_seed(110201)
    e = build_env(1, args, torch.device('cpu')); e.reset('dark')
    return e


def test_initial_occlusion_and_lateral_reveal():
    e = make_env()
    for y in (0., .4, -.4):
        images=[]
        e.p=e.p.new_tensor([[-1., y, 1.5]]);e.R=torch.eye(3)[None]
        for side in (-1,1):
            build_occluded_route(e,[side])
            images.append(render_scene_irradiance(e,e.p,e.R@e.R_cam)[0])
        delta=(images[0]-images[1]).abs().max()
        assert delta == 0 if y == 0 else delta > .001


def test_collision_routes_and_no_vertical_or_lateral_bypass():
    e=make_env();build_occluded_route(e,[1])
    def hit(a,b):
        return swept_box_collision(e.p.new_tensor([a]),e.p.new_tensor([b]),e.voxels,e.drone_radius).item()
    assert hit([1.,.975,1.5],[2.2,.975,1.5])
    assert not hit([1.,-.975,1.5],[2.2,-.975,1.5])
    assert hit([-1.,0.,1.5],[1.,0.,1.5])
    assert hit([1.,-.975,1.5],[1.,-2.,1.5])
    assert hit([1.,-.975,1.5],[1.,-.975,3.])


@pytest.mark.parametrize('delay',[0,1,2])
def test_prediction_matches_nominal_queue_execution_without_future_oracle(delay):
    e=make_env(); e.dg.zero_()
    q=[e.act.clone(),e.act.clone()+e.p.new_tensor([[0.,.1,0.]])]
    proposed=e.act.clone()+e.p.new_tensor([[.2,-.1,0.]])
    p=predict_exposure_motion(e,q,proposed,delay,exposure_us=2000.)
    # Actual run with disturbance innovations suppressed only for this parity test.
    original=torch.randn_like
    torch.randn_like=lambda t,*a,**k: torch.zeros_like(t)
    try:
        for k in range(delay+1):
            e.run(q[k] if k<len(q) else proposed,1/15,e.p_target-e.p)
    finally:
        torch.randn_like=original
    torch.testing.assert_close(p['position'],e.p)
    torch.testing.assert_close(p['velocity'],e.v)
    torch.testing.assert_close(p['rotation'],e.R)
    assert p['exposure_end_s']-p['exposure_start_s']==pytest.approx(.002)


def test_prediction_ignores_hidden_blockage_and_unknown_disturbance():
    e=make_env();q=[e.act]*2
    a=predict_exposure_motion(e,q,e.act,2)
    build_occluded_route(e,[-e.route_blocked_side.item()]);e.dg.fill_(900)
    b=predict_exposure_motion(e,q,e.act,2)
    torch.testing.assert_close(a['position'],b['position'],rtol=0,atol=0)


def test_aligned_intent_uses_queue_and_keeps_flight_output_unchanged():
    from intent_model import build_policy
    from rollout_ops import build_state_vector, build_local_frame, compute_target_velocity, policy_forward
    e=make_env();args=load_intent_args('configs/gray_occluded_route.args',batch_size=1)
    model=build_policy(args)
    r=build_local_frame(e)
    state,_,camera,motion=build_state_vector(e,compute_target_velocity(e.p_target-e.p,e),r,
        e.p.new_tensor([.1]),e.p.new_tensor([.02]),False,False)
    image=torch.zeros(1,2,36,48)
    e.camera_queue_aligned_intent=False
    a=policy_forward(model,state,None,e,gray_obs=image,camera_state=camera,camera_motion_state=motion)
    e.camera_queue_aligned_intent=True
    b=policy_forward(model,state,None,e,gray_obs=image,camera_state=camera,camera_motion_state=motion,
        pending_flight_commands=[e.act,e.act])
    torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
    with pytest.raises(ValueError,match='pending flight'):
        policy_forward(model,state,None,e,gray_obs=image,camera_state=camera,camera_motion_state=motion)


def test_occlusion_gradient_cut_preserves_forward_sensor_image():
    from rollout_ops import render_gray_sensor
    e=make_env();e.p=e.p.clone().requires_grad_()
    exposure=e.p.new_tensor([.3],requires_grad=True);gain=e.p.new_tensor([.02],requires_grad=True)
    full,_=render_gray_sensor(e,exposure,gain)
    e.observation_grad_mode='detached'
    cut,_=render_gray_sensor(e,exposure,gain)
    torch.testing.assert_close(full,cut,rtol=0,atol=0)
    gradients=torch.autograd.grad(cut.square().mean(),(e.p,exposure,gain),allow_unused=True)
    assert gradients[0] is None
    assert gradients[1].isfinite().all() and gradients[2].isfinite().all()


def test_invalid_route_and_prediction_inputs_rejected():
    e=make_env()
    with pytest.raises(ValueError):
        build_occluded_route(e,[0])
    with pytest.raises(ValueError):
        predict_exposure_motion(e,[e.act]*2,e.act,exposure_us=-1)

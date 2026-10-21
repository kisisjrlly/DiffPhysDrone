"""Local derivative correctness, separate from policy-learning performance.

Consistent backward does not promise differentiable visibility/termination or
an exact derivative when warp/STE estimators are selected.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from config import set_global_seed
from losses import compute_physics_losses
from rollout_ops import (update_camera_params, render_gray_sensor, build_local_frame,
                         build_state_vector, compute_target_velocity, decode_action_direct)
from tools.intent_common import load_intent_args
from intent_model import build_policy
from train_utils import build_env
from trainer import _rollout, _loss_from_rollout


def test_camera_filter_and_queue_match_finite_difference_without_changing_forward():
    def sequence(command, mode):
        env = SimpleNamespace(camera_control_mode="learned", camera_smoothing_alpha=.7,
                              joint_backward=mode, _camera_effective_delay_frames=1,
                              _camera_command_queue=[])
        e, g = command.new_tensor([.08]), command.new_tensor([.05])
        values = []
        for _ in range(8):
            e, g, _ = update_camera_params(command[None], e, g, env)
            values.append(torch.stack([e,g],-1))
        return torch.stack(values)
    eg = torch.tensor([.4,.2],dtype=torch.double,requires_grad=True)
    expected = sequence(eg,"legacy")
    actual = sequence(eg,"consistent")
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.autograd.gradcheck(lambda v:sequence(v,"consistent"),(eg,))
    legacy_grad = torch.autograd.grad(expected.sum(),eg)[0]
    correct_grad = torch.autograd.grad(actual.sum(),eg)[0]
    assert (correct_grad > legacy_grad).all()


def test_approach_weight_derivative_matches_its_forward_objective():
    def objective(distance, consistent):
        zero = distance.new_zeros(3,1,3)
        pre_distance = distance.new_tensor([[.35],[.4],[.45]])
        pre = torch.stack([pre_distance,torch.zeros_like(distance),torch.zeros_like(distance)],-1)
        values = compute_physics_losses(zero,zero,zero,pre,zero,distance.new_zeros(1),zero[0],
            valid_mask=torch.ones(3,1,dtype=torch.bool),post_vec_chunk=pre,
            post_clearance_chunk=distance,win=1,differentiate_approach_weight=consistent)
        return values["loss_avoid"]+values["loss_collide"]
    d = torch.tensor([[.15],[.2],[.25]],dtype=torch.double,requires_grad=True)
    torch.testing.assert_close(objective(d,True),objective(d,False),rtol=0,atol=0)
    assert torch.autograd.gradcheck(lambda v:objective(v,True),(d,))
    correct = torch.autograd.grad(objective(d,True),d)[0]
    truncated = torch.autograd.grad(objective(d,False),d)[0]
    assert not torch.allclose(correct,truncated)


def test_consistent_backward_preserves_rollout_pixels_actions_loss_and_terminals():
    args = load_intent_args("configs/gray_joint_parallax_lookahead.args",batch_size=2,
                           timesteps=12,loss_v_window=3,gray_width=24,gray_height=18)
    set_global_seed(727)
    template = build_policy(args)
    results = []
    for mode, decay in (("legacy",.3),("consistent",1.)):
        args.joint_backward,args.grad_decay = mode,decay
        set_global_seed(728)
        env = build_env(2,args,torch.device("cpu"));env.reset("nominal")
        ro = _rollout(env,copy.deepcopy(template),args,2,torch.device("cpu"),False,None,False)
        loss,_,positions,_ = _loss_from_rollout(ro,env,args)
        results.append((loss,positions,torch.stack(ro["act_history"]),torch.stack(ro["exposure_history"]),ro["terminal_collision"]))
    for old,new in zip(*results):
        torch.testing.assert_close(old,new,rtol=0,atol=0)


def test_consistent_mode_requires_undecayed_joint_dynamics():
    with pytest.raises(ValueError,match="grad_decay=1"):
        load_intent_args("configs/gray_joint_parallax_lookahead.args",joint_backward="consistent")
    with pytest.raises(ValueError,match="differentiable scene"):
        load_intent_args("configs/gray_joint_parallax_lookahead.args",joint_backward="consistent",
                         grad_decay=1.,scene_layout="single_wall_slit")


def test_camera_to_closed_loop_flight_matches_local_finite_difference_on_plane():
    args = load_intent_args("configs/gray_joint_intent.args",batch_size=1,gray_width=8,gray_height=6,
                           gray_nn_width=8,gray_nn_height=6,joint_backward="consistent",grad_decay=1.,
                           gray_enable_noise=False,gray_saturation_mode="hard",pose_gradient_estimator="analytic")
    set_global_seed(820)
    actor = build_policy(args).double().requires_grad_(False)
    def execute(command):
        set_global_seed(821)
        env = build_env(1,args,torch.device("cpu"));env.reset("nominal")
        for name,value in list(vars(env).items()):
            if isinstance(value,torch.Tensor) and value.is_floating_point():
                setattr(env,name,value.double())
        env.gray_camera.double()
        env.voxels = command.new_tensor([[[2.,0.,1.5,.1,10.,10.]]])
        env.surface_lights = command.new_tensor([[.4]])
        env.gray_light_factor.fill_(1.)
        env.dg.zero_();env.v_wind.zero_()
        env._camera_effective_delay_frames = 1
        env._camera_command_queue = []
        queue,hx,previous = [env.act]*2,None,None
        e,g = command.new_tensor([.08]),command.new_tensor([.05])
        for step in range(14):
            image,_ = render_gray_sensor(env,e,g)
            pair = torch.cat([image,image if previous is None else previous],1)
            target = env.p_target-env.p
            frame = build_local_frame(env)
            state = build_state_vector(env,compute_target_velocity(target,env),frame,e,g,False,False)[0]
            action,_,hx,_ = actor(state,hx,gray_obs=pair)
            queue.append(decode_action_direct(action,frame,env,1,args.max_acc_cmd))
            e,g,_ = update_camera_params(command[None],e,g,env)
            env.run(queue[step],1/args.base_control_freq,target)
            previous = image
        return (env.p*command.new_tensor([[.7,-.3,.1]])).sum()
    command = torch.tensor([.11,.07],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(execute,(command,),eps=1e-5,atol=2e-6,rtol=1e-3)
    gradient = torch.autograd.grad(execute(command),command)[0]
    assert gradient.abs().sum() > 1e-8

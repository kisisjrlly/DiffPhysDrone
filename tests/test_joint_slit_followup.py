"""Mechanism isolation before any matched-arm result is interpreted."""
import copy

import pytest
import torch

from config import set_global_seed
from intent_model import build_policy
from tools.run_joint_slit_followup import ARMS, CrossPolicy, arm_args
from tools.run_joint_slit_followup import evaluate_with_frames
from tools.run_joint_slit_learning import evaluate, rng_state
from train_utils import build_env
from trainer import _rollout, _loss_from_rollout


def small_args(arm):
    args = arm_args(arm)
    args.batch_size = 1
    args.gray_width, args.gray_height = 20, 16
    args.gray_nn_width, args.gray_nn_height = 16, 12
    args.timesteps = 10
    args.loss_v_window = 3
    args.gray_enable_noise = False
    return args


def test_arm_config_only_cuts_the_named_path():
    a, b, c, d = (arm_args(arm) for arm in ARMS)
    assert a.camera_control_mode == 'mean_ae' and a.train_flight_only
    assert all(x.camera_control_mode == 'learned' for x in (b,c,d))
    assert b.sensor_grad_mode == c.sensor_grad_mode == 'full'
    assert d.sensor_grad_mode == 'detached'
    assert b.observation_grad_mode == d.observation_grad_mode == 'full'
    assert c.observation_grad_mode == 'detached'


def test_gradient_cuts_leave_entire_rollout_forward_unchanged_and_camera_path_isolated():
    trajectories = []
    camera_norms = []
    for arm in ARMS[1:]:
        args = small_args(arm)
        set_global_seed(422, True)
        env = build_env(1, args, torch.device('cpu'))
        model = build_policy(args)
        model.initialize_camera_output(.1, .02)
        env.reset('dark')
        rollout = _rollout(env, model, args, 1, torch.device('cpu'), False, None, False)
        loss, _, positions, _ = _loss_from_rollout(rollout, env, args)
        loss.backward()
        trajectories.append((positions.detach(), torch.stack(rollout['cam_history']).detach(), loss.detach()))
        camera_norms.append(sum(float(p.grad.square().sum()) for name,p in model.named_parameters()
                                if model._is_camera_parameter(name) and p.grad is not None))
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for trajectory in trajectories[1:]:
        for left,right in zip(trajectories[0], trajectory):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert camera_norms[0] > 0 and camera_norms[1] > 0
    assert camera_norms[2] == 0


def test_cross_camera_uses_actual_flight_intent_not_its_own_flight_proposal():
    args = small_args(ARMS[1])
    flight, donor = build_policy(args), build_policy(args)
    cross = CrossPolicy(flight, donor)
    state, image = torch.randn(1,10), torch.rand(1,2,16,20)
    observed = []
    def predictor(action):
        observed.append(action.detach().clone())
        return torch.cat([action, action], -1)
    result = cross(state, gray_obs=image, intent_motion_predictor=predictor)
    # Each branch can compute its own proposal internally, but donor receives
    # an explicit overriding intent made from the actual returned action.
    torch.testing.assert_close(observed[1], result[0])
    expected = donor(state, gray_obs=image, intent_override=torch.cat([result[0], result[0]], -1))[1]
    torch.testing.assert_close(result[1], expected)
    assert cross.camera_hx is not None
    cross.reset()
    assert cross.camera_hx is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual sensor/eval parity uses verified CUDA environment')
def test_frame_and_command_logging_preserves_evaluation_and_rng(tmp_path):
    args = small_args(ARMS[1])
    model = build_policy(args).cuda()
    specs = [dict(seed=990000, scenario='dark', side=-1)]
    before = rng_state()
    plain = evaluate(model, args, specs, tmp_path/'plain.json')
    captured = evaluate_with_frames(model, args, specs, tmp_path/'captured.json')
    after = rng_state()
    for key in plain[0]:
        assert plain[0][key] == captured[0][key]
    assert torch.equal(before['cpu'], after['cpu'])
    assert all(torch.equal(a,b) for a,b in zip(before['cuda'], after['cuda']))
    import json
    import numpy as np
    trace = json.loads((tmp_path/'captured_trace_000.json').read_text())
    with np.load(tmp_path/'captured_frames_000.npz') as frames:
        assert len(frames['frames']) == len(trace) == captured[0]['steps']
    assert all('executed_command' in row and 'attitude' in row and 'exposure_interval_s' in row for row in trace)

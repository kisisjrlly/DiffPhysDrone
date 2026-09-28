#!/usr/bin/env python3
"""Check the real navigation-loss path with a frozen, untrained flight policy.

This is a wiring test, not evidence of navigation performance.
"""
import copy
import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from config import build_parser, parse_scenarios, set_global_seed, validate_args
from model import Model
from train_utils import build_env
from trainer import _rollout, _loss_from_rollout


def main():
    args = build_parser().parse_args(shlex.split(Path('configs/gray_gate_fixed.args').read_text(), comments=True))
    args.batch_size = 2
    args.timesteps = 12
    args.loss_v_window = 4
    args.camera_control_mode = 'learned'
    args.coef_cam_smooth = 0.0
    args.scenarios = parse_scenarios(args.scenarios)
    validate_args(args)
    device = torch.device('cuda')
    set_global_seed(71, True)
    template = Model(dim_obs=10, gray_nn_width=args.gray_nn_width, gray_nn_height=args.gray_nn_height).to(device)
    # Start in an unsaturated region of the provisional response.
    with torch.no_grad():
        template.fc_cam.bias.copy_(torch.logit(torch.tensor([0.10, 0.02], device=device)))
    output = {}
    trajectories = []
    for mode in ('full', 'detached'):
        args.sensor_grad_mode = mode
        set_global_seed(72, True)
        env = build_env(2, args, device)
        env.reset()
        model = copy.deepcopy(template)
        model.freeze_flight_for_camera_only()
        rollout = _rollout(env, model, args, 2, device, False, None, False)
        loss, _, positions, _ = _loss_from_rollout(rollout, env, args)
        parameters = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, parameters, allow_unused=True) if loss.requires_grad else []
        norm = sum(float(g.square().sum()) for g in grads if g is not None) ** 0.5
        output[mode] = dict(navigation_loss=float(loss.detach()), camera_gradient_norm=norm)
        trajectories.append(positions.detach())
    torch.testing.assert_close(trajectories[0], trajectories[1], rtol=0, atol=0)
    assert output['full']['camera_gradient_norm'] > 0
    assert output['detached']['camera_gradient_norm'] == 0
    assert output['full']['navigation_loss'] == output['detached']['navigation_loss']
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

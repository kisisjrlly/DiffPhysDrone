#!/usr/bin/env python3
"""Verify observation-mediated action gradients through the real joint rollout.

This checks learning paths, not policy performance or superiority over RL.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from config import set_global_seed
from intent_model import build_policy
from tools.intent_common import load_intent_args
from train_utils import build_env, write_run_provenance
from trainer import _rollout, _loss_from_rollout


def verify(device="cuda"):
    device = torch.device(device)
    args = load_intent_args(batch_size=1, timesteps=24, loss_v_window=4,
                            gray_width=32, gray_height=24, gray_nn_width=24,
                            gray_nn_height=18, coef_cam_smooth=0.0)
    set_global_seed(182)
    template = build_policy(args).to(device)
    with torch.no_grad():
        template.fc.bias.copy_(template.fc.bias.new_tensor([.25, .1, 0]))
    outputs, trajectories = {}, []
    early_gradients = {}
    for name, obs_mode, sensor_mode in (
        ("full", "full", "full"),
        ("motion_observation_cut", "detached", "full"),
        ("camera_boundary_cut", "full", "detached"),
    ):
        args.observation_grad_mode, args.sensor_grad_mode = obs_mode, sensor_mode
        set_global_seed(183)
        env = build_env(1, args, device)
        env.reset("dark_to_bright")
        model = copy.deepcopy(template)
        rollout = _rollout(env, model, args, 1, device, False, None, False)
        loss, _, positions, _ = _loss_from_rollout(rollout, env, args)
        parameters = [model.fc_cam.weight, model.fc_cam.bias]
        early = rollout["act_history"][0]
        grads = torch.autograd.grad(loss, [early, *parameters], allow_unused=True)
        early_gradients[name] = grads[0].detach()
        camera_norm = sum(float(g.square().sum()) for g in grads[1:] if g is not None) ** .5
        outputs[name] = {"navigation_loss": float(loss.detach()), "camera_head_gradient_norm": camera_norm,
                         "first_flight_action_gradient": grads[0].detach().cpu().tolist()}
        trajectories.append(positions.detach())
    for value in trajectories[1:]:
        torch.testing.assert_close(trajectories[0], value, rtol=0, atol=0)
    observation_contribution = (early_gradients["full"] - early_gradients["motion_observation_cut"]).norm()
    assert torch.isfinite(observation_contribution) and observation_contribution > 1e-9
    assert outputs["full"]["camera_head_gradient_norm"] > 0
    assert outputs["camera_boundary_cut"]["camera_head_gradient_norm"] == 0
    outputs["motion_to_observation_gradient_contribution_norm"] = float(observation_contribution)
    outputs["forward_trajectories_identical"] = True
    outputs["template_seed"] = 182
    outputs["rollout_seed"] = 183
    # The same realized rollout/loss is used for each gradient cut.
    args.observation_grad_mode, args.sensor_grad_mode = "full", "full"
    outputs["interpretation"] = "Gradient wiring only; camera-boundary-cut is not a trained RL baseline."
    return outputs, args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default="logs/intent_joint/gradient_check")
    cli = parser.parse_args()
    result, args = verify(cli.device)
    out = Path(cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gradient_check.json").write_text(json.dumps(result, indent=2) + "\n")
    write_run_provenance(out, args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

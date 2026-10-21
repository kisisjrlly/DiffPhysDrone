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


def verify(device="cuda", config="configs/gray_joint_intent.args", checkpoint=None,
           rollout_steps=24, batch_size=1, training_resolution=False):
    device = torch.device(device)
    overrides = dict(batch_size=batch_size, timesteps=rollout_steps, coef_cam_smooth=0.0)
    if not training_resolution:
        overrides.update(loss_v_window=4, gray_width=32, gray_height=24,
                         gray_nn_width=24, gray_nn_height=18)
    args = load_intent_args(config, **overrides)
    set_global_seed(182)
    template = build_policy(args).to(device)
    if checkpoint:
        template.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    else:
        with torch.no_grad():
            template.fc.bias.copy_(template.fc.bias.new_tensor([.25, .1, 0]))
    outputs, trajectories = {}, []
    early_gradients = {}
    flight_gradients = {}
    for name, obs_mode, sensor_mode in (
        ("full", "full", "full"),
        ("motion_observation_cut", "detached", "full"),
        ("camera_boundary_cut", "full", "detached"),
        ("nondifferentiable_sensing", "detached", "detached"),
    ):
        args.observation_grad_mode, args.sensor_grad_mode = obs_mode, sensor_mode
        set_global_seed(183)
        env = build_env(batch_size, args, device)
        env.reset("dark_to_bright")
        model = copy.deepcopy(template)
        rollout = _rollout(env, model, args, batch_size, device, False, None, False)
        loss, _, positions, _ = _loss_from_rollout(rollout, env, args)
        parameters = list(model.named_parameters())
        early = rollout["act_history"][0]
        grads = torch.autograd.grad(loss, [early, *[p for _, p in parameters]], allow_unused=True)
        early_gradients[name] = grads[0].detach()
        groups = {"flight": [], "camera": []}
        for (param_name, param), grad in zip(parameters, grads[1:]):
            group = "camera" if model._is_camera_parameter(param_name) else "flight"
            groups[group].append(torch.zeros_like(param).flatten() if grad is None else grad.flatten())
        vectors = {key: torch.cat(value).detach() for key, value in groups.items()}
        flight_gradients[name] = vectors["flight"]
        camera_norm = float(vectors["camera"].norm())
        outputs[name] = {"navigation_loss": float(loss.detach()), "camera_head_gradient_norm": camera_norm,
                         "flight_parameter_gradient_norm": float(vectors["flight"].norm()),
                         "first_flight_action_gradient": grads[0].detach().cpu().tolist()}
        trajectories.append(positions.detach())
    for value in trajectories[1:]:
        torch.testing.assert_close(trajectories[0], value, rtol=0, atol=0)
    observation_contribution = (early_gradients["full"] - early_gradients["motion_observation_cut"]).norm()
    assert torch.isfinite(observation_contribution) and observation_contribution > 1e-9
    assert outputs["full"]["camera_head_gradient_norm"] > 0
    assert outputs["camera_boundary_cut"]["camera_head_gradient_norm"] == 0
    assert outputs["nondifferentiable_sensing"]["camera_head_gradient_norm"] == 0
    outputs["motion_to_observation_gradient_contribution_norm"] = float(observation_contribution)
    a, b = flight_gradients["full"], flight_gradients["motion_observation_cut"]
    outputs["flight_gradient_comparison"] = {"motion_path_contribution_norm": float((a - b).norm()),
                                             "cosine_full_vs_motion_cut": float(torch.nn.functional.cosine_similarity(a, b, dim=0))}
    outputs["checkpoint"] = str(Path(checkpoint).resolve()) if checkpoint else None
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
    parser.add_argument("--config", default="configs/gray_joint_intent.args")
    parser.add_argument("--out_dir", default="logs/intent_joint/gradient_check")
    parser.add_argument("--checkpoint")
    parser.add_argument("--rollout_steps", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--training_resolution", action="store_true")
    parser.add_argument("--cpu_threads", type=int, default=4)
    cli = parser.parse_args()
    torch.set_num_threads(cli.cpu_threads)
    result, args = verify(cli.device, cli.config, cli.checkpoint, cli.rollout_steps,
                          cli.batch_size, cli.training_resolution)
    out = Path(cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gradient_check.json").write_text(json.dumps(result, indent=2) + "\n")
    write_run_provenance(out, args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

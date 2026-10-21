#!/usr/bin/env python3
"""Frozen-actor linear decoding diagnostic; never trains camera or flight.

A bounded probe motion is independent of the hidden obstacle order. Fit a
training-only linear readout on one subset of memories and test it on another.
The deployed actor and its parameter count are unchanged.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
import torch.nn.functional as F

from config import set_global_seed
from intent_model import build_policy
from rollout_ops import (build_local_frame, build_state_vector, compute_target_velocity,
                         decode_action_direct, render_gray_sensor)
from tools.evaluate_pole_coordination import frozen_view_renderer
from tools.intent_common import load_intent_args
from train_utils import build_env, write_run_provenance


@torch.no_grad()
def collect(args, model, episodes=384, batch=32, steps=24, seed=785000):
    memories = {k: [] for k in ("initial", "moving_view", "frozen_view")}
    labels, amplitudes, max_accelerations = [], [], []
    min_clearance = 100.
    for offset in range(0, episodes, batch):
        set_global_seed(seed + offset)
        env = build_env(batch, args, torch.device("cpu")); env.reset("nominal")
        amplitude = torch.empty(batch).uniform_(.08, .16) * (torch.randint(0, 2, (batch,)) * 2 - 1)
        queue, h, previous = [env.act] * 2, {"moving_view": None, "frozen_view": None}, {}
        frozen = frozen_view_renderer()
        for step in range(steps):
            rng = torch.get_rng_state()
            normal, _ = render_gray_sensor(env, torch.full((batch,), .08), torch.full((batch,), .05))
            after = torch.get_rng_state(); torch.set_rng_state(rng)
            held, _ = frozen(env, torch.full((batch,), .08), torch.full((batch,), .05))
            torch.set_rng_state(after)
            state = build_state_vector(env, compute_target_velocity(env.p_target - env.p, env),
                                       build_local_frame(env), torch.full((batch,), .08),
                                       torch.full((batch,), .05), args.no_odom, False)[0]
            for name, image in (("moving_view", normal), ("frozen_view", held)):
                pair = torch.cat([image, previous.get(name, image)], 1)
                _, _, h[name], _ = model(state, h[name], gray_obs=pair)
                previous[name] = image
            if step == 0:
                memories["initial"].append(h["moving_view"].cpu().clone())
            horizon = 2 / args.base_control_freq + 1 / env.fixed_pitch_ctl_delay
            u = min((step / args.base_control_freq + horizon) / .9, 1.)
            s = 10*u**3 - 15*u**4 + 6*u**5
            ds = (30*u**2 - 60*u**3 + 30*u**4) / .9
            dds = (60*u - 180*u**2 + 120*u**3) / .9**2
            p, v, a = env.p.new_zeros(batch, 3), env.p.new_zeros(batch, 3), env.p.new_zeros(batch, 3)
            p[:, 0], p[:, 1], p[:, 2] = args.simple_start_x, amplitude*s, args.simple_slit_center_z
            v[:, 1], a[:, 1] = amplitude*ds, amplitude*dds
            command = a + 6*(p - env.p - env.v*horizon - .5*env.a*horizon**2) + 4*(v - env.v - env.a*horizon) + env.fixed_drag_linear*env.v
            frame = build_local_frame(env)
            local = (command[:, None] @ frame).squeeze(1).clamp(-args.max_acc_cmd, args.max_acc_cmd)
            max_accelerations.append(float(local.abs().max()))
            queue.append(decode_action_direct(local, frame, env, batch, args.max_acc_cmd))
            env.run(queue[step], 1 / args.base_control_freq, env.p_target - env.p)
            min_clearance = min(min_clearance, float(env.transition_clearance().min()))
        for name in h:
            memories[name].append(h[name].cpu().clone())
        labels.append((env.cyl[:, 0, 1] > 0).float())
        amplitudes.extend(amplitude.tolist())
    return {k: torch.cat(v) for k,v in memories.items()}, torch.cat(labels), {
        "episodes": episodes, "steps_per_episode": steps, "generated_transitions": episodes*steps,
        "sensor_rendered_frames": 2*episodes*steps, "motion_independent_of_hidden_order": True,
        "probe_amplitude_range_m": [.08, .16], "probe_duration_s": steps/args.base_control_freq,
        "maximum_command_component": max(max_accelerations), "min_clearance_m": min_clearance,
        "camera_settings_fixed": [.08,.05], "collection_seed": seed}


def probe(features, labels, split):
    x, y = features[:split], labels[:split]
    mean, std = x.mean(0), x.std(0).clamp_min(.01)
    x, test = (x-mean)/std, (features[split:]-mean)/std
    head = nn.Linear(x.shape[1], 1)
    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
    optim = torch.optim.LBFGS(head.parameters(), max_iter=75, line_search_fn="strong_wolfe")
    def closure():
        optim.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head(x).squeeze(-1), y) + .01*head.weight.square().sum()
        loss.backward(); return loss
    optim.step(closure)
    with torch.no_grad():
        return {"training_accuracy": float(((head(x).squeeze(-1)>0)==y.bool()).float().mean()),
                "holdout_accuracy": float(((head(test).squeeze(-1)>0)==labels[split:].bool()).float().mean()),
                "training_only_readout_parameters": sum(p.numel() for p in head.parameters())}


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--out_dir",required=True)
    p.add_argument("--episodes",type=int,default=384);p.add_argument("--batch_size",type=int,default=32)
    cli=p.parse_args()
    if cli.episodes % cli.batch_size or cli.episodes < 128:
        raise ValueError("use at least 128 episodes, divisible by batch size")
    torch.set_num_threads(4)
    args=load_intent_args("configs/gray_joint_parallax.args")
    model=build_policy(args).eval();model.load_state_dict(torch.load(cli.checkpoint,map_location="cpu"),strict=True)
    before={k:v.clone() for k,v in model.state_dict().items()}
    memories,labels,metadata=collect(args,model,cli.episodes,cli.batch_size)
    split=cli.episodes*2//3
    results={k:probe(v,labels,split) for k,v in memories.items()}
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    out=Path(cli.out_dir);out.mkdir(parents=True,exist_ok=False)
    torch.save({"memories":memories,"labels":labels,"train_test_split":split},out/"latent_probe_dataset.pt")
    report={"purpose":"Frozen actor observability diagnostic; no policy training", "collection":metadata,
            "train_samples":split,"holdout_samples":cli.episodes-split,"results":results,
            "checkpoint_sha256":hashlib.sha256(Path(cli.checkpoint).read_bytes()).hexdigest(),
            "actor_parameters_unchanged":True,"camera_or_flight_action_labels":False,
            "limitation":"Linear decodability is not evidence that the existing action head uses depth or learns a measurement maneuver."}
    (out/"audit.json").write_text(json.dumps(report,indent=2)+'\n');write_run_provenance(out,args)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()

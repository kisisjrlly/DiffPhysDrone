#!/usr/bin/env python3
"""Small held-out visual decoder probe; never trains the navigation policy.

Earlier noiseless templates knew exact pose/texture. This probe instead fits
a regularized linear decoder on one batch of nuisance images and tests a
disjoint batch. Same nuisance/noise draws across exposure and motion settings.
The result is decoder-specific cue accessibility, not a navigation result.
"""

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from config import set_global_seed
from render.intent_scene import render_exposure
from render.pose import rotation_exp
from tools.audit_joint_slit import CONFIG, EXPOSURES, gain_for_exposure, policy_resolution
from tools.intent_common import load_intent_args
from train_utils import build_env, write_run_provenance


def ridge_probe(train_x, train_y, test_x, test_y):
    """Fixed ridge strength; train-only normalization, no test model selection."""
    train_x, test_x = train_x.double(), test_x.double()
    mean = train_x.mean(0)
    scale = train_x.std(0).clamp_min(.005)
    a, b = (train_x - mean) / scale, (test_x - mean) / scale
    kernel = a @ a.T
    alpha = (.1 * kernel.diagonal().mean()).clamp_min(1e-9)
    coeff = torch.linalg.solve(kernel + torch.eye(len(a), device=a.device) * alpha, train_y.double())
    scores = b @ a.T @ coeff
    correct = ((scores > 0) == (test_y > 0)).double()
    correct = torch.where(scores == 0, torch.full_like(correct, .5), correct)
    return correct, scores


@torch.no_grad()
def nuisance_images(args, scenario, distance_x, motion, exposure, gain, center_delta,
                    center_class, seed, device, count, spatial_supersample=1):
    set_global_seed(seed)
    render_args = copy.deepcopy(args)
    render_args.gray_width *= spatial_supersample
    render_args.gray_height *= spatial_supersample
    env = build_env(count, render_args, device)
    env.reset(scenario)
    centers = env.slit_center_y.new_full((count,), .65 + center_class * center_delta / 2)
    delta = centers - env.slit_center_y
    env.voxels[:, :2, 1] += delta[:, None]
    env.slit_center_y = centers
    env.p = env.p.new_tensor([distance_x, .55, 1.5])[None].expand(count, -1)
    velocity, yaw_rate = motion
    env.v = env.p.new_tensor(velocity)[None].expand(count, -1)
    # Unknown residual yaw +/-0.25deg and fixed-per-world material phase.
    angles = torch.zeros(count, 3, device=device)
    angles[:, 2] = (torch.rand(count, device=device) * 2 - 1) * math.radians(.25)
    env.R = rotation_exp(angles)
    env.R_old = env.R @ rotation_exp(env.p.new_tensor([[0., 0., -yaw_rate * env.last_control_dt]]).expand(count, -1))
    env.gray_texture_phase = torch.rand(count, device=device) * (2 * math.pi)
    env.gray_light_factor = .92 + .16 * torch.rand(count, device=device)
    e, g = env.p.new_full((count,), exposure), env.p.new_full((count,), gain)
    ideal = render_exposure(env, e)[0]
    # Area integration BEFORE photon/read noise, keeping the sensor's original
    # dimensions/noise law fixed. This is anti-aliasing, not lens calibration.
    ideal = torch.nn.functional.adaptive_avg_pool2d(ideal, (args.gray_height, args.gray_width))
    # Explicit noise seed keeps draws paired despite different renderer work.
    set_global_seed(seed + 10000)
    noisy = env.gray_camera(ideal, e, g, motion=None, enable_noise=True)[0]
    return policy_resolution(noisy, args).flatten(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train_per_class", type=int, default=64)
    parser.add_argument("--test_per_class", type=int, default=64)
    parser.add_argument("--spatial_supersample", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--focused", action="store_true",
                        help="Only x=-1m, 5mm cue, stationary vs observed-rate 1.5rad/s motion")
    cli = parser.parse_args()
    if min(cli.train_per_class, cli.test_per_class) < 32:
        raise ValueError("at least 32 independent train and test samples per class")
    out = Path(cli.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(cli.device)
    args = load_intent_args(CONFIG)
    template_env = build_env(1, args, device)
    motions = {
        "stationary": ([0., 0., 0.], 0.),
        "translation_yaw_1p5": ([.8, .6, 0.], 1.5),
        "translation_yaw_3_stress": ([.8, .6, 0.], 3.),
    }
    if cli.focused:
        motions.pop("translation_yaw_3_stress")
    rows, paired = [], []
    for scenario in ("dark", "bright_to_dark"):
        for x in ((-1.,) if cli.focused else (-1., -.4)):
            for delta in ((.005,) if cli.focused else (.005, .02)):
                for gain_mode in ("fixed_gain", "matched_signal"):
                    condition_correct = {}
                    for name, motion in motions.items():
                        for exposure in EXPOSURES:
                            gain = gain_for_exposure(template_env, exposure, gain_mode)
                            data = {}
                            for split, count, split_seed in (("train", cli.train_per_class, 391000),
                                                              ("test", cli.test_per_class, 491000)):
                                images, labels = [], []
                                for side in (-1, 1):
                                    images.append(nuisance_images(args, scenario, x, motion, exposure, gain,
                                                                 delta, side, split_seed + (side + 1) * 100,
                                                                 device, count, cli.spatial_supersample))
                                    labels.append(torch.full((count,), side, device=device))
                                data[split] = (torch.cat(images), torch.cat(labels))
                            correct, scores = ridge_probe(*data["train"], *data["test"])
                            condition_correct[(name, exposure)] = correct.cpu().numpy()
                            row = dict(scenario=scenario, x=x, center_difference_m=delta, gain_mode=gain_mode,
                                       motion=name, exposure01=exposure, gain01=gain,
                                       test_accuracy=float(correct.mean()), test_examples=len(correct))
                            rows.append(row)
                            print(json.dumps(row), flush=True)
                    # Paired descriptive intervals on independent nuisance test
                    # examples. Not cross-training uncertainty or navigation.
                    rng = np.random.default_rng(391029)
                    for name in motions:
                        long = condition_correct[(name, .85)]
                        for exposure in EXPOSURES[:-1]:
                            difference = condition_correct[(name, exposure)] - long
                            draws = difference[rng.integers(0, len(difference), (2000, len(difference)))].mean(1)
                            paired.append(dict(scenario=scenario, x=x, center_difference_m=delta,
                                               gain_mode=gain_mode, motion=name, short_exposure01=exposure,
                                               short_minus_long_accuracy=float(difference.mean()),
                                               paired_nuisance_example_bootstrap_95=np.quantile(draws, [.025, .975]).tolist()))
                    result = {"claim": "Held-out linear cue decoder only; NOT flight training or policy superiority.",
                              "calibrated": False, "train_per_class": cli.train_per_class,
                              "test_per_class": cli.test_per_class,
                              "spatial_supersample": cli.spatial_supersample,
                              "focused": cli.focused,
                              "sensor_resolution": [args.gray_height, args.gray_width],
                              "spatial_filter": "subpixel box-area integration before sensor noise",
                              "train_seed": 391000, "test_seed": 491000,
                              "nuisance": {"yaw_uniform_deg": [-.25, .25], "texture_phase_uniform_rad": [0, 2 * math.pi],
                                           "light_factor_uniform": [.92, 1.08], "frame_noise": True},
                              "ridge_rule": "train-only feature standard deviation with .005 floor; alpha=.1*mean(diag(X X^T)); fixed before evaluating test examples",
                              "rows": rows, "paired_short_minus_long": paired,
                              "limitation": "A weak decoder may miss usable cues; examples reuse common physical nuisance/noise seeds across conditions. No learning-rate or test-accuracy tuning."}
                    (out / "precision_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    write_run_provenance(out, args)
    (out / "result_sha256.json").write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in (out / "precision_probe.json", out / "provenance.json")}, indent=2) + "\n")


if __name__ == "__main__":
    main()

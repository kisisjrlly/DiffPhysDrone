#!/usr/bin/env python3
"""Compare pose Jacobians to finite pixel changes, before navigation training.

Hard rasterization finite differences depend on step size. This is a local
diagnostic, not proof of an exact derivative or learned active sensing.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F

from config import set_global_seed
from render.intent_scene import render_with_pose_gradient
from tools.intent_common import load_intent_args
from train_utils import build_env, write_run_provenance


def audit(config, batch, device):
    args = load_intent_args(config, batch_size=batch)
    set_global_seed(4921)
    env = build_env(batch, args, torch.device(device))
    env.reset("nominal")
    position = env.p.detach().clone()
    position[:, 1].uniform_(-.12, .12)
    position.requires_grad_()
    rotation = env.R_scene_T @ env.R @ env.R_cam
    # Smooth random linear image probes include both interior and edge changes.
    weights = torch.randn(batch, 1, 6, 8, device=device)
    weights = F.interpolate(weights, (env.height, env.width), mode="bilinear", align_corners=False)
    derivatives = {}
    images = []
    for mode in ("analytic", "warp_ste"):
        env.pose_gradient_estimator = mode
        image, _ = render_with_pose_gradient(env, position, rotation)
        images.append(image.detach())
        values = (image * weights).mean((1, 2, 3))
        derivatives[mode] = torch.autograd.grad(values.sum(), position)[0][:, 1].detach()
    torch.testing.assert_close(images[0], images[1], rtol=0, atol=0)
    report = {"same_forward_pixels": True, "batch": batch, "seed": 4921,
              "interpretation": "Local image sensitivity only; surrogate is not the exact hard-raster derivative.",
              "finite_difference": {}}
    with torch.no_grad():
        for step in (.005, .01, .02):
            offset = position.new_tensor([0., step, 0.])
            plus = render_with_pose_gradient(env, position + offset, rotation)[0]
            minus = render_with_pose_gradient(env, position - offset, rotation)[0]
            fd = ((plus - minus) * weights).mean((1, 2, 3)) / (2 * step)
            report["finite_difference"][str(step)] = {
                mode: {"rmse": float((grad - fd).square().mean().sqrt()),
                       "cosine": float(F.cosine_similarity(grad, fd, dim=0)),
                       "same_sign_fraction": float(((grad * fd) > 0).float().mean()),
                       "gradient_rms": float(grad.square().mean().sqrt()),
                       "finite_difference_rms": float(fd.square().mean().sqrt())}
                for mode, grad in derivatives.items()}
    return report, args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gray_joint_parallax.args")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    cli = parser.parse_args()
    torch.set_num_threads(4)
    result, args = audit(cli.config, cli.batch, cli.device)
    out = Path(cli.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    write_run_provenance(out, args)
    (out / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

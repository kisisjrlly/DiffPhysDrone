#!/usr/bin/env python3
"""Privileged tracking sensitivity to slit width; never a learned baseline."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from config import set_global_seed
from tools.audit_intent_scene import executed_dynamics_audit
from tools.intent_common import load_intent_args
from train_utils import build_env, write_run_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=64)
    cli = parser.parse_args()
    if cli.episodes < 1:
        raise ValueError("positive episode count required")
    out = Path(cli.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device, rows = torch.device(cli.device), []
    for half in (.15, .18, .20, .24):
        args = load_intent_args("configs/gray_joint_slit.args", batch_size=cli.episodes,
                               simple_slit_half_y=half)
        set_global_seed(291101)
        env = build_env(cli.episodes, args, device)
        env.reset("bright_to_dark")
        centers, z = env.slit_center_y, env.slit_center_y * 0 + 1.5
        points = torch.stack([
            torch.stack([z * 0 + args.simple_start_x, z * 0, z], -1),
            torch.stack([z * 0 - .5, centers, z], -1),
            torch.stack([z * 0 + .5, centers, z], -1),
            torch.stack([z * 0 + args.simple_goal_x, z * 0, z], -1),
        ], 1)
        for compensated in (False, True):
            set_global_seed(291102)
            report = executed_dynamics_audit(env, args, local_points=points,
                                            durations=[2.6, 1.4, 1.6],
                                            compensate_current_disturbance=compensated)
            report.pop("local_trajectory")
            report.update(half_width_m=half, slit_centers_y=centers.cpu().tolist(),
                          reset_seed=291101, dynamics_seed=291102)
            rows.append(report)
            print(half, compensated, sum(report["success"]), "/", cli.episodes, flush=True)
    (out / "tracking_width_sweep.json").write_text(json.dumps({
        "purpose": "Privileged diagnostic tracker sensitivity; NOT trained navigation or an actor teacher",
        "calibrated": False, "rows": rows,
        "limitations": "One disturbance/layout batch, fixed diagnostic route/controller. Wider slits are separate diagnostic geometries, not a replacement for the original benchmark.",
    }, indent=2) + "\n")
    write_run_provenance(out, args)


if __name__ == "__main__":
    main()

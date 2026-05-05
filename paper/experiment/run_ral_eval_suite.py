#!/usr/bin/env python3
"""Minimal active-sensing eval sweep.

Runs checkpoints over glare/specular/dark x far_left/left/right/far_right and
writes episode_metrics.csv plus summary_metrics.csv.
"""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from config import build_parser, parse_diff_sensor_impl, parse_scenarios, canonicalize_sun_glare_slot, validate_args, set_global_seed
from eval import run_one_episode
from model import Model
from train_utils import build_env


METHODS = {
    'ours': ('Ours', {'camera_control_mode': 'learned', 'sensor_grad_mode': 'full', 'policy_depth_mode': 'depth'}),
    'fixed': ('Fixed Camera', {'camera_control_mode': 'fixed', 'sensor_grad_mode': 'full', 'policy_depth_mode': 'depth'}),
    'fixed_random': ('Random Static Camera', {'camera_control_mode': 'fixed_random_static', 'sensor_grad_mode': 'full', 'policy_depth_mode': 'depth'}),
    'nondiff': ('Non-Diff Active', {'camera_control_mode': 'learned', 'sensor_grad_mode': 'detached', 'policy_depth_mode': 'depth'}),
    'ours_zero': ('Ours Zero Depth', {'camera_control_mode': 'learned', 'sensor_grad_mode': 'full', 'policy_depth_mode': 'zero'}),
}


def _read_args_tokens(config_path: Path) -> list[str]:
    tokens: list[str] = []
    with config_path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.split('#', 1)[0].strip()
            if line:
                tokens.extend(line.split())
    return tokens


def _load_args(config_path: Path):
    parser = build_parser()
    args = parser.parse_args(_read_args_tokens(config_path))
    args.diff_sensor_impl = parse_diff_sensor_impl(args.diff_sensor_impl)
    args.scenarios = parse_scenarios(args.scenarios)
    args.sun_glare_eval_slot = canonicalize_sun_glare_slot(args.sun_glare_eval_slot)
    validate_args(args)
    return args


def _build_model(args, device):
    obs_dim = 7 if args.no_odom else 10
    return Model(
        obs_dim,
        4,
        include_camera_state_in_obs=args.include_camera_state_in_obs,
        use_policy_intent=False,
        depth_nn_width=args.depth_nn_width,
        depth_nn_height=args.depth_nn_height,
        depth_use_pipeline=args.depth_use_pipeline,
        depth_min_valid=args.depth_min_valid,
        depth_max_range=args.depth_max_range,
    ).to(device)


def _load_ckpt(model, ckpt: Path, device):
    state = torch.load(str(ckpt), map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'[suite][warn] {ckpt.name} missing keys: {missing}')
    if unexpected:
        print(f'[suite][warn] {ckpt.name} unexpected keys: {unexpected}')
    model.eval()


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = row['method_key']
        groups.setdefault(key, []).append(row)
    out = []
    metrics = ['success_rate', 'collision_rate', 'goal_reach_rate', 'final_goal_dist', 'avg_speed', 'fill_rate']
    for key, vals in groups.items():
        item = {'method_key': key, 'method_label': vals[0]['method_label'], 'n': len(vals)}
        for metric in metrics:
            item[metric] = sum(float(v[metric]) for v in vals) / max(len(vals), 1)
        out.append(item)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=Path('configs/slit_active_sensing.args'))
    ap.add_argument('--ours_ckpt', type=Path, required=True)
    ap.add_argument('--fixed_ckpt', type=Path, required=True)
    ap.add_argument('--fixed_random_ckpt', type=Path, default=None)
    ap.add_argument('--nondiff_ckpt', type=Path, default=None)
    ap.add_argument('--include_ours_zero_ablation', action='store_true')
    ap.add_argument('--episodes_per_condition', type=int, default=20)
    ap.add_argument('--scenarios', nargs='*', default=['glare', 'specular', 'dark'])
    ap.add_argument('--slots', nargs='*', default=['far_left', 'left', 'right', 'far_right'])
    ap.add_argument('--output_dir', type=Path, default=Path('paper/experiment/results/active_sensing_eval'))
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def main():
    cli = parse_args()
    base_args = _load_args(cli.config)
    set_global_seed(cli.seed, base_args.deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpts = {'ours': cli.ours_ckpt, 'fixed': cli.fixed_ckpt}
    if cli.fixed_random_ckpt is not None:
        ckpts['fixed_random'] = cli.fixed_random_ckpt
    if cli.nondiff_ckpt is not None:
        ckpts['nondiff'] = cli.nondiff_ckpt
    if cli.include_ours_zero_ablation:
        ckpts['ours_zero'] = cli.ours_ckpt
    for key, ckpt in ckpts.items():
        if not ckpt.is_file():
            raise FileNotFoundError(f'{key} checkpoint not found: {ckpt}')

    rows = []
    for method_key, ckpt in ckpts.items():
        label, overrides = METHODS[method_key]
        args = copy.deepcopy(base_args)
        for k, v in overrides.items():
            setattr(args, k, v)
        args.batch_size = 1
        args.eval_episodes = int(cli.episodes_per_condition)
        args.vis_enable = False
        args.vis_episode_idx = -2
        model = _build_model(args, device)
        _load_ckpt(model, ckpt, device)
        env = build_env(args.batch_size, args, device, eval_mode=True)
        dummy_vis = type('DummyVis', (), {'enabled': False})()
        with torch.no_grad():
            for scene in cli.scenarios:
                for slot in cli.slots:
                    args.scenarios = [scene]
                    args.sun_glare_eval_slot = canonicalize_sun_glare_slot(slot)
                    env.sun_glare_eval_slot = args.sun_glare_eval_slot
                    for ep in range(cli.episodes_per_condition):
                        row, _ = run_one_episode(ep, scene, args, model, env, dummy_vis, device, collect_trace=False)
                        row.update({
                            'method_key': method_key,
                            'method_label': label,
                            'condition': f'{scene}_{slot}',
                            'checkpoint': str(ckpt),
                        })
                        rows.append(row)
    _write_csv(cli.output_dir / 'episode_metrics.csv', rows)
    _write_csv(cli.output_dir / 'summary_metrics.csv', _summaries(rows))
    print(f'[suite] wrote {cli.output_dir}')


if __name__ == '__main__':
    main()

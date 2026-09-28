#!/usr/bin/env python3
"""Paired grayscale/blind evaluation; no simulator truth enters the policy."""
import argparse
import copy
import csv
import hashlib
import json
import random
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from config import build_parser, parse_scenarios, set_global_seed, validate_args
from eval import run_one_episode
from model import Model
from train_utils import build_env


def paired_interval(differences, seed=0):
    """Episode bootstrap CI, conditional on these trained checkpoints."""
    rng = random.Random(seed)
    n = len(differences)
    samples = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(2000))
    return [samples[49], samples[1949]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/gray_gate_fixed.args')
    parser.add_argument('--gray_checkpoint', type=Path, required=True)
    parser.add_argument('--blind_checkpoint', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--out_dir', type=Path, required=True)
    cli = parser.parse_args()
    if cli.episodes < 2:
        parser.error('at least two paired episodes are required')
    cli.out_dir.mkdir(parents=True, exist_ok=False)
    tokens = shlex.split(cli.config.read_text(), comments=True)
    args = build_parser().parse_args(tokens + ['--batch_size', '1', '--deterministic', '--no-amp', '--wandb_disabled'])
    args.seed = cli.seed
    args.scenarios = parse_scenarios(args.scenarios)
    args.camera_control_mode = 'fixed'
    args.sensor_grad_mode = 'detached'
    args.vis_enable = False
    args.vis_episode_idx = -1
    validate_args(args)
    if args.include_camera_state_in_obs:
        raise ValueError('vision gate forbids direct camera state in flight input')
    device = torch.device('cuda')
    methods = [('gray', cli.gray_checkpoint, 'gray'),
               ('gray_zeroed', cli.gray_checkpoint, 'zero'),
               ('blind', cli.blind_checkpoint, 'zero')]
    manifest = {'args': vars(args), 'config_text': cli.config.read_text(),
                'calibration': json.loads(Path(args.imx900_calibration).read_text()),
                'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_diff': subprocess.check_output(['git', 'diff'], cwd=ROOT, text=True),
                'torch': torch.__version__, 'episodes': cli.episodes, 'checkpoints': {}}
    manifest['source_sha256'] = {str(path.relative_to(ROOT)):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT / 'eval.py', Path(__file__), ROOT / 'env_cuda.py',
                     ROOT / 'model.py', ROOT / 'rollout_ops.py']}
    rows = []
    for name, checkpoint, vision in methods:
        method_args = copy.deepcopy(args)
        method_args.policy_gray_mode = vision
        set_global_seed(cli.seed, True)
        env = build_env(1, method_args, device, eval_mode=True)
        model = Model(7 if args.no_odom else 10, 3,
                      include_camera_state_in_obs=False,
                      gray_nn_width=args.gray_nn_width, gray_nn_height=args.gray_nn_height).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
        model.eval()
        manifest['checkpoints'][name] = {'path': str(checkpoint.resolve()),
            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        snapshot = checkpoint.parent / 'imx900_calibration.json'
        if not snapshot.is_file() or json.loads(snapshot.read_text()) != manifest['calibration']:
            raise ValueError('checkpoint calibration snapshot missing or differs from evaluation profile')
        training_args = checkpoint.parent / 'args.json'
        manifest['checkpoints'][name]['training_args'] = (
            json.loads(training_args.read_text()) if training_args.exists() else None)
        with torch.no_grad():
            for episode in range(cli.episodes):
                scene = args.scenarios[episode % len(args.scenarios)]
                row, _ = run_one_episode(episode, scene, method_args, model, env, None, device)
                if episode == 0:
                    # Extra unrelated draws must not shift a paired episode.
                    torch.rand(137, device=device)
                    random.random()
                    repeated, _ = run_one_episode(episode, scene, method_args, model, env, None, device)
                    if row != repeated:
                        raise RuntimeError('episode reproducibility check failed')
                rows.append(dict(method=name, **row))
        print(name, 'success', sum(r['success_rate'] for r in rows if r['method'] == name), '/', cli.episodes, flush=True)
    with (cli.out_dir / 'episodes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = {name: [r for r in rows if r['method'] == name] for name, _, _ in methods}
    summary = {'scope': 'provisional simulator; conditional on checkpoints; not across training seeds',
               'methods': {}, 'paired_success_difference': {}}
    for name, group in groups.items():
        summary['methods'][name] = {k: sum(r[k] for r in group) / len(group)
            for k in ('success_rate', 'collision_rate', 'final_goal_dist')}
    for control in ('blind', 'gray_zeroed'):
        differences = [a['success_rate'] - b['success_rate'] for a, b in zip(groups['gray'], groups[control])]
        summary['paired_success_difference'][control] = {
            'mean': sum(differences) / len(differences), 'bootstrap_95_ci': paired_interval(differences)}
    # This is a necessary screen, not a publication-level multi-training-seed claim.
    summary['vision_screen_passed'] = all(v['bootstrap_95_ci'][0] > 0
        for v in summary['paired_success_difference'].values())
    (cli.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (cli.out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

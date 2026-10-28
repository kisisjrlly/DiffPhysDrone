#!/usr/bin/env python3
"""Evaluate saved learning curves and paired holdouts without training actors."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from config import build_parser, parse_scenarios, validate_args
from eval import run_one_episode
from intent_model import build_policy
from train_utils import build_env
from tools.evaluate_pole_coordination import frozen_view_renderer
from tools.run_intent_pilot import bootstrap_difference


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_args(directory):
    args = build_parser().parse_args([])
    vars(args).update(json.loads((directory / 'args.json').read_text()))
    args.batch_size, args.vis_enable, args.vis_episode_idx = 1, False, -1
    args.scenarios = parse_scenarios(args.scenarios)
    args.imx900_calibration = str(directory / 'imx900_calibration.json')
    validate_args(args)
    return args


def evaluate(args, checkpoint, seed, n, condition, device):
    args = copy.deepcopy(args)
    model = build_policy(args).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    if condition in ('fixed_nominal', 'zero_image'):
        args.camera_control_mode = 'fixed'
        args.fixed_camera_exposure, args.fixed_camera_gain = .08, .05
    if condition == 'zero_image':
        args.policy_gray_mode = 'zero'
    env = build_env(1, args, device, eval_mode=True)
    rows, spans = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(n):
            renderer = frozen_view_renderer() if condition == 'frozen_view' else None
            row, trace = run_one_episode(
                i, 'nominal', args, model, env, None, device,
                episode_seed=seed + i, collect_trace=True, sensor_renderer=renderer)
            row['near_pole_sign'] = int(env.cyl[0, 0, 1].sign())
            rows.append(row)
            spans.append({k: max(x[k] for x in trace) - min(x[k] for x in trace)
                          for k in ('exposure', 'gain')})
    result = {
        'checkpoint_sha256': digest(checkpoint), 'checkpoint': str(checkpoint),
        'calibration_sha256': digest(args.imx900_calibration),
        'condition': condition, 'seed_range_inclusive': [seed, seed + n - 1],
        'episodes': rows, 'n': n,
        'successes': int(sum(r['success_rate'] for r in rows)),
        'collisions': int(sum(r['collision_rate'] for r in rows)),
        'timeouts': sum(r['stop_reason'] == 'timeout' for r in rows),
        'evaluation_wall_time_s': time.perf_counter() - start,
        'mean_goal_distance': sum(r['final_goal_dist'] for r in rows) / n,
        'mean_elapsed_s': sum(r['elapsed_s'] for r in rows) / n,
        'mean_exposure': sum(r['exposure_mean'] for r in rows) / n,
        'mean_gain': sum(r['gain_mean'] for r in rows) / n,
        'mean_camera_range': {k: sum(v[k] for v in spans) / n for k in spans[0]},
        'by_near_pole_sign': {
            str(s): {
                'n': sum(r['near_pole_sign'] == s for r in rows),
                'successes': int(sum(r['success_rate'] for r in rows
                                     if r['near_pole_sign'] == s)),
                'collisions': int(sum(r['collision_rate'] for r in rows
                                      if r['near_pole_sign'] == s))}
            for s in (-1, 1)}}
    del model, env
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_dir', required=True)
    parser.add_argument('--horizons', nargs='+', type=int, default=[4, 8, 16, 120])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--interventions', action='store_true',
                        help='Also evaluate fixed camera/zero image/frozen view for the last horizon.')
    cli = parser.parse_args()
    study = Path(cli.study_dir).resolve()
    torch.set_num_threads(4)
    device = torch.device(cli.device)
    directory = study / 'evaluation'
    directory.mkdir(exist_ok=True)
    report_path = directory / 'results.json'
    script_hash = digest(__file__)
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        'purpose': 'conditional single-training-seed method comparison',
        'calibrated': False, 'device': str(device),
        'evaluation_script_sha256': script_hash, 'methods': {}, 'baselines': {}}
    if report['device'] != str(device):
        raise ValueError('cannot mix evaluation devices in one report')
    if report.get('evaluation_script_sha256', script_hash) != script_hash:
        raise ValueError('evaluation script changed; use a separate output directory')

    def save():
        temporary = report_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(report_path)

    def collect(entry, label, args, checkpoint, seed, n, condition='learned'):
        if label in entry:
            if entry[label]['checkpoint_sha256'] != digest(checkpoint):
                raise ValueError('cached evaluation checkpoint changed')
            return
        entry[label] = evaluate(args, checkpoint, seed, n, condition, device)
        save()
        print(label, entry[label]['successes'], '/', n, flush=True)

    base_args = load_args(study / f'h{cli.horizons[0]}' / 'joint')
    checkpoint = Path(base_args.resume)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    for label, n, seed in [('common_validation', 40, 800000),
                           ('common_holdout', 100, 801000)]:
        collect(report['baselines'], label, base_args, checkpoint, seed, n)

    for h in cli.horizons:
        for method in ('joint', 'camera_gradient_cut'):
            run = study / f'h{h}' / method
            args = load_args(run)
            key = f'h{h}/{method}'
            entry = report['methods'].setdefault(key, {})
            for name, n, seed, label in [
                    ('checkpoint0000.pth', 40, 800000, 'validation_50'),
                    ('checkpoint0001.pth', 40, 800000, 'validation_100'),
                    ('final.pth', 100, 801000, 'holdout_150')]:
                print(key, flush=True)
                collect(entry, label, args, run / name, seed, n)
        full = report['methods'][f'h{h}/joint']['holdout_150']['episodes']
        cut = report['methods'][f'h{h}/camera_gradient_cut']['holdout_150']['episodes']
        assert [r['episode_seed'] for r in full] == [r['episode_seed'] for r in cut]
        report.setdefault('paired_holdout_difference', {})[str(h)] = bootstrap_difference(
            [r['success_rate'] for r in full], [r['success_rate'] for r in cut])
        save()

    if cli.interventions:
        h = cli.horizons[-1]
        run = study / f'h{h}' / 'joint'
        args = load_args(run)
        entry = report['methods'][f'h{h}/joint']
        for condition in ('fixed_nominal', 'zero_image', 'frozen_view'):
            collect(entry, 'holdout_' + condition, args, run / 'final.pth',
                    801000, 100, condition)
    print('saved', report_path, flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Bounded joint-slit experiment. No automatic search or budget extension.

Common flight training is a fresh in-distribution experiment, not a claim
about old-checkpoint transfer. Camera imitation targets are never used.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from config import set_global_seed
from intent_model import build_policy
from main_cuda import _parameter_hash
from tools.intent_common import load_intent_args
from train_utils import build_env
from trainer import _rollout, _loss_from_rollout, _compute_success_collision

CONFIG = ROOT / 'configs/gray_joint_slit_learning.args'
LIGHTS = ('dark', 'bright', 'dark_to_bright', 'bright_to_dark')
DIRECT_CHECKS = (500, 1000, 1500, 2000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_save(value, path):
    temporary = Path(str(path) + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def sample_specs(count, first_seed):
    """Balance sign and illumination by construction, never by flight outcome."""
    assert count % 8 == 0
    return [dict(seed=first_seed+i, scenario=LIGHTS[i % 4],
                 side=-1 if (i // 4) % 2 == 0 else 1) for i in range(count)]


def numerical_sources():
    files = list(ROOT.glob('*.py'))
    for folder in ('render', 'sensors', 'controllers'):
        files.extend((ROOT/folder).glob('*.py'))
    files.extend([CONFIG, ROOT/'tools/run_joint_slit_learning.py', ROOT/'tools/intent_common.py'])
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def lock_protocol(out):
    import quadsim_cuda
    args = load_intent_args(CONFIG)
    locked = dict(config=vars(args), sources=numerical_sources(),
                  calibration_sha256=sha(ROOT/args.imx900_calibration),
                  development=sample_specs(32, 910000), confirmation=sample_specs(64, 920000),
                  demonstrations=sample_specs(128, 930000),
                  direct=dict(seed=42, updates=2000, checks=list(DIRECT_CHECKS), lr=3e-4),
                  fallback=dict(imitation_updates=800, imitation_lr=1e-3, batch=8,
                                task_updates=500, task_lr=3e-5, camera_labels=False),
                  arms=dict(updates=1000, batch=4, flight_lr=1e-5, camera_lr=1e-4,
                            checks=[200, 400, 600, 800, 1000], handoff_updates=200,
                            groups=['A_mean_ae', 'B_full', 'C_no_motion_image', 'D_no_eg_image']),
                  criterion=dict(normal=16, zero_difference=8, per_side=6, per_light=2),
                  optimizer=dict(name='AdamW', weight_decay=.01, clip=5, cosine_end_factor=.01),
                  runtime=dict(python=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
                               extension=quadsim_cuda.__file__, extension_sha256=sha(quadsim_cuda.__file__),
                               gpu=torch.cuda.get_device_name(0), amp=False, tf32=True),
                  no_training_seed_expansion=True, calibrated=False)
    locked = json.loads(json.dumps(locked))
    path = out/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != locked:
        raise RuntimeError('Locked protocol/source changed; do not reuse these results')
    if not path.exists():
        write_json(path, locked)
        sources = out/'source'
        for relative in locked['sources']:
            destination = sources/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT/relative).read_bytes())
        (out/'calibration.json').write_bytes((ROOT/args.imx900_calibration).read_bytes())
        write_json(out/'git_at_lock.json', dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)))
    return locked


def gate(normal, zero):
    successes = lambda rows: sum(int(r['success_rate']) for r in rows)
    n, z = successes(normal), successes(zero)
    sides = {str(side): successes([r for r in normal if r['side'] == side]) for side in (-1, 1)}
    lights = {light: successes([r for r in normal if r['scenario'] == light]) for light in LIGHTS}
    return dict(passed=n >= 16 and n-z >= 8 and min(sides.values()) >= 6 and min(lights.values()) >= 2,
                normal=n, zero=z, per_side=sides, per_light=lights)


@torch.no_grad()
def evaluate(model, args, specs, target, zero=False, traces=False):
    from eval import run_one_episode
    target = Path(target)
    state, was_training = rng_state(), model.training
    evaluation_args = copy.deepcopy(args)
    evaluation_args.batch_size = 1
    evaluation_args.policy_gray_mode = 'zero' if zero else 'gray'
    evaluation_args.vis_enable = False
    evaluation_args.camera_handoff_alpha = 1.
    env = build_env(1, evaluation_args, torch.device('cuda'), eval_mode=True)
    rows = []
    model.eval()
    try:
        for i, spec in enumerate(specs):
            env.simple_slit_center_y_min = -.75 if spec['side'] == -1 else 0.
            env.simple_slit_center_y_max = 0. if spec['side'] == -1 else .75
            row, trace = run_one_episode(i, spec['scenario'], evaluation_args, model, env, None,
                                         torch.device('cuda'), episode_seed=spec['seed'], collect_trace=traces)
            row.update(side=spec['side'], slit_center_y=float(env.slit_center_y.item()),
                       failed_penalized_time_s=row['elapsed_s'] if row['success_rate'] else 6.)
            rows.append(row)
            if traces:
                write_json(target.parent/(target.stem+f'_trace_{i:03d}.json'), trace)
        write_json(target, dict(checkpoint_parameters=_parameter_hash(model), zero=zero, rows=rows))
    finally:
        model.train(was_training)
        restore_rng(state)
    return rows


def initialize(args, checkpoint=None):
    set_global_seed(42, True)
    model = build_policy(args).cuda()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location='cuda'), strict=True)
    else:
        model.initialize_camera_output(.10, .02)
    return model


def train_update(model, env, args, optimizer, scheduler, iteration, stream_base):
    seed = stream_base + iteration
    set_global_seed(seed, True)
    env.reset()
    args.audit_rollout_seed = seed
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    rollout = _rollout(env, model, args, 4, torch.device('cuda'), False, None, False)
    loss, terms, positions, clearance = _loss_from_rollout(rollout, env, args)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(f'Nonfinite loss at update {iteration+1}')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    success, collision, distance = _compute_success_collision(positions, clearance, env, args, rollout)
    torch.cuda.synchronize()
    return dict(iteration=iteration+1, loss=float(loss.detach()), gradient_norm=float(norm),
                success_rate=float(success), collision_rate=float(collision), final_goal_dist=float(distance),
                valid_steps=int(torch.stack(rollout['valid_history']).sum()), generated_steps=360,
                optimization_seconds=time.perf_counter()-start,
                peak_cuda_memory_mib=torch.cuda.max_memory_allocated()/1024**2,
                losses={k: float(v.detach()) for k, v in terms.items()})


def common(out, protocol):
    collision = out/'collision/correction.json'
    if not collision.exists() or not all(w['old_stop_is_now_safe'] for w in json.loads(collision.read_text())['witnesses']):
        raise RuntimeError('Finish the prescribed collision correction audit first')
    target = out/'common_direct'
    target.mkdir(exist_ok=True)
    args = load_intent_args(CONFIG)
    model = initialize(args)
    model.freeze_camera_for_flight_only()
    camera_hash = _parameter_hash(model, True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2000, eta_min=3e-6)
    env = build_env(4, args, torch.device('cuda'))
    latest = target/'resume.pt'
    first, accumulated_seconds = 0, 0.
    if latest.exists():
        saved = torch.load(latest, map_location='cuda')
        if saved['protocol_sha256'] != sha(out/'protocol.json'):
            raise RuntimeError('Resume protocol mismatch')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng'])
        first, accumulated_seconds = saved['iteration'], saved['optimization_seconds']
    else:
        atomic_save(model.state_dict(), target/'initial.pth')
        write_json(target/'initialization.json', dict(total_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            camera_parameters_sha256=camera_hash, initial_sha256=sha(target/'initial.pth')))
    # A crash may leave a log row after the last atomic optimizer snapshot.
    metrics_path = target/'metrics.jsonl'
    if metrics_path.exists():
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        metrics_path.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['iteration'] <= first))
    for iteration in range(first, 2000):
        row = train_update(model, env, args, optimizer, scheduler, iteration, 100000)
        accumulated_seconds += row['optimization_seconds']
        row['cumulative_optimization_seconds'] = accumulated_seconds
        with metrics_path.open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
        completed = iteration+1
        if completed % 10 == 0 or completed == 1:
            print(f"[common] {completed}/2000 loss={row['loss']:.5f} success={row['success_rate']:.2f} update={row['optimization_seconds']:.2f}s", flush=True)
        if completed % 25 == 0 or completed in DIRECT_CHECKS:
            if _parameter_hash(model, True) != camera_hash:
                raise RuntimeError('Frozen camera-specific parameters changed')
            atomic_save(dict(iteration=completed, model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), rng=rng_state(), optimization_seconds=accumulated_seconds,
                protocol_sha256=sha(out/'protocol.json')), latest)
        if completed in DIRECT_CHECKS:
            checkpoint = target/f'checkpoint{completed:04d}.pth'
            atomic_save(model.state_dict(), checkpoint)
            normal = evaluate(model, args, protocol['development'], target/f'eval_{completed:04d}_gray.json')
            zero = evaluate(model, args, protocol['development'], target/f'eval_{completed:04d}_zero.json', zero=True)
            decision = gate(normal, zero)
            decision.update(iteration=completed, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                            optimization_seconds=accumulated_seconds, generated_steps=completed*360,
                            camera_specific_unchanged=_parameter_hash(model, True) == camera_hash)
            write_json(target/f'gate_{completed:04d}.json', decision)
            print('[common gate] '+json.dumps(decision), flush=True)
            if decision['passed']:
                write_json(out/'common_locked.json', dict(**decision, source='direct', camera_labels=False))
                return
    write_json(out/'common_direct_decision.json', dict(passed=False, next_stage='single_imitation_warmup',
        final_checkpoint=str(target/'checkpoint2000.pth'), updates=2000))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['lock', 'common'], required=True)
    parser.add_argument('--out_dir', type=Path, required=True)
    cli = parser.parse_args()
    out = cli.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    protocol = lock_protocol(out)
    if cli.stage == 'common':
        if (out/'common_locked.json').exists():
            print('Common checkpoint already locked; no further preparation.', flush=True)
        else:
            common(out, protocol)


if __name__ == '__main__':
    main()

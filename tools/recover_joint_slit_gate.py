#!/usr/bin/env python3
"""Finish an interrupted scheduled gate before resuming any training updates."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tools.run_joint_slit_learning import (CONFIG, DIRECT_CHECKS, lock_protocol, initialize,
    atomic_save, evaluate, gate, sha, write_json)
from tools.intent_common import load_intent_args
from main_cuda import _parameter_hash


def recover(out):
    protocol = lock_protocol(out)
    target = out/'common_direct'
    latest = target/'resume.pt'
    if not latest.exists() or (out/'common_locked.json').exists():
        return
    saved = torch.load(latest, map_location='cpu')
    if saved['protocol_sha256'] != sha(out/'protocol.json'):
        raise RuntimeError('Resume protocol mismatch')
    metrics = target/'metrics.jsonl'
    if metrics.exists():
        discarded = [json.loads(line) for line in metrics.read_text().splitlines()
                     if json.loads(line)['iteration'] > saved['iteration']]
        if discarded:
            write_json(target/f'metrics_discarded_{time.time_ns()}.json', dict(
                reason='Work after last atomic snapshot; retain cost before base runner truncates log', rows=discarded))
    initialization = json.loads((target/'initialization.json').read_text())
    for completed in DIRECT_CHECKS:
        if completed > saved['iteration']:
            break
        path = target/f'gate_{completed:04d}.json'
        checkpoint = target/f'checkpoint{completed:04d}.pth'
        if not path.exists():
            if not checkpoint.exists():
                if completed != saved['iteration']:
                    raise RuntimeError('Cannot recover a missing historical scheduled checkpoint')
                atomic_save(saved['model'], checkpoint)
            args = load_intent_args(CONFIG)
            model = initialize(args, checkpoint)
            normal = evaluate(model, args, protocol['development'], target/f'eval_{completed:04d}_gray.json')
            zero = evaluate(model, args, protocol['development'], target/f'eval_{completed:04d}_zero.json', zero=True)
            decision = gate(normal, zero)
            decision.update(iteration=completed, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                camera_specific_unchanged=_parameter_hash(model, True) == initialization['camera_parameters_sha256'],
                recovered_interrupted_gate=True)
            write_json(path, decision)
        else:
            decision = json.loads(path.read_text())
        if not decision['camera_specific_unchanged']:
            raise RuntimeError('Frozen camera changed')
        if decision['passed']:
            write_json(out/'common_locked.json', dict(**decision, source='direct', camera_labels=False))
            return


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=Path, required=True)
    recover(parser.parse_args().out_dir.resolve())

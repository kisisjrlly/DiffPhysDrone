#!/usr/bin/env python3
"""Reexecute only the two prescribed 16-episode motion cells after sphere fix."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_gray_transfer_cross import variants, source_cell, load_trace
from tools.audit_gray_chain_regression import run_migration_job, write_json, sha


def run(out):
    import torch
    from render.box_scene import swept_box_clearance
    old = ROOT / 'logs/gray_transfer_cross_20261103'
    geometry = json.loads((old / 'terminal_geometry.json').read_text())
    results = {cell: run_migration_job(out, cell, variants()[cell]) for cell in ('OD', 'WD')}
    witnesses = []
    for witness in geometry['rows']:
        cell, index = witness['cell'], witness['seed'] - 30000
        trace = load_trace(source_cell(old, cell) / f'ep_{index:03d}.npz')
        start, end, boxes = (torch.from_numpy(trace[key][-1].copy()).double()
                            if key != 'initial_voxels' else torch.from_numpy(trace[key].copy()).double()
                            for key in ('pre_p', 'post_p', 'initial_voxels'))
        gap = float(swept_box_clearance(start, end, boxes, .12).item())
        row = results[cell]['rows'][index]
        witnesses.append(dict(cell=cell, seed=witness['seed'], corrected_clearance_m=gap,
                              old_stop_is_now_safe=gap > .0011, continued_steps=row['steps'],
                              old_steps=len(trace['dt']), outcome=row))
    result = dict(cells={k: dict(successes=v['successes'], collisions=v['collisions'],
                                 episodes=len(v['rows'])) for k, v in results.items()},
                  witnesses=witnesses, source_sha256={p: sha(ROOT/p) for p in
                  ('render/box_scene.py', 'env_cuda.py', 'tools/audit_sphere_collision_correction.py')},
                  claim='Collision correction only; no joint-learning or generalized flight claim.')
    write_json(out / 'correction.json', result)
    if len(witnesses) != 4 or not all(w['old_stop_is_now_safe'] and w['continued_steps'] > w['old_steps'] for w in witnesses):
        raise RuntimeError('Prespecified false stops did not continue safely')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=Path, required=True)
    run(parser.parse_args().out_dir.resolve())

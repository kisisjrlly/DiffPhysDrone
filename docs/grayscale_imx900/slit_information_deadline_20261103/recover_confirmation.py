"""Resume missing locked confirmation files; never reselect or rerun completed arms."""
import json
from pathlib import Path
import numpy as np
import torch
from tools.audit_slit_information_deadline import (
    HistoricalBatch, MOTIONS, SPLITS, SCENARIOS, require_protocol,
    rollout, write_json, success_count, sha,
)

torch.set_num_threads(4)
out=Path('logs/slit_information_deadline_20261103').resolve()
require_protocol(out)
assert not (out/'confirmation_rows.json').exists()
lock=json.loads((out/'locked_candidates.json').read_text())
lock_hash=sha(out/'locked_candidates.json')
completed=sorted(f.name for f in out.glob('confirm_*.json'))
write_json(out/'confirmation_recovery.json',dict(reason='Executor disappeared without exit status.',
    completed_files_reused=completed,locked_candidates_sha256=lock_hash,
    no_reselection=True,no_parameter_changes=True,no_new_episodes=True))
history=HistoricalBatch('cuda')
rows=[]
for index,arm in enumerate(lock['arms']):
    motion=next(m for m in MOTIONS if m['name']==arm['motion'])
    combined=dict(motion=motion['name'],camera=arm['camera'],episodes=[])
    for scenario in SCENARIOS:
        path=out/f'confirm_{index}_{scenario}.json'
        if path.exists():
            r=json.loads(path.read_text())
            assert r['motion']==motion['name'] and r['camera']==arm['camera'] and r['scenario']==scenario
        else:
            r,frames=rollout(history,motion,mode='image',indices=SPLITS['confirmation'],
                scenario=scenario,camera=arm['camera'])
            write_json(path,r)
            np.savez_compressed(out/f'confirm_{index}_{scenario}_frames.npz',
                steps=[f['step'] for f in frames],images=[f['images'] for f in frames])
        combined['episodes'].extend(r['episodes'])
    rows.append(combined)
    print('LOCKED CONFIRMATION',index,success_count(combined),'/8',flush=True)
assert lock_hash==sha(out/'locked_candidates.json')
write_json(out/'confirmation_rows.json',rows)
write_json(out/'confirmation_lock.json',dict(locked_candidates_sha256=lock_hash))

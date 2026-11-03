"""Resume only missing predeclared combinations after the executor disappeared."""
import json
from pathlib import Path
import numpy as np
import torch
from tools.audit_slit_information_deadline import (
    HistoricalBatch, MOTIONS, SPLITS, SCENARIOS, require_protocol, camera_options,
    rollout, write_json, lock_candidates, success_count,
)

torch.set_num_threads(4)
out=Path('logs/slit_information_deadline_20261103').resolve()
require_protocol(out)
assert not (out/'locked_candidates.json').exists()
history=HistoricalBatch('cuda')
gates=json.loads((out/'control_gates.json').read_text())['development']
eligible=[m for m,ok in gates.items() if ok]
rows=json.loads((out/'development_rows.json').read_text())
completed_before=[(r['motion'],r['camera']['name']) for r in rows]
write_json(out/'recovery_manifest.json',dict(reason='Execution session disappeared during development; no exit status available.',
    completed_combinations_reused=completed_before,parameters_changed=False,additional_episodes=False,
    script_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()))
for motion in MOTIONS:
    if motion['name'] not in eligible:
        continue
    for camera in camera_options(history.env.gray_camera.calibration):
        if not camera['available'] or (motion['name'],camera['name']) in completed_before:
            continue
        combined=dict(motion=motion['name'],camera=camera,episodes=[])
        for scenario in SCENARIOS:
            path=out/f'dev_{motion["name"]}_{camera["name"]}_{scenario}.json'
            if path.exists():
                r=json.loads(path.read_text())
                assert r['motion']==motion['name'] and r['camera']==camera and r['scenario']==scenario
            else:
                r,frames=rollout(history,motion,mode='image',indices=SPLITS['development'],
                                scenario=scenario,camera=camera)
                write_json(path,r)
                np.savez_compressed(out/f'dev_{motion["name"]}_{camera["name"]}_{scenario}_frames.npz',
                    steps=[f['step'] for f in frames],images=[f['images'] for f in frames])
            combined['episodes'].extend(r['episodes'])
        rows.append(combined)
        write_json(out/'development_rows.json',rows)
        print('RESUMED DEVELOPMENT',motion['name'],camera['name'],success_count(combined),'/8',flush=True)
write_json(out/'locked_candidates.json',lock_candidates(rows,eligible))
print('CANDIDATES LOCKED',flush=True)

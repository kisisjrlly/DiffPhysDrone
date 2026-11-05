#!/usr/bin/env python3
"""Check executed evidence against the bounded plan, never infer from status alone."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from intent_model import build_policy
from main_cuda import _parameter_hash
from tools.intent_common import load_intent_args
from tools.run_joint_slit_learning import CONFIG, sha, gate, write_json
from tools.run_joint_slit_followup import ARMS


def load(path):
    return json.loads(Path(path).read_text())


def check_metrics(path, count, generated_steps=True):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if [r['iteration'] for r in rows] != list(range(1,count+1)):
        raise ValueError(f'Incorrect or incomplete update sequence: {path}')
    if generated_steps and any(r['generated_steps'] != 360 for r in rows):
        raise ValueError('Effective four-episode budget changed')
    if any(not torch.isfinite(torch.tensor(r['loss'])).item() for r in rows):
        raise ValueError('Nonfinite losses in accepted evidence')
    return rows


def check_episodes(path, specs):
    rows = load(path)['rows']
    if len(rows) != len(specs):
        raise ValueError(f'Incorrect evaluation sample count: {path}')
    for row,spec in zip(rows,specs):
        if (row['episode_seed'],row['scenario'],row['side']) != (spec['seed'],spec['scenario'],spec['side']):
            raise ValueError(f'Evaluation identity mismatch: {path}')
        if row['slit_center_y']*spec['side'] < 0:
            raise ValueError('Reported layout sign differs from generated layout')
        if not row['success_rate'] and row['failed_penalized_time_s'] != 6.:
            raise ValueError('Failures missing from completion cost')
        if row['camera_profile_calibrated']:
            raise ValueError('Provisional profile mislabeled as calibrated')
    return rows


def parameters(path):
    model = build_policy(load_intent_args(CONFIG))
    model.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
    if sum(p.numel() for p in model.parameters()) != 200605:
        raise ValueError('Network capacity changed')
    return model


def verify(out):
    checks = []
    protocol = load(out/'protocol.json')
    for relative,digest in protocol['sources'].items():
        if sha(out/'source'/relative) != digest or sha(ROOT/relative) != digest:
            raise ValueError(f'Numerical source mismatch: {relative}')
    if sha(out/'calibration.json') != protocol['calibration_sha256']:
        raise ValueError('Calibration snapshot changed')
    if load(out/'calibration.json').get('calibrated') is not False:
        raise ValueError('Calibration is not explicitly provisional')
    checks.append('locked numerical sources, full configuration and provisional calibration')
    collision = load(out/'collision/correction.json')
    if any(value['episodes'] != 16 for value in collision['cells'].values()) or len(collision['witnesses']) != 4:
        raise ValueError('Collision rerun scope changed')
    if not all(w['old_stop_is_now_safe'] and w['continued_steps'] > w['old_steps'] for w in collision['witnesses']):
        raise ValueError('Collision witnesses not actually continued')
    for cell in ('OD','WD'):
        inputs = load(out/f'collision/migration/{cell}/inputs.json')
        for name,digest in inputs['dependencies'].items():
            if sha(out/'collision/source'/name) != digest:
                raise ValueError(f'Collision execution source provenance missing: {name}')
    checks.append('32 collision reruns; four original false stops continued')
    shared_file = out/'common_locked.json'
    if shared_file.exists():
        shared = load(shared_file)
        checkpoint = Path(shared['checkpoint'])
        if sha(checkpoint) != shared['checkpoint_sha256']:
            raise ValueError('Common checkpoint changed')
        if shared['source'] == 'direct':
            step = shared['iteration']
            check_metrics(out/'common_direct/metrics.jsonl', step)
            prefix = out/f'common_direct/eval_{step:04d}'
            normal = check_episodes(Path(str(prefix)+'_gray.json'), protocol['development'])
            zero = check_episodes(Path(str(prefix)+'_zero.json'), protocol['development'])
        else:
            check_metrics(out/'common_direct/metrics.jsonl', 2000)
            check_metrics(out/'fallback/imitation_metrics.jsonl', 800, generated_steps=False)
            check_metrics(out/'fallback/task_metrics.jsonl', 500)
            normal = check_episodes(out/'fallback/eval_gray.json', protocol['development'])
            zero = check_episodes(out/'fallback/eval_zero.json', protocol['development'])
        if not gate(normal,zero)['passed']:
            raise ValueError('Common checkpoint does not meet its declared gate')
        camera_hash = load(out/'common_direct/initialization.json')['camera_parameters_sha256']
        if _parameter_hash(parameters(checkpoint), True) != camera_hash:
            raise ValueError('Common preparation changed frozen camera weights')
        checks.append('common checkpoint strictly loaded, visual/bilateral/light gates passed')
        counts = {}
        for arm in ARMS:
            directory = out/'arms'/arm
            done = load(directory/'done.json')
            if done['updates'] != 1000 or done['generated_steps'] != 360000:
                raise ValueError('Arm budget changed')
            check_metrics(directory/'metrics.jsonl', 1000)
            if done['common_sha256'] != sha(checkpoint):
                raise ValueError('Arm has a different common start')
            if _parameter_hash(parameters(directory/'initial.pth')) != _parameter_hash(parameters(checkpoint)):
                raise ValueError('Arm initial weights differ')
            if done['final_sha256'] != sha(directory/'final.pth'):
                raise ValueError('Arm endpoint changed')
            parameters(directory/'final.pth')
            for step in (200,400,600,800,1000):
                check_episodes(directory/f'eval_{step:04d}.json', protocol['development'])
            rows = check_episodes(directory/'confirmation/results.json', protocol['confirmation'])
            if load(directory/'confirmation/results.json')['checkpoint_parameters'] != _parameter_hash(parameters(directory/'final.pth')):
                raise ValueError('Final evaluation did not use the final endpoint')
            counts[arm] = sum(int(r['success_rate']) for r in rows)
            for index in range(64):
                if not (directory/f'confirmation/results_frames_{index:03d}.npz').exists():
                    raise ValueError('Missing true sensor-frame recording')
                trace = load(directory/f'confirmation/results_trace_{index:03d}.json')
                if not trace or not all('executed_command' in row and 'exposure_interval_s' in row for row in trace):
                    raise ValueError('Missing action/exposure timeline')
        results = load(out/'final_results.json')
        if counts != results['successes']:
            raise ValueError('Reported final counts differ from episodes')
        crosses = results['crosses']
        if counts[ARMS[1]]-counts[ARMS[0]] >= 4:
            for name in ('B_flight_AE','A_flight_B_camera'):
                rows = check_episodes(out/f'cross/{name}.json', protocol['confirmation'])
                if sum(int(r['success_rate']) for r in rows) != crosses[name]:
                    raise ValueError('Crossed-policy result mismatch')
        elif crosses:
            raise ValueError('Untriggered crossed-policy evaluation')
        checks.append('four matched 1000-update arms, fixed 64-episode final results and conditional crosses')
    else:
        decision = load(out/'decision.json')
        if decision['category'] != 'common_flight_not_established':
            raise ValueError('Unsupported early stop')
        check_metrics(out/'common_direct/metrics.jsonl', 2000)
        check_metrics(out/'fallback/imitation_metrics.jsonl', 800, generated_steps=False)
        check_metrics(out/'fallback/task_metrics.jsonl', 500)
        for step in (500,1000,1500,2000):
            prefix = out/f'common_direct/eval_{step:04d}'
            gray_rows = check_episodes(Path(str(prefix)+'_gray.json'), protocol['development'])
            zero_rows = check_episodes(Path(str(prefix)+'_zero.json'), protocol['development'])
            observed = gate(gray_rows,zero_rows)
            recorded = load(out/f'common_direct/gate_{step:04d}.json')
            if any(recorded[key] != value for key,value in observed.items()) or observed['passed']:
                raise ValueError('Direct scheduled gate contradicts the fallback decision')
            if sha(out/f'common_direct/checkpoint{step:04d}.pth') != recorded['checkpoint_sha256']:
                raise ValueError('Direct scheduled checkpoint changed')
        demonstration = load(out/'fallback/demonstration_manifest.json')
        stored = torch.load(out/'fallback/demonstrations.pt', map_location='cpu')
        if (stored['specs'] != protocol['demonstrations'] or demonstration['episodes'] != 128
                or stored['camera_labels'] or demonstration['camera_labels']
                or sha(out/'fallback/demonstrations.pt') != demonstration['sha256']):
            raise ValueError('Flight-only demonstration scope or provenance changed')
        if set(stored['data']) != {'frames','state','action','valid'}:
            raise ValueError('Unexpected demonstration labels')
        if sum(r['success'] for r in stored['outcomes']) != demonstration['successes']:
            raise ValueError('Teacher results differ from stored demonstrations')
        normal = check_episodes(out/'fallback/eval_gray.json', protocol['development'])
        zero = check_episodes(out/'fallback/eval_zero.json', protocol['development'])
        if gate(normal,zero)['passed']:
            raise ValueError('Failed gate contradicts executed results')
        frozen = load(out/'common_direct/initialization.json')['camera_parameters_sha256']
        if _parameter_hash(parameters(out/'fallback/final.pth'), True) != frozen:
            raise ValueError('Failed warmup changed frozen camera weights')
        if (out/'arms').exists():
            raise ValueError('Joint arms started without a valid common checkpoint')
        if (out/'final_results.json').exists() or load(out/'demo_status.json')['produced']:
            raise ValueError('Joint result or learned demonstration claimed after failed preparation')
        checks.append('prescribed direct + one warmup exhausted, gate failure stops arms')
    report_path = ROOT/'docs/grayscale_imx900/JOINT_SLIT_LEARNING_2026_11_05.md'
    report_status = load(out/'report_status.json')
    if report_status['report_sha256'] != sha(report_path):
        raise ValueError('Conclusion report hash mismatch')
    artifact = ROOT/'docs/grayscale_imx900/artifacts/joint_slit_learning_20261105'
    for name in ('learning_curves.png','costs.json','run_inventory.json'):
        if not (artifact/name).is_file():
            raise ValueError(f'Missing report artifact: {name}')
    checks.append('report, learning curve and cost inventory')
    value = dict(executed_evidence_passed=True, checks=checks,
                 remaining_manual_audit=['test coverage and full pytest','interpretation and conditional learned demonstration',
                                         'commit/push result','Bark receipt'])
    write_json(out/'execution_verification.json', value)
    print(json.dumps(value,ensure_ascii=False,indent=2))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=Path, required=True)
    verify(parser.parse_args().out_dir.resolve())

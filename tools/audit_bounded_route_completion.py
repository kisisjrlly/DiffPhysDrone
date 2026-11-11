#!/usr/bin/env python
"""Audit concrete evidence for the bounded route; never infer completion from a marker.

This is read-only unless --write is supplied. Does not run/extend any experiment.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
import torch

ARMS=('A1','A2','A3','B','C','D')
SCENARIOS=('dark','bright','dark_to_bright','bright_to_dark')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(out):
    out=Path(out);checks=[]
    def check(name,ok,evidence):checks.append(dict(requirement=name,passed=bool(ok),evidence=evidence))
    def read(name,default=None):
        path=out/name
        return json.loads(path.read_text()) if path.exists() else default
    protocol=read('protocol.json',{})
    check('locked physical/task configuration',protocol.get('physical')==dict(exposure_us=[100,50000],gain=[1,8],
        initial_exposure_us=890,initial_gain=1,body_radius=.12,slit_half_width=.15,clearance=.0011),'protocol.json')
    files=protocol.get('source_hashes',{})
    wrong=[name for name,value in files.items() if not (out/'source'/name).exists() or digest(out/'source'/name)!=value]
    check('frozen actual source including dirty state',bool(files) and not wrong,dict(files=len(files),mismatch=wrong))
    prepared=read('prepared.json',{})
    rh=digest(out/'reader.pth') if (out/'reader.pth').exists() else None
    check('reader hash frozen',rh is not None and rh==prepared.get('reader_hash'),'reader.pth vs prepared.json')
    dh=digest(out/'reader_data.pth') if (out/'reader_data.pth').exists() else None
    check('reader data hash frozen',dh is not None and dh==prepared.get('data_hash'),'reader_data.pth vs prepared.json')
    split=read('reader_split.json',{})
    fit,val=split.get('training',[]),split.get('validation',[])
    check('96/32 whole-trajectory reader split',len(fit)==96 and len(val)==32 and not set(fit)&set(val)
          and set(fit)|set(val)==set(range(128)) and split.get('data_hash')==dh,'reader_split.json')
    if rh:
        reader=torch.load(out/'reader.pth',map_location='cpu',weights_only=False)
        progress=torch.load(out/'reader_progress.pth',map_location='cpu',weights_only=False) if (out/'reader_progress.pth').exists() else {}
        check('1000 reader updates and validated selected checkpoint',progress.get('update')==1000 and 0<reader.get('update',0)<=1000
              and reader.get('data_hash')==dh and reader.get('split_hash')==digest(out/'reader_split.json'),
              dict(actual_updates=progress.get('update'),selected_update=reader.get('update')))
    integration=read('integration.json',{})
    check('record/batch invariant closed-loop check',integration.get('record_and_batch_invariant') is True,'integration.json + behavioral tests')
    check('declared gradient paths and raw finite differences',bool(read('gradient_paths.json',{})) and
          len(integration.get('finite_difference',[]))==6 and len(read('fixed_state_gradient_audit.json',{}).get('rows',[]))==27,
          'gradient_paths.json, integration.json, fixed_state_gradient_audit.json; direction agreement is not a scientific gate')
    expected=integration.get('locked_rounds',0)
    ledger=[json.loads(line) for line in (out/'ledger.jsonl').read_text().splitlines()] if (out/'ledger.jsonl').exists() else []
    seconds=sum(x.get('seconds',0) for x in ledger)
    counts=Counter()
    for r in ledger:
        if r.get('arm'):counts[r['arm']]+=r.get('count',0)
    check('total charged allocation <=24h',bool(ledger) and seconds<=24*3600,dict(hours=seconds/3600,episode_counts=dict(counts)))
    check('per-arm charged episodes <=4096',all(n<=4096 for n in counts.values()),dict(counts))
    check('all six arms complete locked common rounds',expected>0 and all((out/'training'/f'{arm}.pth').exists() for arm in ARMS)
        and all(torch.load(out/'training'/f'{arm}.pth',map_location='cpu',weights_only=False)['round_done']==expected for arm in ARMS
                if (out/'training'/f'{arm}.pth').exists()),dict(expected_rounds=expected))
    train_issues=[]
    for arm in ARMS:
        path=out/'training'/f'{arm}.pth'
        if not path.exists():train_issues.append(f'{arm}: missing');continue
        state=torch.load(path,map_location='cpu',weights_only=False)
        curve=read(f'training/{arm}_curve.json',[])
        if state.get('rollouts')!=expected*480 or len(curve)!=expected:train_issues.append(f'{arm}: incomplete budget/curve')
        if state.get('reader_hash')!=rh or state.get('config_hash')!=protocol.get('config_hash'):train_issues.append(f'{arm}: checkpoint hash')
        for row in curve:
            if sorted(x['id'] for x in row['episodes'])!=list(range(16)):train_issues.append(f'{arm}: development samples')
    check('matched checkpoint hashes, budgets, all development samples',not train_issues,train_issues)
    selection=read('selection.json',{})
    check('one development-selected AE and six locked checkpoints',selection.get('ae_winner') in ('A1','A2','A3') and
          set(selection.get('arms',{}))==set(ARMS) and selection.get('reader_hash')==rh,'selection.json')
    expected_conditions=list(ARMS)+[f'{j}_cross_{part}' for j in ('C','D') for part in ('camera','motion')]
    results={name:read(f'confirmation/{name}.json',{}) for name in expected_conditions}
    bad=[]
    for name,result in results.items():
        episodes=result.get('episodes',[])
        if sorted(x['id'] for x in episodes)!=list(range(64)):bad.append(f'{name}: missing/duplicate ids');continue
        if result.get('reader_hash')!=rh:bad.append(f'{name}: reader hash')
        if result['successes']!=sum(x['success'] for x in episodes):bad.append(f'{name}: wrong aggregate')
        for row in episodes:
            if row['scenario']!=SCENARIOS[row['id']%4] or row['namespace']!='confirmation':bad.append(f'{name}: mismatched sample/scenario')
            if row['success'] and (row['collision'] or not row['crossed']):bad.append(f'{name}: invalid success')
    check('ten paired confirmation conditions,64 episodes each',not bad,bad)
    check('all ten confirmation timelines exist',all((out/'timelines'/f'{n}_confirmation.pth').exists() for n in expected_conditions),expected_conditions)
    check('prespecified development id0 examples for C/D and all crossings',all((out/'examples'/f'{j}_{tag}.pth').exists()
        for j in ('C','D') for tag in ('joint','cross_camera','cross_motion','ae')),'examples/*.pth')
    artifacts=['episodes.csv','RESULTS_ZH.md','summary.json','task_comparison.png','C_mechanism_timeline.png','D_mechanism_timeline.png',
               'C_frames.png','D_frames.png','C_example.mp4','D_example.mp4']
    missing=[name for name in artifacts if not (out/name).exists() or (out/name).stat().st_size==0]
    check('task table, Chinese conclusions, curves, timeline figures and simulation videos',not missing,missing)
    video=[]
    for name in ('C_example.mp4','D_example.mp4'):
        path=out/name
        if path.exists():
            probe=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','json',str(path)],capture_output=True,text=True)
            valid=probe.returncode==0 and float(json.loads(probe.stdout).get('format',{}).get('duration',0))>0
            video.append(valid)
    check('videos decode to nonempty duration',len(video)==2 and all(video),'ffprobe on both simulation videos')
    tests=out/'verification_tests.txt'
    passed_test_count = max((int(n) for n in re.findall(r'(\d+) passed', tests.read_text())),default=0) if tests.exists() else 0
    check('full pytest evidence',passed_test_count>=326 and 'FAILED ' not in tests.read_text(),dict(path='verification_tests.txt',passed=passed_test_count))
    repo=read('repository_delivery.json',{})
    check('implementation/results committed and push attempted',bool(repo.get('commit')) and repo.get('push_attempted') is True,'repository_delivery.json')
    return dict(complete=all(x['passed'] for x in checks),passed=sum(x['passed'] for x in checks),
                required=len(checks),checks=checks)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out_dir',required=True);parser.add_argument('--write',action='store_true');args=parser.parse_args()
    result=audit(args.out_dir)
    if args.write:(Path(args.out_dir)/'completion_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False));sys.exit(0 if result['complete'] else 1)

if __name__=='__main__':main()

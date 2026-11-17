#!/usr/bin/env python
"""Attach to a VERIFIED live trainer, enforce remaining allocation, finish artifacts.
No scheduler, no automatic training restart, no new experiments. State/logs live
in the output directory so a host reboot cannot erase recovery evidence.
"""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def identity(pid):
    base=Path(f'/proc/{pid}')
    try:
        stat=(base/'stat').read_text().split()
        if stat[2]=='Z':return None
        cmd=(base/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        return dict(pid=pid,start_ticks=stat[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),command=cmd)
    except FileNotFoundError:return None


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--train-pid',type=int,required=True);p.add_argument('--out_dir',required=True);args=p.parse_args()
    out=Path(args.out_dir).resolve();expected=identity(args.train_pid)
    if not expected or 'run_bounded_joint_route.py --stage train' not in expected['command']:
        raise RuntimeError('Refuse to attach to an unverified live trainer')
    ledger=[json.loads(s) for s in (out/'ledger.jsonl').read_text().splitlines()]
    used=sum(r.get('seconds',0) for r in ledger)
    train_used=sum(r.get('seconds',0) for r in ledger if r['kind'] in ('train','train_interrupted'))
    ticks=os.sysconf(os.sysconf_names['SC_CLK_TCK'])
    up=float(Path('/proc/uptime').read_text().split()[0]);elapsed=up-int(expected['start_ticks'])/ticks
    now=time.time();process_start=now-elapsed
    train_deadline=process_start+max(0,12*3600-train_used)
    total_deadline=process_start+max(0,24*3600-used)
    supervision=dict(identity=expected,supervisor_pid=os.getpid(),attached_at_unix=now,
        charged_before_process_seconds=used,charged_training_before_process_seconds=train_used,
        training_deadline_unix=train_deadline,total_deadline_unix=total_deadline,
        limitation='Completed ledger includes conservative reboot-interruption charge.')
    (out/'supervision.json').write_text(json.dumps(supervision,indent=2)+'\n')
    print('ATTACHED',json.dumps(supervision),flush=True)
    while identity(args.train_pid)==expected:
        if time.time()>min(train_deadline,total_deadline):
            os.kill(args.train_pid,signal.SIGINT)
            (out/'budget_interruption.json').write_text(json.dumps(dict(reason='Allocation cap reached; no restart',time=time.time()))+'\n')
            break
        time.sleep(15)
    if not (out/'trained.json').exists():
        print('TRAINING_NOT_COMPLETE; no automatic restart or false negative conclusion',flush=True)
        return
    for stage in ('evaluate','report'):
        remaining=total_deadline-time.time()
        if remaining<=0:raise RuntimeError('Total allocation exhausted before final artifacts')
        timeout=min(remaining,3.2*3600) if stage=='evaluate' else remaining
        log=out/f'{stage}_supervised.log'
        with log.open('a') as stream:
            print('STAGE',stage,flush=True)
            result=subprocess.run([sys.executable,'tools/run_bounded_joint_route.py','--stage',stage,'--out_dir',str(out)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        if result.returncode:raise RuntimeError(f'{stage} failed, see {log}')
    if not (out/'evaluated.json').exists():raise RuntimeError('No complete evaluation evidence')
    from bounded_route.finalize import finalize
    from bounded_route.video import export_videos
    finalize(out);export_videos(out)
    (out/'supervised_artifacts_finished.json').write_text(json.dumps(dict(time=time.time(),complete=True,
        remaining='Final requirement audit, rendered preview review, result commit/push attempt and completion notification remain.'))+'\n')
    print('ARTIFACTS_FINISHED; final delivery audit still required',flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Sequential bounded round; never run beside a common/arm training worker."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(tool, out, stage=None):
    command = [sys.executable, str(ROOT/'tools'/tool), '--out_dir', str(out)]
    if stage:
        command.extend(['--stage', stage])
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=Path, required=True)
    cli = parser.parse_args()
    out = cli.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out/'pipeline.lock').open('w') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another pipeline holds this round')
        if not (out/'collision/correction.json').exists():
            run('audit_sphere_collision_correction.py', out/'collision')
        run('recover_joint_slit_gate.py', out)
        if not (out/'common_locked.json').exists() and not (out/'common_direct_decision.json').exists():
            run('run_joint_slit_learning.py', out, 'common')
        if not (out/'common_locked.json').exists() and not (out/'decision.json').exists():
            run('run_joint_slit_followup.py', out, 'fallback')
        if (out/'common_locked.json').exists():
            run('run_joint_slit_followup.py', out, 'arms')
            run('run_joint_slit_followup.py', out, 'evaluate')
        run('render_joint_slit_video.py', out)
        run('report_joint_slit_learning.py', out)


if __name__ == '__main__':
    main()

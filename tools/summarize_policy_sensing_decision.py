#!/usr/bin/env python3
"""Package the gated actual-network audit without launching experiments."""
import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tools.audit_slit_information_deadline import write_json, sha, HISTORICAL


def summarize(run_dir,destination):
    run_dir,destination=Path(run_dir),Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    results=[]
    labels=[]
    for name,label in [('baseline','Original checkpoint'),('adapted_baseline','Fixed-camera adaptation')]:
        result_path=run_dir/name/'result.json'
        if not result_path.exists():continue
        result=json.loads(result_path.read_text());results.append(result);labels.append(label)
        shutil.copy2(result_path,destination/f'{name}_result.json')
        for mode in ('normal','zero'):
            path=run_dir/name/f'{mode}.json'
            data=json.loads(path.read_text())
            write_json(destination/f'{name}_{mode}_episodes.json',data['episodes'])
            with gzip.open(destination/f'{name}_{mode}_timelines.json.gz','wt') as handle:
                json.dump(data['timelines'],handle,separators=(',',':'))
    if not results:raise RuntimeError('No completed stage to summarize')
    for name in ('baseline','adaptation'):
        for source in ('provenance.json','args.json','initialization_manifest.json',
                       'single_run_authorization.json','verified.json','calibration_snapshot.json'):
            path=run_dir/name/source
            if path.exists():shutil.copy2(path,destination/f'{name}_{source}')
    # Baseline provenance is stored at the run root.
    for name in ('provenance.json','calibration_snapshot.json','archive_manifest.json'):
        if (run_dir/name).exists():shutil.copy2(run_dir/name,destination/name)
    metrics=[]
    if (run_dir/'adaptation/metrics.jsonl').exists():
        metrics=[json.loads(line) for line in (run_dir/'adaptation/metrics.jsonl').read_text().splitlines()]
        shutil.copy2(run_dir/'adaptation/metrics.jsonl',destination/'adaptation_metrics.jsonl')
    invalid=run_dir/'invalid_configuration_attempt/INVALID_RUN.json'
    if invalid.exists():
        shutil.copy2(invalid,destination/'invalid_configuration_attempt.json')
        for name in ('args.json','metrics.jsonl','aborted.json','provenance.json'):
            path=invalid.parent/name
            if path.exists():shutil.copy2(path,destination/f'invalid_{name}')
    final=results[-1]
    if final['gate']=='needs_flight_baseline':
        decision='STOP: no viable image-driven flight baseline under this bounded protocol'
        next_priority='Diagnose fixed-camera neural flight transfer/training; no sensing-motion advantage claim.'
    elif final['gate']=='ceiling_stop':
        decision='STOP: baseline ceiling; no failure to diagnose'
        next_priority='Report lack of discrimination; do not manufacture failures.'
    else:
        decision='Baseline gate passed; bounded interventions remain required'
        next_priority='Run the prespecified real-network intervention stage before gradient claims.'
    summary=dict(decision=decision,next_priority=next_priority,baseline_results=results,
        intervention_stage='not_started',confirmation_stage='not_started',gradient_stage='not_started',
        trained_joint_policy=False,calibrated=False,
        limitation='8 selected historical development episodes, not independent generalization evidence.',
        effective_updates=len(metrics),invalid_attempt=json.loads(invalid.read_text()) if invalid.exists() else None)
    write_json(destination/'summary.json',summary)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold'})
    fig=plt.figure(figsize=(13,8),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=[1.1,1])
    ax=fig.add_subplot(grid[0,0]);x=np.arange(len(results));w=.32
    for offset,key,label,color in [(-w/2,'normal_success','Grayscale','#246E9F'),
                                   (w/2,'zero_success','Zero pixels','#9AA7B1')]:
        vals=[r[key] for r in results]
        bars=ax.bar(x+offset,vals,w,label=label,color=color)
        for bar,value in zip(bars,vals):ax.text(bar.get_x()+w/2,max(value,.08)+.10,f'{value}/8',ha='center')
    ax.set(xticks=x,xticklabels=labels,ylim=(0,8.7),ylabel='Safe arrivals / 8',
           title='A. Same network, with and without image information')
    ax.axhline(4,color='#777777',ls=':',lw=1);ax.legend(frameon=False,loc='upper right')
    ax=fig.add_subplot(grid[0,1])
    selected='adapted_baseline' if len(results)>1 else 'baseline'
    data=json.loads((run_dir/selected/'normal.json').read_text())
    ids=(1,2,3,12)
    centers=json.loads(HISTORICAL.read_text())['reference_dynamics_witness']['slit_centers_y']
    colors=['#246E9F','#3C8C75','#B77B37','#785C91']
    for i,c in zip(ids,colors):
        ax.plot([0,0],[centers[i]-.15,centers[i]+.15],color=c,lw=6,alpha=.45)
        timeline=data['timelines']['dark'];points=[]
        for row in timeline:
            if row['alive'][i]:points.append(row['position'][i][:2])
        if points:
            a=np.asarray(points);ax.plot(a[:,0],a[:,1],color=c,label=f'Layout {i}')
            ax.scatter(*a[-1],color=c,s=16)
    ax.axvline(0,color='#40474F',lw=2,label='Wall plane')
    ax.scatter([1.5],[0],marker='*',s=100,color='#40474F')
    ax.set(xlabel='Forward x (m)',ylabel='Lateral y (m)',title='B. Dark-condition trajectories (colored marks: slit openings)')
    ax.legend(frameon=False,fontsize=8,ncol=2)
    ax=fig.add_subplot(grid[1,0])
    if metrics:
        iterations=np.array([r['iteration'] for r in metrics]);loss=np.array([r['loss'] for r in metrics])
        ax.plot(iterations,loss,color='#ABC3D3',lw=.5,alpha=.65)
        window=25
        if len(loss)>=window:
            ax.plot(iterations[window-1:],np.convolve(loss,np.ones(window)/window,'valid'),color='#246E9F',lw=1.4,label='25-update mean')
        ax.set(xlabel='Fixed-camera update',ylabel='Original navigation loss',title='C. Bounded adaptation (not joint learning)')
        ax.legend(frameon=False)
    else:ax.text(.1,.5,'Adaptation not run',transform=ax.transAxes);ax.axis('off')
    ax=fig.add_subplot(grid[1,1]);ax.axis('off')
    boxes=[('Camera request k',.04,.79),('Exposure / image k+1',.49,.79),
           ('Next network decision',.49,.45),('Flight queue + actuator',.04,.45),
           ('Trajectory / collision / arrival',.04,.12)]
    for text,x,y in boxes:
        ax.text(x,y,text,transform=ax.transAxes,fontsize=9,va='center',
                bbox=dict(boxstyle='round,pad=.5',fc='#F0F4F7',ec='#BAC7D1'))
    for start,end in [((.38,.79),(.48,.79)),((.72,.72),(.72,.52)),((.48,.45),(.43,.45)),((.24,.38),(.24,.19))]:
        ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#586B7C'))
    ax.set_title('D. Timing path that later interventions must test',loc='left')
    fig.suptitle('Real-network prerequisite audit | joint_slit | provisional camera',fontsize=15,fontweight='bold')
    fig.savefig(destination/'comparison.png',dpi=180)
    fig.savefig(destination/'comparison_preview.jpg',dpi=85,pil_kwargs={'quality':80})
    plt.close(fig)
    manifest={p.name:sha(p) for p in sorted(destination.iterdir()) if p.is_file() and p.name!='delivery_manifest.json'}
    write_json(destination/'delivery_manifest.json',dict(files=manifest,run_directory=str(run_dir.resolve()),
        frozen_checkpoint_sha256=results[0]['checkpoint_sha256'],
        checkpoint_files_stay_local=True))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();summarize(args.run_dir,args.destination)

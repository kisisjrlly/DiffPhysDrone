"""Auditable task tables and compact scientific figures, never proxy success."""
import csv,json,math
from pathlib import Path
import numpy as np
import torch


def make_report(out):
    out=Path(out)
    selection=json.loads((out/'selection.json').read_text()) if (out/'selection.json').exists() else {}
    results={p.stem:json.loads(p.read_text()) for p in (out/'confirmation').glob('*.json')}
    rows=[]
    for name,r in sorted(results.items()):
        for ep in r['episodes']:rows.append(dict(condition=name,**ep))
    if rows:
        with (out/'episodes.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    curves={p.stem.removesuffix('_curve'):json.loads(p.read_text()) for p in (out/'training').glob('*_curve.json')}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(13,3.6))
    for name,curve in curves.items():
        ax[0].plot([r['optimization_rollouts'] for r in curve],[r['successes']/16 for r in curve],label=name,marker='o')
        ax[1].plot([r['arm_seconds']/3600 for r in curve],[r['successes']/16 for r in curve],label=name,marker='o')
    ax[0].set(xlabel='Optimization rollouts',ylabel='Development success rate',ylim=(-.03,1.03))
    ax[1].set(xlabel='Per-arm allocated GPU wall hours',ylabel='Development success rate',ylim=(-.03,1.03))
    names=[a for a in ('A1','A2','A3','B','C','D') if a in results]
    ax[2].bar(names,[results[n]['successes'] for n in names]);ax[2].set(ylabel='Confirmation successes / 64',ylim=(0,64))
    if curves:ax[0].legend(ncol=3)
    fig.tight_layout();fig.savefig(out/'task_comparison.png',dpi=150);plt.close(fig)
    summary=dict(complete=(out/'evaluated.json').exists(),ae_winner=selection.get('ae_winner'),
                 successes={n:r['successes'] for n,r in results.items()},interactions={})
    a=selection.get('ae_winner')
    if a in results:
        for joint in ('C','D'):
            if all(n in results for n in (joint,f'{joint}_cross_camera',f'{joint}_cross_motion')):
                interaction=(results[joint]['successes']-results[f'{joint}_cross_camera']['successes']-
                             results[f'{joint}_cross_motion']['successes']+results[a]['successes'])
                summary['interactions'][joint]=dict(success_count=interaction,rate=interaction/64)
        if all(n in results for n in ('B','C','D')):
            summary['joint_candidate']=results['C']['successes']>=max(results[a]['successes'],results['B']['successes'])+8
            summary['differentiable_success_count_difference']=results['C']['successes']-results['D']['successes']
    ledger=[json.loads(s) for s in (out/'ledger.jsonl').read_text().splitlines()]
    summary['allocated_gpu_hours']=sum(r.get('seconds',0) for r in ledger)/3600
    summary['counted_rollouts']=sum(r.get('count',0) for r in ledger)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    text=['# 小闭环执行结果','',f"完整预定比较：{'已完成' if summary['complete'] else '未完成'}。",
          f"已计费GPU占用时间：{summary['allocated_gpu_hours']:.3f}小时。",
          '', '| 条件 | 安全到达 | 碰撞 | 失败按4秒计费的平均时间 |','|---|---:|---:|---:|']
    for name,r in sorted(results.items()):text.append(f"| {name} | {r['successes']}/64 | {r['collisions']} | {r['time']:.3f}s |")
    text+=['','读图器、参数化、控制器、动力学和评分共用。单训练种子结果不构成论文级优越性证据。',
           '仿真使用未标定研究相机和历史全局光照规则，不代表真实无人机表现。',
           '交叉执行包含输入分布变化，需结合时间线解释，不能单独作为预见性测量的证明。']
    if not summary['complete']:text+=['','比较未完成；缺失结果不能解释为联合方法无效。']
    (out/'RESULTS_ZH.md').write_text('\n'.join(text)+'\n')
    # Deterministic representative: development id0, regardless of success.
    for joint in ('C','D'):
        paths=[out/'examples'/f'{joint}_{tag}.pth' for tag in ('joint','cross_camera','cross_motion','ae')]
        if not all(p.exists() for p in paths):continue
        examples=[torch.load(p,weights_only=False) for p in paths]
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for item,label in zip(examples,('joint','cross camera','cross motion','AE')):
            tr=item['timeline'];t=[r['time_s'] for r in tr]
            axes[0,0].plot([r['position'][0][0] for r in tr],[r['position'][0][1] for r in tr],label=label)
            axes[0,1].plot(t,[100*r['perception_error'][0] for r in tr],label=label)
            axes[1,0].plot(t,[r['exposure_us'][0]/1000 for r in tr],label=label)
            axes[1,1].plot(t,[r['submitted'][0][1] for r in tr],label=label)
        axes[0,0].set(xlabel='x (m)',ylabel='y (m)');axes[0,0].legend()
        axes[0,1].set(xlabel='time (s)',ylabel='perception error (cm)')
        axes[1,0].set(xlabel='time (s)',ylabel='exposure (ms)')
        axes[1,1].set(xlabel='time (s)',ylabel='submitted lateral acceleration')
        fig.suptitle(f'{joint}: prespecified development episode 0');fig.tight_layout()
        fig.savefig(out/f'{joint}_mechanism_timeline.png',dpi=130);plt.close(fig)
        # Image contact sheet remains modest; a simulation-labelled video uses the same fixed example.
        tr=examples[0]['timeline'];fig,axes=plt.subplots(2,4,figsize=(8,3.5))
        for ax,k in zip(axes.flat,np.linspace(0,len(tr)-1,8,dtype=int)):
            ax.imshow(tr[k]['frame'][0,0],cmap='gray',vmin=0,vmax=1);ax.set_title(f"{tr[k]['time_s']:.2f}s");ax.axis('off')
        fig.suptitle('Research simulation; development episode 0');fig.tight_layout();fig.savefig(out/f'{joint}_frames.png',dpi=100);plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

"""Finish the fixed experiment report from recorded evidence, without new trials."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch


def read(path):return json.loads(Path(path).read_text())


def paired_difference(left,right):
    a={r['id']:r for r in left['episodes']};b={r['id']:r for r in right['episodes']}
    if set(a)!=set(b):raise ValueError('Paired comparison has mismatched episode IDs')
    ids=sorted(a);dif=np.array([int(a[i]['success'])-int(b[i]['success']) for i in ids],float)
    gen=np.random.default_rng(42)
    means=dif[gen.integers(0,len(ids),(10000,len(ids)))].mean(1)
    return dict(success_count=int(dif.sum()),rate=float(dif.mean()),
        left_only=sum(a[i]['success'] and not b[i]['success'] for i in ids),
        right_only=sum(b[i]['success'] and not a[i]['success'] for i in ids),
        episode_resampling_95_interval=np.quantile(means,[.025,.975]).tolist(),
        limitation='Single trained seed; episode resampling does not quantify training-seed uncertainty.')


def summarize_example(path):
    item=torch.load(path,map_location='cpu',weights_only=False)
    rows=item['timeline'];active=[r for r in rows if r['alive'][0]]
    if not active:return dict(episodes=item['episodes'],active_steps=0)
    error=np.array([r['perception_error'][0] for r in active])
    lateral=np.array([r['actual_offset'][0] for r in active])
    closest=min(active,key=lambda r:abs(r['position'][0][0]))
    return dict(episode=item['episodes'][0],active_steps=len(active),
        perception_rmse_cm=float(np.sqrt((error**2).mean())*100),
        actual_lateral_rmse_cm=float(np.sqrt((lateral**2).mean())*100),
        nearest_wall=dict(step=closest['step'],time_s=closest['time_s'],position=closest['position'][0],
            perception_error_cm=100*closest['perception_error'][0],actual_offset_cm=100*closest['actual_offset'][0],
            exposure_us=closest['exposure_us'][0],gain=closest['gain'][0],
            submitted_command=closest['submitted'][0],executed_command=closest['executed'][0]),
        first_camera_request=dict(step=active[0]['step'],requested_us=active[0]['requested_exposure_us'][0],
            applied_us=active[0]['exposure_us'][0],effective_time_s=1/15),
        interpretation='Recorded temporal ordering, not a counterfactual causal proof.')


def finalize(out):
    out=Path(out)
    if not (out/'evaluated.json').exists():raise RuntimeError('Full planned evaluation evidence required')
    selection=read(out/'selection.json');winner=selection['ae_winner']
    results={p.stem:read(p) for p in (out/'confirmation').glob('*.json')}
    summary=read(out/'summary.json')
    locked=selection['arms']
    comparisons={f'C_vs_{other}':paired_difference(results['C'],results[other]) for other in (winner,'B','D')}
    comparisons['D_vs_A']=paired_difference(results['D'],results[winner])
    summary['paired_comparisons']=comparisons
    summary['development_direction_consistent']=locked['C']['successes']>locked[winner]['successes'] and locked['C']['successes']>locked['B']['successes']
    summary['joint_candidate']=summary['development_direction_consistent'] and results['C']['successes']>=max(results[winner]['successes'],results['B']['successes'])+8
    for joint in ('C','D'):
        count=(results[joint]['successes']-results[f'{joint}_cross_camera']['successes']-
               results[f'{joint}_cross_motion']['successes']+results[winner]['successes'])
        drops=dict(camera_rollback=results[joint]['successes']-results[f'{joint}_cross_motion']['successes'],
                   motion_rollback=results[joint]['successes']-results[f'{joint}_cross_camera']['successes'])
        summary['interactions'][joint]=dict(success_count=count,rate=count/64,**drops,
            engineering_support=count>=4 and min(drops.values())>0)
    rates=[results[a]['successes'] for a in ('A1','A2','A3','B','C','D')]
    if summary['joint_candidate'] and summary['interactions']['C']['engineering_support']:
        category='joint_candidate'
        claim='本轮出现联合任务收益候选；可微学习是否更有效另看C/D曲线和配对结果，单seed不能宣布方法优越。'
        next_work='只复验这套已锁定系统的训练种子与学习效率，不再修改场景或相机物理。'
    elif max(rates)<=8:
        category='low_capability'
        claim='六组都未建立足够的安全到达能力。本轮没有支持联合收益，也不足以判断任务是否不存在配合价值。'
        next_work='只处理共同闭环的任务能力：结合已保存的真值控制、读图误差和跟踪误差定位限制；不先换场景或学习算法。'
    elif min(rates)>=60:
        category='ceiling'
        claim='各组接近成功率上限；继续按既定次级指标报告时间和控制代价，不能从成功数接近推断配合没有价值。'
        next_work='结束本固定版本，先审查既有次级任务指标是否承载主张；不自动追加场景或调参。'
    else:
        category='no_prespecified_joint_support'
        claim='完整比较未达到预先声明的联合收益尺度；报告所有实际差异，不将其解释为普遍无效。'
        next_work='结束本固定版本，根据已记录的配对差异和误差时间线做一次研究取舍，不再在本协议内寻找正面配置。'
    summary.update(conclusion_category=category,claim_zh=claim,next_work_zh=next_work,
        scope='single training seed42; generic uncalibrated software camera; one original slit distribution')
    examples={}
    for joint in ('C','D'):
        examples[joint]={tag:summarize_example(out/'examples'/f'{joint}_{tag}.pth') for tag in ('joint','cross_camera','cross_motion','ae')}
    (out/'mechanism_evidence.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    ledger=[json.loads(line) for line in (out/'ledger.jsonl').read_text().splitlines()]
    count=Counter();phases=Counter();time_by_arm=Counter()
    for row in ledger:
        phases[row.get('phase',row['kind'])]+=row.get('count',0)
        if row.get('arm'):
            count[row['arm']]+=row.get('count',0);time_by_arm[row['arm']]+=row.get('seconds',0)
    summary['charged_rollouts_by_arm']=dict(count);summary['charged_rollouts_by_phase']=dict(phases)
    summary['training_and_selection_seconds_by_arm']=dict(time_by_arm)
    summary['allocated_gpu_hours']=sum(r.get('seconds',0) for r in ledger)/3600
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# 小闭环完整比较结论','',claim,'',
        '| 条件 | 开发集锁定检查点 | 确认成功/64 | 碰撞 | 超时 | 失败按4秒计费 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ('A1','A2','A3','B','C','D'):
        r=results[arm];timeout=sum(ep['timeout'] for ep in r['episodes'])
        lines.append(f"| {arm} | 第{locked[arm]['round']}轮 | {r['successes']} | {r['collisions']} | {timeout} | {r['time']:.3f}s |")
    lines+=['',f'自动曝光胜者由开发集一次锁定为{winner}；没有在确认集逐episode挑曝光。',
        f"C相对自动曝光多成功{comparisons[f'C_vs_{winner}']['success_count']}次，相对只学相机多成功{comparisons['C_vs_B']['success_count']}次，相对完全试跑学习多成功{comparisons['C_vs_D']['success_count']}次。",
        f"C交互量为{summary['interactions']['C']['success_count']}/64；两个交叉回退差异均记录在summary.json。",
        '', '已证实：同一共享读图器、控制器、任务代价和外部随机流完成六组及交叉比较；结果与预算有逐episode账目。',
        '尚未证实：真实相机效果、跨训练种子的优越性，以及仅凭交叉执行即可成立的预见性测量因果主张。',
        '位置到图像使用warp近似；固定状态与整任务有限差分的原始不一致均保留，不把梯度非零当作有效性证明。',
        '',f'下一项工作：{next_work}',
        '',f"总计费占用{summary['allocated_gpu_hours']:.3f} GPU小时；各组全量账目见summary.json。原checkpoint、原失败报告和calibration状态保留。"]
    (out/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(category=category,successes=summary['successes'],claim=claim),ensure_ascii=False),flush=True)

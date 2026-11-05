#!/usr/bin/env python3
"""Export bounded-round evidence only after a prescribed terminal condition."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_joint_slit_learning import sha, write_json
from tools.run_joint_slit_followup import ARMS


def load(path):
    return json.loads(Path(path).read_text())


def metrics(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def summarize_episodes(rows):
    return dict(episodes=len(rows), successes=sum(int(r['success_rate']) for r in rows),
                collisions=sum(int(r['collision_rate']) for r in rows),
                timeouts=sum(r['stop_reason']=='timeout' for r in rows),
                penalized_completion_s=sum(r['failed_penalized_time_s'] for r in rows)/len(rows),
                mean_control_cost=sum(r['control_cost'] for r in rows)/len(rows),
                per_light={light: sum(int(r['success_rate']) for r in rows if r['scenario']==light)
                           for light in sorted({r['scenario'] for r in rows})},
                per_side={str(side): sum(int(r['success_rate']) for r in rows if r['side']==side) for side in (-1,1)})


def classify(results, development_curves, baseline_successes):
    """Engineering categories, not significance or best-algorithm claims."""
    successes = results['successes']
    if results['joint_candidate']:
        return '出现拍摄—飞行配合候选', '下一轮只验证候选收益对应的关键学习信号。'
    if min(successes.values()) >= 60:
        return '各组接近全成功，检查缺乏区分能力', '停止扩展本场景，不收紧时限或增加噪声。'
    if all(curve[-1][1] < baseline_successes for curve in development_curves.values()):
        return '后续训练使各组开发集成绩均低于共同起点', '下一轮只定位更新过程为何破坏已有飞行能力。'
    if successes[ARMS[1]]-successes[ARMS[0]] >= 4:
        return '学习拍摄有任务收益，尚未证明飞行动作配合', '只定位当前拍摄收益来源，不宣称联合配合成立。'
    # A predeclared descriptive curve marker, never a checkpoint selector.
    # If preparation already meets 24/32, all first-hit values are zero.
    first_hit = {name: next((step for step,n in curve if n >= 24), None)
                 for name,curve in development_curves.items()}
    a, b = first_hit[ARMS[0]], first_hit[ARMS[1]]
    if abs(successes[ARMS[1]]-successes[ARMS[0]]) < 4 and b is not None and (a is None or b < a):
        return '最终成绩相近，完整联合组较早达到开发集24/32', '先核对达到该成绩的实际训练耗时；只报告候选学习效率收益。'
    return '当前场景与预算未显示明确联合优势', '停止扩大本轮实验；依据现有失败记录定位一个具体障碍。'


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    stopped = out/'decision.json'
    final = out/'final_results.json'
    if not stopped.exists() and not final.exists():
        raise RuntimeError('The experiment is still running; do not publish a terminal conclusion')
    artifact = ROOT/'docs/grayscale_imx900/artifacts/joint_slit_learning_20261105'
    artifact.mkdir(parents=True, exist_ok=True)
    protocol = load(out/'protocol.json')
    collision = load(out/'collision/correction.json')
    common = load(out/'common_locked.json') if (out/'common_locked.json').exists() else None
    common_rows = metrics(out/'common_direct/metrics.jsonl')
    fallback_imitation = metrics(out/'fallback/imitation_metrics.jsonl')
    fallback_task = metrics(out/'fallback/task_metrics.jsonl')
    costs = dict(common_generated_steps=sum(r['generated_steps'] for r in common_rows+fallback_task),
                 common_valid_steps=sum(r['valid_steps'] for r in common_rows+fallback_task),
                 common_optimization_seconds=sum(r['optimization_seconds'] for r in common_rows+fallback_task),
                 imitation_updates=len(fallback_imitation),
                 imitation_optimization_seconds=fallback_imitation[-1]['cumulative_optimization_seconds'] if fallback_imitation else 0,
                 arms={})
    if (out/'fallback/demonstration_manifest.json').exists():
        costs['demonstrations'] = load(out/'fallback/demonstration_manifest.json')
    fig, axes = plt.subplots(1, 3, figsize=(13,3.6), constrained_layout=True)
    axes[0].plot([r['iteration'] for r in common_rows], [r['loss'] for r in common_rows], color='#354e6b', lw=.8)
    axes[0].set(xlabel='Common-flight updates', ylabel='Task objective', title='Shared preparation')
    colors = ('#646c76', '#246c9b', '#ce8c3c', '#9b678e')
    rows, development_curves = [], {}
    if final.exists():
        results = load(final)
        for arm, color in zip(ARMS, colors):
            directory = out/'arms'/arm
            arm_metrics = metrics(directory/'metrics.jsonl')
            done = load(directory/'done.json')
            costs['arms'][arm] = dict(updates=len(arm_metrics),
                generated_steps=sum(r['generated_steps'] for r in arm_metrics),
                valid_steps=sum(r['valid_steps'] for r in arm_metrics),
                optimization_seconds=done['optimization_seconds'], final_sha256=done['final_sha256'])
            updates, successes, seconds = [0], [common['normal']/32], [0.]
            by_update = {r['iteration']:r['cumulative_optimization_seconds'] for r in arm_metrics}
            for step in (200,400,600,800,1000):
                development = load(directory/f'eval_{step:04d}.json')['rows']
                updates.append(step)
                successes.append(sum(r['success_rate'] for r in development)/32)
                seconds.append(by_update[step])
            development_curves[arm] = list(zip(updates, [round(value*32) for value in successes]))
            axes[1].plot([i*360 for i in updates], successes, 'o-', color=color, label=arm)
            axes[2].plot(seconds, successes, 'o-', color=color, label=arm)
            episodes = load(directory/'confirmation/results.json')['rows']
            rows.extend(dict(arm=arm, **row) for row in episodes)
        axes[1].legend(fontsize=7)
        category, next_work = classify(results, development_curves, common['normal'])
        write_json(artifact/'task_metrics.json', {arm:summarize_episodes([r for r in rows if r['arm']==arm]) for arm in ARMS})
    else:
        results = load(stopped)
        category, next_work = '共同视觉飞行起点未建立', results['next_work']
        for ax in axes[1:]:
            ax.text(.5,.5,'Joint training not started\nCommon-flight gate failed', ha='center', va='center', transform=ax.transAxes)
    axes[1].set(xlabel='Generated control steps', ylabel='Development success', ylim=(-.02,1.02), title='Matched interaction budget')
    axes[2].set(xlabel='Optimization seconds', ylabel='Development success', ylim=(-.02,1.02), title='Actual training cost')
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(artifact/'learning_curves.png', dpi=180)
    plt.close(fig)
    for name in ('protocol.json','collision/correction.json','common_locked.json','common_direct_decision.json',
                 'fallback/decision.json','fallback/demonstration_manifest.json','final_results.json','decision.json'):
        path = out/name
        if path.exists():
            destination = artifact/name.replace('/','_')
            shutil.copy2(path, destination)
    for path in sorted((out/'common_direct').glob('gate_*.json')):
        shutil.copy2(path, artifact/path.name)
    discarded = {str(path.relative_to(out)):load(path) for path in out.rglob('*_discarded_*.json')}
    costs['discarded_update_records'] = discarded
    costs['discarded_optimization_seconds'] = sum(
        row.get('optimization_seconds',0.) for value in discarded.values() for row in value['rows'])
    write_json(artifact/'costs.json', costs)
    if rows:
        keys = sorted({key for row in rows for key in row})
        with (artifact/'episodes.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    text = ['# 当前窄缝联合学习：2026-11-05执行结果', '', f'**结论：{category}。**', '',
        '## 已证实', '',
        f'- 球体连续碰撞修正后，两组历史复算为 {collision["cells"]["OD"]["successes"]:.0f}/16、'
        f'{collision["cells"]["WD"]["successes"]:.0f}/16；四次误停均继续运行，未直接改记成功。',
        f'- 共同准备：直接任务更新{len(common_rows)}次；模仿更新{len(fallback_imitation)}次；热身后任务更新{len(fallback_task)}次。']
    if common:
        text.append(f'- 共同起点开发集：正常{common["normal"]}/32、零图{common["zero"]}/32。')
    if final.exists():
        text.extend(['', '|方法|最终安全到达|训练生成步|训练秒数|', '|---|---:|---:|---:|'])
        for arm in ARMS:
            cost = costs['arms'][arm]
            text.append(f'|{arm}|{results["successes"][arm]}/64|{cost["generated_steps"]}|{cost["optimization_seconds"]:.1f}|')
        text.extend(['', f'交叉结果：`{json.dumps(results["crosses"], ensure_ascii=False)}`；'
                     f'交互成功数差：{results["interaction_success_count"]}。'])
    text.extend(['', '## 尚未证明', '',
        '单个训练seed、开发样本筛选；没有跨seed统计、RL算法竞争或真机证据。'
        '梯度切断组是机制检查，策略交叉下降也可能包含分布失配。相机仍未实测标定。',
        '', '## 下一项唯一工作', '', next_work, '',
        '完整配置、源码快照、逐步训练指标和检查点在运行目录：', f'`{out}`', '',
        '![学习过程](artifacts/joint_slit_learning_20261105/learning_curves.png)', ''])
    report_path = ROOT/'docs/grayscale_imx900/JOINT_SLIT_LEARNING_2026_11_05.md'
    report_path.write_text('\n'.join(text))
    inventory = {str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix in
                 ('.json','.jsonl','.pth','.pt','.npz')}
    write_json(artifact/'run_inventory.json', inventory)
    write_json(out/'report_status.json', dict(category=category, report=str(report_path),
        report_sha256=sha(report_path), remaining_delivery=['completion audit','commit/push','Bark']))
    print(category, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=Path, required=True)
    report(parser.parse_args().out_dir.resolve())

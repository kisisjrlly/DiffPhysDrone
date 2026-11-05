"""Reporting must separate task success, learning cost and preparation failures."""
from tools.report_joint_slit_learning import classify, summarize_episodes, discarded_cost
from tools.run_joint_slit_followup import ARMS


def curves(final=20):
    return {arm:[(0,20), (200,final), (1000,final)] for arm in ARMS}


def test_retry_cost_counts_only_discarded_work_not_prior_training(tmp_path):
    import json
    from tools.run_joint_slit_followup import truncate_metrics
    path = tmp_path/'imitation_metrics.jsonl'
    rows = [dict(iteration=1,cumulative_optimization_seconds=10.,valid_steps=8),
            dict(iteration=2,cumulative_optimization_seconds=12.,valid_steps=9)]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    truncate_metrics(path, 1)
    record = json.loads(next(tmp_path.glob('*_discarded_*.json')).read_text())
    assert discarded_cost(record) == dict(updates=1,optimization_seconds=2.,generated_steps=0,valid_steps=9)
    assert len(path.read_text().splitlines()) == 1
    direct = dict(rows=[dict(optimization_seconds=3.,generated_steps=360,valid_steps=240)])
    assert discarded_cost(direct)['optimization_seconds'] == 3.


def test_all_arm_regression_is_not_reported_as_perception_failure():
    result = dict(joint_candidate=False, successes=dict(zip(ARMS,(10,12,11,9))))
    category, next_work = classify(result, curves(final=12), 20)
    assert '低于共同起点' in category
    assert '更新过程' in next_work


def test_earlier_sample_threshold_does_not_imply_compute_efficiency():
    development = curves(final=25)
    development[ARMS[0]] = [(0,20),(200,22),(1000,25)]
    result = dict(joint_candidate=False, successes=dict(zip(ARMS,(40,42,39,36))))
    category, next_work = classify(result, development, 20)
    assert '较早' in category
    assert '实际训练耗时' in next_work
    # Already above the threshold at shared initialization: no early-learning claim.
    development = {arm:[(0,25),(1000,25)] for arm in ARMS}
    assert '较早' not in classify(result, development, 25)[0]


def test_summary_counts_failures_in_completion_cost_and_keeps_layout_splits():
    rows = [dict(success_rate=1,collision_rate=0,stop_reason='goal',failed_penalized_time_s=4.,
                 control_cost=2.,scenario='dark',side=-1),
            dict(success_rate=0,collision_rate=1,stop_reason='collision',failed_penalized_time_s=6.,
                 control_cost=3.,scenario='bright',side=1)]
    summary = summarize_episodes(rows)
    assert summary['successes']==summary['collisions']==1
    assert summary['penalized_completion_s']==5
    assert summary['per_side']=={'-1':1,'1':0}
    assert summary['per_light']=={'bright':0,'dark':1}

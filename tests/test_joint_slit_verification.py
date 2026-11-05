"""Result checks reject missing samples, wrong identities and optimistic failure cost."""
import json

import pytest

from tools.verify_joint_slit_round import check_episodes, check_metrics


def test_incomplete_or_duplicate_training_updates_cannot_pass(tmp_path):
    path = tmp_path/'metrics.jsonl'
    path.write_text('\n'.join(json.dumps(dict(iteration=i,generated_steps=360,loss=1.)) for i in (1,2,2)))
    with pytest.raises(ValueError, match='sequence'):
        check_metrics(path, 3)


def test_evaluation_identity_and_failure_penalty_are_checked(tmp_path):
    path = tmp_path/'evaluation.json'
    spec = dict(seed=3,scenario='dark',side=-1)
    row = dict(episode_seed=3,scenario='dark',side=-1,slit_center_y=-.1,
               success_rate=0,failed_penalized_time_s=6.,camera_profile_calibrated=False)
    path.write_text(json.dumps(dict(rows=[row])))
    assert len(check_episodes(path,[spec])) == 1
    path.write_text(json.dumps(dict(rows=[dict(row,failed_penalized_time_s=.3)])))
    with pytest.raises(ValueError, match='Failures'):
        check_episodes(path,[spec])
    path.write_text(json.dumps(dict(rows=[dict(row,episode_seed=4)])))
    with pytest.raises(ValueError, match='identity'):
        check_episodes(path,[spec])

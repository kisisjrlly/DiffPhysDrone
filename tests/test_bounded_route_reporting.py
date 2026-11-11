"""Guard against invalid research conclusions from unmatched or floor results."""
import pytest
from bounded_route.finalize import paired_difference


def result(values,ids=None):
    ids=range(len(values)) if ids is None else ids
    return dict(episodes=[dict(id=i,success=bool(value)) for i,value in zip(ids,values)])


def test_paired_comparison_counts_discordant_outcomes_and_preserves_sign():
    difference=paired_difference(result([1,1,0,0]),result([0,1,1,0]))
    assert difference['success_count']==0
    assert difference['left_only']==1 and difference['right_only']==1
    assert difference['episode_resampling_95_interval'][0]<=0<=difference['episode_resampling_95_interval'][1]


def test_paired_comparison_refuses_mismatched_episode_ids():
    with pytest.raises(ValueError,match='mismatched'):
        paired_difference(result([1,0]),result([0,1],[1,2]))


def test_all_failure_comparison_is_not_reported_as_a_positive_effect():
    difference=paired_difference(result([0]*64),result([0]*64))
    assert difference['success_count']==0
    assert difference['episode_resampling_95_interval']==[0,0]
    assert 'training-seed uncertainty' in difference['limitation']

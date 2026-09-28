from types import SimpleNamespace

import pytest
import torch

import eval as evaluation


class TerminalEnv:
    batch_size = 1

    def __init__(self, goal, clearance):
        self.p = torch.zeros(1, 3)
        self.p_target = torch.tensor([[goal, 0., 0.]])
        self.act = torch.zeros(1, 3)
        self.clearance = clearance

    def reset(self, scene_name):
        pass

    def find_vec_to_nearest_pt(self):
        return torch.tensor([[self.clearance, 0., 0.]])


@pytest.mark.parametrize('goal,clearance,reason,success', [
    (0., 1., 'goal', 1.), (1., 0., 'collision', 0.), (0., 0., 'collision', 0.),
])
def test_terminal_episode_stops_independently(monkeypatch, goal, clearance, reason, success):
    monkeypatch.setattr(evaluation, 'init_camera_params',
                        lambda *args: (torch.zeros(1), torch.zeros(1)))
    args = SimpleNamespace(seed=7, deterministic=False, amp=False,
        vis_enable=False, vis_episode_idx=-1, timesteps=2,
        base_control_freq=15., collision_clearance=0.1)
    env = TerminalEnv(goal, clearance)
    row, _ = evaluation.run_one_episode(2, 'nominal', args,
        SimpleNamespace(reset=lambda: None), env, None, torch.device('cpu'))
    assert row['episode_seed'] == 9
    assert row['stop_reason'] == reason
    assert row['success_rate'] == success
    assert row['steps'] == 0
    assert row['min_clearance'] == clearance


def test_batched_early_stopping_is_rejected():
    with pytest.raises(ValueError, match='batch_size=1'):
        evaluation.run_one_episode(0, 'nominal', None, None,
            SimpleNamespace(batch_size=2), None, torch.device('cpu'))

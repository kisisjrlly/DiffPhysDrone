"""Protocol and handoff invariants for the bounded training round."""
import pytest
import torch

from rollout_ops import blend_camera_request
from tools.run_joint_slit_learning import CONFIG, LIGHTS, gate, sample_specs
from tools.intent_common import load_intent_args


def test_handoff_preserves_bounds_and_only_trains_learned_request():
    learned = torch.tensor([[.2, .8]], requires_grad=True)
    automatic = torch.tensor([[.9, .1]], requires_grad=True)
    result = blend_camera_request(learned, automatic, .25)
    torch.testing.assert_close(result, torch.tensor([[.725, .275]]))
    result.sum().backward()
    torch.testing.assert_close(learned.grad, torch.full_like(learned, .25))
    assert automatic.grad is None
    with pytest.raises(ValueError):
        blend_camera_request(learned, automatic, 1.1)


def test_locked_specs_are_balanced_and_disjoint():
    dev, confirm, demos = (sample_specs(n, start) for n, start in ((32, 910000), (64, 920000), (128, 930000)))
    for specs in (dev, confirm, demos):
        assert all(sum(s['scenario'] == light and s['side'] == sign for s in specs) == len(specs)//8
                   for light in LIGHTS for sign in (-1, 1))
    assert not {s['seed'] for s in dev} & {s['seed'] for s in confirm+demos}


def test_gate_requires_visual_and_bilateral_capability():
    normal = [dict(**s, success_rate=1.) for s in sample_specs(32, 910000)]
    zero = [dict(**s, success_rate=0.) for s in sample_specs(32, 910000)]
    assert gate(normal, zero)['passed']
    assert not gate(normal, normal)['passed']
    one_side = [dict(r, success_rate=float(r['side'] == 1)) for r in normal]
    assert not gate(one_side, zero)['passed']


def test_training_config_uses_task_goal_without_forcing_observation_speed():
    args = load_intent_args(CONFIG)
    assert args.coef_v == args.coef_obj_avoidance == args.coef_cam_smooth == 0
    assert args.coef_terminal_goal == 10 and args.coef_collide == 6
    assert args.batch_size == 4 and args.timesteps == 90 and args.base_control_freq == 15
    assert args.camera_control_mode == 'mean_ae' and args.train_flight_only
    assert args.camera_queue_aligned_intent and args.observation_grad_mode == 'full'


def test_resume_finishes_pending_gate_before_allowing_more_updates(tmp_path, monkeypatch):
    import json
    import tools.recover_joint_slit_gate as recovery
    from tools.run_joint_slit_learning import sha
    (tmp_path/'protocol.json').write_text('{}')
    target = tmp_path/'common_direct'
    target.mkdir()
    (target/'initialization.json').write_text(json.dumps({'camera_parameters_sha256':'camera'}))
    torch.save(dict(iteration=500, model={'weight':torch.tensor([1.])},
                    protocol_sha256=sha(tmp_path/'protocol.json')), target/'resume.pt')
    specs = sample_specs(32, 910000)
    monkeypatch.setattr(recovery, 'lock_protocol', lambda out: {'development':specs})
    monkeypatch.setattr(recovery, 'initialize', lambda args, checkpoint: object())
    monkeypatch.setattr(recovery, '_parameter_hash', lambda *args: 'camera')
    calls = []
    def fake_evaluate(model, args, samples, path, zero=False):
        calls.append(zero)
        return [dict(**s, success_rate=float(not zero)) for s in samples]
    monkeypatch.setattr(recovery, 'evaluate', fake_evaluate)
    recovery.recover(tmp_path)
    assert calls == [False, True]
    locked = json.loads((tmp_path/'common_locked.json').read_text())
    assert locked['passed'] and locked['iteration'] == 500
    assert (target/'checkpoint0500.pth').exists()

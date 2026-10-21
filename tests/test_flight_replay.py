"""Retain flight knowledge without prescribing measurement actions or E/G."""

import torch

from flight_replay import FlightReplayAnchor, flight_sequence, masked_action_anchor
from intent_model import build_policy
from tools.intent_common import load_intent_args


def test_anchor_leaves_measurement_actions_unsupervised_and_has_no_camera_labels():
    args = load_intent_args("configs/gray_joint_parallax_lookahead.args")
    model = build_policy(args)
    data = {"frames": torch.rand(2, 28, 1, 36, 48), "state": torch.rand(2, 28, 10)}
    indices = torch.arange(2)
    with torch.no_grad():
        targets = flight_sequence(model, data, indices, torch.device("cpu"))
    actions = flight_sequence(model, data, indices, torch.device("cpu"))
    valid = torch.arange(28)[None].expand(2, -1) >= 24
    assert masked_action_anchor(actions, targets, valid) == 0
    perturbed = actions.detach().clone().requires_grad_()
    with torch.no_grad():
        perturbed[:, :24] += 10
        perturbed[:, 24:] += .1
    loss = masked_action_anchor(perturbed, targets, valid)
    grad = torch.autograd.grad(loss, perturbed)[0]
    assert grad[:, :24].abs().sum() == 0
    assert grad[:, 24:].abs().sum() > 0
    with torch.no_grad():
        model.fc.bias.add_(.1)
    loss = masked_action_anchor(flight_sequence(model, data, indices, torch.device("cpu")), targets, valid)
    loss.backward()
    assert model.fc.weight.grad.abs().sum() > 0
    assert all(p.grad is None for n,p in model.named_parameters() if model._is_camera_parameter(n))
    assert sum(p.numel() for p in model.parameters()) == 200605


def test_anchor_keeps_rollout_rng_and_initial_actor_unchanged(tmp_path):
    args = load_intent_args("configs/gray_joint_parallax_lookahead.args", gray_nn_width=12, gray_nn_height=9)
    model = build_policy(args)
    args.resume = str(tmp_path / "common.pth")
    torch.save(model.state_dict(), args.resume)
    args.flight_replay_dataset = str(tmp_path / "data.pt")
    args.flight_replay_weight = 1000.
    data = {"frames": torch.rand(2, 28, 1, 9, 12), "state": torch.rand(2, 28, 10),
            "valid": (torch.arange(28)[None].expand(2, -1) >= 24).clone()}
    torch.save({"data": data, "dataset_configuration": {"gray_nn_width":12,"gray_nn_height":9},
                "metadata": {"unlabelled_initial_probe_steps":24}}, args.flight_replay_dataset)
    initial = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    anchor = FlightReplayAnchor(model, args, torch.device("cpu"), tmp_path)
    value = anchor.loss(model)
    assert value < 1e-10
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(initial[k],v) for k,v in model.state_dict().items())
    assert anchor.sequence_samples == args.flight_replay_batch
    assert anchor.valid_samples == args.flight_replay_batch * 4

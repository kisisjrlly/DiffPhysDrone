"""Common flight warm start must preserve the deployed actor and camera head."""

import json
import shutil

import torch

from intent_model import build_policy
from tools.intent_common import load_intent_args
from tools.pole_navigation_teacher import PoleNavigationTeacher
from train_utils import build_env
from tools.train_pole_navigation_imitation import collect, dataset_configuration
from tools.evaluate_pole_coordination import evaluate


def pole_args(**overrides):
    values = dict(batch_size=1, gray_width=20, gray_height=16, gray_nn_width=16,
                  gray_nn_height=12, timesteps=12, loss_v_window=3, gray_enable_noise=False)
    values.update(overrides)
    return load_intent_args("configs/gray_poles_navigation.args", **values)


def test_batched_imitation_encoding_matches_actor_flight_forward():
    args = pole_args()
    model = build_policy(args)
    state, images = torch.randn(3, 10), torch.rand(3, 2, 16, 20)
    history = torch.randn(3, 128)
    expected, _, hx, _ = model(state, history, gray_obs=images)
    actual, direct_hx = model.flight_from_encoded(model.encode_observation(state, images), history)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(direct_hx, hx, rtol=0, atol=0)
    before = {name: value.clone() for name, value in model.named_parameters() if model._is_camera_parameter(name)}
    model.freeze_camera_for_flight_only()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    actual.square().sum().backward()
    optim.step()
    for name, value in model.named_parameters():
        if name in before:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            assert value.grad is None


def test_batched_teacher_uses_bounded_commands_and_completes_actual_dynamics():
    args = pole_args(batch_size=3, timesteps=120)
    env = build_env(3, args, torch.device("cpu"))
    torch.manual_seed(721220)
    env.reset("nominal")
    teacher = PoleNavigationTeacher(env, args)
    queue = [env.act] * 2
    alive = torch.ones(3, dtype=torch.bool)
    reached, collision = ~alive, ~alive
    for step in range(120):
        local, command = teacher.command(env, step / 15)
        assert local.abs().max() <= args.max_acc_cmd
        queue.append(command)
        env.run(queue[step], 1 / 15, env.p_target - env.p)
        hit = env.transition_clearance() <= args.collision_clearance
        goal = (env.p - env.p_target).norm(dim=-1) < .35
        collision |= alive & hit
        reached |= alive & ~hit & goal
        alive &= ~(hit | goal)
    assert reached.all() and not collision.any()


def test_recovery_data_keeps_teacher_labels_when_learner_flies():
    args = pole_args(timesteps=8, batch_size=2)
    actor = build_policy(args).eval()
    with torch.no_grad():
        actor.fc.weight.zero_()
        actor.fc.bias.copy_(torch.tensor([0., 10., 0.]))
    data, meta = collect(args, 2, 2, 6203, torch.device("cpu"),
                         policy=actor, teacher_probability=0.)
    assert data["frames"].shape[:3] == (2, 8, 1)
    assert data["action"].shape == (2, 8, 3)
    assert data["valid"].dtype == torch.bool
    assert not torch.allclose(data["action"][..., 1], torch.full((2, 8), args.max_acc_cmd))
    assert meta["collection_actor"] == "current learned policy with teacher mixing"
    assert meta["teacher_execution_probability"] == 0
    assert meta["generated_transitions"] == 16
    assert not meta["camera_parameter_labels"]
    assert len(dataset_configuration(args)["calibration_sha256"]) == 64


def test_initial_measurement_motion_has_no_flight_or_camera_labels():
    args = load_intent_args("configs/gray_joint_parallax.args", batch_size=2, timesteps=30,
                            gray_enable_noise=False, gray_width=20, gray_height=15,
                            gray_nn_width=16, gray_nn_height=12)
    args.imitation_initial_probe_steps, args.imitation_clearance_margin = 24, .06
    data, meta = collect(args, 2, 2, 6301, torch.device("cpu"))
    assert not data["valid"][:, :24].any()
    assert data["valid"][:, 24:].all()
    assert not data["action"][:, :24].any()
    assert meta["unlabelled_initial_probe_steps"] == 24
    assert not meta["initial_probe_motion_labels"] and not meta["camera_parameter_labels"]


def test_paired_behavior_evaluation_preserves_checkpoint_scenario_order(tmp_path):
    scenarios = ["dark_to_bright", "bright_to_dark", "dark", "bright"]
    args = pole_args(timesteps=4, loss_v_window=1, scenarios=scenarios)
    checkpoint = tmp_path / "final.pth"
    torch.save(build_policy(args).state_dict(), checkpoint)
    (tmp_path / "args.json").write_text(json.dumps(vars(args)))
    shutil.copy2(args.imx900_calibration, tmp_path / "imx900_calibration.json")
    result = evaluate(checkpoint, ["nominal", "no_motion_integration"], None,
                      1, 758000, torch.device("cpu"), tmp_path / "evaluation")
    for condition in result["conditions"].values():
        assert [r["scenario"] for r in condition["episodes"]] == scenarios
        assert [r["episode_seed"] for r in condition["episodes"]] == list(range(758000, 758004))
    for name in ("nominal", "no_motion_integration"):
        trace = json.loads((tmp_path / "evaluation" / f"{name}_traces.json").read_text())
        assert abs(trace[0]["requested_exposure"] - args.fixed_camera_exposure) < 1e-7
        assert abs(trace[0]["exposure_us"] - 732) < .001
        assert trace[0]["time_s"] == 0

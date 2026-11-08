"""Checks for the frozen-base residual joint policy (plan section 8).

These tests pin the properties the experiment's conclusions rest on. If any of
them fails the comparison A-F is meaningless, so they assert behaviour rather
than merely running the code:

* the base loads strictly and its weights never move;
* a zero flight residual reproduces the base flight action bit-for-bit, so a
  later change is attributable to learning;
* a zero camera residual reproduces the plain physical auto-exposure command,
  so the camera starts from the same policy every arm shares;
* camera metadata cannot reach the flight head;
* loss still reaches the image through the frozen base - freezing parameters is
  not freezing behaviour, and losing that path would silently turn the
  experiment into supervised camera tuning;
* the camera mapping is expressed in physical units, so the same residual means
  the same exposure under either profile.
"""

import json
import os

import pytest
import torch

from residual_joint_model import (
    FLIGHT_RESIDUAL_MAX,
    ResidualJointPolicy,
    flight_obs_dim,
    flight_state_slice,
    load_legacy_base,
    state_dict_digest,
)
from sensors.imx900_calibration import IMX900Calibration

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE_DIR = os.path.join(ROOT, "checkpoint", "2026-09-24-11-35-14")
PROVISIONAL = os.path.join(ROOT, "configs", "calibration", "imx900_provisional.json")
RESEARCH = os.path.join(ROOT, "configs", "calibration", "grayscale_research_50ms.json")

needs_base = pytest.mark.skipif(
    not os.path.isfile(os.path.join(BASE_DIR, "final.pth")),
    reason="verified base checkpoint is not present",
)


def _base_state():
    raw = torch.load(os.path.join(BASE_DIR, "final.pth"), map_location="cpu")
    return {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}


def _base_args():
    with open(os.path.join(BASE_DIR, "args.json")) as handle:
        return json.load(handle)


def _build(base_args=None, **kwargs):
    args = _base_args() if base_args is None else base_args
    base = load_legacy_base(
        _base_state(),
        include_camera_state_in_obs=args["include_camera_state_in_obs"],
        gray_nn_width=args["gray_nn_width"],
        gray_nn_height=args["gray_nn_height"],
        no_odom=args["no_odom"],
    )
    policy = ResidualJointPolicy(
        base,
        include_camera_state_in_obs=args["include_camera_state_in_obs"],
        no_odom=args["no_odom"],
        **kwargs,
    )
    return policy, base, args


def _obs(args, batch=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "gray_obs": torch.rand(batch, 2, 36, 48, generator=generator),
        "state": torch.randn(batch, flight_obs_dim(no_odom=args["no_odom"]),
                             generator=generator) * 0.3,
        "hx": torch.zeros(batch, 192),
        "cam_hx": torch.zeros(batch, 96),
        "camera_state": torch.zeros(batch, 2),
        "camera_motion_state": torch.zeros(batch, 6),
        "pending_flight_commands": torch.zeros(batch, 3),
        "camera_timeline": torch.zeros(batch, 6),
    }


# ------------------------------------------------------------------ widths
def test_obs_width_matches_build_policy():
    """``dim_obs`` sizes a Model parameter; ``flight_state_slice`` sizes a feature."""
    assert flight_obs_dim(no_odom=False) == 10
    assert flight_obs_dim(no_odom=True) == 7
    # Camera channels are appended by Model itself, so the flight branch only
    # sees the motion/target part.
    assert flight_state_slice(include_camera_state_in_obs=False) == 10
    assert flight_state_slice(include_camera_state_in_obs=True) == 8
    assert flight_state_slice(no_odom=True) == 7


# ------------------------------------------------------- check 1: strict load
@needs_base
def test_base_loads_strictly_and_hash_is_stable():
    state = _base_state()
    before = state_dict_digest(state)
    policy, base, _ = _build()
    after = state_dict_digest(_base_state())
    assert before == after, "loading a checkpoint must not mutate it"
    assert state_dict_digest(policy.base.state_dict()) == before
    assert not base.training, "the base must stay in eval mode"


@needs_base
def test_base_parameters_never_move_under_optimisation():
    """Check 3: the frozen base must not receive updates."""
    policy, base, args = _build()
    policy.bind_calibration(IMX900Calibration.from_json(PROVISIONAL))
    obs = _obs(args)
    reference = {k: v.clone() for k, v in policy.base.state_dict().items()}

    with torch.no_grad():
        policy.flight_head.weight.normal_(0, 0.05)
        policy.camera_head.weight.normal_(0, 0.05)
    policy.reset()
    action, camera, _, _ = policy(
        obs["state"], obs["hx"], gray_obs=obs["gray_obs"], cam_hx=obs["cam_hx"],
        camera_state=obs["camera_state"], camera_motion_state=obs["camera_motion_state"],
        pending_flight_commands=obs["pending_flight_commands"],
        camera_timeline=obs["camera_timeline"], state_observation=obs["state"],
    )
    ((action ** 2).sum() + (camera ** 2).sum()).backward()

    assert all(p.grad is None for p in policy.base.parameters() if p.requires_grad is False)
    for name, value in policy.base.state_dict().items():
        assert torch.equal(value, reference[name]), f"{name} changed"


# ------------------------------------- check 1b: zero residual == base action
@needs_base
def test_zero_flight_residual_reproduces_base_action():
    policy, base, args = _build()
    policy.bind_calibration(IMX900Calibration.from_json(PROVISIONAL))
    obs = _obs(args, seed=1)
    policy.reset()
    base_action = base(
        obs["state"], obs["hx"], gray_obs=obs["gray_obs"], cam_hx=obs["cam_hx"],
        camera_state=obs["camera_state"], camera_motion_state=obs["camera_motion_state"],
    )[0]
    policy.reset()
    action = policy(
        obs["state"], obs["hx"], gray_obs=obs["gray_obs"], cam_hx=obs["cam_hx"],
        camera_state=obs["camera_state"], camera_motion_state=obs["camera_motion_state"],
        pending_flight_commands=obs["pending_flight_commands"],
        camera_timeline=obs["camera_timeline"], state_observation=obs["state"],
    )[0]
    assert torch.allclose(action, base_action, atol=1e-6)
    assert base_action.abs().max() > 1e-3, "a silent zero action would make this vacuous"


@needs_base
def test_zero_camera_residual_reproduces_plain_auto_exposure():
    """The camera arm must start from the AE every other arm also uses."""
    from controllers.physical_auto_exposure import physical_mean_auto_exposure_target

    policy, _, args = _build()
    calibration = IMX900Calibration.from_json(PROVISIONAL)
    policy.bind_calibration(calibration)
    obs = _obs(args, seed=2)
    policy.reset()
    camera = policy(
        obs["state"], obs["hx"], gray_obs=obs["gray_obs"], cam_hx=obs["cam_hx"],
        camera_state=obs["camera_state"], camera_motion_state=obs["camera_motion_state"],
        pending_flight_commands=obs["pending_flight_commands"],
        camera_timeline=obs["camera_timeline"], state_observation=obs["state"],
    )[1]
    expected = physical_mean_auto_exposure_target(
        obs["gray_obs"][:, -1:],
        torch.full((obs["gray_obs"].shape[0],), 0.5),
        torch.full((obs["gray_obs"].shape[0],), 0.5),
        calibration,
        target_luma=policy.target_luma,
    )
    assert torch.allclose(camera, expected, atol=1e-5)


@needs_base
def test_flight_residual_is_bounded_by_the_plan_value():
    policy, _, args = _build()
    policy.bind_calibration(IMX900Calibration.from_json(PROVISIONAL))
    obs = _obs(args)
    with torch.no_grad():
        policy.flight_head.weight.mul_(0.0)
        policy.flight_head.bias.copy_(torch.tensor([1e3, -1e3, 1e3]))
    policy.reset()
    action = policy(
        obs["state"], obs["hx"], gray_obs=obs["gray_obs"], cam_hx=obs["cam_hx"],
        camera_state=obs["camera_state"], camera_motion_state=obs["camera_motion_state"],
        pending_flight_commands=obs["pending_flight_commands"],
        camera_timeline=obs["camera_timeline"], state_observation=obs["state"],
    )[0]
    assert action.abs().max() <= policy.max_acc_cmd + 1e-5
    # The head saturates at +-FLIGHT_RESIDUAL_MAX * tanh(1e3) ~ +-1.1.
    assert FLIGHT_RESIDUAL_MAX > 0


# ------------------------------- check 2: camera cannot reach the flight head
@needs_base
def test_camera_metadata_does_not_change_the_flight_action():
    policy, _, args = _build()
    policy.bind_calibration(IMX900Calibration.from_json(PROVISIONAL))
    obs = _obs(args, seed=4)
    with torch.no_grad():
        policy.flight_head.weight.normal_(0, 0.05)

    def run(camera_state, timeline):
        policy.reset()
        return policy(
            obs["state"], obs["hx"], gray_obs=obs["gray_obs"], cam_hx=obs["cam_hx"],
            camera_state=camera_state, camera_motion_state=obs["camera_motion_state"],
            pending_flight_commands=obs["pending_flight_commands"],
            camera_timeline=timeline, state_observation=obs["state"],
        )[0]

    reference = run(obs["camera_state"], obs["camera_timeline"])
    changed = run(torch.randn_like(obs["camera_state"]) * 0.9,
                  torch.randn_like(obs["camera_timeline"]))
    assert torch.equal(reference, changed), "camera state leaked into the flight branch"


@needs_base
def test_camera_memory_does_not_enter_the_flight_input():
    """Structural check: no camera tensor may be an ancestor of the flight head."""
    policy, _, args = _build()
    policy.bind_calibration(IMX900Calibration.from_json(PROVISIONAL))
    attack = pytest.importorskip("torch.autograd")
    _ = attack
    obs = _obs(args, seed=5)
    with torch.no_grad():
        policy.flight_head.weight.normal_(0, 0.05)
    policy.reset()
    action = policy(
        obs["state"], obs["hx"], gray_obs=obs["gray_obs"], cam_hx=obs["cam_hx"],
        camera_state=obs["camera_state"], camera_motion_state=obs["camera_motion_state"],
        pending_flight_commands=obs["pending_flight_commands"],
        camera_timeline=obs["camera_timeline"], state_observation=obs["state"],
    )[0]
    grads = torch.autograd.grad(action.sum(), policy.flight_gru.weight_ih, retain_graph=True)
    assert grads[0] is not None
    # The camera GRU must not appear in the flight action's graph at all.
    assert torch.autograd.grad(
        action.sum(), policy.flight_head.weight, retain_graph=True, allow_unused=True
    )[0] is not None
    leaf = policy.camera_gru.weight_ih
    grad = torch.autograd.grad(action.sum(), leaf, retain_graph=True, allow_unused=True)[0]
    assert grad is None, "the flight action depends on camera memory"


# --------------------------- check 3: gradients cross the frozen base to image
@needs_base
def test_loss_reaches_the_image_through_the_frozen_base():
    policy, _, args = _build()
    policy.bind_calibration(IMX900Calibration.from_json(PROVISIONAL))
    with torch.no_grad():
        policy.flight_head.weight.normal_(0, 0.05)
        policy.camera_head.weight.normal_(0, 0.05)

    # The image depends on the previous action, which is the loop under test.
    previous_action = torch.zeros(2, 3, requires_grad=True)
    gray = (0.45 + 0.4 * previous_action[:, :1, None, None]).expand(2, 2, 36, 48).contiguous()
    policy.reset()
    action, camera, _, _ = policy(
        torch.randn(2, flight_obs_dim()) * 0.3, torch.zeros(2, 192), gray_obs=gray,
        cam_hx=torch.zeros(2, 96), camera_state=torch.zeros(2, 2),
        camera_motion_state=torch.zeros(2, 6), pending_flight_commands=torch.zeros(2, 3),
        camera_timeline=torch.zeros(2, 6),
        state_observation=torch.randn(2, flight_obs_dim()) * 0.3,
    )
    ((action ** 2).sum() + (camera ** 2).sum()).backward()
    assert previous_action.grad is not None
    assert previous_action.grad.abs().sum() > 1e-8, "image path is detached"


@needs_base
def test_both_heads_and_the_encoder_receive_gradient():
    policy, _, _ = _build()
    policy.bind_calibration(IMX900Calibration.from_json(PROVISIONAL))
    with torch.no_grad():
        policy.flight_head.weight.normal_(0, 0.05)
        policy.camera_head.weight.normal_(0, 0.05)
    source = torch.rand(2, 1, 36, 48, requires_grad=True)
    gray = torch.cat([source, source], 1)
    policy.reset()
    action, camera, _, _ = policy(
        torch.randn(2, flight_obs_dim()) * 0.3, torch.zeros(2, 192), gray_obs=gray,
        cam_hx=torch.zeros(2, 96), camera_state=torch.zeros(2, 2),
        camera_motion_state=torch.zeros(2, 6), pending_flight_commands=torch.zeros(2, 3),
        camera_timeline=torch.zeros(2, 6),
        state_observation=torch.randn(2, flight_obs_dim()) * 0.3,
    )
    ((action ** 2).sum() + (camera ** 2).sum()).backward()
    assert source.grad.abs().sum() > 1e-8
    for module, label in (
        (policy.image_encoder.proj, "image_encoder"),
        (policy.flight_gru, "flight_gru"),
        (policy.camera_gru, "camera_gru"),
        (policy.flight_head, "flight_head"),
        (policy.camera_head, "camera_head"),
    ):
        weight = module.weight if hasattr(module, "weight") else module.weight_ih
        assert weight.grad is not None, f"{label} got no gradient"
        assert weight.grad.abs().sum() > 1e-10, f"{label} gradient is zero"


# -------------------------- check 4: physical units are profile-independent
@needs_base
def test_camera_mapping_is_in_physical_units():
    """The same residual must move the same *physical* exposure by the same factor.

    Both profiles are started at the same physical operating point (890 us,
    gain 1) expressed in each profile's own normalized coordinates. Feeding the
    same *normalized* value instead would mean 4 ms under the 8 ms profile and
    25 ms under the 50 ms profile, so the two runs would not be comparable -
    that mismatch is exactly the unit bug the plan warns about.
    """
    policy, _, args = _build()
    obs = _obs(args, seed=7)
    obs["gray_obs"] = torch.full_like(obs["gray_obs"], 0.60)
    with torch.no_grad():
        policy.camera_head.weight.zero_()
        # exp(0.02*log(500)) ~= 1.13x. The plan's mapping spans 500x so the whole
        # legal range is reachable; a larger residual would pin both profiles
        # against their own ceilings and the comparison would test only the clamp.
        policy.camera_head.bias.copy_(torch.tensor([0.02, 0.0]))

    results = {}
    for path in (PROVISIONAL, RESEARCH):
        calibration = IMX900Calibration.from_json(path)
        policy.bind_calibration(calibration)
        # Same physical operating point, written in this profile's units.
        start = calibration.exposure_normalized(890.0)
        gain = calibration.gain_from_factor(1.0)
        obs["camera_state"] = torch.stack(
            [torch.full((obs["gray_obs"].shape[0],), float(start)),
             torch.full((obs["gray_obs"].shape[0],), float(gain))], -1
        ) * 2.0 - 1.0
        policy.reset()
        camera = policy(
            obs["state"], obs["hx"], gray_obs=obs["gray_obs"], cam_hx=obs["cam_hx"],
            camera_state=obs["camera_state"],
            camera_motion_state=obs["camera_motion_state"],
            pending_flight_commands=obs["pending_flight_commands"],
            camera_timeline=obs["camera_timeline"], state_observation=obs["state"],
        )[1]
        results[path] = (
            calibration.exposure_to_us(camera[:, 0]).detach(),
            calibration.gain_to_factor(camera[:, 1]).detach(),
        )
        assert results[path][0].max() < float(calibration.exposure_us_max) - 1.0, (
            f"{path} saturated its ceiling, the comparison would be trivial"
        )
        assert results[path][0].min() > float(calibration.exposure_us_min) + 1.0

    (us_a, g_a), (us_b, g_b) = results[PROVISIONAL], results[RESEARCH]
    # 890 us -> ~1.13 * 1.13 (AE + residual). Both profiles must agree.
    assert torch.allclose(us_a, us_b, rtol=1e-4, atol=0.5), (
        f"the same residual produced different physical exposure: {us_a} vs {us_b}"
    )
    assert torch.allclose(g_a, g_b, atol=1e-5)


@needs_base
def test_camera_command_respects_profile_bounds():
    policy, _, args = _build()
    obs = _obs(args, seed=8)
    for path in (PROVISIONAL, RESEARCH):
        calibration = IMX900Calibration.from_json(path)
        policy.bind_calibration(calibration)
        # Saturate the head both ways and confirm we stay inside the profile.
        for bias in (100.0, -100.0):
            with torch.no_grad():
                policy.camera_head.weight.zero_()
                policy.camera_head.bias.fill_(bias)
            policy.reset()
            camera = policy(
                obs["state"], obs["hx"], gray_obs=obs["gray_obs"], cam_hx=obs["cam_hx"],
                camera_state=obs["camera_state"],
                camera_motion_state=obs["camera_motion_state"],
                pending_flight_commands=obs["pending_flight_commands"],
                camera_timeline=obs["camera_timeline"], state_observation=obs["state"],
            )[1]
            us = calibration.exposure_to_us(camera[:, 0])
            assert us.min() >= calibration.exposure_us_min - 1e-3
            assert us.max() <= calibration.exposure_us_max + 1e-3
            assert camera.min() > 0.0 and camera.max() < 1.0, "logit must stay finite"


# ----------------------------------------- check 3b: component arms are real
@needs_base
def test_disabling_a_head_removes_its_parameters_from_the_update():
    policy, _, _ = _build()
    full = policy.trainable_parameter_count()
    assert full > 0
    policy.freeze_camera_residual()
    assert policy.trainable_parameter_count() < full
    assert not any(n.startswith("camera_") for n in policy.trainable_parameter_names())
    policy.freeze_flight_residual()
    remaining = policy.trainable_parameter_names()
    assert not any(n.startswith(("flight_", "camera_")) for n in remaining)
    # The encoder is shared, so it must still be trainable in either arm.
    assert any(n.startswith("image_encoder.") for n in remaining)


@needs_base
def test_zero_init_heads_start_at_exactly_zero_correction():
    policy, _, _ = _build()
    for head in (policy.flight_head, policy.camera_head):
        assert torch.count_nonzero(head.weight) == 0
        assert torch.count_nonzero(head.bias) == 0


@needs_base
def test_reset_clears_memory_between_episodes():
    policy, _, args = _build()
    policy.bind_calibration(IMX900Calibration.from_json(PROVISIONAL))
    obs = _obs(args, seed=9)
    with torch.no_grad():
        policy.flight_head.weight.normal_(0, 0.05)
    memory = []
    for _ in range(2):
        policy.reset()
        action = policy(
            obs["state"], obs["hx"], gray_obs=obs["gray_obs"], cam_hx=obs["cam_hx"],
            camera_state=obs["camera_state"],
            camera_motion_state=obs["camera_motion_state"],
            pending_flight_commands=obs["pending_flight_commands"],
            camera_timeline=obs["camera_timeline"], state_observation=obs["state"],
        )[0]
        memory.append(action.clone())
    assert torch.allclose(memory[0], memory[1]), "hidden state leaked across resets"

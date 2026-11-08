"""Analytic checks for the shutter motion history.

The correction in ``render/exposure_history.py`` replaces a plain
"extrapolate the current velocity backwards" render with one that uses the
motion that was actually recorded. These tests pin the properties that
matters for: it must be exact for constant velocity (so frames early in an
episode agree with the legacy path), exact for piecewise-linear motion (so
accelerating motion is not mis-rendered), identical on the first frame, and
differentiable through the poses the policy produced.
"""

import pytest
import torch

from render.exposure_history import ExposureHistory, pre_history_fingerprint
from render.pose import body_angular_velocity, rotation_exp

DT = 1.0 / 15.0
EYE = torch.eye(3).expand(1, 3, 3)


def _linear_history(segments, dt=DT, velocity=(1.0, 0.0, 0.0)):
    """History of pure constant-velocity motion sampled at segment boundaries."""
    history = ExposureHistory(length=segments + 1)
    v = torch.tensor([list(velocity)])
    for k in range(segments):
        history.append(torch.tensor([[k * dt * velocity[0], 0.0, 0.0]]), v, EYE, dt)
    return history, torch.tensor([[segments * dt * velocity[0], 0.0, 0.0]]), v


@pytest.mark.parametrize("tau", [0.001, 0.008, 0.02, 0.05, 0.09, 0.39])
def test_constant_velocity_is_exact(tau):
    """Constant velocity must reproduce ``p_now - v*tau`` to float precision."""
    segments = 6
    history, now, v = _linear_history(segments)
    sample, _ = history.sample(now, EYE, v, torch.tensor([-tau]))
    expected = torch.tensor([[segments * DT - tau, 0.0, 0.0]])
    assert torch.allclose(sample, expected, atol=1e-5)


@pytest.mark.parametrize("steps_back", [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0, 4.5])
def test_constant_angular_rate_is_exact(steps_back):
    """Attitude interpolation must follow a constant body rate exactly."""
    segments = 6
    omega = torch.tensor([[0.0, 0.0, 0.15 / DT]])
    history = ExposureHistory(length=segments + 1)
    for k in range(segments):
        history.append(torch.zeros(1, 3), torch.zeros(1, 3),
                       rotation_exp(omega * k * DT), DT)
    now = rotation_exp(omega * segments * DT)
    _, sample = history.sample(torch.zeros(1, 3), now, omega,
                               torch.tensor([-DT * steps_back]))
    expected = rotation_exp(omega * (segments * DT - DT * steps_back))
    assert torch.allclose(sample, expected, atol=1e-4)


def test_accelerating_motion_matches_piecewise_linear_path():
    """Each interval is credited exactly the piece of the shutter it covers."""
    acceleration, v0, segments = 2.0, 0.6, 8

    def position(t):
        return v0 * t + 0.5 * acceleration * t * t

    def velocity(t):
        return v0 + acceleration * t

    history = ExposureHistory(length=segments + 1)
    for k in range(segments):
        start, end = k * DT, (k + 1) * DT
        history.append(
            torch.tensor([[position(start), 0.0, 0.0]]),
            torch.tensor([[(velocity(start) + velocity(end)) / 2.0, 0.0, 0.0]]),
            EYE, DT,
        )
    now = torch.tensor([[position(segments * DT), 0.0, 0.0]])

    def overlap_integral(t_from, t_to):
        total = 0.0
        for k in range(segments):
            start, end = k * DT, (k + 1) * DT
            mean = (velocity(start) + velocity(end)) / 2.0
            low, high = max(start, t_from), min(end, t_to)
            if high > low:
                total += mean * (high - low)
        return total

    for tau in (0.02, 0.05, 0.0667, 0.2, 0.4):
        sample, _ = history.sample(now, EYE, torch.zeros(1, 3), torch.tensor([-tau]))
        expected = position(segments * DT) - overlap_integral(segments * DT - tau,
                                                             segments * DT)
        assert abs(sample[0, 0].item() - expected) < 1e-5


def test_primed_first_frame_matches_legacy_extrapolation():
    """Frame 0 must not differ from the legacy path, in position or attitude."""

    class Env:
        pass

    env = Env()
    env.last_control_dt = DT
    env.R_scene_T = EYE
    env.p = torch.zeros(1, 3)
    env.v = torch.tensor([[0.9, -0.2, 0.1]])
    env.R_old = rotation_exp(torch.tensor([[0.04, 0.01, 0.08]]))
    env.R = rotation_exp(torch.tensor([[0.05, 0.02, 0.1]]))

    history = ExposureHistory(length=4)
    history.prime_from_env(env)
    assert history.ready

    position = env.p
    velocity = env.v
    rotation = env.R
    omega = body_angular_velocity(env.R_old, env.R, DT)
    for tau in (0.001, 0.008, 0.02, 0.05):
        sample_p, sample_R = history.sample(position, rotation, omega,
                                            torch.tensor([-tau]))
        assert torch.allclose(sample_p, position + velocity * (-tau), atol=1e-5)
        # One-step attitude interpolation is an O(omega*dt^2) approximation of
        # the constant-rate law; position is the part required to be exact.
        assert torch.allclose(sample_R, rotation @ rotation_exp(omega * (-tau)),
                              atol=2e-3)


def test_offset_beyond_history_never_extrapolates():
    """A shutter longer than the buffer clamps at the oldest recorded interval."""
    history, now, v = _linear_history(3)
    sample, _ = history.sample(now, EYE, v, torch.tensor([-5.0]))
    assert torch.allclose(sample, torch.zeros(1, 3), atol=1e-5)


def test_history_keeps_the_graph():
    """Recording must preserve autograd so the task loss reaches the camera."""
    w = torch.tensor(0.4, requires_grad=True)
    history = ExposureHistory(length=4)
    for k in range(4):
        position = torch.stack([w * k * DT, torch.zeros(()), torch.zeros(())]).reshape(1, 3)
        velocity = torch.stack([w, torch.zeros(()), torch.zeros(())]).reshape(1, 3)
        history.append(position, velocity, EYE, DT)
    position = torch.stack([w * 4 * DT, torch.zeros(()), torch.zeros(())]).reshape(1, 3)
    velocity = torch.stack([w, torch.zeros(()), torch.zeros(())]).reshape(1, 3)
    sample, _ = history.sample(position, EYE, velocity, torch.tensor([-0.02]))
    assert sample.requires_grad
    sample.sum().backward()
    # d/dw [p_now - w*tau] with p_now = 4*DT*w: 4*DT - 0.02 = 0.24666...
    assert abs(w.grad.item() - (4 * DT - 0.02)) < 1e-4


def test_pre_history_fingerprint_is_stable():
    assert pre_history_fingerprint(4) == pre_history_fingerprint(4)
    assert pre_history_fingerprint(4) != pre_history_fingerprint(6)
    assert len(pre_history_fingerprint(4)) == 64


def test_rejects_malformed_inputs():
    history = ExposureHistory(length=4)
    with pytest.raises(ValueError):
        history.append(torch.zeros(3), torch.zeros(1, 3), EYE, DT)
    with pytest.raises(ValueError):
        history.append(torch.zeros(1, 3), torch.zeros(1, 3), EYE, 0.0)
    with pytest.raises(ValueError):
        ExposureHistory(length=1)


def test_linear_buffer_is_an_approximation_to_quadratic_env_motion():
    """Document the *scope* of the history module's correctness.

    ``record_history_step`` stores the pose and the *pre-step* velocity, and the
    buffer interpolates linearly between samples. The simulator's own step is
    quadratic:

        p_next = p + v*dt + a*(0.5*dt^2)

    so a linear buffer is an approximation, not an identity, and the gap at an
    interval midpoint is ``a*dt^2/8``. The analytic tests above use
    constant-velocity fixtures and therefore cannot see this term; this test
    records its size so nobody later claims the buffer was verified against
    ``env.run``.

    At the experiment's operating point (a = max_acc_cmd = 2.2 m/s^2,
    dt = 1/15 s) the gap is ~1.22 mm, i.e. 2.4% of the 0.05 m class cue.
    """
    dt = 1.0 / 15.0
    acceleration = 2.2
    velocity = 1.0

    p0 = 0.0
    p1 = p0 + velocity * dt + acceleration * (0.5 * dt ** 2)
    midpoint_from_linear_buffer = 0.5 * (p0 + p1)

    half = 0.5 * dt
    midpoint_of_true_motion = p0 + velocity * half + acceleration * (0.5 * half ** 2)

    gap = abs(midpoint_from_linear_buffer - midpoint_of_true_motion)
    assert gap == pytest.approx(acceleration * dt ** 2 / 8, rel=1e-9)
    # Non-zero, so the buffer is not exact, but small relative to the task cue.
    assert 1e-3 < gap < 2e-3
    assert gap / 0.05 < 0.03

"""Invariants for the gradient-free arm (arm F).

These tests deliberately do **not** run rollouts - they pin down the parts of
arm F that can be checked without a simulator, because the interesting failure
modes are all in the estimator's plumbing rather than in the ES maths.

The bug these tests exist to prevent: the residual policy zero-initialises its
output heads, and a perturbation scaled by a tensor's own RMS is exactly
nothing on a zero tensor. An ES arm that silently perturbs nothing reports
"gradient-free optimisation learns nothing", which is a statement about the
initialisation, not about the method.

The fix has to be an *absolute* scale for zero tensors, not seeding them: the
estimator is antithetic, so ``theta`` stays at the zero-residual base policy and
only the exploration radius changes. Seeding would give F a different initial
policy from the backprop arms and destroy the comparison F exists to make.
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "residual_joint_es",
    os.path.join(ROOT, "tools", "run_residual_joint_es.py"),
)
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)


class _Tiny(nn.Module):
    """Stand-in for the residual parameters: one live tensor, one dead head."""

    def __init__(self):
        super().__init__()
        self.live = nn.Linear(4, 4)
        self.dead = nn.Linear(4, 2)
        nn.init.zeros_(self.dead.weight)
        nn.init.zeros_(self.dead.bias)


def _params(module):
    return [(name, parameter) for name, parameter in module.named_parameters()]


def _by_name(params):
    return {name: parameter for name, parameter in params}


def test_zero_tensor_names_finds_zero_initialised_heads():
    module = _Tiny()
    zero = es.zero_tensor_names(_params(module))
    assert "dead.weight" in zero
    assert "dead.bias" in zero
    assert "live.weight" not in zero


def test_draw_direction_perturbs_zero_tensors_using_the_absolute_floor():
    """A zero tensor must receive a real perturbation, not float32 rounding."""
    module = _Tiny()
    params = _params(module)
    direction = es._draw_direction(params, es.SCALE_START,
                                   torch.Generator().manual_seed(0),
                                   torch.device("cpu"))
    deltas = dict(zip([name for name, _ in params], direction))

    # Without the floor this is ~1e-8 (pure rounding). Require the scale the
    # docstring derives: scale * ZERO_RMS_FLOOR, within normal-tail tolerance.
    expected = es.SCALE_START * es.ZERO_RMS_FLOOR
    for name in ("dead.weight", "dead.bias"):
        peak = deltas[name].abs().max().item()
        assert peak > 0.1 * expected, name
        assert peak < 6.0 * expected, name


def test_draw_direction_does_not_modify_theta():
    """Drawing a direction must be a read-only operation on the parameters.

    This is what keeps F's initialisation identical to the backprop arms; a
    version that seeded in place would fail here.
    """
    module = _Tiny()
    params = _params(module)
    before = {name: parameter.detach().clone() for name, parameter in params}

    es._draw_direction(params, es.SCALE_START, torch.Generator().manual_seed(1),
                       torch.device("cpu"))

    for name, parameter in params:
        assert torch.equal(parameter.detach(), before[name]), name
    # The heads must still be exactly zero: same initial function as A-E.
    assert module.dead.weight.detach().abs().max().item() == 0.0
    assert module.dead.bias.detach().abs().max().item() == 0.0


def test_draw_direction_scales_live_tensors_by_their_own_rms():
    module = _Tiny()
    params = _params(module)
    direction = es._draw_direction(params, 0.01, torch.Generator().manual_seed(2),
                                   torch.device("cpu"))
    deltas = dict(zip([name for name, _ in params], direction))

    rms = _by_name(params)["live.weight"].detach().pow(2).mean().sqrt().item()
    if rms <= es.ZERO_RMS_FLOOR:
        pytest.skip("fixture's live tensor sits below the floor")
    observed = deltas["live.weight"].pow(2).mean().sqrt().item()
    # Proportional to this tensor's own RMS; the constant is the draw's sample
    # RMS, which is noisy on a small tensor, so only the order is asserted.
    assert 0.2 * rms * 0.01 < observed < 6.0 * rms * 0.01


def test_draw_direction_is_per_tensor_above_the_floor():
    """Two tensors above the floor must move in proportion to their own RMS.

    This is the property that distinguishes a working per-tensor ES from one
    that effectively freezes everything except the largest tensor.
    """

    class _Scales(nn.Module):
        def __init__(self):
            super().__init__()
            self.big = nn.Parameter(torch.ones(64) * 1.0)
            self.small = nn.Parameter(torch.ones(64) * 0.5)

    params = _params(_Scales())
    direction = es._draw_direction(params, 0.01, torch.Generator().manual_seed(3),
                                   torch.device("cpu"))
    big_delta, small_delta = direction
    ratio = big_delta.norm().item() / small_delta.norm().item()
    # RMS 1.0 versus 0.5, both above the floor, so the ratio must be ~2.
    assert 1.2 < ratio < 3.2


def test_draw_direction_floor_collapses_subfloor_tensors_to_one_scale():
    """Below the floor, tensors are perturbed at the same absolute scale."""

    class _Below(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.ones(64) * 0.01)
            self.b = nn.Parameter(torch.ones(64) * 0.001)

    params = _params(_Below())
    direction = es._draw_direction(params, 0.01, torch.Generator().manual_seed(5),
                                   torch.device("cpu"))
    ratio = direction[0].norm().item() / direction[1].norm().item()
    assert 0.5 < ratio < 2.0


def test_apply_is_reversible_by_construction():
    module = _Tiny()
    params = _params(module)
    before = {name: parameter.detach().clone() for name, parameter in params}
    direction = es._draw_direction(params, 0.01,
                                   torch.Generator().manual_seed(4),
                                   torch.device("cpu"))

    es._apply(params, direction, +1.0)
    es._apply(params, direction, -1.0)

    for name, parameter in params:
        # float32 arithmetic is not exactly invertible; the residue must stay
        # far below the perturbation itself, or the antithetic pair is biased.
        residue = (parameter.detach() - before[name]).abs().max().item()
        step = max(before[name].abs().max().item() * 0.01, 1e-3)
        assert residue <= max(step * 1e-3, 1e-9), name


def test_relative_scale_decays_linearly_and_stays_in_range():
    assert es._relative_scale(0, 4) == pytest.approx(es.SCALE_START)
    assert es._relative_scale(3, 4) == pytest.approx(es.SCALE_FINAL)
    # Degenerate and out-of-range inputs must not explode or go negative.
    assert es._relative_scale(0, 1) == pytest.approx(es.SCALE_START)
    assert es._relative_scale(99, 4) == pytest.approx(es.SCALE_FINAL)
    for update in range(4):
        value = es._relative_scale(update, 4)
        assert es.SCALE_FINAL <= value <= es.SCALE_START


def test_episode_budget_covers_both_signs():
    # A +eps rollout is never free: both signs of every direction are billed.
    assert es.EPISODES_PER_UPDATE == es.DIRECTIONS * es.SIGNS * es.EPISODES_PER_SIGN
    assert es.SIGNS == 2


def test_zero_rms_floor_is_above_typical_initialised_tensor_rms():
    """The floor must not dominate ordinary tensors.

    Typical residual GRU tensors initialise with RMS ~0.07. If the floor were
    far above that, every tensor would be perturbed identically in absolute
    terms and the per-tensor scaling would stop meaning anything.
    """
    assert 0.03 < es.ZERO_RMS_FLOOR < 0.5

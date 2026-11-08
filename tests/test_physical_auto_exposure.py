"""Tests for the physical-unit auto-exposure controllers.

The point of these controllers is that a step means the same relative change
under any calibration profile. That is what these tests pin: the same law must
reach the same *physical* exposure on the 8 ms and 50 ms profiles, and it must
not depend on the profile's coordinate range.
"""

import math

import pytest
import torch

from controllers.physical_auto_exposure import (
    physical_gradient_auto_exposure_target,
    physical_mean_auto_exposure_target,
)
from sensors.imx900_calibration import IMX900Calibration

PROVISIONAL = "configs/calibration/imx900_provisional.json"
RESEARCH_50MS = "configs/calibration/grayscale_research_50ms.json"


@pytest.fixture(scope="module", params=[PROVISIONAL, RESEARCH_50MS])
def profile(request):
    return IMX900Calibration.from_json(request.param)


def _sensor_luma(calibration, exposure01, gain01, irradiance):
    """Locally linear sensor: brightness = irradiance * T * gain_factor."""
    us = calibration.exposure_to_us(torch.as_tensor(exposure01))
    factor = calibration.gain_to_factor(torch.as_tensor(gain01))
    return (torch.as_tensor(irradiance) * us * factor).clamp(0.0, 1.0)


def _settle(calibration, irradiance, steps=60, target=0.35):
    exposure = torch.tensor([0.5])
    gain = torch.tensor([0.2])
    history = []
    for _ in range(steps):
        luma = _sensor_luma(calibration, exposure, gain, irradiance)
        history.append(float(luma))
        update = physical_mean_auto_exposure_target(
            luma.reshape(1, 1, 1, 1), exposure, gain, calibration
        )
        exposure, gain = update[0, 0:1], update[0, 1:2]
    return history, exposure, gain


def test_mean_ae_reaches_the_target(profile):
    history, _, _ = _settle(profile, irradiance=1.5e-4)
    assert abs(history[-1] - 0.35) < 2e-3


def test_mean_ae_converges_to_the_same_physical_exposure_on_both_profiles():
    """The same scene must settle on the same microseconds under any profile."""
    results = {}
    for path in (PROVISIONAL, RESEARCH_50MS):
        calibration = IMX900Calibration.from_json(path)
        _, exposure, _ = _settle(calibration, irradiance=1.5e-4)
        results[path] = calibration.exposure_to_us(exposure).item()
    assert abs(results[PROVISIONAL] - results[RESEARCH_50MS]) < 1.0


def test_mean_ae_is_monotone_in_irradiance(profile):
    """Brighter scenes must settle on shorter exposures."""
    exposures = []
    for irradiance in (1.0e-4, 2.0e-4, 4.0e-4):
        _, exposure, _ = _settle(profile, irradiance=irradiance)
        exposures.append(profile.exposure_to_us(exposure).item())
    assert exposures[0] > exposures[1] > exposures[2]


def test_mean_ae_increases_exposure_for_a_dark_frame(profile):
    dark = torch.full((2, 1, 8, 8), 0.05)
    exposure = torch.full((2,), 0.3)
    gain = torch.full((2,), 0.2)
    update = physical_mean_auto_exposure_target(dark, exposure, gain, profile)
    assert torch.all(update[:, 0] > exposure)


def test_mean_ae_decreases_exposure_for_a_bright_frame(profile):
    bright = torch.full((1, 1, 8, 8), 0.90)
    update = physical_mean_auto_exposure_target(
        bright, torch.tensor([0.6]), torch.tensor([0.3]), profile
    )
    assert update[0, 0] < 0.6


def test_mean_ae_uses_gain_only_when_exposure_is_pinned(profile):
    """An almost-black frame with exposure at the ceiling must move gain."""
    dark = torch.full((1, 1, 8, 8), 0.001)
    update = physical_mean_auto_exposure_target(
        dark, torch.tensor([1.0]), torch.tensor([0.0]), profile
    )
    assert update[0, 0].item() == pytest.approx(1.0)
    assert update[0, 1] > 0.0


def test_mean_ae_leaves_gain_alone_while_exposure_can_still_move(profile):
    dark = torch.full((1, 1, 8, 8), 0.05)
    update = physical_mean_auto_exposure_target(
        dark, torch.tensor([0.2]), torch.tensor([0.0]), profile
    )
    assert update[0, 1].item() == pytest.approx(0.0, abs=1e-6)


def test_controllers_are_detached_and_bounded(profile):
    gray = torch.rand((3, 1, 8, 8), requires_grad=True)
    exposure = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    gain = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    for fn in (physical_mean_auto_exposure_target,
               physical_gradient_auto_exposure_target):
        update = fn(gray, exposure, gain, profile)
        assert not update.requires_grad
        assert torch.all((update >= 0.0) & (update <= 1.0))


def test_gradient_ae_brightens_an_underexposed_edge(profile):
    gray = torch.zeros((1, 1, 16, 16))
    gray[:, :, :, 8:] = 0.04
    update = physical_gradient_auto_exposure_target(
        gray, torch.tensor([0.3]), torch.tensor([0.1]), profile
    )
    assert update.shape == (1, 2)
    assert update[0, 0] >= 0.3


def test_gradient_ae_steps_are_octaves_of_exposure_time(profile):
    """A doubling candidate must land at most ``exposure_step_octaves`` away."""
    gray = torch.zeros((1, 1, 16, 16))
    gray[:, :, :, 4:] = 0.30
    exposure = torch.tensor([0.4])
    update = physical_gradient_auto_exposure_target(
        gray, exposure, torch.tensor([0.1]), profile, exposure_step_octaves=0.25
    )
    current_us = profile.exposure_to_us(exposure).item()
    target_us = profile.exposure_to_us(update[0, 0]).item()
    assert abs(math.log2(target_us / current_us)) <= 0.25 + 1e-6


def test_missing_calibration_is_a_clear_error():
    with pytest.raises(TypeError):
        physical_mean_auto_exposure_target(
            torch.rand(1, 1, 4, 4), torch.tensor([0.5]), torch.tensor([0.5]),
            object(),
        )

import torch

from controllers.auto_exposure import (
    gradient_auto_exposure_target,
    mean_auto_exposure_target,
)


def test_mean_ae_increases_exposure_for_dark_image():
    gray = torch.full((2, 1, 8, 8), 0.05)
    exposure = torch.full((2,), 0.3)
    gain = torch.full((2,), 0.2)
    target = mean_auto_exposure_target(gray, exposure, gain)
    assert target.shape == (2, 2)
    assert torch.all(target[:, 0] > exposure)


def test_mean_ae_decreases_exposure_for_bright_image():
    gray = torch.full((1, 1, 8, 8), 0.9)
    exposure = torch.tensor([0.6])
    gain = torch.tensor([0.3])
    target = mean_auto_exposure_target(gray, exposure, gain)
    assert target[0, 0] < exposure[0]


def test_gradient_ae_prefers_brightening_underexposed_edges():
    gray = torch.zeros((1, 1, 16, 16))
    gray[:, :, :, 8:] = 0.04
    exposure = torch.tensor([0.3])
    gain = torch.tensor([0.1])
    target = gradient_auto_exposure_target(gray, exposure, gain)
    assert target.shape == (1, 2)
    assert target[0, 0] >= exposure[0]


def test_classical_controllers_are_detached_and_bounded():
    gray = torch.rand((3, 1, 8, 8), requires_grad=True)
    exposure = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    gain = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    for fn in (mean_auto_exposure_target, gradient_auto_exposure_target):
        target = fn(gray, exposure, gain)
        assert not target.requires_grad
        assert torch.all((target >= 0.0) & (target <= 1.0))

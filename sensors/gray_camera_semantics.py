"""Normalized <-> physical semantics for the grayscale active camera.

The defaults in this module are *provisional simulation defaults*. They are not
claimed to be the factory limits of the Sony IMX900/e-con camera. Replace them
with values measured/reported by the deployed driver after hardware bring-up.
"""

from dataclasses import dataclass
from typing import Union

import torch

TensorOrFloat = Union[torch.Tensor, float]


def _clamp01(value: TensorOrFloat) -> TensorOrFloat:
    if isinstance(value, torch.Tensor):
        return value.clamp(0.0, 1.0)
    return min(max(float(value), 0.0), 1.0)


@dataclass(frozen=True)
class GrayCameraSemantics:
    """Mapping between normalized policy actions and camera-model quantities."""

    # Provisional simulation range. Override from calibration/driver metadata.
    exposure_us_min: float = 100.0
    exposure_us_max: float = 8000.0
    exposure_reference_us: float = 1000.0

    # Effective amplification used by the differentiable sensor model.
    # This is intentionally not labelled dB or a V4L2 control unit.
    gain_factor_min: float = 1.0
    gain_factor_max: float = 8.0

    def __post_init__(self):
        if self.exposure_us_min <= 0:
            raise ValueError("exposure_us_min must be > 0")
        if self.exposure_us_max <= self.exposure_us_min:
            raise ValueError("exposure_us_max must be > exposure_us_min")
        if self.exposure_reference_us <= 0:
            raise ValueError("exposure_reference_us must be > 0")
        if self.gain_factor_min <= 0:
            raise ValueError("gain_factor_min must be > 0")
        if self.gain_factor_max < self.gain_factor_min:
            raise ValueError("gain_factor_max must be >= gain_factor_min")

    def exposure_to_us(self, exposure01: TensorOrFloat) -> TensorOrFloat:
        """Linearly map a normalized exposure command to microseconds."""
        x = _clamp01(exposure01)
        return self.exposure_us_min + (
            self.exposure_us_max - self.exposure_us_min
        ) * x

    def exposure_to_relative(self, exposure01: TensorOrFloat) -> TensorOrFloat:
        """Exposure relative to the calibration reference exposure."""
        return self.exposure_to_us(exposure01) / self.exposure_reference_us

    def gain_to_factor(self, gain01: TensorOrFloat) -> TensorOrFloat:
        """Smooth log-domain interpolation of effective sensor gain."""
        x = _clamp01(gain01)
        ratio = self.gain_factor_max / self.gain_factor_min
        if isinstance(x, torch.Tensor):
            return self.gain_factor_min * torch.exp(
                x * math_log(ratio, device=x.device, dtype=x.dtype)
            )
        return self.gain_factor_min * (ratio ** float(x))


def math_log(value: float, *, device=None, dtype=None):
    """Return log(value) as float or tensor without creating CPU/device mismatches."""
    import math

    out = math.log(float(value))
    if device is None and dtype is None:
        return out
    return torch.as_tensor(out, device=device, dtype=dtype)


def from_args(args) -> GrayCameraSemantics:
    return GrayCameraSemantics(
        exposure_us_min=float(args.gray_exposure_us_min),
        exposure_us_max=float(args.gray_exposure_us_max),
        exposure_reference_us=float(args.gray_exposure_reference_us),
        gain_factor_min=float(args.gray_gain_factor_min),
        gain_factor_max=float(args.gray_gain_factor_max),
    )

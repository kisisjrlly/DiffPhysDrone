"""Calibration profile for the e-con/Sony IMX900 differentiable camera.

The project intentionally separates:
1) policy actions in normalized [0, 1] coordinates;
2) a real-camera calibration profile;
3) the differentiable surrogate that consumes those physical quantities.

The default JSON shipped with the repository is explicitly *provisional*.
It must not be reported as measured IMX900 behavior.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import List, Optional, Union

import torch

TensorOrFloat = Union[torch.Tensor, float]


def _clamp01(value: TensorOrFloat) -> TensorOrFloat:
    if isinstance(value, torch.Tensor):
        return value.clamp(0.0, 1.0)
    return min(max(float(value), 0.0), 1.0)


def _ste_quantize(value: TensorOrFloat, step: float) -> TensorOrFloat:
    """Quantize in forward while preserving identity gradient for tensors."""
    step = float(step)
    if step <= 0.0:
        return value
    if isinstance(value, torch.Tensor):
        snapped = torch.round(value / step) * step
        return value + (snapped - value).detach()
    return round(float(value) / step) * step


@dataclass(frozen=True)
class IMX900Calibration:
    schema_version: int = 1
    profile_name: str = "IMX900 provisional"
    calibrated: bool = False
    source: str = "unmeasured placeholder"

    exposure_us_min: float = 100.0
    exposure_us_max: float = 8000.0
    exposure_reference_us: float = 1000.0
    exposure_step_us: float = 0.0
    signal_scale: float = 1.0

    gain_mapping: str = "log"
    gain_factor_min: float = 1.0
    gain_factor_max: float = 8.0
    gain_factor_step: float = 0.0
    gain_lut_x: Optional[List[float]] = None
    gain_lut_factor: Optional[List[float]] = None

    shot_noise_alpha: float = 0.015
    shot_noise_beta: float = 0.0
    read_noise_std_base: float = 0.005
    read_noise_gain_exponent: float = 1.0

    black_level: float = 0.0
    saturation_level: float = 1.0
    quantization_bits: int = 0
    blur_scale: float = 0.08
    command_delay_frames: int = 0

    def __post_init__(self):
        if int(self.schema_version) != 1:
            raise ValueError("unsupported IMX900 calibration schema_version")
        if self.exposure_us_min <= 0 or self.exposure_us_max <= self.exposure_us_min:
            raise ValueError("invalid exposure range")
        if self.exposure_reference_us <= 0 or self.exposure_step_us < 0:
            raise ValueError("invalid exposure reference/step")
        if self.signal_scale <= 0:
            raise ValueError("signal_scale must be > 0")
        if self.gain_factor_min <= 0 or self.gain_factor_max < self.gain_factor_min:
            raise ValueError("invalid gain factor range")
        if self.gain_factor_step < 0:
            raise ValueError("gain_factor_step must be >= 0")
        if self.gain_mapping not in {"linear", "log", "lut"}:
            raise ValueError("gain_mapping must be linear, log, or lut")
        if self.shot_noise_alpha < 0 or self.shot_noise_beta < 0:
            raise ValueError("shot noise parameters must be >= 0")
        if self.read_noise_std_base < 0 or self.read_noise_gain_exponent < 0:
            raise ValueError("read-noise parameters must be >= 0")
        if self.saturation_level <= 0:
            raise ValueError("saturation_level must be > 0")
        if self.quantization_bits < 0:
            raise ValueError("quantization_bits must be >= 0")
        if self.blur_scale < 0:
            raise ValueError("blur_scale must be >= 0")
        if self.command_delay_frames < 0:
            raise ValueError("command_delay_frames must be >= 0")
        if self.gain_mapping == "lut":
            if not self.gain_lut_x or not self.gain_lut_factor:
                raise ValueError("lut gain mapping requires gain_lut_x/factor")
            if len(self.gain_lut_x) != len(self.gain_lut_factor) or len(self.gain_lut_x) < 2:
                raise ValueError("gain LUT arrays must have equal length >= 2")
            if abs(self.gain_lut_x[0]) > 1e-9 or abs(self.gain_lut_x[-1] - 1.0) > 1e-9:
                raise ValueError("gain_lut_x must span normalized [0, 1]")
            if any(b <= a for a, b in zip(self.gain_lut_x, self.gain_lut_x[1:])):
                raise ValueError("gain_lut_x must be strictly increasing")
            if any(v <= 0 for v in self.gain_lut_factor):
                raise ValueError("gain_lut_factor entries must be > 0")

    @classmethod
    def from_dict(cls, data):
        exposure = data.get("exposure", {})
        gain = data.get("gain", {})
        noise = data.get("noise", {})
        motion_blur = data.get("motion_blur", {})
        actuator = data.get("actuator", {})
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            profile_name=str(data.get("profile_name", "IMX900 calibration")),
            calibrated=bool(data.get("calibrated", False)),
            source=str(data.get("source", "")),
            exposure_us_min=float(exposure.get("min_us", 100.0)),
            exposure_us_max=float(exposure.get("max_us", 8000.0)),
            exposure_reference_us=float(exposure.get("reference_us", 1000.0)),
            exposure_step_us=float(exposure.get("step_us", 0.0)),
            signal_scale=float(exposure.get("signal_scale", 1.0)),
            gain_mapping=str(gain.get("mapping", "log")).lower(),
            gain_factor_min=float(gain.get("min_factor", 1.0)),
            gain_factor_max=float(gain.get("max_factor", 8.0)),
            gain_factor_step=float(gain.get("step_factor", 0.0)),
            gain_lut_x=gain.get("lut_x"),
            gain_lut_factor=gain.get("lut_factor"),
            shot_noise_alpha=float(noise.get("shot_alpha", 0.015)),
            shot_noise_beta=float(noise.get("shot_beta", 0.0)),
            read_noise_std_base=float(noise.get("read_std_base", 0.005)),
            read_noise_gain_exponent=float(noise.get("read_gain_exponent", 1.0)),
            black_level=float(data.get("black_level", 0.0)),
            saturation_level=float(data.get("saturation_level", 1.0)),
            quantization_bits=int(data.get("quantization_bits", 0)),
            blur_scale=float(motion_blur.get("scale", 0.08)),
            command_delay_frames=int(actuator.get("command_delay_frames", 0)),
        )

    @classmethod
    def from_json(cls, path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def exposure_to_us(self, exposure01: TensorOrFloat) -> TensorOrFloat:
        x = _clamp01(exposure01)
        value = self.exposure_us_min + (self.exposure_us_max - self.exposure_us_min) * x
        return _ste_quantize(value, self.exposure_step_us)

    def exposure_to_relative(self, exposure01: TensorOrFloat) -> TensorOrFloat:
        return self.exposure_to_us(exposure01) / self.exposure_reference_us

    def _gain_lut(self, x: torch.Tensor) -> torch.Tensor:
        xp = torch.as_tensor(self.gain_lut_x, device=x.device, dtype=x.dtype)
        yp = torch.as_tensor(self.gain_lut_factor, device=x.device, dtype=x.dtype)
        idx = torch.bucketize(x, xp).sub(1).clamp(0, xp.numel() - 2)
        x0, x1 = xp[idx], xp[idx + 1]
        y0, y1 = yp[idx], yp[idx + 1]
        w = (x - x0) / (x1 - x0).clamp_min(torch.finfo(x.dtype).eps)
        return y0 + w * (y1 - y0)

    def gain_to_factor(self, gain01: TensorOrFloat) -> TensorOrFloat:
        x = _clamp01(gain01)
        if isinstance(x, torch.Tensor):
            if self.gain_mapping == "linear":
                value = self.gain_factor_min + (self.gain_factor_max - self.gain_factor_min) * x
            elif self.gain_mapping == "log":
                ratio = self.gain_factor_max / self.gain_factor_min
                value = self.gain_factor_min * torch.exp(x * math.log(ratio))
            else:
                value = self._gain_lut(x)
            return _ste_quantize(value, self.gain_factor_step)

        x = float(x)
        if self.gain_mapping == "linear":
            value = self.gain_factor_min + (self.gain_factor_max - self.gain_factor_min) * x
        elif self.gain_mapping == "log":
            value = self.gain_factor_min * (
                self.gain_factor_max / self.gain_factor_min
            ) ** x
        else:
            xs, ys = self.gain_lut_x, self.gain_lut_factor
            idx = max(0, min(len(xs) - 2, next(
                (i - 1 for i, v in enumerate(xs) if v >= x),
                len(xs) - 2,
            )))
            x0, x1 = xs[idx], xs[idx + 1]
            y0, y1 = ys[idx], ys[idx + 1]
            value = y0 + (x - x0) * (y1 - y0) / max(x1 - x0, 1e-12)
        return _ste_quantize(value, self.gain_factor_step)

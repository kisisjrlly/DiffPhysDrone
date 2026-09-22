"""Calibration profile for the e-con/Sony IMX900 differentiable camera.

The profile is the single source of truth for measured camera behavior.
Policy actions remain normalized in [0, 1]; this layer maps them to physical
exposure/gain and measured sensor-response quantities.

The repository ships a provisional profile for software development only.
Real-camera claims must use a profile marked calibrated=true.
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
    """Quantize the forward value while preserving identity tensor gradients."""
    step = float(step)
    if step <= 0.0:
        return value
    if isinstance(value, torch.Tensor):
        snapped = torch.round(value / step) * step
        return value + (snapped - value).detach()
    return round(float(value) / step) * step


def _validate_lut(xs, ys, *, name, x_min=None, x_max=None, positive_y=False):
    if not xs or not ys or len(xs) != len(ys) or len(xs) < 2:
        raise ValueError(f"{name} LUT arrays must have equal length >= 2")
    if any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError(f"{name} LUT x values must be strictly increasing")
    if x_min is not None and abs(float(xs[0]) - float(x_min)) > 1e-9:
        raise ValueError(f"{name} LUT must start at {x_min}")
    if x_max is not None and abs(float(xs[-1]) - float(x_max)) > 1e-9:
        raise ValueError(f"{name} LUT must end at {x_max}")
    if positive_y and any(float(v) <= 0.0 for v in ys):
        raise ValueError(f"{name} LUT y values must be > 0")


def _interp_lut_tensor(x, xs, ys):
    xp = torch.as_tensor(xs, device=x.device, dtype=x.dtype)
    yp = torch.as_tensor(ys, device=x.device, dtype=x.dtype)
    idx = torch.bucketize(x, xp).sub(1).clamp(0, xp.numel() - 2)
    x0, x1 = xp[idx], xp[idx + 1]
    y0, y1 = yp[idx], yp[idx + 1]
    w = (x - x0) / (x1 - x0).clamp_min(torch.finfo(x.dtype).eps)
    return y0 + w * (y1 - y0)


def _interp_lut_float(x, xs, ys):
    x = float(x)
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    idx = next(i for i in range(len(xs) - 1) if xs[i] <= x <= xs[i + 1])
    x0, x1 = float(xs[idx]), float(xs[idx + 1])
    y0, y1 = float(ys[idx]), float(ys[idx + 1])
    return y0 + (x - x0) * (y1 - y0) / max(x1 - x0, 1e-12)


@dataclass(frozen=True)
class IMX900Calibration:
    schema_version: int = 2
    profile_name: str = "IMX900 provisional"
    calibrated: bool = False
    source: str = "unmeasured placeholder"

    # Exposure command -> physical exposure. Linear is the development default;
    # a measured LUT is preferred if the deployed driver is not well modeled
    # by a linear action-to-time map.
    exposure_mapping: str = "linear"
    exposure_us_min: float = 100.0
    exposure_us_max: float = 8000.0
    exposure_reference_us: float = 1000.0
    exposure_step_us: float = 0.0
    exposure_lut_x: Optional[List[float]] = None
    exposure_lut_us: Optional[List[float]] = None
    signal_scale: float = 1.0

    # Gain command -> effective amplification.
    gain_mapping: str = "log"
    gain_factor_min: float = 1.0
    gain_factor_max: float = 8.0
    gain_factor_step: float = 0.0
    gain_lut_x: Optional[List[float]] = None
    gain_lut_factor: Optional[List[float]] = None

    # Compact photon/read-noise model. Read noise can be either a fitted power
    # law or a measured LUT indexed by effective gain factor.
    shot_noise_alpha: float = 0.015
    shot_noise_beta: float = 0.0
    read_noise_mapping: str = "power"
    read_noise_std_base: float = 0.005
    read_noise_gain_exponent: float = 1.0
    read_noise_lut_gain: Optional[List[float]] = None
    read_noise_lut_std: Optional[List[float]] = None

    black_level: float = 0.0
    saturation_level: float = 1.0
    quantization_bits: int = 0

    # Optional fixed monotonic response curve for an unavoidable ISP/nonlinear
    # capture path. Keep "linear" for RAW/linear monochrome capture.
    response_mapping: str = "linear"
    response_lut_x: Optional[List[float]] = None
    response_lut_y: Optional[List[float]] = None

    blur_scale: float = 0.08
    command_delay_frames: int = 0
    command_delay_jitter_frames: int = 0

    def __post_init__(self):
        if int(self.schema_version) not in {1, 2}:
            raise ValueError("unsupported IMX900 calibration schema_version")

        if self.exposure_mapping not in {"linear", "lut"}:
            raise ValueError("exposure_mapping must be linear or lut")
        if self.exposure_us_min <= 0 or self.exposure_us_max <= self.exposure_us_min:
            raise ValueError("invalid exposure range")
        if self.exposure_reference_us <= 0 or self.exposure_step_us < 0:
            raise ValueError("invalid exposure reference/step")
        if self.signal_scale <= 0:
            raise ValueError("signal_scale must be > 0")
        if self.exposure_mapping == "lut":
            _validate_lut(
                self.exposure_lut_x,
                self.exposure_lut_us,
                name="exposure",
                x_min=0.0,
                x_max=1.0,
                positive_y=True,
            )
            if abs(float(self.exposure_lut_us[0]) - self.exposure_us_min) > 1e-6:
                raise ValueError("exposure LUT first value must equal exposure_us_min")
            if abs(float(self.exposure_lut_us[-1]) - self.exposure_us_max) > 1e-6:
                raise ValueError("exposure LUT last value must equal exposure_us_max")

        if self.gain_factor_min <= 0 or self.gain_factor_max < self.gain_factor_min:
            raise ValueError("invalid gain factor range")
        if self.gain_factor_step < 0:
            raise ValueError("gain_factor_step must be >= 0")
        if self.gain_mapping not in {"linear", "log", "lut"}:
            raise ValueError("gain_mapping must be linear, log, or lut")
        if self.gain_mapping == "lut":
            _validate_lut(
                self.gain_lut_x,
                self.gain_lut_factor,
                name="gain",
                x_min=0.0,
                x_max=1.0,
                positive_y=True,
            )

        if self.shot_noise_alpha < 0 or self.shot_noise_beta < 0:
            raise ValueError("shot noise parameters must be >= 0")
        if self.read_noise_mapping not in {"power", "lut"}:
            raise ValueError("read_noise_mapping must be power or lut")
        if self.read_noise_std_base < 0 or self.read_noise_gain_exponent < 0:
            raise ValueError("read-noise parameters must be >= 0")
        if self.read_noise_mapping == "lut":
            _validate_lut(
                self.read_noise_lut_gain,
                self.read_noise_lut_std,
                name="read-noise",
                positive_y=True,
            )
            if float(self.read_noise_lut_gain[0]) > self.gain_factor_min:
                raise ValueError("read-noise LUT must cover gain_factor_min")
            if float(self.read_noise_lut_gain[-1]) < self.gain_factor_max:
                raise ValueError("read-noise LUT must cover gain_factor_max")

        if self.saturation_level <= 0:
            raise ValueError("saturation_level must be > 0")
        if self.quantization_bits < 0:
            raise ValueError("quantization_bits must be >= 0")
        if self.response_mapping not in {"linear", "lut"}:
            raise ValueError("response_mapping must be linear or lut")
        if self.response_mapping == "lut":
            _validate_lut(
                self.response_lut_x,
                self.response_lut_y,
                name="response",
                x_min=0.0,
                x_max=1.0,
            )
            if any(b < a for a, b in zip(self.response_lut_y, self.response_lut_y[1:])):
                raise ValueError("response LUT must be monotonic non-decreasing")
            if any(float(v) < 0.0 or float(v) > 1.0 for v in self.response_lut_y):
                raise ValueError("response LUT y values must remain in [0,1]")

        if self.blur_scale < 0:
            raise ValueError("blur_scale must be >= 0")
        if self.command_delay_frames < 0 or self.command_delay_jitter_frames < 0:
            raise ValueError("command delay/jitter must be >= 0")

    @classmethod
    def from_dict(cls, data):
        exposure = data.get("exposure", {})
        gain = data.get("gain", {})
        noise = data.get("noise", {})
        response = data.get("response", {})
        motion_blur = data.get("motion_blur", {})
        actuator = data.get("actuator", {})
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            profile_name=str(data.get("profile_name", "IMX900 calibration")),
            calibrated=bool(data.get("calibrated", False)),
            source=str(data.get("source", "")),
            exposure_mapping=str(exposure.get("mapping", "linear")).lower(),
            exposure_us_min=float(exposure.get("min_us", 100.0)),
            exposure_us_max=float(exposure.get("max_us", 8000.0)),
            exposure_reference_us=float(exposure.get("reference_us", 1000.0)),
            exposure_step_us=float(exposure.get("step_us", 0.0)),
            exposure_lut_x=exposure.get("lut_x"),
            exposure_lut_us=exposure.get("lut_us"),
            signal_scale=float(exposure.get("signal_scale", 1.0)),
            gain_mapping=str(gain.get("mapping", "log")).lower(),
            gain_factor_min=float(gain.get("min_factor", 1.0)),
            gain_factor_max=float(gain.get("max_factor", 8.0)),
            gain_factor_step=float(gain.get("step_factor", 0.0)),
            gain_lut_x=gain.get("lut_x"),
            gain_lut_factor=gain.get("lut_factor"),
            shot_noise_alpha=float(noise.get("shot_alpha", 0.015)),
            shot_noise_beta=float(noise.get("shot_beta", 0.0)),
            read_noise_mapping=str(noise.get("read_mapping", "power")).lower(),
            read_noise_std_base=float(noise.get("read_std_base", 0.005)),
            read_noise_gain_exponent=float(noise.get("read_gain_exponent", 1.0)),
            read_noise_lut_gain=noise.get("read_lut_gain"),
            read_noise_lut_std=noise.get("read_lut_std"),
            black_level=float(data.get("black_level", 0.0)),
            saturation_level=float(data.get("saturation_level", 1.0)),
            quantization_bits=int(data.get("quantization_bits", 0)),
            response_mapping=str(response.get("mapping", "linear")).lower(),
            response_lut_x=response.get("lut_x"),
            response_lut_y=response.get("lut_y"),
            blur_scale=float(motion_blur.get("scale", 0.08)),
            command_delay_frames=int(actuator.get("command_delay_frames", 0)),
            command_delay_jitter_frames=int(actuator.get("command_delay_jitter_frames", 0)),
        )

    @classmethod
    def from_json(cls, path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def exposure_to_us(self, exposure01: TensorOrFloat) -> TensorOrFloat:
        x = _clamp01(exposure01)
        if self.exposure_mapping == "lut":
            if isinstance(x, torch.Tensor):
                value = _interp_lut_tensor(x, self.exposure_lut_x, self.exposure_lut_us)
            else:
                value = _interp_lut_float(x, self.exposure_lut_x, self.exposure_lut_us)
        else:
            value = self.exposure_us_min + (self.exposure_us_max - self.exposure_us_min) * x
        return _ste_quantize(value, self.exposure_step_us)

    def exposure_to_relative(self, exposure01: TensorOrFloat) -> TensorOrFloat:
        return self.exposure_to_us(exposure01) / self.exposure_reference_us

    def gain_to_factor(self, gain01: TensorOrFloat) -> TensorOrFloat:
        x = _clamp01(gain01)
        if isinstance(x, torch.Tensor):
            if self.gain_mapping == "linear":
                value = self.gain_factor_min + (self.gain_factor_max - self.gain_factor_min) * x
            elif self.gain_mapping == "log":
                ratio = self.gain_factor_max / self.gain_factor_min
                value = self.gain_factor_min * torch.exp(x * math.log(ratio))
            else:
                value = _interp_lut_tensor(x, self.gain_lut_x, self.gain_lut_factor)
            return _ste_quantize(value, self.gain_factor_step)

        x = float(x)
        if self.gain_mapping == "linear":
            value = self.gain_factor_min + (self.gain_factor_max - self.gain_factor_min) * x
        elif self.gain_mapping == "log":
            value = self.gain_factor_min * (
                self.gain_factor_max / self.gain_factor_min
            ) ** x
        else:
            value = _interp_lut_float(x, self.gain_lut_x, self.gain_lut_factor)
        return _ste_quantize(value, self.gain_factor_step)

    def read_noise_std(self, gain_factor: TensorOrFloat) -> TensorOrFloat:
        if self.read_noise_mapping == "lut":
            if isinstance(gain_factor, torch.Tensor):
                return _interp_lut_tensor(
                    gain_factor,
                    self.read_noise_lut_gain,
                    self.read_noise_lut_std,
                )
            return _interp_lut_float(
                gain_factor,
                self.read_noise_lut_gain,
                self.read_noise_lut_std,
            )
        return self.read_noise_std_base * gain_factor ** self.read_noise_gain_exponent

    def apply_response(self, image: torch.Tensor) -> torch.Tensor:
        """Apply an optional measured fixed response curve before quantization."""
        if self.response_mapping == "linear":
            return image
        x = image.clamp(0.0, 1.0)
        return _interp_lut_tensor(x, self.response_lut_x, self.response_lut_y)

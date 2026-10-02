"""Small, explicit IMX900 capture interface for shadow-mode integration.

This module deliberately does not infer effective exposure/gain from requested
values.  Driver-specific controls and metadata must be supplied by the camera
integration on the target Jetson.  The generic implementation is useful for
development and records unavailable values as ``None`` until that integration
exists.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol


class FrameBackend(Protocol):
    def read(self) -> tuple[bool, Any]: ...


@dataclass
class FrameRecord:
    frame_id: int
    camera_timestamp: float
    jetson_receive_timestamp: float
    requested_exposure_us: Optional[float]
    requested_gain: Optional[float]
    effective_exposure_us: Optional[float]
    effective_gain: Optional[float]
    command_timestamp: Optional[float]
    pixel_format: Optional[str]
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    temperature: Optional[float]


class Imx900Camera:
    """Camera adapter with safe bookkeeping and an injectable frame backend."""

    def __init__(
        self,
        backend: FrameBackend,
        *,
        control: Optional[Callable[[str, float], None]] = None,
        effective_settings: Optional[Callable[[], dict[str, float]]] = None,
        pixel_format: Optional[str] = None,
        fps: Optional[float] = None,
    ) -> None:
        self.backend = backend
        self.control = control
        self.effective_settings = effective_settings
        self.pixel_format = pixel_format
        self.fps = fps
        self._requested = {"exposure_us": None, "gain": None}
        self._command_timestamp = None
        self._frame_id = 0

    def _set(self, name: str, value: float) -> None:
        value = float(value)
        if self.control is not None:
            self.control(name, value)
        self._requested[name] = value
        self._command_timestamp = time.time()

    def set_exposure_us(self, value: float) -> None:
        self._set("exposure_us", value)

    def set_gain(self, value: float) -> None:
        self._set("gain", value)

    def get_requested_settings(self) -> dict[str, Optional[float]]:
        return dict(self._requested)

    def get_effective_settings_if_available(self) -> Optional[dict[str, float]]:
        if self.effective_settings is None:
            return None
        values = self.effective_settings()
        return dict(values) if values is not None else None

    def grab_frame_with_timestamp(self) -> tuple[Any, FrameRecord]:
        receive_time = time.time()
        ok, frame = self.backend.read()
        if not ok:
            raise RuntimeError("camera backend returned no frame")
        camera_timestamp = getattr(self.backend, "last_timestamp", None)
        if camera_timestamp is None:
            camera_timestamp = receive_time
        effective = self.get_effective_settings_if_available() or {}
        shape = getattr(frame, "shape", ())
        height = int(shape[0]) if len(shape) >= 2 else None
        width = int(shape[1]) if len(shape) >= 2 else None
        record = FrameRecord(
            frame_id=self._frame_id,
            camera_timestamp=float(camera_timestamp),
            jetson_receive_timestamp=float(receive_time),
            requested_exposure_us=self._requested["exposure_us"],
            requested_gain=self._requested["gain"],
            effective_exposure_us=effective.get("exposure_us"),
            effective_gain=effective.get("gain"),
            command_timestamp=self._command_timestamp,
            pixel_format=self.pixel_format,
            width=width,
            height=height,
            fps=self.fps,
            temperature=None,
        )
        self._frame_id += 1
        return frame, record


class V4L2Control:
    """Best-effort generic v4l2-ctl controller; names are device-specific."""

    def __init__(self, device: str, exposure_control: str, gain_control: str) -> None:
        self.device = device
        self.controls = {"exposure_us": exposure_control, "gain": gain_control}

    def __call__(self, name: str, value: float) -> None:
        control = self.controls[name]
        subprocess.run(
            ["v4l2-ctl", "-d", self.device, "--set-ctrl", f"{control}={value}"],
            check=True,
            capture_output=True,
            text=True,
        )


def append_jsonl(path: str | Path, record: FrameRecord) -> None:
    """Append one frame metadata record without embedding image bytes."""
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

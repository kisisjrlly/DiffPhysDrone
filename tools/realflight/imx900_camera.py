"""Auditable IMX900 capture interface for bench and shadow-mode integration."""

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
class CameraCommand:
    command_id: int
    issue_monotonic_ns: int
    issue_wall_time: float
    requested_exposure_us: Optional[float]
    requested_gain: Optional[float]


@dataclass
class FrameRecord:
    frame_id: int
    camera_timestamp: Optional[float]
    jetson_receive_monotonic_ns: int
    jetson_receive_wall_time: float
    requested_exposure_us: Optional[float]
    requested_gain: Optional[float]
    effective_exposure_us: Optional[float]
    effective_gain: Optional[float]
    effective_settings_source: str
    command_id: Optional[int]
    command_issue_monotonic_ns: Optional[int]
    pixel_format: Optional[str]
    dtype: Optional[str]
    bit_depth: Optional[int]
    stride_bytes: Optional[int]
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    sequence: Optional[int]
    temperature: Optional[float]
    timestamp_source: str


class Imx900Camera:
    """Camera adapter with explicit command and frame provenance."""

    def __init__(self, backend: FrameBackend, *,
                 control: Optional[Callable[[str, float], None]] = None,
                 effective_settings: Optional[Callable[[], dict[str, float]]] = None,
                 pixel_format: Optional[str] = None, bit_depth: Optional[int] = None,
                 fps: Optional[float] = None) -> None:
        self.backend = backend
        self.control = control
        self.effective_settings = effective_settings
        self.pixel_format = pixel_format
        self.bit_depth = bit_depth
        self.fps = fps
        self._requested = {"exposure_us": None, "gain": None}
        self._command: Optional[CameraCommand] = None
        self._next_command_id = 0
        self._frame_id = 0

    def set_settings(self, *, exposure_us: float, gain: float) -> CameraCommand:
        issue_monotonic_ns = time.monotonic_ns()
        issue_wall_time = time.time()
        exposure_us, gain = float(exposure_us), float(gain)
        if self.control is not None:
            self.control("exposure_us", exposure_us)
            self.control("gain", gain)
        command = CameraCommand(self._next_command_id, issue_monotonic_ns, issue_wall_time,
                                exposure_us, gain)
        self._next_command_id += 1
        self._requested = {"exposure_us": exposure_us, "gain": gain}
        self._command = command
        return command

    def set_exposure_us(self, value: float) -> CameraCommand:
        return self.set_settings(exposure_us=value, gain=float(self._requested["gain"] or 0.0))

    def set_gain(self, value: float) -> CameraCommand:
        return self.set_settings(exposure_us=float(self._requested["exposure_us"] or 0.0), gain=value)

    def get_requested_settings(self) -> dict[str, Optional[float]]:
        return dict(self._requested)

    def get_effective_settings_if_available(self) -> Optional[dict[str, float]]:
        if self.effective_settings is None:
            return None
        values = self.effective_settings()
        return dict(values) if values is not None else None

    def grab_frame_with_timestamp(self, *, raw_path: str | Path | None = None,
                                  metadata_path: str | Path | None = None) -> tuple[Any, FrameRecord]:
        ok, frame = self.backend.read()
        receive_monotonic_ns, receive_wall_time = time.monotonic_ns(), time.time()
        if not ok:
            raise RuntimeError("camera backend returned no frame")
        camera_timestamp = getattr(self.backend, "last_timestamp", None)
        effective = self.get_effective_settings_if_available()
        shape = getattr(frame, "shape", ())
        strides = getattr(frame, "strides", None)
        command = self._command
        record = FrameRecord(
            frame_id=self._frame_id,
            camera_timestamp=float(camera_timestamp) if camera_timestamp is not None else None,
            jetson_receive_monotonic_ns=receive_monotonic_ns,
            jetson_receive_wall_time=receive_wall_time,
            requested_exposure_us=self._requested["exposure_us"],
            requested_gain=self._requested["gain"],
            effective_exposure_us=effective.get("exposure_us") if effective else None,
            effective_gain=effective.get("gain") if effective else None,
            effective_settings_source=("driver_query_after_frame" if effective is not None else "unavailable"),
            command_id=command.command_id if command else None,
            command_issue_monotonic_ns=command.issue_monotonic_ns if command else None,
            pixel_format=self.pixel_format,
            dtype=str(getattr(getattr(frame, "dtype", None), "name", "unknown")),
            bit_depth=self.bit_depth,
            stride_bytes=int(strides[0]) if strides else None,
            width=int(shape[1]) if len(shape) >= 2 else None,
            height=int(shape[0]) if len(shape) >= 2 else None,
            fps=self.fps,
            sequence=getattr(self.backend, "last_sequence", None),
            temperature=None,
            timestamp_source=("backend" if camera_timestamp is not None else "unavailable"),
        )
        if raw_path is not None:
            save_raw_frame(frame, raw_path)
        if metadata_path is not None:
            append_jsonl(metadata_path, record)
        self._frame_id += 1
        return frame, record


class V4L2Control:
    """Generic v4l2-ctl controller; control names remain device-specific."""
    def __init__(self, device: str, exposure_control: str, gain_control: str) -> None:
        self.device = device
        self.controls = {"exposure_us": exposure_control, "gain": gain_control}

    def __call__(self, name: str, value: float) -> None:
        subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl",
                        f"{self.controls[name]}={value}"], check=True,
                       capture_output=True, text=True)


def save_raw_frame(frame: Any, path: str | Path) -> None:
    """Write raw array bytes only; never silently convert to JPEG/8-bit."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(frame, "tobytes"):
        raise TypeError("RAW capture requires an array-like frame with tobytes()")
    target.write_bytes(frame.tobytes(order="C"))


def append_jsonl(path: str | Path, record: FrameRecord) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

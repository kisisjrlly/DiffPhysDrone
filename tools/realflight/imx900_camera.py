"""Auditable IMX900 camera interface for bench and shadow-mode capture."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol


@dataclass(frozen=True)
class FramePacket:
    frame: Any
    camera_timestamp_value: Optional[int] = None
    camera_timestamp_unit: Optional[str] = None
    camera_timestamp_clock_domain: Optional[str] = None
    camera_timestamp_source: Optional[str] = None
    sequence: Optional[int] = None
    pixel_format: Optional[str] = None
    frame_metadata_source: str = "unavailable"
    effective_settings: Optional[Mapping[str, float]] = None


class FrameBackend(Protocol):
    def read(self) -> FramePacket | tuple[bool, Any]: ...


class CameraControlBackend(Protocol):
    atomic_controls: bool

    def set_controls(self, values: Mapping[str, float]) -> None: ...

    def set_control(self, name: str, value: float) -> None: ...


@dataclass(frozen=True)
class CameraCommand:
    command_id: int
    transaction_start_monotonic_ns: int
    transaction_end_monotonic_ns: int
    exposure_write_start_monotonic_ns: Optional[int]
    exposure_write_end_monotonic_ns: Optional[int]
    gain_write_start_monotonic_ns: Optional[int]
    gain_write_end_monotonic_ns: Optional[int]
    requested_exposure_us: float
    requested_gain: float
    backend_type: str
    atomic_controls: bool


@dataclass
class FrameRecord:
    frame_id: int
    camera_timestamp_value: Optional[int]
    camera_timestamp_unit: Optional[str]
    camera_timestamp_clock_domain: Optional[str]
    camera_timestamp_source: Optional[str]
    receive_monotonic_ns: int
    receive_wall_time: float
    sequence: Optional[int]
    frame_metadata_source: str
    requested_exposure_us: Optional[float]
    requested_gain: Optional[float]
    frame_effective_exposure_us: Optional[float]
    frame_effective_gain: Optional[float]
    queried_exposure_after_frame: Optional[float]
    queried_gain_after_frame: Optional[float]
    command_id: Optional[int]
    pixel_format: Optional[str]
    raw_file: Optional[str]
    raw_sha256: Optional[str]
    raw_bytes: Optional[int]
    dtype: Optional[str]
    bit_depth: Optional[int]
    stride_bytes: Optional[int]
    width: Optional[int]
    height: Optional[int]


class BenchControlAdapter:
    """Adapter for a single-control callback; never claims atomicity."""

    atomic_controls = False

    def __init__(self, callback):
        self.callback = callback

    def set_control(self, name: str, value: float) -> None:
        self.callback(name, float(value))

    def set_controls(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            self.set_control(name, value)


class Imx900Camera:
    def __init__(self, backend: FrameBackend, *, control: Optional[CameraControlBackend] = None,
                 effective_settings=None, pixel_format=None, bit_depth=None) -> None:
        self.backend = backend
        self.control = BenchControlAdapter(control) if callable(control) else control
        self.effective_settings = effective_settings
        self.pixel_format = pixel_format
        self.bit_depth = bit_depth
        self._requested = {"exposure_us": None, "gain": None}
        self._command: Optional[CameraCommand] = None
        self._next_command_id = 0
        self._frame_id = 0

    def set_settings(self, *, exposure_us: float, gain: float) -> CameraCommand:
        exposure_us, gain = float(exposure_us), float(gain)
        start = time.monotonic_ns()
        exposure_start = exposure_end = gain_start = gain_end = None
        if self.control is not None:
            if getattr(self.control, "atomic_controls", False):
                self.control.set_controls({"exposure_us": exposure_us, "gain": gain})
                exposure_start = gain_start = start
                exposure_end = gain_end = time.monotonic_ns()
            else:
                exposure_start = time.monotonic_ns()
                self.control.set_control("exposure_us", exposure_us)
                exposure_end = time.monotonic_ns()
                gain_start = time.monotonic_ns()
                self.control.set_control("gain", gain)
                gain_end = time.monotonic_ns()
        end = time.monotonic_ns()
        command = CameraCommand(self._next_command_id, start, end, exposure_start,
                                exposure_end, gain_start, gain_end, exposure_us, gain,
                                type(self.control).__name__ if self.control else "none",
                                bool(getattr(self.control, "atomic_controls", False)))
        self._next_command_id += 1
        self._requested = {"exposure_us": exposure_us, "gain": gain}
        self._command = command
        return command

    def set_exposure_us(self, value: float) -> CameraCommand:
        if self._requested["gain"] is None:
            raise ValueError("gain must be initialized before using set_exposure_us")
        return self.set_settings(exposure_us=value, gain=self._requested["gain"])

    def set_gain(self, value: float) -> CameraCommand:
        if self._requested["exposure_us"] is None:
            raise ValueError("exposure_us must be initialized before using set_gain")
        return self.set_settings(exposure_us=self._requested["exposure_us"], gain=value)

    def get_requested_settings(self):
        return dict(self._requested)

    def get_effective_settings_if_available(self):
        return dict(self.effective_settings()) if self.effective_settings else None

    def grab_frame_with_timestamp(self, *, raw_path=None, metadata_path=None):
        result = self.backend.read()
        receive_ns, receive_wall = time.monotonic_ns(), time.time()
        if isinstance(result, FramePacket):
            packet = result
        else:
            ok, frame = result
            if not ok:
                raise RuntimeError("camera backend returned no frame")
            packet = FramePacket(frame=frame, pixel_format=self.pixel_format)
        effective = packet.effective_settings
        queried = self.get_effective_settings_if_available()
        frame = packet.frame
        shape = getattr(frame, "shape", ())
        strides = getattr(frame, "strides", None)
        raw_info = save_raw_frame(frame, raw_path) if raw_path else None
        command = self._command
        record = FrameRecord(
            frame_id=self._frame_id,
            camera_timestamp_value=packet.camera_timestamp_value,
            camera_timestamp_unit=packet.camera_timestamp_unit,
            camera_timestamp_clock_domain=packet.camera_timestamp_clock_domain,
            camera_timestamp_source=packet.camera_timestamp_source,
            receive_monotonic_ns=receive_ns,
            receive_wall_time=receive_wall,
            sequence=packet.sequence,
            frame_metadata_source=packet.frame_metadata_source,
            requested_exposure_us=self._requested["exposure_us"],
            requested_gain=self._requested["gain"],
            frame_effective_exposure_us=effective.get("exposure_us") if effective else None,
            frame_effective_gain=effective.get("gain") if effective else None,
            queried_exposure_after_frame=queried.get("exposure_us") if queried else None,
            queried_gain_after_frame=queried.get("gain") if queried else None,
            command_id=command.command_id if command else None,
            pixel_format=packet.pixel_format or self.pixel_format,
            raw_file=raw_info[0] if raw_info else None,
            raw_sha256=raw_info[1] if raw_info else None,
            raw_bytes=raw_info[2] if raw_info else None,
            dtype=str(getattr(getattr(frame, "dtype", None), "name", "unknown")),
            bit_depth=self.bit_depth,
            stride_bytes=int(strides[0]) if strides else None,
            width=int(shape[1]) if len(shape) >= 2 else None,
            height=int(shape[0]) if len(shape) >= 2 else None,
        )
        if metadata_path:
            append_jsonl(metadata_path, record)
        self._frame_id += 1
        return frame, record


class V4L2Control:
    atomic_controls = False

    def __init__(self, device: str, exposure_control: str, gain_control: str):
        self.device = device
        self.controls = {"exposure_us": exposure_control, "gain": gain_control}

    def set_control(self, name: str, value: float) -> None:
        subprocess.run(["v4l2-ctl", "-d", self.device, "--set-ctrl",
                        f"{self.controls[name]}={value}"], check=True,
                       capture_output=True, text=True)

    def set_controls(self, values):
        for name, value in values.items():
            self.set_control(name, value)


def save_raw_frame(frame, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(frame, "tobytes"):
        raise TypeError("RAW capture requires array-like frame with tobytes()")
    payload = frame.tobytes(order="C")
    target.write_bytes(payload)
    return str(target), hashlib.sha256(payload).hexdigest(), len(payload)


def append_jsonl(path, record: FrameRecord) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

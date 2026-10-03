import time
import pytest

from tools.realflight.imx900_camera import Imx900Camera


class Backend:
    last_timestamp = None

    def read(self):
        return True, Frame()


class Frame:
    shape = (2, 2)
    dtype = type("DType", (), {"name": "uint16"})()
    strides = (4, 2)

    def tobytes(self, order="C"):
        return b"raw-frame"

    def __eq__(self, other):
        return other == [[1, 2], [3, 4]]


def test_adapter_records_requested_and_unknown_effective_values():
    commands = []
    camera = Imx900Camera(Backend(), control=lambda name, value: commands.append((name, value)))
    with pytest.raises(ValueError):
        camera.set_exposure_us(2500)
    camera.set_settings(exposure_us=2500, gain=2.0)
    frame, record = camera.grab_frame_with_timestamp()
    assert frame == [[1, 2], [3, 4]]
    assert commands == [("exposure_us", 2500.0), ("gain", 2.0)]
    assert record.camera_timestamp_value is None
    assert record.requested_exposure_us == 2500.0
    assert record.frame_effective_exposure_us is None
    assert record.width == 2 and record.height == 2
    assert record.camera_timestamp_source is None


def test_effective_settings_are_only_reported_from_driver_callback():
    camera = Imx900Camera(
        Backend(), effective_settings=lambda: {"exposure_us": 2400.0, "gain": 1.9}
    )
    _, record = camera.grab_frame_with_timestamp()
    assert record.queried_exposure_after_frame == 2400.0
    assert record.queried_gain_after_frame == 1.9
    assert record.frame_effective_exposure_us is None


def test_receive_timestamp_is_after_blocking_read():
    class SlowBackend(Backend):
        def read(self):
            time.sleep(0.01)
            return super().read()

    camera = Imx900Camera(SlowBackend())
    before = time.monotonic_ns()
    _, record = camera.grab_frame_with_timestamp()
    assert record.receive_monotonic_ns >= before + 9_000_000


def test_raw_metadata_is_hashable_and_settings_can_be_atomic(tmp_path):
    class Atomic:
        atomic_controls = True

        def __init__(self):
            self.values = []

        def set_controls(self, values):
            self.values.append(dict(values))

        def set_control(self, name, value):
            raise AssertionError("atomic backend must not use partial writes")

    backend = Atomic()
    camera = Imx900Camera(Backend(), control=backend, pixel_format="unpacked_linear_uint16", bit_depth=12)
    command = camera.set_settings(exposure_us=1000, gain=2)
    _, record = camera.grab_frame_with_timestamp(
        raw_path=tmp_path / "frames" / "frame_000000.raw",
        metadata_path=tmp_path / "metadata.jsonl",
    )
    assert command.atomic_controls is True
    assert backend.values == [{"exposure_us": 1000.0, "gain": 2.0}]
    assert record.raw_bytes == len(b"raw-frame")
    assert record.raw_sha256
    assert record.raw_file.endswith("frame_000000.raw")
    assert record.frame_id == 0

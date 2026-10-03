import time

from tools.realflight.imx900_camera import Imx900Camera


class Backend:
    last_timestamp = None

    def read(self):
        return True, Frame()


class Frame:
    shape = (2, 2)

    def __eq__(self, other):
        return other == [[1, 2], [3, 4]]


def test_adapter_records_requested_and_unknown_effective_values():
    commands = []
    camera = Imx900Camera(Backend(), control=lambda name, value: commands.append((name, value)))
    camera.set_exposure_us(2500)
    camera.set_gain(2.0)
    frame, record = camera.grab_frame_with_timestamp()
    assert frame == [[1, 2], [3, 4]]
    assert commands == [("exposure_us", 2500.0), ("gain", 0.0),
                        ("exposure_us", 2500.0), ("gain", 2.0)]
    assert record.camera_timestamp is None
    assert record.requested_exposure_us == 2500.0
    assert record.effective_exposure_us is None
    assert record.width == 2 and record.height == 2
    assert record.timestamp_source == "unavailable"


def test_effective_settings_are_only_reported_from_driver_callback():
    camera = Imx900Camera(
        Backend(), effective_settings=lambda: {"exposure_us": 2400.0, "gain": 1.9}
    )
    _, record = camera.grab_frame_with_timestamp()
    assert record.effective_exposure_us == 2400.0
    assert record.effective_gain == 1.9


def test_receive_timestamp_is_after_blocking_read():
    class SlowBackend(Backend):
        def read(self):
            time.sleep(0.01)
            return super().read()

    camera = Imx900Camera(SlowBackend())
    before = time.monotonic_ns()
    _, record = camera.grab_frame_with_timestamp()
    assert record.jetson_receive_monotonic_ns >= before + 9_000_000

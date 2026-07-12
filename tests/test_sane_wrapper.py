import pytest

from auto_sane.sane_wrapper import initialize


def test_sane_context():
    with initialize() as sane:
        assert isinstance(sane.version, tuple)


@pytest.mark.slow
def test_get_devices():
    with initialize() as sane:
        devices = sane.get_devices()
        assert len(devices) >= 0


@pytest.mark.slow
def test_open_device():
    with initialize() as sane:
        devices = sane.get_devices()
        assert len(devices) > 0
        with devices[0].open() as _:
            pass


@pytest.mark.slow
def test_scan():
    with initialize() as sane:
        devices = sane.get_devices()
        assert len(devices) > 0
        with devices[0].open() as device:
            for page in device.multi_scan():
                page.save("page.png")

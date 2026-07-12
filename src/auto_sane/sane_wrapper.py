from contextlib import contextmanager
from typing import Iterator, NamedTuple, Protocol

import sane

__all__ = ["AvailableDevice", "Sane", "initialize"]

_VersionTuple = tuple[int, int, int, int]

# TODO: Check whether the python-sane module does anything to
#       prevent overlapping sane sessions.


class AvailableDevice(NamedTuple):
    device_name: str
    vendor: str
    mode: str
    type: str

    @contextmanager
    def open(self) -> Iterator[sane.SaneDev]:
        device = sane.open(self.device_name)
        try:
            yield device
        finally:
            device.close()

    def __str__(self) -> str:
        return self.device_name


class Sane(Protocol):
    @property
    def version(self) -> _VersionTuple: ...

    def get_devices(self, local_only: bool = False) -> list[AvailableDevice]: ...


class _SaneImpl(Sane):
    def __init__(self, version: _VersionTuple):
        super().__init__()
        self._version = version

    def get_devices(self, local_only: bool = False) -> list[AvailableDevice]:
        devices = sane.get_devices()
        return [
            AvailableDevice(
                device_name=device[0], vendor=device[1], mode=device[2], type=device[3]
            )
            for device in devices
        ]

    @property
    def version(self) -> _VersionTuple:
        return self._version


@contextmanager
def initialize() -> Iterator[Sane]:
    version = sane.init()
    try:
        yield _SaneImpl(version)
    finally:
        sane.exit()

import asyncio
from pathlib import Path

from PIL.Image import Image

from auto_sane.sane_wrapper import AvailableDevice

type ImageList = list[Image]
ImageListQueue = asyncio.Queue[ImageList]
DirectoryQueue = asyncio.Queue[Path]


class NoDeviceFoundError(RuntimeError):
    pass


class TooManyMatchingDevicesError(RuntimeError):
    def __init__(self, devices: list[AvailableDevice]):
        devices_str = ", ".join(str(device) for device in devices)
        super().__init__(f"Found too many matching devices ({devices_str}).")


class NoDocumentsInFeedError(RuntimeError):
    pass


class DeveloperWarning(RuntimeWarning):
    pass

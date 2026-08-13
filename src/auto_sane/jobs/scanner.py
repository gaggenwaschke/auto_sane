import asyncio
import logging
import re
import warnings
from collections.abc import AsyncIterator

from PIL.Image import Image
from sane import SaneDev

from auto_sane.config import Config
from auto_sane.sane_wrapper import AvailableDevice, Sane, initialize
from auto_sane.types import (
    DeveloperWarning,
    ImageListQueue,
    NoDeviceFoundError,
    TooManyMatchingDevicesError,
)

logger = logging.getLogger(__name__)


class Scanner:
    def __init__(self, config: Config, queue: ImageListQueue):
        self._config = config
        self._queue = queue

    async def run(self) -> None:
        with initialize() as sane:
            logger.info("Initialized Sane. Will search for devices now.")
            available_device = await self._wait_for_device(sane)
            logger.info('Found device "%s"', available_device)
            with available_device.open() as device:
                logger.info('Connected to device "%s"', available_device.device_name)
                self._configure_device(device)

                while True:
                    pages = [page async for page in self._scan_document(device)]
                    try:
                        self._queue.put_nowait(pages)
                    except asyncio.QueueFull:
                        msg = "Scanner to pdf queue full, will block on back pressure. PDF merging too slow!"
                        logger.warning(msg)
                        warnings.warn(msg, category=DeveloperWarning)
                        await self._queue.put(pages)

    async def _scan_document(self, device: SaneDev) -> AsyncIterator[Image]:
        """Will scan all current pages in feeder into one document."""
        logger.info("Please feed new document!")
        await self._start_when_first_doc_in_feeder(device)
        logger.info("Starting to process new document")
        more_pages = True
        page_index = 0

        try:
            while more_pages:
                yield device.snap(True)
                logger.debug("Scanned page #%d", page_index)

                try:
                    device.start()
                    page_index += 1
                except Exception as e:
                    if str(e) == "Document feeder out of documents":
                        more_pages = False
                    else:
                        raise
        finally:
            device.cancel()

        logger.debug("Finished scanning document")

    async def _start_when_first_doc_in_feeder(self, device: SaneDev):
        paper_in_feed = False
        while not paper_in_feed:
            try:
                device.start()
                paper_in_feed = True
            except Exception as e:
                if str(e) == "Document feeder out of documents":
                    paper_in_feed = False
                else:
                    raise
            await asyncio.sleep(1.0)

    async def _wait_for_device(self, sane: Sane) -> AvailableDevice:
        device: AvailableDevice | None = None
        while device is None:
            try:
                device = self._get_device(sane)
            except NoDeviceFoundError:
                logger.info(
                    "No device found, will search again in %d seconds.",
                    self._config.device_poll_interval.total_seconds(),
                )
                await asyncio.sleep(self._config.device_poll_interval.total_seconds())
        return device

    def _get_device(self, sane: Sane) -> AvailableDevice:
        devices = sane.get_devices()

        user_chosen_device = self._config.device

        if user_chosen_device is None:
            if len(devices) > 1:
                raise TooManyMatchingDevicesError(devices)
            if len(devices) == 0:
                raise NoDeviceFoundError("No devices found!")
            return devices[0]

        if isinstance(user_chosen_device, int):
            if user_chosen_device >= len(devices):
                raise NoDeviceFoundError(
                    f"No device with index {user_chosen_device} found, only got {len(devices)} devices."
                )
            return devices[user_chosen_device]

        # Regex
        regex = re.compile(user_chosen_device)
        devices = [device for device in devices if regex.match(device.device_name)]
        if len(devices) == 0:
            raise NoDeviceFoundError(
                f'No device matching the regex "{user_chosen_device}" was found.'
            )
        if len(devices) > 1:
            raise TooManyMatchingDevicesError(devices)
        return devices[0]

    def _configure_device(self, device: SaneDev) -> None:
        logger.debug("Using Configuration:")
        device.mode = "Color"
        device.resolution = self._config.dpi
        device.source = self._config.page_mode

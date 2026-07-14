import asyncio
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, CliApp
from sane import SaneDev

from auto_sane.sane_wrapper import AvailableDevice, Sane, initialize

PageMode = Literal["ADF", "Duplex"]


class NoDeviceFoundError(RuntimeError):
    pass


class TooManyMatchingDevicesError(RuntimeError):
    def __init__(self, devices: list[AvailableDevice]):
        devices_str = ", ".join(str(device) for device in devices)
        super().__init__(f"Found too many matching devices ({devices_str}).")


class NoDocumentsInFeedError(RuntimeError):
    pass


logger = logging.getLogger(__name__)


class App(BaseSettings):
    dpi: int = Field(
        default=300, description="Dots per inch used for scanning the documents"
    )
    page_mode: PageMode = Field(default="Duplex", description="Which sides to scan")
    device: None | int | str = Field(
        default=0,
        description="Index of the device (starting at 0) or regex matching ONE device.",
    )
    device_poll_interval: timedelta = Field(
        default_factory=lambda: timedelta(seconds=5),
        description="Time to wait between attempts to connect to scanner device.",
    )
    target_dir: Path = Field(
        default=Path("./scans"), description="Target directory to save the scans to."
    )

    async def cli_cmd(self):
        logging.basicConfig(level=logging.DEBUG)
        with initialize() as sane:
            logger.info("Initialized Sane. Will search for devices now.")
            available_device = await self._wait_for_device(sane)
            logger.info('Found device "%s"', available_device)
            with available_device.open() as device:
                logger.info('Connected to device "%s"', available_device.device_name)
                self._configure_device(device)

                while True:
                    await self._process_document(device)

    async def _process_document(self, device: SaneDev) -> None:
        logger.info("Please feed new document!")
        await self._start_when_first_doc_in_feeder(device)
        document_name = datetime.now(UTC).isoformat()
        logger.info("Starting to process new document #%s", document_name)
        target_dir = self.target_dir / str(document_name)
        target_dir.mkdir(parents=True, exist_ok=False)
        more_pages = True
        page_index = 0
        while more_pages:
            page = device.snap(True)
            logger.debug("Scanned page #%d", page_index)
            page.save(target_dir / f"{page_index}.png")
            try:
                device.start()
                page_index += 1
            except Exception as e:
                if str(e) == "Document feeder out of documents":
                    more_pages = False
                else:
                    raise
        device.cancel()
        logger.debug('Finished "%s"', document_name)

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
                    self.device_poll_interval.total_seconds(),
                )
                await asyncio.sleep(self.device_poll_interval.total_seconds())
        return device

    def _get_device(self, sane: Sane) -> AvailableDevice:
        devices = sane.get_devices()

        if self.device is None:
            if len(devices) > 1:
                raise TooManyMatchingDevicesError(devices)
            if len(devices) == 0:
                raise NoDeviceFoundError("No devices found!")
            return devices[0]

        if isinstance(self.device, int):
            if self.device >= len(devices):
                raise NoDeviceFoundError(
                    f"No device with index {self.device} found, only got {len(devices)} devices."
                )
            return devices[self.device]

        # Regex
        regex = re.compile(self.device)
        devices = [device for device in devices if regex.match(device.device_name)]
        if len(devices) == 0:
            raise NoDeviceFoundError(
                f'No device matching the regex "{self.device}" was found.'
            )
        if len(devices) > 1:
            raise TooManyMatchingDevicesError(devices)
        return devices[0]

    def _configure_device(self, device: SaneDev) -> None:
        logger.debug("Using Configuration:")
        device.mode = "Color"
        device.resolution = self.dpi
        device.source = self.page_mode


def _forever_index():
    index = 0
    while True:
        yield index
        index += 1


def main():
    CliApp.run(App)


if __name__ == "__main__":
    main()

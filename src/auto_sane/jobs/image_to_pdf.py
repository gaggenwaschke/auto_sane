import logging
from datetime import UTC, datetime
from pathlib import Path

from auto_sane.types import DirectoryQueue, ImageListQueue

logger = logging.getLogger(__name__)


class ImageToPdf:
    def __init__(
        self,
        input_queue: ImageListQueue,
        output_dir: Path,
        output_queue: DirectoryQueue,
    ):
        self._input = input_queue
        self._output_dir = output_dir
        self._output = output_queue

    async def run(self) -> None:
        while True:
            document_dir_name = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
            target_dir = self._output_dir / document_dir_name
            target_dir.mkdir(parents=True, exist_ok=False)
            images = await self._input.get()
            for index, image in enumerate(images):
                file_name = f"{index}.pdf"
                image.save(target_dir / file_name)
            logger.debug("Finished writing single page PDF's to %s", target_dir)
            await self._output.put(target_dir)

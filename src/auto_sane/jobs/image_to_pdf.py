import logging
from datetime import UTC, datetime

from auto_sane.config import Config
from auto_sane.types import ImageListQueue

logger = logging.getLogger(__name__)


class ImageToPdf:
    def __init__(
        self,
        config: Config,
        input_queue: ImageListQueue,
    ):
        self._config = config
        self._input = input_queue

    async def run(self) -> None:
        while True:
            images = await self._input.get()
            document_name = f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}.pdf"
            document_path = self._config.target_dir / document_name
            images[0].save(
                document_path,
                save_all=True,
                append_images=images[1:],
                optimize=True,
                quality=self._config.pdf_quality,
            )
            logger.debug("Finished writing single page PDF's to %s", document_path)

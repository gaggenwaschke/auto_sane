import logging
from contextlib import closing

from pypdf import PdfWriter

from auto_sane.config import Config
from auto_sane.types import DirectoryQueue

logger = logging.getLogger(__name__)


class PdfMerger:
    def __init__(self, config: Config, input_queue: DirectoryQueue):
        self._config = config
        self._queue = input_queue

    async def run(self) -> None:
        target_dir = self._config.target_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        while True:
            single_pages_dir = await self._queue.get()
            document_name = f"{single_pages_dir.stem}.pdf"
            target_file = target_dir / str(document_name)

            logger.debug('Writing doc to disc "%s"', target_file)
            with closing(PdfWriter()) as writer:
                for file in single_pages_dir.iterdir():
                    writer.merge(position=None, fileobj=file)
                writer.write(target_file)
            logger.debug('Finished writing "%s" to disk.', target_file)

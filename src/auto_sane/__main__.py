import asyncio
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic_settings import CliApp

from auto_sane.config import Config
from auto_sane.jobs.deskew import Deskew
from auto_sane.jobs.image_to_pdf import ImageToPdf
from auto_sane.jobs.pdf_merger import PdfMerger
from auto_sane.jobs.scanner import Scanner
from auto_sane.types import DirectoryQueue, ImageListQueue

logger = logging.getLogger(__name__)


class App(Config):
    async def cli_cmd(self):
        logging.basicConfig(level=logging.DEBUG)

        scanner_to_deskew_queue = ImageListQueue(2)
        deskew_to_pdf_queue = ImageListQueue(2)
        pdf_to_merger_queue = DirectoryQueue(2)
        with TemporaryDirectory() as scanned_docs_str:
            scanned_docs_dir = Path(scanned_docs_str)
            scanner = Scanner(self, scanner_to_deskew_queue)
            deskew = Deskew(self, scanner_to_deskew_queue, deskew_to_pdf_queue)
            image_to_pdf = ImageToPdf(
                deskew_to_pdf_queue, scanned_docs_dir, pdf_to_merger_queue
            )
            merger = PdfMerger(self, pdf_to_merger_queue)

            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(scanner.run())
                tasks.create_task(deskew.run())
                tasks.create_task(image_to_pdf.run())
                tasks.create_task(merger.run())


def main():
    CliApp.run(App)


if __name__ == "__main__":
    main()

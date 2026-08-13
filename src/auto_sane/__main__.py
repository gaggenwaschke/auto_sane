import asyncio
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic_settings import CliApp

from auto_sane.config import Config
from auto_sane.jobs.pdf_merger import PdfMerger
from auto_sane.jobs.scanner import Scanner
from auto_sane.types import ScannerToPdfQueue

logger = logging.getLogger(__name__)


class App(Config):
    async def cli_cmd(self):
        logging.basicConfig(level=logging.DEBUG)

        scanner_to_pdf_queue = ScannerToPdfQueue(16)
        with TemporaryDirectory() as scanned_docs_str:
            scanned_docs_dir = Path(scanned_docs_str)
            scanner = Scanner(self, scanner_to_pdf_queue, scanned_docs_dir)
            merger = PdfMerger(self, scanner_to_pdf_queue)

            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(scanner.run())
                tasks.create_task(merger.run())


def main():
    CliApp.run(App)


if __name__ == "__main__":
    main()

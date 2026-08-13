import asyncio
import logging

from pydantic_settings import CliApp

from auto_sane.config import Config
from auto_sane.jobs.deskew import Deskew
from auto_sane.jobs.image_to_pdf import ImageToPdf
from auto_sane.jobs.scanner import Scanner
from auto_sane.types import ImageListQueue

logger = logging.getLogger(__name__)


class App(Config):
    async def cli_cmd(self):
        logging.basicConfig(level=logging.DEBUG)

        scanner_to_deskew_queue = ImageListQueue(2)
        deskew_to_pdf_queue = ImageListQueue(2)
        scanner = Scanner(self, scanner_to_deskew_queue)
        deskew = Deskew(self, scanner_to_deskew_queue, deskew_to_pdf_queue)
        image_to_pdf = ImageToPdf(self, deskew_to_pdf_queue)

        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(scanner.run())
            tasks.create_task(deskew.run())
            tasks.create_task(image_to_pdf.run())


def main():
    CliApp.run(App)


if __name__ == "__main__":
    main()

from pathlib import Path

import pytest
from PIL import Image

DATA_PATH = Path(__file__).parent / "data"
SMALL_IMAGE = DATA_PATH / "small.png"
DIN_A4_CROOKET = DATA_PATH / "din_a4_crooket.png"


@pytest.fixture(scope="session")
def din_a4_crooket():
    with Image.open(DIN_A4_CROOKET) as image:
        yield image

from datetime import timedelta
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

PageMode = Literal["ADF", "Duplex"]


class Config(BaseSettings):
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
    canny_lower: int = Field(
        default=40000, description="Canny edge detection lower limit"
    )
    canny_upper: int = Field(
        default=60000, description="Canny edge detection upper limit"
    )
    canny_apperature: int = Field(
        default=7, description="Canny edge detection aperature"
    )
    pdf_quality: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Quality of the images exported to PDF [0..100]",
    )

import logging

import cv2
import numpy as np
from PIL import Image

from auto_sane.config import Config
from auto_sane.transform import quad_to_rect
from auto_sane.types import ImageListQueue

type QuadPoints = np.ndarray


logger = logging.getLogger(__name__)


class Deskew:
    def __init__(
        self, config: Config, input_queue: ImageListQueue, output_queue: ImageListQueue
    ) -> None:
        self._config = config
        self._input = input_queue
        self._output = output_queue

    async def run(self) -> None:
        while True:
            images = await self._input.get()
            logger.debug("Starting to deskew next document")
            deskewed_images = list[Image.Image]()
            for image in images:
                numpy_image = np.asarray(image)
                detected_edges = self._deskew_minarea_on_edge_detection(numpy_image)
                deskewed_image = quad_to_rect(numpy_image, detected_edges)
                deskewed_images.append(Image.fromarray(deskewed_image))
            logger.debug("Finished deskew")
            await self._output.put(deskewed_images)

    def _edge_detection(self, image) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(
            gray,
            self._config.canny_lower,
            self._config.canny_upper,
            apertureSize=self._config.canny_apperature,
        )

    def _deskew_minarea_on_edge_detection(self, image) -> QuadPoints:
        edges = self._edge_detection(image)

        y_indices, x_indices = np.where(edges > 0)
        coords = np.column_stack((x_indices, y_indices))

        if len(coords) == 0:
            # Returning image corners as a fallback to keep type consistent,
            # although normally we'd handle this upstream or via Optional
            h, w = image.shape[:2]
            return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        rect = cv2.minAreaRect(coords)
        box = cv2.boxPoints(rect).astype(np.float32)

        pts = box
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        return np.array([tl, tr, br, bl], dtype=np.float32)

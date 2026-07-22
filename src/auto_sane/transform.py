import cv2
import numpy as np


def quad_to_rect(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """
    Transforms a quadrilateral region of an image into a rectangular one.

    Args:
        image: Input image array.
        quad: Array of 4 points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
              representing the corners in order (TL, TR, BR, BL).

    Returns:
        The transformed rectangular image array.
    """
    # Ensure quad is float32 for OpenCV
    src_pts = quad.astype(np.float32)

    # Calculate target dimensions based on the max of opposite sides
    # Width: distance between (TL, TR) and (BL, BR)
    w1 = np.linalg.norm(src_pts[0] - src_pts[1])
    w2 = np.linalg.norm(src_pts[3] - src_pts[2])
    max_w = max(int(w1), int(w2))

    # Height: distance between (TL, BL) and (TR, BR)
    h1 = np.linalg.norm(src_pts[0] - src_pts[3])
    h2 = np.linalg.norm(src_pts[1] - src_pts[2])
    max_h = max(int(h1), int(h2))

    dst_pts = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )

    # Compute the perspective transform matrix and warp the image
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(image, matrix, (max_w, max_h))

    return result

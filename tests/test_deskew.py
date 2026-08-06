from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
import pytest

from auto_sane.transform import quad_to_rect

DATA_DIR = Path("./tests/data")
OUTPUT_DIR = Path("./tests/outputs")

QuadPoints: TypeAlias = np.ndarray


def intersect(l1, l2):
    # L1: p1 + t*v1 | L2: p2 + s*v2
    p1, v1 = l1
    p2, v2 = l2
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if abs(cross) < 1e-6:
        return None
    dp = p2 - p1
    t = (dp[0] * v2[1] - dp[1] * v2[0]) / cross
    return p1 + t * v1


def edge_detection(image) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 40000, 60000, apertureSize=7)


def deskew_minarea_on_diff_quantile(image: np.ndarray) -> QuadPoints:
    diff = np.diff(image).sum(axis=2)

    threshold = np.quantile(diff, 0.99)
    binary_image = diff > threshold
    cv2.imwrite("tests/outputs/binary.png", binary_image * 255)
    y_indices, x_indices = np.where(diff > 0)
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


def deskew_minarea_on_edge_detection(image) -> QuadPoints:
    edges = edge_detection(image)

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


def deskew_minarea_on_edge_detection_refined(image) -> QuadPoints:
    edges = edge_detection(image)

    y_indices, x_indices = np.where(edges > 0)
    coords = np.column_stack((x_indices, y_indices))
    if len(coords) == 0:
        h, w = image.shape[:2]
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    rect = cv2.minAreaRect(coords)
    box = cv2.boxPoints(rect).astype(np.float32)

    pts = box
    segments = [(pts[0], pts[1]), (pts[1], pts[2]), (pts[2], pts[3]), (pts[3], pts[0])]
    refined_lines = []

    for p1, p2 in segments:
        mask = np.zeros_like(edges)
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]), int(p2[1]))
        cv2.line(mask, pt1, pt2, [255], thickness=10)
        strip_edges = cv2.bitwise_and(edges, mask)
        y_s, x_s = np.where(strip_edges > 0)
        pts_s = np.column_stack((x_s, y_s)).astype(np.float32)

        if len(pts_s) >= 2:
            line = cv2.fitLine(pts_s, cv2.DIST_L2, 0, 0.01, 0.01)
            v = line[:2].flatten()
            p = line[2:].flatten()
            if len(v) == 2 and len(p) == 2:
                dir_vec = v
                point = p
                t_vals = [-100, 100]
                l_pts = [point + t * dir_vec for t in t_vals]
                refined_lines.append((l_pts[0], l_pts[1]))
            else:
                refined_lines.append((p1, p2))
        else:
            refined_lines.append((p1, p2))

    def intersect(l1, l2):
        p1, p2 = l1
        p3, p4 = l2
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return np.array([px, py])

    new_pts = []
    for i in range(4):
        p = intersect(refined_lines[i - 1], refined_lines[i])
        new_pts.append(p if p is not None else box[i])

    src_pts = np.array(new_pts, dtype=np.float32)
    s = src_pts.sum(axis=1)
    diff = np.diff(src_pts, axis=1)
    tl = src_pts[np.argmin(s)]
    br = src_pts[np.argmax(s)]
    tr = src_pts[np.argmin(diff)]
    bl = src_pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def deskew_hough(image: cv2.Mat) -> QuadPoints:
    edges = edge_detection(image)

    lines = cv2.HoughLinesP(
        edges, 3, np.pi / 180, threshold=300, minLineLength=500, maxLineGap=1000
    )
    if lines is None:
        h, w = image.shape[:2]
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    horiz_candidates = []
    vert_candidates = []
    line_image = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_image = cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 30 or angle > 150:
            horiz_candidates.append(line[0])
        elif 60 < angle < 120:
            vert_candidates.append(line[0])
    cv2.imwrite(str(OUTPUT_DIR / "line_image.png"), line_image)

    if not horiz_candidates or not vert_candidates:
        h, w = image.shape[:2]
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    avg_y = np.mean([(line[1] + line[3]) / 2 for line in horiz_candidates])
    top_lines = [line for line in horiz_candidates if (line[1] + line[3]) / 2 < avg_y]
    bottom_lines = [
        line for line in horiz_candidates if (line[1] + line[3]) / 2 >= avg_y
    ]
    avg_x = np.mean([(line[0] + line[2]) / 2 for line in vert_candidates])
    left_lines = [line for line in vert_candidates if (line[0] + line[2]) / 2 < avg_x]
    right_lines = [line for line in vert_candidates if (line[0] + line[2]) / 2 >= avg_x]

    final_bundles = {
        "top": top_lines,
        "bottom": bottom_lines,
        "left": left_lines,
        "right": right_lines,
    }
    fitted_lines = {}
    for key, segments in final_bundles.items():
        if not segments:
            h, w = image.shape[:2]
            return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        pts = []
        for s in segments:
            pts.extend([[s[0], s[1]], [s[2], s[3]]])
        pts = np.array(pts, dtype=np.float32)
        line_params = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        fitted_lines[key] = (line_params[2:].flatten(), line_params[:2].flatten())

    tl = intersect(fitted_lines["top"], fitted_lines["left"])
    tr = intersect(fitted_lines["top"], fitted_lines["right"])
    br = intersect(fitted_lines["bottom"], fitted_lines["right"])
    bl = intersect(fitted_lines["bottom"], fitted_lines["left"])

    if any(x is None for x in [tl, tr, br, bl]):
        h, w = image.shape[:2]
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    return np.array([tl, tr, br, bl], dtype=np.float32)


def deskew_floodfill_crop(image) -> QuadPoints:
    h, w = image.shape[:2]

    # Sample background colors from the middle of left and right sides
    left_sample = np.mean(image[h // 2 - 5 : h // 2 + 6, 0:5], axis=(0, 1))
    right_sample = np.mean(image[h // 2 - 5 : h // 2 + 6, w - 6 : w], axis=(0, 1))
    bg_color = ((left_sample + right_sample) / 2).astype(np.uint8)

    # Create a mask of pixels similar to the sampled background color
    lower = np.clip(bg_color.astype(np.int16) - 30, 0, 255).astype(np.uint8)
    upper = np.clip(bg_color.astype(np.int16) + 30, 0, 255).astype(np.uint8)
    bg_mask = cv2.inRange(image, lower, upper)

    # Morphological closing to fill holes inside the document that might be bg-colored
    kernel = np.ones((5, 5), np.uint8)
    closed_bg = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)

    # The document is everything that's NOT the closed background
    doc_mask = cv2.bitwise_not(closed_bg)

    y_indices, x_indices = np.where(doc_mask > 0)
    coords = np.column_stack((x_indices, y_indices))

    if len(coords) == 0:
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


implementations = {
    "minarea_on_edge_detection": deskew_minarea_on_edge_detection,
    "minarea_on_edge_detection_refined": deskew_minarea_on_edge_detection_refined,
    "floodfill_crop": deskew_floodfill_crop,
    "hough": deskew_hough,
    "deskew_minarea_on_diff_quantile": deskew_minarea_on_diff_quantile,
}


@pytest.mark.parametrize("img_path", [f for f in DATA_DIR.glob("*.png")])
@pytest.mark.parametrize("name, func", implementations.items())
def test_deskew_benchmark(benchmark, img_path, name, func):
    image = cv2.imread(str(img_path))
    assert image is not None, f"Could not read {img_path}"

    # Benchmark only the point detection algorithm
    quad_pts = benchmark(func, image)

    # Perform transform outside of the benchmarked section
    result = quad_to_rect(image, quad_pts)

    output_path = OUTPUT_DIR / f"{img_path.stem}_{name}.png"
    cv2.imwrite(str(output_path), result)

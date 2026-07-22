import numpy as np

from auto_sane.transform import quad_to_rect


def create_test_image(width=500, height=500):
    """Creates a black image with colored markers at the corners for verification."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # TL: Red, TR: Green, BR: Blue, BL: Yellow
    img[0:20, 0:20] = [0, 0, 255]  # Red (OpenCV is BGR)
    img[0:20, width - 20 :] = [0, 255, 0]  # Green
    img[height - 20 :, width - 20 :] = [255, 0, 0]  # Blue
    img[height - 20 :, 0:20] = [0, 255, 255]  # Yellow
    return img


def test_quad_to_rect_perfect_rectangle():
    """Case: Input quad is already a perfect rectangle."""
    img = create_test_image()
    # TL, TR, BR, BL - Use coordinates that hit the marker zones (0-20 and 480-500)
    quad = np.array([[10, 10], [490, 10], [490, 490], [10, 490]], dtype=np.float32)
    result = quad_to_rect(img, quad)

    assert result.shape[0] == 480
    assert result.shape[1] == 480
    # Check corners survive by checking the mean color of a small corner area
    assert np.allclose(np.mean(result[0:5, 0:5], axis=(0, 1)), [0, 0, 255], atol=10)
    assert np.allclose(np.mean(result[0:5, -5:], axis=(0, 1)), [0, 255, 0], atol=10)
    assert np.allclose(np.mean(result[-5:, -5:], axis=(0, 1)), [255, 0, 0], atol=10)
    assert np.allclose(np.mean(result[-5:, 0:5], axis=(0, 1)), [0, 255, 255], atol=10)


def test_quad_to_rect_trapezoid():
    """Case: Input is a trapezoid (perspective warp)."""
    img = create_test_image()
    # Top edge shorter than bottom edge
    quad = np.array(
        [
            [100, 100],
            [400, 100],  # TL, TR
            [450, 400],
            [50, 400],  # BR, BL
        ],
        dtype=np.float32,
    )

    result = quad_to_rect(img, quad)

    # Max width is approx 400 (bottom edge), max height is approx 300
    assert result.shape[1] >= 350  # Bottom edge length ~400
    assert result.shape[0] >= 280  # Height difference ~300

    # Verification: the corners of the quad should map to corners of the result
    # We simulate this by putting points in the image and checking if they are at corners afoer transform
    test_img = np.zeros((500, 500, 3), dtype=np.uint8)
    test_img[100, 100] = [255, 0, 0]  # TL
    test_img[100, 400] = [0, 255, 0]  # TR
    test_img[400, 450] = [0, 0, 255]  # BR
    test_img[400, 50] = [255, 255, 0]  # BL

    result = quad_to_rect(test_img, quad)
    # Note: warpPerspective might not map a single pixel exactly to (0,0) if coordinates are float
    # but it should be very close. We check small regions.
    assert np.any(np.all(result[0:5, 0:5] == [255, 0, 0], axis=-1))
    assert np.any(np.all(result[0:5, -5:] == [0, 255, 0], axis=-1))
    assert np.any(np.all(result[-5:, -5:] == [0, 0, 255], axis=-1))
    assert np.any(np.all(result[-5:, 0:5] == [255, 255, 0], axis=-1))


def test_quad_to_rect_rotated():
    """Case: Input quad is a rectangle but rotated."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    # Define a diamond shape (square rotated by 45 deg)
    quad = np.array(
        [
            [250, 100],
            [400, 250],  # TL, TR
            [250, 400],
            [100, 250],  # BR, BL
        ],
        dtype=np.float32,
    )

    # Put colors at these points
    img[100, 250] = [255, 0, 0]
    img[250, 400] = [0, 255, 0]
    img[400, 250] = [0, 0, 255]
    img[250, 100] = [255, 255, 0]

    result = quad_to_rect(img, quad)

    # The result should be a square (roughly 212x212 based on distance sqrt(150^2 + 150^2))
    assert abs(result.shape[0] - result.shape[1]) <= 2

    assert np.any(np.all(result[0:5, 0:5] == [255, 0, 0], axis=-1))
    assert np.any(np.all(result[0:5, -5:] == [0, 255, 0], axis=-1))
    assert np.any(np.all(result[-5:, -5:] == [0, 0, 255], axis=-1))
    assert np.any(np.all(result[-5:, 0:5] == [255, 255, 0], axis=-1))


def test_quad_to_rect_extreme_skew():
    """Case: Highly distorted quad."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    quad = np.array(
        [
            [10, 10],
            [490, 20],  # TL, TR
            [470, 480],
            [30, 450],  # BR, BL
        ],
        dtype=np.float32,
    )

    img[10, 10] = [255, 0, 0]
    img[20, 490] = [0, 255, 0]
    img[480, 470] = [0, 0, 255]
    img[450, 30] = [255, 255, 0]

    result = quad_to_rect(img, quad)

    assert np.any(np.all(result[0:5, 0:5] == [255, 0, 0], axis=-1))
    assert np.any(np.all(result[0:5, -5:] == [0, 255, 0], axis=-1))
    assert np.any(np.all(result[-5:, -5:] == [0, 0, 255], axis=-1))
    assert np.any(np.all(result[-5:, 0:5] == [255, 255, 0], axis=-1))


def test_quad_to_rect_very_small():
    """Case: Very small quad."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    quad = np.array([[100, 100], [105, 100], [105, 105], [100, 105]], dtype=np.float32)

    img[100, 100] = [255, 0, 0]
    img[100, 105] = [0, 255, 0]
    img[105, 105] = [0, 0, 255]
    img[105, 100] = [255, 255, 0]

    result = quad_to_rect(img, quad)

    assert result.shape == (5, 5, 3)  # max_w=5, max_h=5
    assert np.any(np.all(result[0:2, 0:2] == [255, 0, 0], axis=-1))

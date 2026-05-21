from __future__ import annotations

import numpy as np
from PIL import Image

from beltmap._driver_runtime import read_gray


def test_read_gray_preserves_uint16_grayscale_values(tmp_path):
    image = np.asarray(
        [
            [0, 1, 255],
            [256, 4096, 65535],
        ],
        dtype=np.uint16,
    )
    path = tmp_path / "frame_16bit.tif"
    Image.fromarray(image).save(path)

    loaded = read_gray(path)

    assert loaded.dtype == np.float32
    assert loaded.shape == image.shape
    np.testing.assert_array_equal(loaded, image.astype(np.float32))


def test_read_gray_preserves_float_grayscale_values(tmp_path):
    image = np.asarray(
        [
            [0.0, 1.25, 12.5],
            [128.0, 1024.5, 4096.75],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "frame_float.tif"
    Image.fromarray(image).save(path)

    loaded = read_gray(path)

    assert loaded.dtype == np.float32
    assert loaded.shape == image.shape
    np.testing.assert_allclose(loaded, image)


def test_read_gray_converts_rgb_inputs_to_luminance(tmp_path):
    image = np.asarray(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [10, 20, 30]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "frame_rgb.png"
    Image.fromarray(image, mode="RGB").save(path)

    loaded = read_gray(path)

    expected = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    assert loaded.dtype == np.float32
    assert loaded.shape == image.shape[:2]
    np.testing.assert_allclose(loaded, expected.astype(np.float32), rtol=0, atol=1e-5)

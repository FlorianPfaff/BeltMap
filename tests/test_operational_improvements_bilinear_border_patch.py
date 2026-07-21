from __future__ import annotations

import numpy as np

import beltmap  # noqa: F401 - imports side-effect patches
from beltmap import operational_improvements as operational


def test_bilinear_border_sampling_patch_is_autoloaded() -> None:
    assert getattr(
        operational._sample_bilinear,
        "_beltmap_bilinear_border_sampling_patched",
        False,
    )


def test_bilinear_identity_warp_preserves_bottom_and_right_borders() -> None:
    image = np.arange(16, dtype=float).reshape(4, 4)
    model = operational.estimate_homography(
        [(0, 0), (3, 0), (3, 3), (0, 3)],
        [(0, 0), (3, 0), (3, 3), (0, 3)],
    )

    warped = operational.warp_perspective(
        image,
        model,
        image.shape,
        interpolation="bilinear",
        fill_value=-999.0,
    )

    np.testing.assert_allclose(warped, image, atol=1e-12)

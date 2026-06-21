from __future__ import annotations

import numpy as np

from beltmap.yolo_recurrence import CropRegion, patch_correlation, patch_stats


def test_crop_region_parse() -> None:
    region = CropRegion.parse("1,2,3,4")
    assert region.top == 1
    assert region.left == 2
    assert region.height == 3
    assert region.width == 4


def test_patch_correlation_identical_patch_is_one() -> None:
    patch = np.arange(16, dtype=float).reshape(4, 4)
    assert patch_correlation(patch, patch) == 1.0


def test_patch_stats_reports_positive_excess() -> None:
    image = np.full((20, 20), 10.0)
    image[8:12, 8:12] = 50.0
    stats = patch_stats(
        image,
        center_y=10.0,
        center_x=10.0,
        half_y=2.0,
        half_x=2.0,
        signal_margin_px=0,
        background_margin_px=6,
        patch_correlation_margin_px=0,
    )
    assert stats.local_max == 50.0
    assert stats.bg99 == 10.0
    assert stats.excess == 40.0

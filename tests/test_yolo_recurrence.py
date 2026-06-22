from __future__ import annotations

import numpy as np

from beltmap.yolo_recurrence import parse_belt_region, patch_correlation, patch_excess
from beltmap.yolo_recurrence_key_patch import duplicate_safe_row_key


def test_parse_belt_region() -> None:
    region = parse_belt_region("1,2,3,4")
    assert region.top == 1
    assert region.left == 2
    assert region.height == 3
    assert region.width == 4


def test_patch_correlation_identical_patch_is_one() -> None:
    patch = np.arange(16, dtype=float).reshape(4, 4)
    value = patch_correlation(patch, patch)
    assert abs(value - 1.0) < 1e-12


def test_patch_excess_reports_positive_excess() -> None:
    raw_patch = np.full((5, 5), 10.0)
    background_patch = np.full((5, 5), 10.0)
    raw_patch[2, 2] = 50.0
    assert patch_excess(raw_patch, background_patch) == 40.0


def test_duplicate_safe_row_key_distinguishes_same_frame_same_label_boxes() -> None:
    base = {
        "frame_index": "12",
        "label": "0",
        "bbox_top": "10",
        "bbox_left": "20",
        "bbox_bottom": "30",
        "bbox_right": "40",
        "y": "20",
        "x": "30",
        "confidence": "0.9",
        "source": "yolo11_raw",
    }
    second = dict(base)
    second.update({"bbox_left": "120", "bbox_right": "140", "x": "130"})

    assert duplicate_safe_row_key(base) != duplicate_safe_row_key(second)

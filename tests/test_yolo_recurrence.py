from __future__ import annotations

import pytest
import numpy as np

from beltmap.yolo_recurrence import error_taxonomy, parse_belt_region, patch_correlation, patch_excess, row_key
from beltmap.yolo_recurrence_key_patch import (
    correlation_supported_high_revisits,
    duplicate_safe_row_key,
)


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


def test_correlation_supported_high_revisits_ignores_uncorrelated_bright_revisits() -> None:
    row = {
        "recurrence_ratio_prev": "1.4",
        "patch_correlation_prev": "0.0",
        "recurrence_ratio_next": "1.2",
        "patch_correlation_next": "-0.5",
    }

    assert correlation_supported_high_revisits(row, threshold=0.4) == 0


def test_correlation_supported_high_revisits_counts_shape_supported_revisits() -> None:
    row = {
        "recurrence_ratio_prev": "0.9",
        "patch_correlation_prev": "0.7",
        "recurrence_ratio_next": "0.8",
        "patch_correlation_next": "0.6",
    }

    assert correlation_supported_high_revisits(row, threshold=0.4) == 2


def test_error_taxonomy_ignores_uncorrelated_bright_revisits() -> None:
    feature = {
        "valid_revisits": "2",
        "hard_reject": "False",
        "max_recurrence_ratio": "1.4",
        "belt_fixedness_score": "0.0",
        "high_recurrence_revisits": "0",
    }

    assert error_taxonomy(feature, role="FP") == "fp_low_shape_supported_recurrence_evidence"


def test_error_taxonomy_reports_shape_supported_recurrence() -> None:
    feature = {
        "valid_revisits": "2",
        "hard_reject": "False",
        "max_recurrence_ratio": "1.4",
        "belt_fixedness_score": "0.45",
        "high_recurrence_revisits": "1",
    }

    assert error_taxonomy(feature, role="TP") == "tp_shape_supported_recurrent_but_not_hard_rejected"


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


def test_duplicate_safe_row_key_rejects_missing_required_detection_identity() -> None:
    row = {
        "frame_index": "12",
        "bbox_top": "10",
        "bbox_left": "20",
        "bbox_bottom": "30",
        "bbox_right": "40",
        "y": "20",
        "x": "30",
        "confidence": "0.9",
        "source": "yolo11_raw",
    }

    with pytest.raises(ValueError, match="label is required"):
        duplicate_safe_row_key(row)


def test_duplicate_safe_row_key_rejects_missing_center_geometry() -> None:
    row = {
        "frame_index": "12",
        "label": "0",
        "bbox_top": "10",
        "bbox_left": "20",
        "bbox_bottom": "30",
        "bbox_right": "40",
        "x": "30",
        "confidence": "0.9",
        "source": "yolo11_raw",
    }

    with pytest.raises(ValueError, match="y is required"):
        duplicate_safe_row_key(row)


def test_duplicate_safe_row_key_rejects_fractional_frame_index() -> None:
    row = {
        "frame_index": "12.5",
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

    with pytest.raises(ValueError, match="frame_index must be integer-valued"):
        duplicate_safe_row_key(row)


def test_direct_yolo_recurrence_row_key_is_duplicate_safe() -> None:
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

    assert row_key(base) != row_key(second)
    assert row_key(base) == duplicate_safe_row_key(base)

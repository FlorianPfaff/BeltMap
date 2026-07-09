from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from beltmap.yolo_recurrence import (
    FEATURE_FIELDNAMES,
    RUN_EXTRA_FIELDS,
    enrich_detection_row,
    error_taxonomy,
    find_source_images,
    parse_belt_region,
    patch_correlation,
    patch_excess,
    row_key,
)
from beltmap.yolo_recurrence_key_patch import (
    THRESHOLD_FIELD,
    correlation_supported_belt_fixedness_score,
    correlation_supported_high_revisits,
    duplicate_safe_row_key,
)


def write_image(path: Path, *, size: tuple[int, int] = (12, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, 32).save(path)


def test_parse_belt_region() -> None:
    region = parse_belt_region("1,2,3,4")
    assert region.top == 1
    assert region.left == 2
    assert region.height == 3
    assert region.width == 4


def test_find_source_images_skips_supported_nonframe_auxiliary_images(tmp_path: Path) -> None:
    write_image(tmp_path / "frame_000001.png")
    write_image(tmp_path / "preview_without_digits.png")

    assert find_source_images(tmp_path) == {1: tmp_path / "frame_000001.png"}


def test_find_source_images_still_rejects_duplicate_parseable_frames(tmp_path: Path) -> None:
    write_image(tmp_path / "raw" / "frame_000001.png")
    write_image(tmp_path / "augmented" / "copy_000001.jpg")

    with pytest.raises(ValueError, match="duplicate source image frame index 1"):
        find_source_images(tmp_path)


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


def test_correlation_supported_belt_fixedness_uses_ranked_supported_revisit() -> None:
    row = {
        "recurrence_ratio_prev": "0.9",
        "patch_correlation_prev": "0.7",  # strength 0.63
        "recurrence_ratio_next": "0.8",
        "patch_correlation_next": "0.6",  # strength 0.48
    }

    assert correlation_supported_belt_fixedness_score(row, min_revisits=2) == pytest.approx(0.48)


def test_correlation_supported_belt_fixedness_uses_one_sided_revisit_when_only_one_is_visible() -> None:
    row = {
        "recurrence_ratio_prev": "0.9",
        "patch_correlation_prev": "0.7",
        "recurrence_ratio_next": "",
        "patch_correlation_next": "",
    }

    assert correlation_supported_belt_fixedness_score(row, min_revisits=2) == pytest.approx(0.63)


def test_error_taxonomy_ignores_uncorrelated_bright_revisits() -> None:
    feature = {
        "valid_revisits": "2",
        "hard_reject": "False",
        "max_recurrence_ratio": "1.4",
        "belt_fixedness_score": "0.0",
        "high_recurrence_revisits": "0",
    }

    assert error_taxonomy(feature, role="FP") == "fp_low_shape_supported_recurrence_evidence"


def test_error_taxonomy_uses_configured_supported_revisit_count_not_default_threshold() -> None:
    feature = {
        "valid_revisits": "2",
        "hard_reject": "False",
        "max_recurrence_ratio": "1.4",
        "belt_fixedness_score": "0.45",
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


def test_threshold_field_is_exported_for_feature_and_filtered_runs() -> None:
    row = {
        "frame_index": "12",
        "label": "0",
        "y": "20",
        "x": "30",
        "area_px": "400",
        "bbox_top": "10",
        "bbox_left": "20",
        "bbox_bottom": "30",
        "bbox_right": "40",
        "score": "0.9",
        "confidence": "0.9",
        "source": "yolo11_raw",
    }
    feature = {
        THRESHOLD_FIELD: "0.7",
        "transient_score": "0.8",
        "belt_fixedness_score": "0.2",
        "max_recurrence_ratio": "0.5",
        "high_recurrence_revisits": "1",
        "hard_reject": "False",
    }

    assert THRESHOLD_FIELD in FEATURE_FIELDNAMES
    assert THRESHOLD_FIELD in RUN_EXTRA_FIELDS
    assert enrich_detection_row(row, feature, rerank=False)[THRESHOLD_FIELD] == "0.7"
    assert enrich_detection_row(row, feature, rerank=True)[THRESHOLD_FIELD] == "0.7"


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


def test_yolo_recurrence_patch_reload_is_idempotent() -> None:
    import beltmap.yolo_recurrence as yolo_recurrence
    import beltmap.yolo_recurrence_key_patch as key_patch

    before = yolo_recurrence.score_detection_recurrence
    before_original = getattr(before, "_beltmap_yolo_recurrence_original", before)

    importlib.reload(key_patch)

    after = yolo_recurrence.score_detection_recurrence
    after_original = getattr(after, "_beltmap_yolo_recurrence_original", after)

    assert getattr(after, "_beltmap_yolo_recurrence_patched", False)
    assert after_original is before_original
    assert after_original is not after

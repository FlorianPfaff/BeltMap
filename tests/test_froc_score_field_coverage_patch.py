from __future__ import annotations

import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.compare_runs as compare_runs


def test_froc_score_field_coverage_patch_is_autoloaded() -> None:
    assert getattr(
        compare_runs.detection_score_field,
        "_beltmap_froc_score_field_coverage_patched",
        False,
    )


def test_froc_prefers_score_field_with_greatest_detection_coverage() -> None:
    truth = {
        "particles": [
            {
                "frame_index": 0,
                "top": 1,
                "left": 2,
                "bottom": 4,
                "right": 5,
            }
        ]
    }
    detections = [
        {
            "frame_index": "0",
            "bbox_top": "1",
            "bbox_left": "2",
            "bbox_bottom": "4",
            "bbox_right": "5",
            "y": "2.0",
            "x": "3.0",
            "peak_signal": "12.0",
            "confidence": "0.9",
        },
        {
            "frame_index": "1",
            "bbox_top": "10",
            "bbox_left": "10",
            "bbox_bottom": "12",
            "bbox_right": "12",
            "y": "11.0",
            "x": "11.0",
            "peak_signal": "",
            "confidence": "0.2",
        },
    ]

    froc = compare_runs.detection_froc_curve(
        detections,
        truth,
        scored_frames={0, 1},
        iou_threshold=0.25,
        max_thresholds=None,
    )

    assert froc["available"] is True
    assert froc["score_field"] == "confidence"
    assert froc["skipped_score_rows"] == 0

    threshold_points = {
        point["score_threshold"]: point
        for point in froc["points"]
        if point["score_threshold"] is not None
    }
    assert set(threshold_points) == {0.9, 0.2}
    assert threshold_points[0.9]["predicted_boxes"] == 1
    assert threshold_points[0.9]["false_positives"] == 0
    assert threshold_points[0.9]["recall"] == pytest.approx(1.0)
    assert threshold_points[0.2]["predicted_boxes"] == 2
    assert threshold_points[0.2]["false_positives"] == 1

from __future__ import annotations

from typing import Any

import pytest

import beltmap  # noqa: F401 - imports side-effect patches
from beltmap import benchmark


def _truth_box(event_id: str, frame_index: int) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "frame_index": frame_index,
        "top": 0,
        "left": 0,
        "bottom": 4,
        "right": 4,
    }


def _prediction_box(track_id: str, frame_index: int) -> dict[str, str]:
    return {
        "track_id": track_id,
        "frame_index": str(frame_index),
        "bbox_top": "0",
        "bbox_left": "0",
        "bbox_bottom": "4",
        "bbox_right": "4",
    }


def test_event_matching_patch_is_autoloaded() -> None:
    assert getattr(
        benchmark.event_metrics,
        "_beltmap_cardinality_optimal_event_matching_patched",
        False,
    )


def test_event_metrics_maximize_match_cardinality_before_temporal_iou() -> None:
    truth = {
        "particles": [
            _truth_box("t0", 0),
            _truth_box("t0", 1),
            _truth_box("t1", 0),
        ]
    }
    predictions = [
        _prediction_box("p0", 0),
        _prediction_box("p0", 1),
        _prediction_box("p1", 1),
    ]

    metrics = benchmark.event_metrics(
        predictions,
        truth,
        iou_threshold=0.5,
    )

    assert metrics["matched_events"] == 2
    assert metrics["false_positive_events"] == 0
    assert metrics["false_negative_events"] == 0
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert {
        (match["pred_event_id"], match["truth_event_id"])
        for match in metrics["matches"]
    } == {
        ("pred:p0", "truth:t1"),
        ("pred:p1", "truth:t0"),
    }

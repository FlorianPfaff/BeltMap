import pytest
import numpy as np

from beltmap.track_diagnostics import (
    accepted_track_quality_summary,
    detection_quality_summary,
    finite_float,
)


def test_finite_float_rejects_boolean_values():
    assert finite_float(True) is None
    assert finite_float(False) is None
    assert finite_float(np.bool_(True)) is None


def test_detection_quality_summary_ignores_boolean_measurements():
    rows = [
        {"area_px": True, "peak_signal": False},
        {"area_px": 10, "peak_signal": 6},
    ]

    summary = detection_quality_summary(rows, detection_threshold=5, near_threshold_margin=1)

    assert summary["area_px"]["count"] == 1
    assert summary["peak_signal"]["count"] == 1
    assert summary["small_detections_area_lt_threshold"] == 1
    assert summary["near_threshold_peak_count"] == 1


def test_accepted_track_quality_summary_ignores_boolean_area_values():
    rows = [
        {"track_id": "bad", "area_px": True},
        {"track_id": "bad", "area_px": False},
        {"track_id": "real", "area_px": 10},
    ]

    summary = accepted_track_quality_summary(rows)

    assert summary["tracks"] == 2
    assert summary["tracks_with_area"] == 1
    assert summary["small_accepted_tracks"] == 1
    assert summary["small_accepted_track_ids_preview"] == ["real"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"small_area_threshold_px": float("nan")}, "small_area_threshold_px"),
        ({"detection_threshold": -0.1}, "detection_threshold"),
        ({"near_threshold_margin": float("inf")}, "near_threshold_margin"),
    ],
)
def test_detection_quality_summary_rejects_invalid_thresholds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        detection_quality_summary([], **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"small_area_threshold_px": -1.0}, "small_area_threshold_px"),
        ({"long_track_min_detections": 1.5}, "long_track_min_detections"),
        ({"very_long_track_min_detections": 1.5}, "very_long_track_min_detections"),
        (
            {"long_track_min_detections": 5, "very_long_track_min_detections": 4},
            "very_long_track_min_detections",
        ),
    ],
)
def test_accepted_track_quality_summary_rejects_invalid_thresholds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        accepted_track_quality_summary([], **kwargs)

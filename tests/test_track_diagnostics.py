import pytest

from beltmap.track_diagnostics import (
    accepted_track_quality_summary,
    detection_quality_summary,
)


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

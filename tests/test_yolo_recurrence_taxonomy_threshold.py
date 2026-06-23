from __future__ import annotations

from beltmap.yolo_recurrence import error_taxonomy


def test_error_taxonomy_uses_configured_threshold_for_low_evidence() -> None:
    feature = {
        "valid_revisits": "2",
        "hard_reject": "False",
        "max_recurrence_ratio": "1.4",
        "belt_fixedness_score": "0.45",
        "high_recurrence_revisits": "0",
        "hard_ratio_threshold": "0.80",
    }

    assert error_taxonomy(feature, role="FP") == "fp_low_shape_supported_recurrence_evidence"


def test_error_taxonomy_uses_configured_threshold_for_supported_recurrence() -> None:
    feature = {
        "valid_revisits": "2",
        "hard_reject": "False",
        "max_recurrence_ratio": "1.4",
        "belt_fixedness_score": "0.45",
        "high_recurrence_revisits": "0",
        "hard_ratio_threshold": "0.30",
    }

    assert error_taxonomy(feature, role="FP") == "fp_shape_supported_recurrent_but_not_hard_rejected"

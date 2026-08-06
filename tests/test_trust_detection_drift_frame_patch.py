from __future__ import annotations

import pytest

import beltmap  # noqa: F401 - imports side-effect patches
from beltmap import trust


def test_detection_drift_frame_axis_patch_is_autoloaded() -> None:
    assert getattr(
        trust.run_drift_report,
        "_beltmap_detection_drift_frame_axis_patched",
        False,
    )


def test_detection_drift_uses_sparse_source_frame_indices(tmp_path) -> None:
    (tmp_path / "detections_per_frame.csv").write_text(
        "frame_index,n_detections\n"
        "100,0\n"
        "1100,1\n",
        encoding="utf-8",
    )

    report = trust.run_drift_report(tmp_path)

    assert report["detection_count_slope_per_frame"] == pytest.approx(0.001)
    assert not any(
        warning.startswith("detection counts drift")
        for warning in report["warnings"]
    )


def test_detection_drift_keeps_positional_fallback_without_frame_indices(
    tmp_path,
) -> None:
    (tmp_path / "detections_per_frame.csv").write_text(
        "n_detections\n"
        "0\n"
        "2\n",
        encoding="utf-8",
    )

    report = trust.run_drift_report(tmp_path)

    assert report["detection_count_slope_per_frame"] == pytest.approx(2.0)
    assert any(
        warning.startswith("detection counts drift")
        for warning in report["warnings"]
    )

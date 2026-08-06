"""Measure detection-count drift against recorded frame indices.

``run_drift_report`` historically regressed ``detections_per_frame.csv`` counts
against their row positions. Subset and strided runs retain sparse or absolute
``frame_index`` values, so the reported ``detection_count_slope_per_frame`` could
be exaggerated by the spacing between rows and trigger false drift warnings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import trust as _trust

_PATCHED_ATTR = "_beltmap_detection_drift_frame_axis_patched"
_ORIGINAL_ATTR = "_beltmap_original_run_drift_report"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original report helper if this patch is reloaded."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_run_drift_report = _unwrap_patched_callable(_trust.run_drift_report)


def frame_aware_run_drift_report(output_dir: Path) -> dict[str, Any]:
    """Report detection-count drift per source frame, not per CSV row.

    When at least two distinct finite frame indices have finite detection counts,
    those aligned pairs define the regression axis. Legacy files without usable
    frame indices retain the historical positional fallback. All finite counts
    still contribute to the descriptive summary.
    """

    detection_counts: list[float] = []
    detection_frames: list[float] = []
    aligned_detection_counts: list[float] = []
    for row in _trust.read_csv_rows(output_dir / "detections_per_frame.csv"):
        count = _trust.finite_float(row.get("n_detections"))
        if count is None:
            continue
        detection_counts.append(count)
        frame = _trust.finite_float(row.get("frame_index"))
        if frame is not None:
            detection_frames.append(frame)
            aligned_detection_counts.append(count)

    phase_rows = _trust.read_csv_rows(output_dir / "phase_estimates.csv")
    corrections: list[float] = []
    correction_frames: list[float] = []
    losses: list[float] = []
    loss_frames: list[float] = []
    for row in phase_rows:
        frame = _trust.finite_float(row.get("frame_index"))
        if frame is None:
            continue
        correction = _trust.finite_float(row.get("correction_px"))
        loss = _trust.finite_float(row.get("loss"))
        if correction is not None:
            correction_frames.append(frame)
            corrections.append(correction)
        if loss is not None:
            loss_frames.append(frame)
            losses.append(loss)

    if len(set(detection_frames)) >= 2:
        detection_slope = _trust._linear_slope(
            detection_frames,
            aligned_detection_counts,
        )
    else:
        detection_slope = _trust._linear_slope(
            list(range(len(detection_counts))),
            detection_counts,
        )
    correction_slope = (
        _trust._linear_slope(correction_frames, corrections) if corrections else None
    )
    loss_slope = _trust._linear_slope(loss_frames, losses) if losses else None

    warnings: list[str] = []
    if detection_slope is not None and abs(detection_slope) > 0.01:
        warnings.append(
            "detection counts drift over time; check contamination, illumination "
            "drift, or threshold stability"
        )
    if loss_slope is not None and loss_slope > 0:
        warnings.append(
            "registration loss increases over time; consider multi-epoch maps or "
            "illumination correction"
        )
    if correction_slope is not None and abs(correction_slope) > 0.05:
        warnings.append("phase correction drift suggests speed or timing mismatch")

    return {
        "output_dir": str(output_dir),
        "detections_per_frame_summary": _trust.summarize_numeric(detection_counts),
        "detection_count_slope_per_frame": detection_slope,
        "phase_correction_summary": _trust.summarize_numeric(corrections),
        "phase_correction_slope_px_per_frame": correction_slope,
        "registration_loss_summary": _trust.summarize_numeric(losses),
        "registration_loss_slope_per_frame": loss_slope,
        "warnings": warnings,
    }


setattr(frame_aware_run_drift_report, _PATCHED_ATTR, True)
setattr(
    frame_aware_run_drift_report,
    _ORIGINAL_ATTR,
    _original_run_drift_report,
)
_trust.run_drift_report = frame_aware_run_drift_report

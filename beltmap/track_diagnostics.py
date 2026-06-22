from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

DEFAULT_SMALL_AREA_THRESHOLD_PX = 50.0
DEFAULT_NEAR_THRESHOLD_MARGIN = 1.0


def finite_float(value: Any) -> float | None:
    """Parse a finite float value, returning ``None`` for blanks."""

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    """Return compact scalar statistics for finite values."""

    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def safe_share(count: int, total: int) -> float | None:
    return None if total <= 0 else float(count / total)


def group_rows_by_track_id(
    rows: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        track_id = str(row.get("track_id", "")).strip() or f"missing:{index}"
        grouped.setdefault(track_id, []).append(row)
    return grouped


def detection_quality_summary(
    rows: list[dict[str, Any]],
    *,
    small_area_threshold_px: float = DEFAULT_SMALL_AREA_THRESHOLD_PX,
    detection_threshold: float | None = None,
    near_threshold_margin: float = DEFAULT_NEAR_THRESHOLD_MARGIN,
) -> dict[str, Any]:
    """Summarize small and weak detections that can survive velocity checks."""

    small_area_threshold_px = _nonnegative_float_value(
        small_area_threshold_px,
        "small_area_threshold_px",
    )
    near_threshold_margin = _nonnegative_float_value(
        near_threshold_margin,
        "near_threshold_margin",
    )
    if detection_threshold is not None:
        detection_threshold = _nonnegative_float_value(
            detection_threshold,
            "detection_threshold",
        )

    areas = [
        value for row in rows if (value := finite_float(row.get("area_px"))) is not None
    ]
    peaks = [
        value
        for row in rows
        if (value := finite_float(row.get("peak_signal"))) is not None
    ]
    small_detections = sum(1 for value in areas if value < small_area_threshold_px)

    near_threshold_count: int | None = None
    near_threshold_share: float | None = None
    if detection_threshold is not None:
        weak_limit = detection_threshold + near_threshold_margin
        near_threshold_count = sum(
            1 for value in peaks if detection_threshold <= value <= weak_limit
        )
        near_threshold_share = safe_share(near_threshold_count, len(peaks))

    return {
        "small_area_threshold_px": small_area_threshold_px,
        "small_detections_area_lt_threshold": small_detections,
        "small_detection_share_area_lt_threshold": safe_share(
            small_detections,
            len(areas),
        ),
        "area_px": describe(areas),
        "peak_signal": describe(peaks),
        "near_threshold_peak_margin": near_threshold_margin,
        "near_threshold_peak_count": near_threshold_count,
        "near_threshold_peak_share": near_threshold_share,
    }


def accepted_track_quality_summary(
    rows: list[dict[str, Any]],
    *,
    small_area_threshold_px: float = DEFAULT_SMALL_AREA_THRESHOLD_PX,
    long_track_min_detections: int = 5,
    very_long_track_min_detections: int = 10,
) -> dict[str, Any]:
    """Summarize accepted tracks that remain suspicious despite plausible velocity."""

    small_area_threshold_px = _nonnegative_float_value(
        small_area_threshold_px,
        "small_area_threshold_px",
    )
    long_track_min_detections = _positive_integer_value(
        long_track_min_detections,
        "long_track_min_detections",
    )
    very_long_track_min_detections = _positive_integer_value(
        very_long_track_min_detections,
        "very_long_track_min_detections",
    )
    if very_long_track_min_detections < long_track_min_detections:
        raise ValueError(
            "very_long_track_min_detections must be greater than or equal to "
            "long_track_min_detections"
        )

    grouped = group_rows_by_track_id(rows)
    track_lengths: list[int] = []
    mean_areas: list[float] = []
    median_areas: list[float] = []
    small_track_ids: list[str] = []
    long_small_track_ids: list[str] = []
    very_long_small_track_ids: list[str] = []
    tracks_with_area = 0

    for track_id, track_rows in grouped.items():
        track_lengths.append(len(track_rows))
        areas = [
            value
            for row in track_rows
            if (value := finite_float(row.get("area_px"))) is not None
        ]
        if not areas:
            continue
        tracks_with_area += 1
        mean_area = float(np.mean(areas))
        mean_areas.append(mean_area)
        median_areas.append(float(np.median(areas)))
        if mean_area < small_area_threshold_px:
            small_track_ids.append(track_id)
            if len(track_rows) >= long_track_min_detections:
                long_small_track_ids.append(track_id)
            if len(track_rows) >= very_long_track_min_detections:
                very_long_small_track_ids.append(track_id)

    return {
        "available": bool(rows),
        "small_area_threshold_px": small_area_threshold_px,
        "track_rows": len(rows),
        "tracks": len(grouped),
        "tracks_with_area": tracks_with_area,
        "track_length": describe(track_lengths),
        "mean_area_px": describe(mean_areas),
        "median_area_px": describe(median_areas),
        "small_accepted_tracks": len(small_track_ids),
        "small_accepted_track_share": safe_share(
            len(small_track_ids),
            tracks_with_area,
        ),
        "long_small_accepted_tracks_ge_5": len(long_small_track_ids),
        "long_small_accepted_tracks_ge_10": len(very_long_small_track_ids),
        "small_accepted_track_ids_preview": small_track_ids[:20],
    }


def _nonnegative_float_value(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _positive_integer_value(value: int, name: str) -> int:
    parsed = float(value)
    if not np.isfinite(parsed) or not parsed.is_integer() or parsed < 1:
        raise ValueError(f"{name} must be a finite positive integer")
    return int(parsed)

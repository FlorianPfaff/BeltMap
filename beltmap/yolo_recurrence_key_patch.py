from __future__ import annotations

import math
from typing import Any, Mapping

from beltmap import yolo_recurrence as _yolo_recurrence


def _required(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"{key} is required for YOLO recurrence row keys")
    return value


def _finite_float(row: Mapping[str, Any], key: str) -> float:
    parsed = float(_required(row, key))
    if not math.isfinite(parsed):
        raise ValueError(f"{key} must be finite for YOLO recurrence row keys")
    return parsed


def _integer_value(row: Mapping[str, Any], key: str) -> int:
    parsed = _finite_float(row, key)
    if not parsed.is_integer():
        raise ValueError(f"{key} must be integer-valued for YOLO recurrence row keys")
    return int(parsed)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _recurrence_strength(ratio: float, correlation: float) -> float:
    return max(0.0, min(1.0, ratio)) * max(0.0, correlation)


def correlation_supported_high_revisits(
    row: Mapping[str, Any],
    *,
    threshold: float,
) -> int:
    """Count revisits whose recurrence is supported by both signal and shape.

    The original hard filter used the recurrence ratio alone.  In dense particle
    scenes, another transient particle can make the same belt coordinate bright
    one revolution away even when the residual patch is not actually correlated
    with the original detection.  The belt-fixedness score already combines the
    clipped ratio with nonnegative patch correlation; the hard-reject decision
    should use the same evidence instead of rejecting on ratio alone.
    """

    count = 0
    for suffix in ("prev", "next"):
        ratio = _optional_float(row.get(f"recurrence_ratio_{suffix}"))
        corr = _optional_float(row.get(f"patch_correlation_{suffix}"))
        if ratio is None or corr is None:
            continue
        if _recurrence_strength(ratio, corr) >= threshold:
            count += 1
    return count


def duplicate_safe_row_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    """Return a stable detection key that does not collapse same-class detections.

    YOLO exports usually have one class label for all particle boxes.  The
    recurrence scorer stores per-detection features and truth-match roles in
    dictionaries, so a key containing only ``(frame_index, label)`` aliases every
    particle candidate in the same frame.  Including the crop-local box geometry
    and confidence keeps ordinary same-frame particle detections distinct while
    remaining deterministic across feature, hard-filter, and rerank passes.

    Do not silently invent missing geometry or label values here.  A malformed
    exported detection row should fail before scoring; otherwise unrelated rows
    could be collapsed onto a default ``label=0`` or ``x/y=0`` key.
    """

    confidence = row.get("confidence", row.get("score", ""))
    return (
        _integer_value(row, "frame_index"),
        _integer_value(row, "label"),
        _integer_value(row, "bbox_top"),
        _integer_value(row, "bbox_left"),
        _integer_value(row, "bbox_bottom"),
        _integer_value(row, "bbox_right"),
        round(_finite_float(row, "y"), 3),
        round(_finite_float(row, "x"), 3),
        str(confidence),
        str(row.get("source", "")),
    )


_original_score_detection_recurrence = _yolo_recurrence.score_detection_recurrence


def correlation_gated_score_detection_recurrence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Patch the hard filter to use the same signal+shape evidence as scoring."""

    result = _original_score_detection_recurrence(*args, **kwargs)
    config = kwargs.get("config")
    threshold = float(
        getattr(config, "hard_ratio_threshold", _yolo_recurrence.DEFAULT_HARD_RATIO_THRESHOLD)
    )
    min_revisits = int(getattr(config, "hard_min_revisits", _yolo_recurrence.DEFAULT_HARD_MIN_REVISITS))
    high_revisits = correlation_supported_high_revisits(result, threshold=threshold)
    result["high_recurrence_revisits"] = high_revisits
    result["hard_reject"] = high_revisits >= min_revisits
    return result


# Backwards-compatible safety patch for older imports.  The production CLI
# imports this module before running the recurrence scorer.  The underlying
# scorer also needs these fixes when used directly; ``beltmap.__init__`` imports
# this module for that side effect until ``beltmap.yolo_recurrence`` is updated
# in place.
_yolo_recurrence.row_key = duplicate_safe_row_key
_yolo_recurrence.score_detection_recurrence = correlation_gated_score_detection_recurrence

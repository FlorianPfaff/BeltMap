from __future__ import annotations

import math
from typing import Any, Mapping

from beltmap import yolo_recurrence as _yolo_recurrence

_PATCHED_ATTR = "_beltmap_yolo_recurrence_patched"
_ORIGINAL_ATTR = "_beltmap_yolo_recurrence_original"


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


def _optional_int(value: Any) -> int:
    parsed = _optional_float(value)
    if parsed is None:
        return 0
    return int(parsed)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _recurrence_strength(ratio: float, correlation: float) -> float:
    return max(0.0, min(1.0, ratio)) * max(0.0, correlation)


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original unpatched callable behind our wrapper, if present.

    This module is imported for side effects from ``beltmap.__init__`` and from
    the YOLO recurrence CLI.  A normal second import is cached, but test suites
    and notebooks can reload modules.  Without unwrapping, a reload would store
    the previous wrapper as ``_original_score_detection_recurrence`` and the new
    wrapper would recursively call itself.
    """

    return getattr(func, _ORIGINAL_ATTR, func)


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


_original_score_detection_recurrence = _unwrap_patched_callable(
    _yolo_recurrence.score_detection_recurrence
)


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


setattr(correlation_gated_score_detection_recurrence, _PATCHED_ATTR, True)
setattr(correlation_gated_score_detection_recurrence, _ORIGINAL_ATTR, _original_score_detection_recurrence)


def correlation_gated_error_taxonomy(feature: Mapping[str, Any], *, role: str) -> str:
    """Classify recurrence errors using the same evidence as the hard filter.

    The original taxonomy used ``max_recurrence_ratio`` alone.  After gating the
    hard filter by signal *and* patch correlation, that made uncorrelated bright
    revisits look like recurrent belt-fixed evidence in reports even though they
    could never trigger the hard filter.  Use ``high_recurrence_revisits`` and
    ``belt_fixedness_score`` instead so the report describes the actual decision
    evidence.
    """

    valid = _optional_int(feature.get("valid_revisits"))
    hard_reject = _bool_value(feature.get("hard_reject"))
    supported_revisits = _optional_int(feature.get("high_recurrence_revisits"))
    supported_score = _optional_float(feature.get("belt_fixedness_score")) or 0.0
    threshold = _yolo_recurrence.DEFAULT_HARD_RATIO_THRESHOLD
    role_lower = role.lower()
    if valid == 0:
        return f"{role_lower}_no_valid_revisits"
    if role == "FP" and hard_reject:
        return "fp_recurrent_removed"
    if role == "FP" and supported_revisits == 0 and supported_score < threshold:
        return "fp_low_shape_supported_recurrence_evidence"
    if role == "TP" and hard_reject:
        return "tp_high_shape_supported_recurrence_accidentally_removed"
    if supported_revisits > 0 or supported_score >= threshold:
        return f"{role_lower}_shape_supported_recurrent_but_not_hard_rejected"
    return f"{role_lower}_inconclusive_low_recurrence"


# Backwards-compatible safety patch for older imports.  The production CLI
# imports this module before running the recurrence scorer.  The underlying
# scorer also needs these fixes when used directly; ``beltmap.__init__`` imports
# this module for that side effect until ``beltmap.yolo_recurrence`` is updated
# in place.  The wrapper is idempotent, so reloading this module cannot wrap a
# previous wrapper and recurse.
_yolo_recurrence.row_key = duplicate_safe_row_key
_yolo_recurrence.score_detection_recurrence = correlation_gated_score_detection_recurrence
_yolo_recurrence.error_taxonomy = correlation_gated_error_taxonomy

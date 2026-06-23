from __future__ import annotations

import math
from typing import Any, Mapping

from PIL import Image

from beltmap import yolo_recurrence as _yolo_recurrence

_PATCHED_ATTR = "_beltmap_yolo_recurrence_patched"
_ORIGINAL_ATTR = "_beltmap_yolo_recurrence_original"
THRESHOLD_FIELD = "hard_ratio_threshold"


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


def _shape_supported_strengths(row: Mapping[str, Any]) -> list[float]:
    strengths: list[float] = []
    for suffix in ("prev", "next"):
        ratio = _optional_float(row.get(f"recurrence_ratio_{suffix}"))
        corr = _optional_float(row.get(f"patch_correlation_{suffix}"))
        if ratio is None or corr is None:
            continue
        strengths.append(_recurrence_strength(ratio, corr))
    return strengths


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original unpatched callable behind our wrapper, if present.

    This module is imported for side effects from ``beltmap.__init__`` and from
    the YOLO recurrence CLI.  A normal second import is cached, but test suites
    and notebooks can reload modules.  Without unwrapping, a reload would store
    the previous wrapper as ``_original_score_detection_recurrence`` and the new
    wrapper would recursively call itself.
    """

    return getattr(func, _ORIGINAL_ATTR, func)


def _ensure_field(fieldnames: list[str], field: str) -> None:
    if field not in fieldnames:
        fieldnames.append(field)


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

    return sum(strength >= threshold for strength in _shape_supported_strengths(row))


def correlation_supported_belt_fixedness_score(
    row: Mapping[str, Any],
    *,
    min_revisits: int,
) -> float:
    """Return the shape-supported recurrence score used for YOLO reranking.

    Hard rejection still requires ``min_revisits`` supported revisits.  The soft
    rerank score should nevertheless use available one-sided recurrence evidence
    when only a previous or next revolution is visible.  The old patched path kept
    the core module's second-largest score, so detections near the beginning or
    end of a split got ``belt_fixedness_score = 0`` even when their only valid
    revisit was strongly shape-supported.  That made the rerank path blind to
    exactly the one-sided cases shown in the contact sheets.
    """

    strengths = sorted(_shape_supported_strengths(row), reverse=True)
    if not strengths:
        return 0.0
    rank = min(max(1, int(min_revisits)), len(strengths))
    return float(strengths[rank - 1])


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
_original_enrich_detection_row = _unwrap_patched_callable(_yolo_recurrence.enrich_detection_row)
_original_load_crop = _unwrap_patched_callable(_yolo_recurrence.load_crop)


def correlation_gated_score_detection_recurrence(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Patch the hard filter and rerank score to use shape-supported recurrence."""

    result = _original_score_detection_recurrence(*args, **kwargs)
    config = kwargs.get("config")
    threshold = float(
        getattr(config, "hard_ratio_threshold", _yolo_recurrence.DEFAULT_HARD_RATIO_THRESHOLD)
    )
    min_revisits = int(getattr(config, "hard_min_revisits", _yolo_recurrence.DEFAULT_HARD_MIN_REVISITS))
    high_revisits = correlation_supported_high_revisits(result, threshold=threshold)
    belt_fixedness = correlation_supported_belt_fixedness_score(result, min_revisits=min_revisits)
    result[THRESHOLD_FIELD] = threshold
    result["high_recurrence_revisits"] = high_revisits
    result["belt_fixedness_score"] = belt_fixedness
    result["transient_score"] = max(0.05, min(1.0, 1.0 - belt_fixedness))
    result["hard_reject"] = high_revisits >= min_revisits
    return result


setattr(correlation_gated_score_detection_recurrence, _PATCHED_ATTR, True)
setattr(correlation_gated_score_detection_recurrence, _ORIGINAL_ATTR, _original_score_detection_recurrence)


def persist_threshold_enrich_detection_row(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Keep the configured recurrence threshold in exported hard/rerank runs."""

    enriched = dict(_original_enrich_detection_row(*args, **kwargs))
    feature = args[1] if len(args) >= 2 else kwargs.get("feature")
    if isinstance(feature, Mapping) and THRESHOLD_FIELD in feature:
        enriched[THRESHOLD_FIELD] = feature[THRESHOLD_FIELD]
    return enriched


setattr(persist_threshold_enrich_detection_row, _PATCHED_ATTR, True)
setattr(persist_threshold_enrich_detection_row, _ORIGINAL_ATTR, _original_enrich_detection_row)


def bounds_checked_load_crop(*args: Any, **kwargs: Any) -> Any:
    """Reject out-of-image belt regions before PIL pads them with black pixels.

    ``PIL.Image.crop`` silently pads regions extending beyond the source image.
    Recurrence scoring would then compare detector boxes against artificial black
    borders, which can suppress or amplify apparent recurrence evidence.  The
    BeltMap crop must be fully contained in the source frame; fail with a clear
    error instead of producing padded crops.
    """

    if args:
        frame_index = int(args[0])
    else:
        frame_index = int(kwargs.get("frame_index"))
    source_images = kwargs.get("source_images")
    region = kwargs.get("region")
    if not isinstance(source_images, Mapping) or region is None:
        return _original_load_crop(*args, **kwargs)
    path = source_images.get(frame_index)
    if path is None:
        return _original_load_crop(*args, **kwargs)

    with Image.open(path) as image:
        image_width, image_height = image.size
    left = int(region.left)
    top = int(region.top)
    right = left + int(region.width)
    bottom = top + int(region.height)
    if left < 0 or top < 0 or right > image_width or bottom > image_height:
        raise ValueError(
            "belt region exceeds source image bounds for frame "
            f"{frame_index}: region=(top={top}, left={left}, "
            f"height={region.height}, width={region.width}), "
            f"image=(height={image_height}, width={image_width})"
        )
    return _original_load_crop(*args, **kwargs)


setattr(bounds_checked_load_crop, _PATCHED_ATTR, True)
setattr(bounds_checked_load_crop, _ORIGINAL_ATTR, _original_load_crop)


def correlation_gated_error_taxonomy(feature: Mapping[str, Any], *, role: str) -> str:
    """Classify recurrence errors using the same evidence as the hard filter.

    The hard-filter threshold is applied before this function runs and is
    represented by ``high_recurrence_revisits``.  Do not re-threshold
    ``belt_fixedness_score`` here with the module default: users can pass a
    different hard-reject threshold, and reapplying the default would make the
    report describe recurrence evidence that the configured hard filter did not
    actually use.
    """

    valid = _optional_int(feature.get("valid_revisits"))
    hard_reject = _bool_value(feature.get("hard_reject"))
    supported_revisits = _optional_int(feature.get("high_recurrence_revisits"))
    supported_score = _optional_float(feature.get("belt_fixedness_score")) or 0.0
    threshold = _optional_float(feature.get(THRESHOLD_FIELD))
    threshold_supported = threshold is not None and supported_score >= threshold
    role_lower = role.lower()
    if valid == 0:
        return f"{role_lower}_no_valid_revisits"
    if role == "FP" and hard_reject:
        return "fp_recurrent_removed"
    if role == "FP" and supported_revisits == 0 and not threshold_supported:
        return "fp_low_shape_supported_recurrence_evidence"
    if role == "TP" and hard_reject:
        return "tp_high_shape_supported_recurrence_accidentally_removed"
    if supported_revisits > 0 or threshold_supported:
        return f"{role_lower}_shape_supported_recurrent_but_not_hard_rejected"
    return f"{role_lower}_inconclusive_low_recurrence"


# Backwards-compatible safety patch for older imports.  The production CLI
# imports this module before running the recurrence scorer.  The underlying
# scorer also needs these fixes when used directly; ``beltmap.__init__`` imports
# this module for that side effect until ``beltmap.yolo_recurrence`` is updated
# in place.  The wrapper is idempotent, so reloading this module cannot wrap a
# previous wrapper and recurse.
_ensure_field(_yolo_recurrence.FEATURE_FIELDNAMES, THRESHOLD_FIELD)
_ensure_field(_yolo_recurrence.RUN_EXTRA_FIELDS, THRESHOLD_FIELD)
_yolo_recurrence.row_key = duplicate_safe_row_key
_yolo_recurrence.score_detection_recurrence = correlation_gated_score_detection_recurrence
_yolo_recurrence.enrich_detection_row = persist_threshold_enrich_detection_row
_yolo_recurrence.load_crop = bounds_checked_load_crop
_yolo_recurrence.error_taxonomy = correlation_gated_error_taxonomy

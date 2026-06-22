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


# Backwards-compatible safety patch for older imports.  The production CLI
# imports this module before running the recurrence scorer.  The underlying
# scorer also needs this duplicate-safe key when used directly; keep this module
# small so it can be removed once ``beltmap.yolo_recurrence.row_key`` is updated
# in place.
_yolo_recurrence.row_key = duplicate_safe_row_key

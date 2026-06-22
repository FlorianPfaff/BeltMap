from __future__ import annotations

from typing import Any, Mapping

from beltmap import yolo_recurrence as _yolo_recurrence


def duplicate_safe_row_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    """Return a stable detection key that does not collapse same-class detections.

    YOLO exports usually have one class label for all particle boxes.  The
    recurrence scorer stores per-detection features and truth-match roles in
    dictionaries, so a key containing only ``(frame_index, label)`` aliases every
    particle candidate in the same frame.  Including the crop-local box geometry
    and confidence keeps ordinary same-frame particle detections distinct while
    remaining deterministic across feature, hard-filter, and rerank passes.
    """

    confidence = row.get("confidence", row.get("score", ""))
    return (
        int(round(float(row["frame_index"]))),
        int(round(float(row.get("label", 0)))),
        int(round(float(row["bbox_top"]))),
        int(round(float(row["bbox_left"]))),
        int(round(float(row["bbox_bottom"]))),
        int(round(float(row["bbox_right"]))),
        round(float(row.get("y", 0.0)), 3),
        round(float(row.get("x", 0.0)), 3),
        str(confidence),
        str(row.get("source", "")),
    )


# Patch the legacy helper at import time.  The public CLI imports this module
# before invoking beltmap.yolo_recurrence.run_yolo_recurrence_filter, so all
# dictionary lookups inside the existing scorer use the duplicate-safe key.
_yolo_recurrence.row_key = duplicate_safe_row_key

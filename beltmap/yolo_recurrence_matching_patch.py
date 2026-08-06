"""Use cardinality-optimal truth matching for YOLO recurrence role labels.

The recurrence diagnostic labels each detection as a true or false positive before
reporting which detections the hard filter removes. Greedy descending-IoU
matching can consume the only valid detection for another truth box and therefore
under-count true positives. Reuse the maximum-cardinality, maximum-total-IoU
matcher used by BeltMap's real-data evaluator.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import yolo_recurrence as _yolo_recurrence
from .advanced_quality_matching_patch import _maximum_cardinality_iou_matches

_PATCHED_ATTR = "_beltmap_yolo_recurrence_cardinality_matching_patched"
_ORIGINAL_ATTR = "_beltmap_original_yolo_recurrence_match_detection_roles"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the role matcher behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_match_detection_roles = _unwrap_patched_callable(
    _yolo_recurrence.match_detection_roles
)


def _box_mapping(box: _yolo_recurrence.PatchBox) -> dict[str, float]:
    """Convert a recurrence patch box to the shared matcher representation."""

    return {
        "top": float(box.top),
        "left": float(box.left),
        "bottom": float(box.bottom),
        "right": float(box.right),
    }


def _truth_box(row: Mapping[str, Any]) -> dict[str, float]:
    """Apply the historical floor/ceil conversion to one truth box."""

    return _box_mapping(
        _yolo_recurrence.PatchBox(
            top=int(math.floor(float(row["top"]))),
            left=int(math.floor(float(row["left"]))),
            bottom=int(math.ceil(float(row["bottom"]))),
            right=int(math.ceil(float(row["right"]))),
        )
    )


def cardinality_optimal_match_detection_roles(
    detection_rows: Sequence[Mapping[str, Any]],
    *,
    truth_path: Path,
    iou_threshold: float = 0.25,
) -> dict[tuple[object, ...], str]:
    """Assign TP/FP roles with maximum-cardinality one-to-one IoU matching."""

    truth = _yolo_recurrence.load_labeled_detection_truth(truth_path)
    roles = {_yolo_recurrence.row_key(row): "FP" for row in detection_rows}
    frames = sorted(
        {
            _yolo_recurrence.int_value(row["frame_index"], name="frame_index")
            for row in detection_rows
        }
    )

    for frame in frames:
        frame_detections = [
            row
            for row in detection_rows
            if _yolo_recurrence.int_value(
                row["frame_index"],
                name="frame_index",
            )
            == frame
        ]
        frame_truth = [
            row
            for row in truth.get("particles", [])
            if int(row["frame_index"]) == frame
        ]
        truth_boxes = [_truth_box(row) for row in frame_truth]
        detection_boxes = [
            _box_mapping(_yolo_recurrence.bbox_from_row(row))
            for row in frame_detections
        ]
        matches = _maximum_cardinality_iou_matches(
            truth_boxes,
            detection_boxes,
            iou_threshold=iou_threshold,
        )
        for _iou, _truth_index, detection_index in matches:
            roles[
                _yolo_recurrence.row_key(frame_detections[detection_index])
            ] = "TP"
    return roles


setattr(cardinality_optimal_match_detection_roles, _PATCHED_ATTR, True)
setattr(
    cardinality_optimal_match_detection_roles,
    _ORIGINAL_ATTR,
    _original_match_detection_roles,
)
_yolo_recurrence.match_detection_roles = cardinality_optimal_match_detection_roles

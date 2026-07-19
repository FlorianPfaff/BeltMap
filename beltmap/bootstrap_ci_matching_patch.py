"""Use cardinality-optimal box matching for labeled bootstrap metrics.

The real-data evaluator already maximizes the number of valid one-to-one box
matches before maximizing total IoU.  The bootstrap path historically retained
the older greedy matcher, so a comparison row could report a point estimate and
confidence interval based on different true-positive counts.  This compatibility
patch makes the bootstrap frame outcomes reuse the same matching objective.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from . import bootstrap_ci as _bootstrap_ci
from .advanced_quality_matching_patch import _maximum_cardinality_iou_matches

_PATCHED_ATTR = "_beltmap_cardinality_optimal_bootstrap_matching_patched"
_ORIGINAL_ATTR = "_beltmap_original_labeled_frame_outcomes"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the outcome builder behind this compatibility patch, if present."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_labeled_frame_outcomes = _unwrap_patched_callable(
    _bootstrap_ci.labeled_frame_outcomes
)


def cardinality_optimal_labeled_frame_outcomes(
    detection_rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Any],
    *,
    scored_frames: set[int],
    iou_threshold: float,
) -> list[_bootstrap_ci.LabeledFrameOutcome]:
    """Compute bootstrap frame outcomes with cardinality-optimal IoU matching."""

    iou_threshold = _bootstrap_ci._iou_threshold_value(iou_threshold)
    normalized_scored_frames = {
        _bootstrap_ci._nonnegative_integer_value(frame, "scored_frames")
        for frame in scored_frames
    }
    truth_by_frame = _bootstrap_ci.group_truth_boxes(dict(truth))
    pred_by_frame = _bootstrap_ci.group_detection_boxes(
        [
            dict(row)
            for row in detection_rows
            if _bootstrap_ci.source_frame_index(dict(row))
            in normalized_scored_frames
        ]
    )
    outcomes: list[_bootstrap_ci.LabeledFrameOutcome] = []

    for frame_index in sorted(normalized_scored_frames):
        truths = truth_by_frame.get(frame_index, [])
        preds = pred_by_frame.get(frame_index, [])
        frame_matches = _maximum_cardinality_iou_matches(
            truths,
            preds,
            iou_threshold=iou_threshold,
        )
        matched_ious: list[float] = []
        centroid_errors: list[float] = []
        for iou, truth_index, pred_index in frame_matches:
            matched_ious.append(float(iou))
            pred_y, pred_x = _bootstrap_ci.predicted_center(preds[pred_index])
            truth_y, truth_x = _bootstrap_ci.truth_center(truths[truth_index])
            centroid_errors.append(
                float(math.hypot(pred_y - truth_y, pred_x - truth_x))
            )

        true_positives = len(frame_matches)
        outcomes.append(
            _bootstrap_ci.LabeledFrameOutcome(
                frame_index=frame_index,
                truth_boxes=len(truths),
                predicted_boxes=len(preds),
                true_positives=true_positives,
                false_positives=len(preds) - true_positives,
                false_negatives=len(truths) - true_positives,
                matched_ious=tuple(matched_ious),
                centroid_errors_px=tuple(centroid_errors),
            )
        )
    return outcomes


setattr(cardinality_optimal_labeled_frame_outcomes, _PATCHED_ATTR, True)
setattr(
    cardinality_optimal_labeled_frame_outcomes,
    _ORIGINAL_ATTR,
    _original_labeled_frame_outcomes,
)
_bootstrap_ci.labeled_frame_outcomes = cardinality_optimal_labeled_frame_outcomes

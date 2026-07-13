"""Use cardinality-optimal matching in benchmark and bootstrap detection metrics.

The comparison and bootstrap paths historically used descending-IoU greedy
matching. A locally best pair can consume the only valid partner for another
box, reducing the reported true-positive count even though a larger valid
one-to-one matching exists. Reuse the real-data evaluator's maximum-cardinality
assignment so point estimates and confidence intervals obey the same matching
semantics.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from . import benchmark as _benchmark
from . import bootstrap_ci as _bootstrap_ci
from .advanced_quality_matching_patch import _maximum_cardinality_iou_matches

_DETECTION_PATCHED_ATTR = "_beltmap_cardinality_optimal_detection_matching_patched"
_BOOTSTRAP_PATCHED_ATTR = "_beltmap_cardinality_optimal_bootstrap_matching_patched"
_ORIGINAL_DETECTION_ATTR = "_beltmap_original_detection_metrics"
_ORIGINAL_BOOTSTRAP_ATTR = "_beltmap_original_labeled_frame_outcomes"


def _unwrap_patched_callable(func: Any, original_attr: str) -> Any:
    """Return the implementation behind this compatibility patch, if present."""

    return getattr(func, original_attr, func)


_original_detection_metrics = _unwrap_patched_callable(
    _benchmark.detection_metrics,
    _ORIGINAL_DETECTION_ATTR,
)
_original_labeled_frame_outcomes = _unwrap_patched_callable(
    _bootstrap_ci.labeled_frame_outcomes,
    _ORIGINAL_BOOTSTRAP_ATTR,
)


def cardinality_optimal_detection_metrics(
    detection_rows: list[dict[str, str]],
    truth: dict[str, Any],
    *,
    iou_threshold: float = 0.25,
    scored_frames: set[int] | None = None,
) -> dict[str, Any]:
    """Compute detection metrics with maximum-cardinality one-to-one matching."""

    iou_threshold = _benchmark.validate_iou_threshold(iou_threshold)
    truth_by_frame = _benchmark.group_truth_boxes(truth)
    pred_by_frame = _benchmark.group_detection_boxes(detection_rows)
    frame_indices = (
        sorted(set(truth_by_frame) | set(pred_by_frame))
        if scored_frames is None
        else sorted(scored_frames)
    )

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    centroid_errors: list[float] = []
    matched_ious: list[float] = []

    for frame_index in frame_indices:
        truths = truth_by_frame.get(frame_index, [])
        preds = pred_by_frame.get(frame_index, [])
        matches = _maximum_cardinality_iou_matches(
            truths,
            preds,
            iou_threshold=iou_threshold,
        )
        true_positives += len(matches)
        false_positives += len(preds) - len(matches)
        false_negatives += len(truths) - len(matches)
        for iou, truth_index, pred_index in matches:
            matched_ious.append(float(iou))
            pred_y, pred_x = _benchmark.predicted_center(preds[pred_index])
            truth_y, truth_x = _benchmark.truth_center(truths[truth_index])
            centroid_errors.append(
                float(math.hypot(pred_y - truth_y, pred_x - truth_x))
            )

    precision = _benchmark.detection_precision(
        true_positives,
        false_positives,
        false_negatives,
    )
    recall = _benchmark.detection_recall(
        true_positives,
        false_positives,
        false_negatives,
    )
    f1 = _benchmark.f1_score(precision, recall)
    centroid_stats = _benchmark.summary_errors(centroid_errors, unit="px")
    iou_values = np.asarray(matched_ious, dtype=np.float64)

    return {
        "available": bool(frame_indices),
        "iou_threshold": iou_threshold,
        "truth_boxes": sum(
            len(truth_by_frame.get(frame, [])) for frame in frame_indices
        ),
        "predicted_boxes": sum(
            len(pred_by_frame.get(frame, [])) for frame in frame_indices
        ),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": None if precision is None else float(precision),
        "recall": None if recall is None else float(recall),
        "f1": None if f1 is None else float(f1),
        "mean_matched_iou": (
            None if iou_values.size == 0 else float(np.mean(iou_values))
        ),
        "mean_centroid_error_px": centroid_stats["mean_abs_error_px"],
        "median_centroid_error_px": centroid_stats["median_abs_error_px"],
        "max_centroid_error_px": centroid_stats["max_abs_error_px"],
    }


def cardinality_optimal_labeled_frame_outcomes(
    detection_rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Any],
    *,
    scored_frames: set[int],
    iou_threshold: float,
) -> list[_bootstrap_ci.LabeledFrameOutcome]:
    """Compute per-frame bootstrap outcomes with maximum-cardinality matching."""

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
        matches = _maximum_cardinality_iou_matches(
            truths,
            preds,
            iou_threshold=iou_threshold,
        )
        matched_ious: list[float] = []
        centroid_errors: list[float] = []
        for iou, truth_index, pred_index in matches:
            matched_ious.append(float(iou))
            pred_y, pred_x = _bootstrap_ci.predicted_center(preds[pred_index])
            truth_y, truth_x = _bootstrap_ci.truth_center(truths[truth_index])
            centroid_errors.append(
                float(math.hypot(pred_y - truth_y, pred_x - truth_x))
            )

        true_positives = len(matches)
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


setattr(
    cardinality_optimal_detection_metrics,
    _DETECTION_PATCHED_ATTR,
    True,
)
setattr(
    cardinality_optimal_detection_metrics,
    _ORIGINAL_DETECTION_ATTR,
    _original_detection_metrics,
)
setattr(
    cardinality_optimal_labeled_frame_outcomes,
    _BOOTSTRAP_PATCHED_ATTR,
    True,
)
setattr(
    cardinality_optimal_labeled_frame_outcomes,
    _ORIGINAL_BOOTSTRAP_ATTR,
    _original_labeled_frame_outcomes,
)

_benchmark.detection_metrics = cardinality_optimal_detection_metrics
_bootstrap_ci.labeled_frame_outcomes = cardinality_optimal_labeled_frame_outcomes

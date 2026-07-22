"""Use cardinality-optimal matching for synthetic benchmark detections.

The synthetic benchmark historically sorted all candidate pairs by IoU and
accepted them greedily. A high-IoU pair can consume a prediction needed by
another truth box, leaving fewer true positives than a valid one-to-one matching
contains. Reuse the maximum-cardinality, maximum-total-IoU matcher already used
by the real-data evaluator.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

from . import benchmark as _benchmark
from .advanced_quality_matching_patch import _maximum_cardinality_iou_matches

_PATCHED_ATTR = "_beltmap_cardinality_optimal_benchmark_matching_patched"
_ORIGINAL_ATTR = "_beltmap_original_benchmark_detection_metrics"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the benchmark evaluator behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_detection_metrics = _unwrap_patched_callable(
    _benchmark.detection_metrics
)


def cardinality_optimal_detection_metrics(
    detection_rows: list[dict[str, str]],
    truth: dict[str, Any],
    *,
    iou_threshold: float = 0.25,
    scored_frames: set[int] | None = None,
) -> dict[str, Any]:
    """Evaluate synthetic detections with cardinality-optimal IoU matching."""

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
        frame_matches = _maximum_cardinality_iou_matches(
            truths,
            preds,
            iou_threshold=iou_threshold,
        )
        true_positives += len(frame_matches)
        false_positives += len(preds) - len(frame_matches)
        false_negatives += len(truths) - len(frame_matches)

        for iou, truth_index, pred_index in frame_matches:
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


setattr(cardinality_optimal_detection_metrics, _PATCHED_ATTR, True)
setattr(
    cardinality_optimal_detection_metrics,
    _ORIGINAL_ATTR,
    _original_detection_metrics,
)
_benchmark.detection_metrics = cardinality_optimal_detection_metrics

# Keep modules that imported the function before this patch synchronized.
for _module_name in ("beltmap.compare_runs", "beltmap.texture_stress"):
    _module = sys.modules.get(_module_name)
    if (
        _module is not None
        and getattr(_module, "detection_metrics", None)
        is _original_detection_metrics
    ):
        _module.detection_metrics = cardinality_optimal_detection_metrics

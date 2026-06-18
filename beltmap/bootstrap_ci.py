from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .benchmark import (
    bbox_iou,
    detection_precision,
    detection_recall,
    f1_score,
    finite_float,
    group_detection_boxes,
    group_truth_boxes,
    predicted_center,
    source_frame_index,
    truth_center,
)

BOOTSTRAP_METRIC_FIELDS = (
    "detections_per_frame_mean",
    "detections_per_frame_median",
    "labeled_precision",
    "labeled_recall",
    "labeled_f1",
    "labeled_false_positives",
    "labeled_false_negatives",
    "labeled_mean_matched_iou",
    "labeled_mean_centroid_error_px",
    "velocity_ratio_median",
    "velocity_ratio_share_0_to_1",
    "filtered_velocity_ratio_median",
    "filtered_velocity_ratio_share_0_to_1",
    "long_velocity_tracks_ge_5",
    "long_velocity_tracks_ge_10",
)

BOOTSTRAP_BASE_FIELDS = (
    "bootstrap_samples",
    "bootstrap_confidence_level",
    "bootstrap_block_length_frames",
)

BOOTSTRAP_SUMMARY_FIELDS = [
    *BOOTSTRAP_BASE_FIELDS,
    *(
        field
        for metric in BOOTSTRAP_METRIC_FIELDS
        for field in (
            f"{metric}_bootstrap_median",
            f"{metric}_ci_low",
            f"{metric}_ci_high",
        )
    ),
]


@dataclass(frozen=True)
class LabeledFrameOutcome:
    """Greedy detection-match counts and matched errors for one scored frame."""

    frame_index: int
    truth_boxes: int
    predicted_boxes: int
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_ious: tuple[float, ...] = ()
    centroid_errors_px: tuple[float, ...] = ()


MetricFunction = Callable[[np.ndarray], float | None]


def empty_bootstrap_metrics() -> dict[str, Any]:
    """Return blank bootstrap fields for stable comparison CSV schemas."""

    return {field: None for field in BOOTSTRAP_SUMMARY_FIELDS}


def finite_values(rows: Iterable[Mapping[str, Any]], field: str) -> list[float]:
    """Collect finite float values from a sequence of CSV-like rows."""

    values: list[float] = []
    for row in rows:
        value = finite_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def resample_indices(
    n_units: int,
    *,
    rng: np.random.Generator,
    block_length: int = 1,
) -> np.ndarray:
    """Return bootstrap indices, optionally using circular contiguous blocks."""

    if n_units <= 0:
        return np.asarray([], dtype=np.int64)
    if block_length <= 1:
        return rng.integers(0, n_units, size=n_units, dtype=np.int64)

    indices: list[int] = []
    while len(indices) < n_units:
        start = int(rng.integers(0, n_units))
        indices.extend((start + offset) % n_units for offset in range(block_length))
    return np.asarray(indices[:n_units], dtype=np.int64)


def ci_summary(
    values: Iterable[float | int | None],
    *,
    confidence_level: float,
) -> tuple[float | None, float | None, float | None]:
    """Return bootstrap median and equal-tailed confidence interval bounds."""

    arr = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    if arr.size == 0:
        return None, None, None
    alpha = 0.5 * (1.0 - confidence_level)
    return (
        float(np.median(arr)),
        float(np.quantile(arr, alpha)),
        float(np.quantile(arr, 1.0 - alpha)),
    )


def add_ci_fields(
    target: dict[str, Any],
    metric_name: str,
    estimates: Iterable[float | int | None],
    *,
    confidence_level: float,
) -> None:
    """Append median/CI fields for one metric to ``target`` in place."""

    median, low, high = ci_summary(estimates, confidence_level=confidence_level)
    target[f"{metric_name}_bootstrap_median"] = median
    target[f"{metric_name}_ci_low"] = low
    target[f"{metric_name}_ci_high"] = high


def bootstrap_numeric_metrics(
    values: Sequence[float],
    metrics: Mapping[str, MetricFunction],
    *,
    samples: int,
    confidence_level: float,
    rng: np.random.Generator,
    block_length: int = 1,
) -> dict[str, Any]:
    """Bootstrap scalar metrics computed from a one-dimensional numeric sample."""

    result: dict[str, Any] = {}
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    estimates: dict[str, list[float | None]] = {name: [] for name in metrics}
    if arr.size == 0:
        for name in metrics:
            add_ci_fields(result, name, [], confidence_level=confidence_level)
        return result

    for _sample_index in range(samples):
        indices = resample_indices(arr.size, rng=rng, block_length=block_length)
        sample = arr[indices]
        for name, function in metrics.items():
            estimates[name].append(function(sample))

    for name, values_for_metric in estimates.items():
        add_ci_fields(result, name, values_for_metric, confidence_level=confidence_level)
    return result


def mean_value(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.mean(values))


def median_value(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.median(values))


def share_0_to_1(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.count_nonzero((values >= 0.0) & (values <= 1.0)) / values.size)


def count_ge(threshold: float) -> MetricFunction:
    def count(values: np.ndarray) -> float | None:
        return None if values.size == 0 else float(np.count_nonzero(values >= threshold))

    return count


def labeled_frame_outcomes(
    detection_rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Any],
    *,
    scored_frames: set[int],
    iou_threshold: float,
) -> list[LabeledFrameOutcome]:
    """Compute per-frame labeled detection outcomes for bootstrap resampling."""

    truth_by_frame = group_truth_boxes(dict(truth))
    pred_by_frame = group_detection_boxes(
        [dict(row) for row in detection_rows if source_frame_index(dict(row)) in scored_frames]
    )
    frame_indices = sorted(scored_frames)
    outcomes: list[LabeledFrameOutcome] = []

    for frame_index in frame_indices:
        truths = truth_by_frame.get(frame_index, [])
        preds = pred_by_frame.get(frame_index, [])
        candidates: list[tuple[float, int, int]] = []
        for pred_index, pred in enumerate(preds):
            for truth_index, target in enumerate(truths):
                candidates.append((bbox_iou(pred, target), pred_index, truth_index))

        matched_preds: set[int] = set()
        matched_truths: set[int] = set()
        matched_ious: list[float] = []
        centroid_errors: list[float] = []
        for iou, pred_index, truth_index in sorted(candidates, reverse=True):
            if iou < iou_threshold:
                break
            if pred_index in matched_preds or truth_index in matched_truths:
                continue
            matched_preds.add(pred_index)
            matched_truths.add(truth_index)
            matched_ious.append(float(iou))
            pred_y, pred_x = predicted_center(preds[pred_index])
            truth_y, truth_x = truth_center(truths[truth_index])
            centroid_errors.append(float(math.hypot(pred_y - truth_y, pred_x - truth_x)))

        true_positives = len(matched_preds)
        outcomes.append(
            LabeledFrameOutcome(
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


def aggregate_labeled_outcomes(outcomes: Sequence[LabeledFrameOutcome]) -> dict[str, Any]:
    """Aggregate per-frame labeled outcomes into detection precision/recall metrics."""

    truth_boxes = sum(outcome.truth_boxes for outcome in outcomes)
    predicted_boxes = sum(outcome.predicted_boxes for outcome in outcomes)
    true_positives = sum(outcome.true_positives for outcome in outcomes)
    false_positives = sum(outcome.false_positives for outcome in outcomes)
    false_negatives = sum(outcome.false_negatives for outcome in outcomes)
    precision = detection_precision(true_positives, false_positives, false_negatives)
    recall = detection_recall(true_positives, false_positives, false_negatives)
    f1 = f1_score(precision, recall)

    matched_ious = [value for outcome in outcomes for value in outcome.matched_ious]
    centroid_errors = [value for outcome in outcomes for value in outcome.centroid_errors_px]
    return {
        "labeled_truth_boxes": truth_boxes,
        "labeled_predicted_boxes": predicted_boxes,
        "labeled_true_positives": true_positives,
        "labeled_false_positives": false_positives,
        "labeled_false_negatives": false_negatives,
        "labeled_precision": None if precision is None else float(precision),
        "labeled_recall": None if recall is None else float(recall),
        "labeled_f1": None if f1 is None else float(f1),
        "labeled_mean_matched_iou": None if not matched_ious else float(np.mean(matched_ious)),
        "labeled_mean_centroid_error_px": None if not centroid_errors else float(np.mean(centroid_errors)),
    }


def bootstrap_labeled_metrics(
    outcomes: Sequence[LabeledFrameOutcome],
    *,
    samples: int,
    confidence_level: float,
    rng: np.random.Generator,
    block_length_frames: int,
) -> dict[str, Any]:
    """Bootstrap labeled detection metrics over scored frames."""

    metric_names = (
        "labeled_precision",
        "labeled_recall",
        "labeled_f1",
        "labeled_false_positives",
        "labeled_false_negatives",
        "labeled_mean_matched_iou",
        "labeled_mean_centroid_error_px",
    )
    result: dict[str, Any] = {}
    estimates: dict[str, list[float | int | None]] = {name: [] for name in metric_names}
    if not outcomes:
        for name in metric_names:
            add_ci_fields(result, name, [], confidence_level=confidence_level)
        return result

    for _sample_index in range(samples):
        indices = resample_indices(
            len(outcomes),
            rng=rng,
            block_length=block_length_frames,
        )
        sample = [outcomes[int(index)] for index in indices]
        metrics = aggregate_labeled_outcomes(sample)
        for name in metric_names:
            estimates[name].append(metrics.get(name))

    for name, values in estimates.items():
        add_ci_fields(result, name, values, confidence_level=confidence_level)
    return result


def bootstrap_run_summary(
    *,
    detections_per_frame: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    velocities: Sequence[Mapping[str, Any]],
    filtered_velocities: Sequence[Mapping[str, Any]],
    labeled_truth: Mapping[str, Any] | None = None,
    scored_frames: set[int] | None = None,
    truth_iou_threshold: float = 0.25,
    samples: int = 0,
    confidence_level: float = 0.95,
    seed: int | None = 0,
    block_length_frames: int = 1,
) -> dict[str, Any]:
    """Return bootstrap median/CI fields for one comparison row."""

    if samples < 0:
        raise ValueError("bootstrap samples must be non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("bootstrap confidence level must be between 0 and 1")
    if block_length_frames < 1:
        raise ValueError("bootstrap block length must be at least 1 frame")

    result = empty_bootstrap_metrics()
    if samples == 0:
        return result

    result.update(
        {
            "bootstrap_samples": int(samples),
            "bootstrap_confidence_level": float(confidence_level),
            "bootstrap_block_length_frames": int(block_length_frames),
        }
    )
    rng = np.random.default_rng(seed)

    result.update(
        bootstrap_numeric_metrics(
            finite_values(detections_per_frame, "n_detections"),
            {
                "detections_per_frame_mean": mean_value,
                "detections_per_frame_median": median_value,
            },
            samples=samples,
            confidence_level=confidence_level,
            rng=rng,
            block_length=block_length_frames,
        )
    )
    result.update(
        bootstrap_numeric_metrics(
            finite_values(velocities, "velocity_ratio_y"),
            {
                "velocity_ratio_median": median_value,
                "velocity_ratio_share_0_to_1": share_0_to_1,
            },
            samples=samples,
            confidence_level=confidence_level,
            rng=rng,
            block_length=1,
        )
    )
    result.update(
        bootstrap_numeric_metrics(
            finite_values(velocities, "n_detections"),
            {
                "long_velocity_tracks_ge_5": count_ge(5.0),
                "long_velocity_tracks_ge_10": count_ge(10.0),
            },
            samples=samples,
            confidence_level=confidence_level,
            rng=rng,
            block_length=1,
        )
    )
    result.update(
        bootstrap_numeric_metrics(
            finite_values(filtered_velocities, "velocity_ratio_y"),
            {
                "filtered_velocity_ratio_median": median_value,
                "filtered_velocity_ratio_share_0_to_1": share_0_to_1,
            },
            samples=samples,
            confidence_level=confidence_level,
            rng=rng,
            block_length=1,
        )
    )
    if labeled_truth is not None and scored_frames:
        result.update(
            bootstrap_labeled_metrics(
                labeled_frame_outcomes(
                    detections,
                    labeled_truth,
                    scored_frames=scored_frames,
                    iou_threshold=truth_iou_threshold,
                ),
                samples=samples,
                confidence_level=confidence_level,
                rng=rng,
                block_length_frames=block_length_frames,
            )
        )
    return result

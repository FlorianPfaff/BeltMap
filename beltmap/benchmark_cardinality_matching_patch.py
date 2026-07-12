"""Use cardinality-optimal IoU matching in benchmark and bootstrap metrics.

The real-data evaluator already maximizes valid one-to-one match cardinality, but
``benchmark.detection_metrics`` and ``bootstrap_ci.labeled_frame_outcomes`` still
used a descending-IoU greedy assignment. A high-IoU pair can consume the only
valid partner of another box, undercounting true positives in comparison reports,
FROC curves, synthetic benchmarks, and bootstrap confidence intervals.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from . import benchmark as _benchmark
from . import bootstrap_ci as _bootstrap_ci

_PATCHED_ATTR = "_beltmap_cardinality_optimal_detection_matching_patched"
_BENCHMARK_ORIGINAL_ATTR = "_beltmap_original_detection_metrics"
_BOOTSTRAP_ORIGINAL_ATTR = "_beltmap_original_labeled_frame_outcomes"


def _unwrap_patched_callable(func: Any, original_attr: str) -> Any:
    """Return the callable behind this compatibility patch, if present."""

    return getattr(func, original_attr, func)


_original_detection_metrics = _unwrap_patched_callable(
    _benchmark.detection_metrics,
    _BENCHMARK_ORIGINAL_ATTR,
)
_original_labeled_frame_outcomes = _unwrap_patched_callable(
    _bootstrap_ci.labeled_frame_outcomes,
    _BOOTSTRAP_ORIGINAL_ATTR,
)


@dataclass
class _ResidualEdge:
    destination: int
    reverse_index: int
    capacity: int
    cost: float


def _add_unit_edge(
    graph: list[list[_ResidualEdge]],
    source: int,
    destination: int,
    *,
    cost: float,
) -> _ResidualEdge:
    forward = _ResidualEdge(
        destination=destination,
        reverse_index=len(graph[destination]),
        capacity=1,
        cost=float(cost),
    )
    reverse = _ResidualEdge(
        destination=source,
        reverse_index=len(graph[source]),
        capacity=0,
        cost=-float(cost),
    )
    graph[source].append(forward)
    graph[destination].append(reverse)
    return forward


def _shortest_augmenting_path(
    graph: list[list[_ResidualEdge]],
    *,
    source: int,
    sink: int,
) -> tuple[list[int], list[int]] | None:
    """Find one minimum-cost residual path with Bellman-Ford relaxation."""

    node_count = len(graph)
    distances = np.full(node_count, np.inf, dtype=np.float64)
    previous_nodes = np.full(node_count, -1, dtype=np.int64)
    previous_edges = np.full(node_count, -1, dtype=np.int64)
    distances[source] = 0.0

    for _iteration in range(node_count - 1):
        changed = False
        for node, edges in enumerate(graph):
            if not np.isfinite(distances[node]):
                continue
            for edge_index, edge in enumerate(edges):
                if edge.capacity <= 0:
                    continue
                candidate = float(distances[node] + edge.cost)
                if candidate < float(distances[edge.destination]) - 1e-15:
                    distances[edge.destination] = candidate
                    previous_nodes[edge.destination] = node
                    previous_edges[edge.destination] = edge_index
                    changed = True
        if not changed:
            break

    if previous_nodes[sink] < 0:
        return None
    return previous_nodes.tolist(), previous_edges.tolist()


def _maximum_cardinality_iou_matches(
    truth_boxes: Sequence[Mapping[str, float]],
    detection_boxes: Sequence[Mapping[str, float]],
    *,
    iou_threshold: float,
) -> list[tuple[float, int, int]]:
    """Return maximum-cardinality matches, maximizing total IoU as a tie-breaker."""

    if not truth_boxes or not detection_boxes:
        return []

    truth_count = len(truth_boxes)
    detection_count = len(detection_boxes)
    source = 0
    first_truth = 1
    first_detection = first_truth + truth_count
    sink = first_detection + detection_count
    graph: list[list[_ResidualEdge]] = [[] for _node in range(sink + 1)]

    for truth_index in range(truth_count):
        _add_unit_edge(graph, source, first_truth + truth_index, cost=0.0)
    for detection_index in range(detection_count):
        _add_unit_edge(graph, first_detection + detection_index, sink, cost=0.0)

    candidate_edges: dict[tuple[int, int], tuple[_ResidualEdge, float]] = {}
    for truth_index, truth_box in enumerate(truth_boxes):
        for detection_index, detection_box in enumerate(detection_boxes):
            iou = float(_benchmark.bbox_iou(dict(truth_box), dict(detection_box)))
            if iou < iou_threshold:
                continue
            edge = _add_unit_edge(
                graph,
                first_truth + truth_index,
                first_detection + detection_index,
                cost=-iou,
            )
            candidate_edges[(truth_index, detection_index)] = (edge, iou)

    while True:
        path = _shortest_augmenting_path(graph, source=source, sink=sink)
        if path is None:
            break
        previous_nodes, previous_edges = path
        node = sink
        while node != source:
            previous_node = previous_nodes[node]
            edge_index = previous_edges[node]
            edge = graph[previous_node][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse_index].capacity += 1
            node = previous_node

    matches = [
        (iou, truth_index, detection_index)
        for (truth_index, detection_index), (edge, iou) in candidate_edges.items()
        if edge.capacity == 0
    ]
    return sorted(matches, key=lambda item: (item[1], item[2]))


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
            matched_ious.append(iou)
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
    """Compute bootstrap frame outcomes with the same matching as point metrics."""

    iou_threshold = _bootstrap_ci._iou_threshold_value(iou_threshold)
    normalized_scored_frames = {
        _bootstrap_ci._nonnegative_integer_value(frame, "scored_frames")
        for frame in scored_frames
    }
    truth_by_frame = _benchmark.group_truth_boxes(dict(truth))
    pred_by_frame = _benchmark.group_detection_boxes(
        [
            dict(row)
            for row in detection_rows
            if _benchmark.source_frame_index(dict(row)) in normalized_scored_frames
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
            pred_y, pred_x = _benchmark.predicted_center(preds[pred_index])
            truth_y, truth_x = _benchmark.truth_center(truths[truth_index])
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


setattr(cardinality_optimal_detection_metrics, _PATCHED_ATTR, True)
setattr(
    cardinality_optimal_detection_metrics,
    _BENCHMARK_ORIGINAL_ATTR,
    _original_detection_metrics,
)
setattr(cardinality_optimal_labeled_frame_outcomes, _PATCHED_ATTR, True)
setattr(
    cardinality_optimal_labeled_frame_outcomes,
    _BOOTSTRAP_ORIGINAL_ATTR,
    _original_labeled_frame_outcomes,
)
_benchmark.detection_metrics = cardinality_optimal_detection_metrics
_bootstrap_ci.labeled_frame_outcomes = cardinality_optimal_labeled_frame_outcomes

# Repair references in modules that may have been imported before this patch was
# loaded explicitly. Normal package imports load the patch first, but this keeps
# reloads and notebook sessions consistent too.
for _module_name in ("beltmap.compare_runs", "beltmap.texture_stress"):
    _module = sys.modules.get(_module_name)
    if (
        _module is not None
        and getattr(_module, "detection_metrics", None) is _original_detection_metrics
    ):
        _module.detection_metrics = cardinality_optimal_detection_metrics

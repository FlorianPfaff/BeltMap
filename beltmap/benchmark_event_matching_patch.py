"""Use cardinality-optimal assignment for synthetic event metrics.

The event benchmark historically accepted candidate event pairs greedily in
 descending temporal-IoU order.  A high-scoring pair can consume the only valid
partner of another event and therefore reduce the number of matched events even
when a larger one-to-one assignment exists.  This patch maximizes match
cardinality first and total temporal IoU second.
"""

from __future__ import annotations

from typing import Any

from . import benchmark as _benchmark
from .advanced_quality_matching_patch import (
    _ResidualEdge,
    _add_unit_edge,
    _shortest_augmenting_path,
)

_PATCHED_ATTR = "_beltmap_cardinality_optimal_event_matching_patched"
_ORIGINAL_ATTR = "_beltmap_original_event_metrics"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the event evaluator behind this compatibility patch, if present."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_event_metrics = _unwrap_patched_callable(_benchmark.event_metrics)


def _maximum_cardinality_event_matches(
    candidates: list[tuple[float, int, int, dict[str, Any]]],
    *,
    predicted_count: int,
    truth_count: int,
) -> list[dict[str, Any]]:
    """Return maximum-cardinality event matches, then maximize temporal IoU."""

    if not candidates or predicted_count <= 0 or truth_count <= 0:
        return []

    source = 0
    first_prediction = 1
    first_truth = first_prediction + predicted_count
    sink = first_truth + truth_count
    graph: list[list[_ResidualEdge]] = [[] for _node in range(sink + 1)]

    for prediction_index in range(predicted_count):
        _add_unit_edge(
            graph,
            source,
            first_prediction + prediction_index,
            cost=0.0,
        )
    for truth_index in range(truth_count):
        _add_unit_edge(
            graph,
            first_truth + truth_index,
            sink,
            cost=0.0,
        )

    candidate_edges: dict[
        tuple[int, int],
        tuple[_ResidualEdge, dict[str, Any]],
    ] = {}
    for score, prediction_index, truth_index, comparison in candidates:
        edge = _add_unit_edge(
            graph,
            first_prediction + prediction_index,
            first_truth + truth_index,
            cost=-float(score),
        )
        candidate_edges[(prediction_index, truth_index)] = (edge, comparison)

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
        comparison
        for (_prediction_index, _truth_index), (edge, comparison) in candidate_edges.items()
        if edge.capacity == 0
    ]
    return sorted(
        matches,
        key=lambda comparison: (
            str(comparison["pred_event_id"]),
            str(comparison["truth_event_id"]),
        ),
    )


def cardinality_optimal_event_metrics(
    prediction_rows: list[dict[str, str]],
    truth: dict[str, Any],
    *,
    iou_threshold: float = 0.25,
    prediction_source: str = "detections.csv",
) -> dict[str, Any]:
    """Compute event metrics with cardinality-optimal one-to-one assignment."""

    result = dict(
        _original_event_metrics(
            prediction_rows,
            truth,
            iou_threshold=iou_threshold,
            prediction_source=prediction_source,
        )
    )
    threshold = _benchmark.validate_iou_threshold(iou_threshold)
    truth_events = _benchmark.build_events_from_boxes(
        _benchmark.truth_event_boxes(truth),
        prefix="truth",
        iou_threshold=threshold,
    )
    predicted_events = _benchmark.build_events_from_boxes(
        _benchmark.predicted_event_boxes(prediction_rows),
        prefix="pred",
        iou_threshold=threshold,
    )

    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    for prediction_index, predicted in enumerate(predicted_events):
        for truth_index, target in enumerate(truth_events):
            comparison = _benchmark.compare_events(
                predicted,
                target,
                iou_threshold=threshold,
            )
            if comparison["matched_frames"] > 0:
                candidates.append(
                    (
                        float(comparison["temporal_iou"]),
                        prediction_index,
                        truth_index,
                        comparison,
                    )
                )

    matches = _maximum_cardinality_event_matches(
        candidates,
        predicted_count=len(predicted_events),
        truth_count=len(truth_events),
    )
    true_positives = len(matches)
    false_positives = len(predicted_events) - true_positives
    false_negatives = len(truth_events) - true_positives
    precision = true_positives / len(predicted_events) if predicted_events else None
    recall = true_positives / len(truth_events) if truth_events else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    def mean_field(name: str) -> float | None:
        values = [_benchmark.finite_float(match.get(name)) for match in matches]
        finite = [value for value in values if value is not None]
        return None if not finite else float(sum(finite) / len(finite))

    result.update(
        {
            "matched_events": true_positives,
            "false_positive_events": false_positives,
            "false_negative_events": false_negatives,
            "birth_false_positive_rate": (
                None
                if precision is None
                else float(false_positives / len(predicted_events))
            ),
            "missed_event_rate": (
                None
                if recall is None
                else float(false_negatives / len(truth_events))
            ),
            "precision": None if precision is None else float(precision),
            "recall": None if recall is None else float(recall),
            "f1": None if f1 is None else float(f1),
            "mean_temporal_iou": mean_field("temporal_iou"),
            "mean_truth_frame_coverage": mean_field("truth_frame_coverage"),
            "mean_predicted_frame_precision": mean_field(
                "predicted_frame_precision"
            ),
            "mean_frame_iou": mean_field("mean_frame_iou"),
            "mean_latency_frames": mean_field("latency_frames"),
            "mean_duration_error_frames": mean_field("duration_error_frames"),
            "matches": matches,
        }
    )
    return result


setattr(cardinality_optimal_event_metrics, _PATCHED_ATTR, True)
setattr(
    cardinality_optimal_event_metrics,
    _ORIGINAL_ATTR,
    _original_event_metrics,
)
_benchmark.event_metrics = cardinality_optimal_event_metrics

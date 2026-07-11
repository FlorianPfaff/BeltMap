"""Use cardinality-optimal assignment for sparse tracklet evaluation.

The original frame-level evaluator accepted candidate pairs greedily in descending
IoU order.  A high-IoU pair can consume the only prediction available to another
truth box even when an alternative assignment would match both boxes.  That
undercounts true positives and distorts every downstream detection and HOTA-style
metric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from . import tracklet_evaluation as _evaluation

_PATCHED_ATTR = "_beltmap_tracklet_cardinality_matching_patched"
_ORIGINAL_ATTR = "_beltmap_original_greedy_frame_matches"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original matcher behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_greedy_frame_matches = _unwrap_patched_callable(
    _evaluation.greedy_frame_matches
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
    """Add a unit-capacity residual edge and return its forward edge."""

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
    distances = [math.inf] * node_count
    previous_nodes = [-1] * node_count
    previous_edges = [-1] * node_count
    distances[source] = 0.0

    for _iteration in range(node_count - 1):
        changed = False
        for node, edges in enumerate(graph):
            if not math.isfinite(distances[node]):
                continue
            for edge_index, edge in enumerate(edges):
                if edge.capacity <= 0:
                    continue
                candidate = distances[node] + edge.cost
                if candidate < distances[edge.destination] - 1e-15:
                    distances[edge.destination] = candidate
                    previous_nodes[edge.destination] = node
                    previous_edges[edge.destination] = edge_index
                    changed = True
        if not changed:
            break

    if previous_nodes[sink] < 0:
        return None
    return previous_nodes, previous_edges


def _optimal_frame_pairs(
    truth_boxes: list[_evaluation.TrackletBox],
    prediction_boxes: list[_evaluation.TrackletBox],
    truth_indices: list[int],
    prediction_indices: list[int],
    *,
    iou_threshold: float,
) -> list[tuple[float, int, int]]:
    """Maximize valid match count, then total IoU, for one frame."""

    if not truth_indices or not prediction_indices:
        return []

    truth_count = len(truth_indices)
    prediction_count = len(prediction_indices)
    source = 0
    first_truth = 1
    first_prediction = first_truth + truth_count
    sink = first_prediction + prediction_count
    graph: list[list[_ResidualEdge]] = [[] for _node in range(sink + 1)]

    for truth_position in range(truth_count):
        _add_unit_edge(
            graph,
            source,
            first_truth + truth_position,
            cost=0.0,
        )
    for prediction_position in range(prediction_count):
        _add_unit_edge(
            graph,
            first_prediction + prediction_position,
            sink,
            cost=0.0,
        )

    candidate_edges: dict[
        tuple[int, int],
        tuple[_ResidualEdge, float],
    ] = {}
    for truth_position, truth_index in enumerate(truth_indices):
        truth = _evaluation.box_dict(truth_boxes[truth_index])
        for prediction_position, prediction_index in enumerate(prediction_indices):
            prediction = _evaluation.box_dict(prediction_boxes[prediction_index])
            iou = float(_evaluation.bbox_iou(prediction, truth))
            if iou < iou_threshold:
                continue
            edge = _add_unit_edge(
                graph,
                first_truth + truth_position,
                first_prediction + prediction_position,
                cost=-iou,
            )
            candidate_edges[(truth_position, prediction_position)] = (edge, iou)

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

    pairs = [
        (
            iou,
            truth_indices[truth_position],
            prediction_indices[prediction_position],
        )
        for (truth_position, prediction_position), (edge, iou) in candidate_edges.items()
        if edge.capacity == 0
    ]
    return sorted(pairs, key=lambda item: (item[1], item[2]))


def cardinality_optimal_frame_matches(
    truth_boxes: list[_evaluation.TrackletBox],
    prediction_boxes: list[_evaluation.TrackletBox],
    *,
    scored_frames: set[int],
    iou_threshold: float,
) -> tuple[list[_evaluation.TrackletMatch], set[int], set[int]]:
    """Match boxes one-to-one, maximizing cardinality before total IoU."""

    truth_by_frame = _evaluation.group_indices_by_frame(truth_boxes)
    prediction_by_frame = _evaluation.group_indices_by_frame(prediction_boxes)
    unmatched_truth = set(range(len(truth_boxes)))
    unmatched_predictions = set(range(len(prediction_boxes)))
    matches: list[_evaluation.TrackletMatch] = []

    for frame_index in sorted(scored_frames):
        pairs = _optimal_frame_pairs(
            truth_boxes,
            prediction_boxes,
            truth_by_frame.get(frame_index, []),
            prediction_by_frame.get(frame_index, []),
            iou_threshold=iou_threshold,
        )
        for iou, truth_index, prediction_index in pairs:
            unmatched_truth.discard(truth_index)
            unmatched_predictions.discard(prediction_index)
            truth_box = truth_boxes[truth_index]
            prediction_box = prediction_boxes[prediction_index]
            matches.append(
                _evaluation.TrackletMatch(
                    frame_index=frame_index,
                    truth_index=truth_index,
                    prediction_index=prediction_index,
                    truth_tracklet_id=truth_box.tracklet_id,
                    predicted_track_id=prediction_box.tracklet_id,
                    iou=iou,
                    centroid_error_px=_evaluation.center_distance_px(
                        prediction_box,
                        truth_box,
                    ),
                )
            )

    return matches, unmatched_truth, unmatched_predictions


setattr(cardinality_optimal_frame_matches, _PATCHED_ATTR, True)
setattr(
    cardinality_optimal_frame_matches,
    _ORIGINAL_ATTR,
    _original_greedy_frame_matches,
)
_evaluation.greedy_frame_matches = cardinality_optimal_frame_matches

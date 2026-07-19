"""Use cardinality-optimal box matching for real-data detection metrics.

The original evaluator greedily accepted candidate pairs in descending IoU order.
That can consume a detection needed by another truth box and therefore report
fewer true positives than a valid one-to-one matching contains. This patch uses
minimum-cost maximum flow: cardinality is maximized first, and total IoU is
maximized among matchings with that cardinality.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import advanced_quality as _advanced_quality

_PATCHED_ATTR = "_beltmap_cardinality_optimal_iou_matching_patched"
_ORIGINAL_ATTR = "_beltmap_original_evaluate_real_detections"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the evaluator behind this compatibility patch, if present."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_evaluate_real_detections = _unwrap_patched_callable(
    _advanced_quality.evaluate_real_detections
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
    """Return maximum-cardinality one-to-one matches, then maximize total IoU."""

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
        _add_unit_edge(
            graph,
            source,
            first_truth + truth_index,
            cost=0.0,
        )
    for detection_index in range(detection_count):
        _add_unit_edge(
            graph,
            first_detection + detection_index,
            sink,
            cost=0.0,
        )

    candidate_edges: dict[tuple[int, int], tuple[_ResidualEdge, float]] = {}
    for truth_index, truth_box in enumerate(truth_boxes):
        for detection_index, detection_box in enumerate(detection_boxes):
            iou = float(_advanced_quality.bbox_iou(truth_box, detection_box))
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


def cardinality_optimal_evaluate_real_detections(
    output_dir: Path,
    labels_path: Path,
    *,
    iou_threshold: float = 0.5,
) -> _advanced_quality.RealLabelMetrics:
    """Evaluate detections using cardinality-optimal one-to-one IoU matching."""

    iou_threshold = _advanced_quality._finite_real(iou_threshold, "iou_threshold")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")

    detections = _advanced_quality.detection_boxes_by_frame(output_dir)
    truth = _advanced_quality.load_real_label_boxes(labels_path)
    matches = 0
    ious: list[float] = []
    centroid_errors: list[float] = []
    labeled_frames = set(truth)
    detection_count = sum(
        len(detections.get(frame_index, []))
        for frame_index in labeled_frames
    )
    truth_count = sum(len(boxes) for boxes in truth.values())

    for frame_index, truth_boxes in truth.items():
        frame_detections = detections.get(frame_index, [])
        frame_matches = _maximum_cardinality_iou_matches(
            truth_boxes,
            frame_detections,
            iou_threshold=iou_threshold,
        )
        matches += len(frame_matches)
        for iou, truth_index, detection_index in frame_matches:
            ious.append(iou)
            truth_y, truth_x = _advanced_quality._box_centroid(
                truth_boxes[truth_index]
            )
            detection_y, detection_x = _advanced_quality._box_centroid(
                frame_detections[detection_index]
            )
            centroid_errors.append(
                float(
                    np.hypot(
                        truth_y - detection_y,
                        truth_x - detection_x,
                    )
                )
            )

    if detection_count == 0 and truth_count == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        precision = None if detection_count == 0 else matches / detection_count
        recall = None if truth_count == 0 else matches / truth_count
        if precision is None or recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

    return _advanced_quality.RealLabelMetrics(
        frames=len(truth),
        truth_boxes=truth_count,
        detection_boxes=detection_count,
        matches=matches,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=None if not ious else float(np.mean(ious)),
        mean_centroid_error_px=(
            None
            if not centroid_errors
            else float(np.mean(centroid_errors))
        ),
    )


setattr(cardinality_optimal_evaluate_real_detections, _PATCHED_ATTR, True)
setattr(
    cardinality_optimal_evaluate_real_detections,
    _ORIGINAL_ATTR,
    _original_evaluate_real_detections,
)
_advanced_quality.evaluate_real_detections = (
    cardinality_optimal_evaluate_real_detections
)

# Import for side effect: keep labeled bootstrap intervals consistent with the
# cardinality-optimal real-data point estimate above.
from . import bootstrap_ci_matching_patch as _bootstrap_ci_matching_patch  # noqa: F401,E402

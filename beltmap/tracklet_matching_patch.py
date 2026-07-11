"""Use cardinality-optimal assignment for sparse tracklet evaluation.

The original frame-level evaluator accepted candidate pairs greedily in descending
IoU order. A high-IoU pair can consume the only prediction available to another
truth box even when an alternative assignment would match both boxes. That
undercounts true positives and distorts every downstream detection and HOTA-style
metric.
"""

from __future__ import annotations

import math
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


def _maximum_weight_square_assignment(weights: list[list[float]]) -> list[int]:
    """Return a maximum-weight one-to-one square assignment.

    This is the O(n^3) Hungarian primal-dual algorithm. The returned list maps
    each row index to exactly one column index.
    """

    size = len(weights)
    if size == 0:
        return []
    if any(len(row) != size for row in weights):
        raise ValueError("assignment weights must form a square matrix")

    maximum_weight = max(max(row) for row in weights)
    costs = [
        [maximum_weight - weight for weight in row]
        for row in weights
    ]

    row_potential = [0.0] * (size + 1)
    column_potential = [0.0] * (size + 1)
    matched_row = [0] * (size + 1)
    previous_column = [0] * (size + 1)

    for row_index in range(1, size + 1):
        matched_row[0] = row_index
        minimum_slack = [math.inf] * (size + 1)
        used_column = [False] * (size + 1)
        current_column = 0

        while True:
            used_column[current_column] = True
            current_row = matched_row[current_column]
            delta = math.inf
            next_column = 0
            for column_index in range(1, size + 1):
                if used_column[column_index]:
                    continue
                reduced_cost = (
                    costs[current_row - 1][column_index - 1]
                    - row_potential[current_row]
                    - column_potential[column_index]
                )
                if reduced_cost < minimum_slack[column_index]:
                    minimum_slack[column_index] = reduced_cost
                    previous_column[column_index] = current_column
                if minimum_slack[column_index] < delta:
                    delta = minimum_slack[column_index]
                    next_column = column_index

            for column_index in range(size + 1):
                if used_column[column_index]:
                    row_potential[matched_row[column_index]] += delta
                    column_potential[column_index] -= delta
                else:
                    minimum_slack[column_index] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break

        while True:
            prior_column = previous_column[current_column]
            matched_row[current_column] = matched_row[prior_column]
            current_column = prior_column
            if current_column == 0:
                break

    assignment = [-1] * size
    for column_index in range(1, size + 1):
        row_index = matched_row[column_index]
        if row_index > 0:
            assignment[row_index - 1] = column_index - 1
    return assignment


def _optimal_frame_pairs(
    truth_boxes: list[_evaluation.TrackletBox],
    prediction_boxes: list[_evaluation.TrackletBox],
    truth_indices: list[int],
    prediction_indices: list[int],
    *,
    iou_threshold: float,
) -> list[tuple[float, int, int]]:
    """Maximize valid match count, then total IoU, for one frame."""

    truth_count = len(truth_indices)
    prediction_count = len(prediction_indices)
    if truth_count == 0 or prediction_count == 0:
        return []

    # Real truth rows plus one dummy row per prediction, and real prediction
    # columns plus one dummy column per truth, permit either side to remain
    # unmatched without ever forcing an invalid real-to-real edge.
    size = truth_count + prediction_count
    cardinality_bonus = float(size + 1)
    forbidden_weight = -cardinality_bonus * float(size + 1)
    weights = [[0.0] * size for _ in range(size)]
    accepted_iou: dict[tuple[int, int], float] = {}

    for truth_position, truth_index in enumerate(truth_indices):
        truth = _evaluation.box_dict(truth_boxes[truth_index])
        for prediction_position, prediction_index in enumerate(prediction_indices):
            prediction = _evaluation.box_dict(prediction_boxes[prediction_index])
            iou = float(_evaluation.bbox_iou(prediction, truth))
            if not math.isfinite(iou) or iou < iou_threshold:
                weights[truth_position][prediction_position] = forbidden_weight
                continue
            weights[truth_position][prediction_position] = cardinality_bonus + iou
            accepted_iou[(truth_position, prediction_position)] = iou

    assignment = _maximum_weight_square_assignment(weights)
    pairs: list[tuple[float, int, int]] = []
    for truth_position in range(truth_count):
        prediction_position = assignment[truth_position]
        iou = accepted_iou.get((truth_position, prediction_position))
        if iou is None:
            continue
        pairs.append(
            (
                iou,
                truth_indices[truth_position],
                prediction_indices[prediction_position],
            )
        )
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

"""Enforce one-to-one confirming-component use in cross-map agreement."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Any, Sequence

from . import cross_map_agreement as _agreement

_PATCHED_ATTR = "_beltmap_cross_map_one_to_one_patched"
_ORIGINAL_ATTR = "_beltmap_original_score_cross_map_agreement"

_MatchingCost = tuple[float, float, float]


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original scorer behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_score_cross_map_agreement = _unwrap_patched_callable(
    _agreement.score_cross_map_agreement
)


@dataclass
class _ResidualEdge:
    destination: int
    reverse_index: int
    capacity: int
    cost: _MatchingCost


def _negate_cost(cost: _MatchingCost) -> _MatchingCost:
    return tuple(-component for component in cost)


def _add_unit_edge(
    graph: list[list[_ResidualEdge]],
    source: int,
    destination: int,
    *,
    cost: _MatchingCost,
) -> _ResidualEdge:
    forward = _ResidualEdge(
        destination=destination,
        reverse_index=len(graph[destination]),
        capacity=1,
        cost=cost,
    )
    reverse = _ResidualEdge(
        destination=source,
        reverse_index=len(graph[source]),
        capacity=0,
        cost=_negate_cost(cost),
    )
    graph[source].append(forward)
    graph[destination].append(reverse)
    return forward


def _add_cost(left: _MatchingCost, right: _MatchingCost) -> _MatchingCost:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _shortest_augmenting_path(
    graph: list[list[_ResidualEdge]],
    *,
    source: int,
    sink: int,
) -> tuple[list[int], list[int]] | None:
    """Return one lexicographically minimum-cost residual path."""

    node_count = len(graph)
    distances: list[_MatchingCost | None] = [None] * node_count
    previous_nodes = [-1] * node_count
    previous_edges = [-1] * node_count
    distances[source] = (0.0, 0.0, 0.0)

    for _iteration in range(node_count - 1):
        changed = False
        for node, edges in enumerate(graph):
            distance = distances[node]
            if distance is None:
                continue
            for edge_index, edge in enumerate(edges):
                if edge.capacity <= 0:
                    continue
                candidate = _add_cost(distance, edge.cost)
                current = distances[edge.destination]
                if current is None or candidate < current:
                    distances[edge.destination] = candidate
                    previous_nodes[edge.destination] = node
                    previous_edges[edge.destination] = edge_index
                    changed = True
        if not changed:
            break

    if previous_nodes[sink] < 0:
        return None
    return previous_nodes, previous_edges


def _matching_cost(
    match: _agreement.CrossMapAgreementMapScore,
) -> _MatchingCost:
    """Minimize negative IoU/peak ratio, then centroid distance."""

    return (
        -float(match.bbox_iou or 0.0),
        -float(match.peak_ratio or 0.0),
        float(match.centroid_distance_px or 0.0),
    )


def _one_to_one_matches_for_map(
    primary_detections: Sequence[_agreement.ParticleDetection],
    candidates: Sequence[_agreement.ParticleDetection],
    *,
    primary_residual: _agreement.ResidualImage | None,
    confirming_residual: _agreement.ResidualImage | None,
    map_index: int,
    config: _agreement.CrossMapAgreementConfig,
    fallback_matches: Sequence[_agreement.CrossMapAgreementMapScore],
) -> list[_agreement.CrossMapAgreementMapScore]:
    """Return cardinality- and quality-optimal one-to-one accepted matches.

    The flow is augmented until no accepted primary/candidate path remains, which
    maximizes match cardinality. Lexicographic path costs then maximize total
    bounding-box IoU, maximize total peak ratio, and minimize total centroid
    distance among matchings with that cardinality.
    """

    if not primary_detections or not candidates:
        return [
            replace(match, accepted=False) if match.accepted else match
            for match in fallback_matches
        ]

    primary_count = len(primary_detections)
    candidate_count = len(candidates)
    source = 0
    first_primary = 1
    first_candidate = first_primary + primary_count
    sink = first_candidate + candidate_count
    graph: list[list[_ResidualEdge]] = [[] for _node in range(sink + 1)]

    zero_cost: _MatchingCost = (0.0, 0.0, 0.0)
    for primary_index in range(primary_count):
        _add_unit_edge(
            graph,
            source,
            first_primary + primary_index,
            cost=zero_cost,
        )
    for candidate_index in range(candidate_count):
        _add_unit_edge(
            graph,
            first_candidate + candidate_index,
            sink,
            cost=zero_cost,
        )

    candidate_signs = [
        _agreement.detection_raw_sign(candidate, confirming_residual)
        for candidate in candidates
    ]
    candidate_edges: dict[
        tuple[int, int],
        tuple[_ResidualEdge, _agreement.CrossMapAgreementMapScore],
    ] = {}

    for primary_index, detection in enumerate(primary_detections):
        primary_sign = _agreement.detection_raw_sign(detection, primary_residual)
        for candidate_index, (candidate, candidate_sign) in enumerate(
            zip(candidates, candidate_signs, strict=True)
        ):
            match = _agreement._score_match(
                detection,
                candidate,
                primary_sign=primary_sign,
                candidate_sign=candidate_sign,
                map_index=map_index,
                config=config,
            )
            if not match.accepted:
                continue
            edge = _add_unit_edge(
                graph,
                first_primary + primary_index,
                first_candidate + candidate_index,
                cost=_matching_cost(match),
            )
            candidate_edges[(primary_index, candidate_index)] = (edge, match)

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

    selected_by_primary = {
        primary_index: match
        for (primary_index, _candidate_index), (edge, match) in candidate_edges.items()
        if edge.capacity == 0
    }
    selected: list[_agreement.CrossMapAgreementMapScore] = []
    for primary_index, fallback in enumerate(fallback_matches):
        match = selected_by_primary.get(primary_index)
        if match is not None:
            selected.append(match)
        elif fallback.accepted:
            selected.append(replace(fallback, accepted=False))
        else:
            selected.append(fallback)
    return selected


def one_to_one_score_cross_map_agreement(
    primary_detections: Sequence[_agreement.ParticleDetection],
    confirming_detections_by_map: Sequence[
        Sequence[_agreement.ParticleDetection]
    ],
    *,
    primary_residual: _agreement.ResidualImage | None = None,
    confirming_residuals: Sequence[_agreement.ResidualImage | None] | None = None,
    config: _agreement.CrossMapAgreementConfig | None = None,
) -> list[_agreement.CrossMapAgreementScore]:
    """Score agreement without reusing one confirming component twice.

    The original scorer selected each primary detection's best confirming component
    independently. Two overlapping primary detections could therefore both count
    the same confirming component as independent evidence. This wrapper computes a
    maximum-cardinality, maximum-quality bipartite matching for each confirming map
    before counting confirmations.
    """

    primary = list(primary_detections)
    confirming = [list(detections) for detections in confirming_detections_by_map]
    materialized_residuals = (
        None if confirming_residuals is None else list(confirming_residuals)
    )
    cfg = config or _agreement.CrossMapAgreementConfig()

    baseline = _original_score_cross_map_agreement(
        primary,
        confirming,
        primary_residual=primary_residual,
        confirming_residuals=materialized_residuals,
        config=cfg,
    )
    if not baseline:
        return []

    residuals = (
        [None] * len(confirming)
        if materialized_residuals is None
        else materialized_residuals
    )
    matches_by_primary: list[list[_agreement.CrossMapAgreementMapScore]] = [
        [score.matches[map_position] for map_position in range(len(confirming))]
        for score in baseline
    ]

    for map_position, (candidates, confirming_residual) in enumerate(
        zip(confirming, residuals, strict=True)
    ):
        selected = _one_to_one_matches_for_map(
            primary,
            candidates,
            primary_residual=primary_residual,
            confirming_residual=confirming_residual,
            map_index=map_position + 1,
            config=cfg,
            fallback_matches=[
                score.matches[map_position] for score in baseline
            ],
        )
        for primary_index, match in enumerate(selected):
            matches_by_primary[primary_index][map_position] = match

    scores: list[_agreement.CrossMapAgreementScore] = []
    for baseline_score, matches in zip(baseline, matches_by_primary, strict=True):
        match_tuple = tuple(matches)
        confirming_maps = sum(1 for match in match_tuple if match.accepted)
        scores.append(
            _agreement.CrossMapAgreementScore(
                detection=baseline_score.detection,
                accepted=confirming_maps >= cfg.min_confirming_maps,
                confirming_maps=confirming_maps,
                matches=match_tuple,
            )
        )
    return scores


setattr(one_to_one_score_cross_map_agreement, _PATCHED_ATTR, True)
setattr(
    one_to_one_score_cross_map_agreement,
    _ORIGINAL_ATTR,
    _original_score_cross_map_agreement,
)
_agreement.score_cross_map_agreement = one_to_one_score_cross_map_agreement

_package = sys.modules.get(__package__)
if _package is not None:
    setattr(
        _package,
        "score_cross_map_agreement",
        one_to_one_score_cross_map_agreement,
    )

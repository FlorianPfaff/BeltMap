"""Enforce one-to-one confirming-component use in cross-map agreement."""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Any, Sequence

from . import cross_map_agreement as _agreement

_PATCHED_ATTR = "_beltmap_cross_map_one_to_one_patched"
_ORIGINAL_ATTR = "_beltmap_original_score_cross_map_agreement"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original scorer behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_score_cross_map_agreement = _unwrap_patched_callable(
    _agreement.score_cross_map_agreement
)


def _match_rank(
    match: _agreement.CrossMapAgreementMapScore,
) -> tuple[float, float, float, float]:
    distance_rank = (
        -float("inf")
        if match.centroid_distance_px is None
        else -match.centroid_distance_px
    )
    return (
        1.0 if match.accepted else 0.0,
        0.0 if match.bbox_iou is None else match.bbox_iou,
        0.0 if match.peak_ratio is None else match.peak_ratio,
        distance_rank,
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
    """Return maximum-cardinality one-to-one accepted matches for one map."""

    candidate_signs = [
        _agreement.detection_raw_sign(candidate, confirming_residual)
        for candidate in candidates
    ]
    accepted_edges: list[
        list[tuple[int, tuple[float, float, float, float]]]
    ] = []
    match_lookup: list[dict[int, _agreement.CrossMapAgreementMapScore]] = []

    for detection in primary_detections:
        primary_sign = _agreement.detection_raw_sign(detection, primary_residual)
        matches_by_candidate: dict[int, _agreement.CrossMapAgreementMapScore] = {}
        edges: list[tuple[int, tuple[float, float, float, float]]] = []
        for candidate_index, (candidate, candidate_sign) in enumerate(
            zip(candidates, candidate_signs)
        ):
            match = _agreement._score_match(
                detection,
                candidate,
                primary_sign=primary_sign,
                candidate_sign=candidate_sign,
                map_index=map_index,
                config=config,
            )
            matches_by_candidate[candidate_index] = match
            if match.accepted:
                edges.append((candidate_index, _match_rank(match)))
        edges.sort(key=lambda item: item[1], reverse=True)
        accepted_edges.append(edges)
        match_lookup.append(matches_by_candidate)

    candidate_to_primary: dict[int, int] = {}
    primary_to_candidate: dict[int, int] = {}

    def augment(primary_index: int, seen_candidates: set[int]) -> bool:
        for candidate_index, _rank in accepted_edges[primary_index]:
            if candidate_index in seen_candidates:
                continue
            seen_candidates.add(candidate_index)
            previous_primary = candidate_to_primary.get(candidate_index)
            if previous_primary is None or augment(
                previous_primary, seen_candidates
            ):
                candidate_to_primary[candidate_index] = primary_index
                primary_to_candidate[primary_index] = candidate_index
                return True
        return False

    primary_order = sorted(
        range(len(primary_detections)),
        key=lambda index: (
            accepted_edges[index][0][1]
            if accepted_edges[index]
            else (-float("inf"),) * 4
        ),
        reverse=True,
    )
    primary_order.sort(key=lambda index: len(accepted_edges[index]))
    for primary_index in primary_order:
        augment(primary_index, set())

    selected: list[_agreement.CrossMapAgreementMapScore] = []
    for primary_index, fallback in enumerate(fallback_matches):
        candidate_index = primary_to_candidate.get(primary_index)
        if candidate_index is not None:
            selected.append(match_lookup[primary_index][candidate_index])
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
    maximum-cardinality bipartite matching for each confirming map before counting
    confirmations.
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
        zip(confirming, residuals)
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
    for baseline_score, matches in zip(baseline, matches_by_primary):
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

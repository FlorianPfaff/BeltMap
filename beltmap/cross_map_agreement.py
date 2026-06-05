
"""Agreement scoring for detections from independently learned belt maps."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Sequence

import numpy as np

from .residual import ResidualImage
from .tracking import ParticleDetection


@dataclass(frozen=True)
class CrossMapAgreementConfig:
    """Gates for confirming detections with independently learned belt maps."""

    max_centroid_distance_px: float = 4.0
    min_bbox_iou: float = 0.0
    min_peak_ratio: float = 0.25
    require_sign_consistency: bool = True
    min_confirming_maps: int = 2
    filter_detections: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.max_centroid_distance_px) or self.max_centroid_distance_px < 0:
            raise ValueError("max_centroid_distance_px must be finite and non-negative")
        if not isfinite(self.min_bbox_iou) or not 0.0 <= self.min_bbox_iou <= 1.0:
            raise ValueError("min_bbox_iou must be finite and in [0, 1]")
        if not isfinite(self.min_peak_ratio) or not 0.0 <= self.min_peak_ratio <= 1.0:
            raise ValueError("min_peak_ratio must be finite and in [0, 1]")
        if self.min_confirming_maps < 1:
            raise ValueError("min_confirming_maps must be at least 1")


@dataclass(frozen=True)
class CrossMapAgreementMapScore:
    """Best-match diagnostics for one confirming map."""

    map_index: int
    matched_label: int | None
    centroid_distance_px: float | None
    bbox_iou: float | None
    peak_ratio: float | None
    sign_consistent: bool | None
    accepted: bool


@dataclass(frozen=True)
class CrossMapAgreementScore:
    """Agreement diagnostics for one primary detection."""

    detection: ParticleDetection
    accepted: bool
    confirming_maps: int
    matches: tuple[CrossMapAgreementMapScore, ...]


def score_cross_map_agreement(
    primary_detections: Sequence[ParticleDetection],
    confirming_detections_by_map: Sequence[Sequence[ParticleDetection]],
    *,
    primary_residual: ResidualImage | None = None,
    confirming_residuals: Sequence[ResidualImage | None] | None = None,
    config: CrossMapAgreementConfig | None = None,
) -> list[CrossMapAgreementScore]:
    """Score whether each primary detection is reproduced by confirming maps.

    The primary detections usually come from the normal full-sample map.  The
    confirming detections should come from independent, disjoint-sample maps
    rendered for the same frame and phase.  A primary detection is accepted only
    when enough confirming maps contain a compatible component.
    """

    cfg = config or CrossMapAgreementConfig()
    if cfg.min_confirming_maps > len(confirming_detections_by_map):
        raise ValueError(
            "min_confirming_maps cannot exceed the number of confirming maps"
        )
    residuals = (
        [None] * len(confirming_detections_by_map)
        if confirming_residuals is None
        else list(confirming_residuals)
    )
    if len(residuals) != len(confirming_detections_by_map):
        raise ValueError(
            "confirming_residuals must have the same length as "
            "confirming_detections_by_map"
        )

    scores: list[CrossMapAgreementScore] = []
    for detection in primary_detections:
        primary_sign = detection_raw_sign(detection, primary_residual)
        matches = tuple(
            _best_match_for_map(
                detection,
                confirming_detections,
                primary_sign=primary_sign,
                confirming_residual=confirming_residual,
                map_index=map_index,
                config=cfg,
            )
            for map_index, (confirming_detections, confirming_residual) in enumerate(
                zip(confirming_detections_by_map, residuals),
                start=1,
            )
        )
        confirming_maps = sum(1 for match in matches if match.accepted)
        scores.append(
            CrossMapAgreementScore(
                detection=detection,
                accepted=confirming_maps >= cfg.min_confirming_maps,
                confirming_maps=confirming_maps,
                matches=matches,
            )
        )
    return scores


def filter_detections_by_agreement(
    scores: Sequence[CrossMapAgreementScore],
    *,
    config: CrossMapAgreementConfig | None = None,
) -> list[ParticleDetection]:
    """Return detections that pass agreement, or all detections in score-only mode."""

    cfg = config or CrossMapAgreementConfig()
    if not cfg.filter_detections:
        return [score.detection for score in scores]
    return [score.detection for score in scores if score.accepted]


def detection_raw_sign(
    detection: ParticleDetection,
    residual: ResidualImage | None,
) -> int | None:
    """Return the sign of the un-oriented residual near a detection centroid."""

    if residual is None:
        return None
    raw = np.asarray(residual.raw, dtype=np.float64)
    mask = np.asarray(residual.mask, dtype=bool)
    if raw.shape != mask.shape or raw.ndim != 2:
        return None

    y = int(round(float(detection.y)))
    x = int(round(float(detection.x)))
    if 0 <= y < raw.shape[0] and 0 <= x < raw.shape[1] and mask[y, x]:
        value = float(raw[y, x])
        if np.isfinite(value) and value != 0.0:
            return 1 if value > 0.0 else -1

    top = max(0, int(detection.bbox_top))
    left = max(0, int(detection.bbox_left))
    bottom = min(raw.shape[0], int(detection.bbox_bottom))
    right = min(raw.shape[1], int(detection.bbox_right))
    if top >= bottom or left >= right:
        return None
    values = raw[top:bottom, left:right]
    valid = mask[top:bottom, left:right] & np.isfinite(values)
    if not np.any(valid):
        return None
    value = float(np.mean(values[valid]))
    if value == 0.0:
        return 0
    return 1 if value > 0.0 else -1


def _best_match_for_map(
    detection: ParticleDetection,
    candidates: Sequence[ParticleDetection],
    *,
    primary_sign: int | None,
    confirming_residual: ResidualImage | None,
    map_index: int,
    config: CrossMapAgreementConfig,
) -> CrossMapAgreementMapScore:
    best: CrossMapAgreementMapScore | None = None
    best_rank: tuple[float, float, float, float] | None = None
    for candidate in candidates:
        match = _score_match(
            detection,
            candidate,
            primary_sign=primary_sign,
            candidate_sign=detection_raw_sign(candidate, confirming_residual),
            map_index=map_index,
            config=config,
        )
        distance_rank = -float("inf") if match.centroid_distance_px is None else -match.centroid_distance_px
        rank = (
            1.0 if match.accepted else 0.0,
            0.0 if match.bbox_iou is None else match.bbox_iou,
            0.0 if match.peak_ratio is None else match.peak_ratio,
            distance_rank,
        )
        if best_rank is None or rank > best_rank:
            best = match
            best_rank = rank
    if best is not None:
        return best
    return CrossMapAgreementMapScore(
        map_index=map_index,
        matched_label=None,
        centroid_distance_px=None,
        bbox_iou=None,
        peak_ratio=None,
        sign_consistent=None,
        accepted=False,
    )


def _score_match(
    detection: ParticleDetection,
    candidate: ParticleDetection,
    *,
    primary_sign: int | None,
    candidate_sign: int | None,
    map_index: int,
    config: CrossMapAgreementConfig,
) -> CrossMapAgreementMapScore:
    distance = hypot(float(detection.y) - float(candidate.y), float(detection.x) - float(candidate.x))
    iou = bbox_iou(detection, candidate)
    ratio = peak_ratio(detection.peak_signal, candidate.peak_signal)
    sign_consistent = (
        None
        if primary_sign is None or candidate_sign is None
        else primary_sign == candidate_sign
    )
    accepted = (
        distance <= config.max_centroid_distance_px
        and iou >= config.min_bbox_iou
        and ratio is not None
        and ratio >= config.min_peak_ratio
        and (
            not config.require_sign_consistency
            or sign_consistent is True
        )
    )
    return CrossMapAgreementMapScore(
        map_index=map_index,
        matched_label=candidate.label,
        centroid_distance_px=float(distance),
        bbox_iou=float(iou),
        peak_ratio=ratio,
        sign_consistent=sign_consistent,
        accepted=accepted,
    )


def bbox_iou(a: ParticleDetection, b: ParticleDetection) -> float:
    """Return intersection-over-union of two detection bounding boxes."""

    top = max(int(a.bbox_top), int(b.bbox_top))
    left = max(int(a.bbox_left), int(b.bbox_left))
    bottom = min(int(a.bbox_bottom), int(b.bbox_bottom))
    right = min(int(a.bbox_right), int(b.bbox_right))
    inter_h = max(0, bottom - top)
    inter_w = max(0, right - left)
    intersection = inter_h * inter_w
    area_a = max(0, int(a.bbox_bottom) - int(a.bbox_top)) * max(0, int(a.bbox_right) - int(a.bbox_left))
    area_b = max(0, int(b.bbox_bottom) - int(b.bbox_top)) * max(0, int(b.bbox_right) - int(b.bbox_left))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def peak_ratio(a: float | None, b: float | None) -> float | None:
    """Return min(abs(a), abs(b)) / max(abs(a), abs(b)) for finite peaks."""

    if a is None or b is None:
        return None
    first = abs(float(a))
    second = abs(float(b))
    if not np.isfinite(first) or not np.isfinite(second):
        return None
    high = max(first, second)
    if high <= 0.0:
        return None
    return float(min(first, second) / high)

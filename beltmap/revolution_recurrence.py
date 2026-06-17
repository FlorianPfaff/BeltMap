"""Track-level belt-revolution recurrence scoring.

This module is intentionally stricter than belt-fixedness diagnostics. A track
that is fixed in belt coordinates is not suspicious by itself because real
particles ride with the belt while visible. The runtime ghost signal here is
leave-current-track-out recurrence: other detections in other exposed belt
revolutions reappear near the same belt coordinate.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .tracking import ParticleDetection
from .tracking import ParticleTrack


@dataclass(frozen=True)
class BeltRevolutionRecurrenceConfig:
    """Settings for runtime track-level recurrence filtering."""

    radius_y_px: float = 8.0
    radius_x_px: float = 8.0
    min_track_detections: int = 5
    min_other_revolutions: int = 2
    min_other_detections: int = 2
    min_recurrence_fraction: float = 1.0


@dataclass(frozen=True)
class BeltRevolutionTrackScore:
    """Recurrence evidence for one track."""

    track_id: int
    n_detections: int
    frame_start: float
    frame_end: float
    belt_y_center_px: float | None
    belt_x_center_px: float | None
    belt_y_rms_px: float | None
    belt_x_std_px: float | None
    self_revolution_count: int
    self_revolutions_json: str
    other_exposed_revolutions: int
    other_hit_revolutions: int
    other_hit_revolutions_json: str
    other_hit_detections: int
    recurrence_fraction: float
    recurrence_score: float
    runtime_recurrence_rejected: bool
    causal_read: str

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _BeltPoint:
    track_id: int
    frame_index: int
    revolution: int
    belt_y: float
    x: float


def validate_recurrence_config(config: BeltRevolutionRecurrenceConfig) -> None:
    if not math.isfinite(config.radius_y_px) or config.radius_y_px < 0:
        raise ValueError("radius_y_px must be finite and non-negative")
    if not math.isfinite(config.radius_x_px) or config.radius_x_px < 0:
        raise ValueError("radius_x_px must be finite and non-negative")
    if config.min_track_detections < 1:
        raise ValueError("min_track_detections must be positive")
    if config.min_other_revolutions < 1:
        raise ValueError("min_other_revolutions must be positive")
    if config.min_other_detections < 1:
        raise ValueError("min_other_detections must be positive")
    if (
        not math.isfinite(config.min_recurrence_fraction)
        or not 0.0 <= config.min_recurrence_fraction <= 1.0
    ):
        raise ValueError("min_recurrence_fraction must be in [0, 1]")


def circular_distance(a: float, b: float, period: float) -> float:
    delta = abs(a - b) % period
    return float(min(delta, period - delta))


def circular_mean(values: Sequence[float], period: float) -> float | None:
    if not values:
        return None
    angles = 2.0 * np.pi * np.asarray(values, dtype=np.float64) / period
    vector = np.mean(np.exp(1j * angles))
    return float((np.angle(vector) % (2.0 * np.pi)) * period / (2.0 * np.pi))


def detection_belt_y(
    detection: ParticleDetection,
    *,
    phase_px: float,
    map_height_px: float,
) -> float:
    return float((float(detection.y) + phase_px) % map_height_px)


def coordinate_visible_in_frame(
    belt_y: float,
    *,
    phase_px: float,
    frame_height_px: float,
    map_height_px: float,
) -> bool:
    image_y = (belt_y - phase_px) % map_height_px
    return 0.0 <= image_y < frame_height_px


def _track_center(
    track: ParticleTrack,
    *,
    phase_px_by_frame: Sequence[float],
    map_height_px: float,
) -> tuple[float | None, float | None, float | None, float | None]:
    belt_y_values: list[float] = []
    x_values: list[float] = []
    for detection in track.detections:
        frame_index = int(round(float(detection.frame_index)))
        if not 0 <= frame_index < len(phase_px_by_frame):
            continue
        belt_y_values.append(
            detection_belt_y(
                detection,
                phase_px=float(phase_px_by_frame[frame_index]),
                map_height_px=map_height_px,
            )
        )
        x_values.append(float(detection.x))
    belt_y_center = circular_mean(belt_y_values, map_height_px)
    belt_x_center = float(np.mean(x_values)) if x_values else None
    if belt_y_center is None:
        belt_y_rms = None
    else:
        distances = [
            circular_distance(value, belt_y_center, map_height_px)
            for value in belt_y_values
        ]
        belt_y_rms = float(np.sqrt(np.mean(np.square(distances)))) if distances else None
    belt_x_std = float(np.std(x_values)) if x_values else None
    return belt_y_center, belt_x_center, belt_y_rms, belt_x_std


def _belt_points(
    tracks: Sequence[ParticleTrack],
    *,
    phase_px_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    map_height_px: float,
) -> list[_BeltPoint]:
    points: list[_BeltPoint] = []
    for track in tracks:
        for detection in track.detections:
            frame_index = int(round(float(detection.frame_index)))
            if not 0 <= frame_index < len(phase_px_by_frame):
                continue
            if not 0 <= frame_index < len(revolution_by_frame):
                continue
            points.append(
                _BeltPoint(
                    track_id=int(track.track_id),
                    frame_index=frame_index,
                    revolution=int(revolution_by_frame[frame_index]),
                    belt_y=detection_belt_y(
                        detection,
                        phase_px=float(phase_px_by_frame[frame_index]),
                        map_height_px=map_height_px,
                    ),
                    x=float(detection.x),
                )
            )
    return points


def _track_revolutions(
    track: ParticleTrack,
    revolution_by_frame: Sequence[int],
) -> set[int]:
    revolutions: set[int] = set()
    for detection in track.detections:
        frame_index = int(round(float(detection.frame_index)))
        if 0 <= frame_index < len(revolution_by_frame):
            revolutions.add(int(revolution_by_frame[frame_index]))
    return revolutions


def _exposed_other_revolutions(
    *,
    belt_y_center: float,
    self_revolutions: set[int],
    phase_px_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    frame_height_px: float,
    map_height_px: float,
) -> set[int]:
    exposed: set[int] = set()
    for frame_index, revolution in enumerate(revolution_by_frame):
        revolution = int(revolution)
        if revolution in self_revolutions:
            continue
        if coordinate_visible_in_frame(
            belt_y_center,
            phase_px=float(phase_px_by_frame[frame_index]),
            frame_height_px=frame_height_px,
            map_height_px=map_height_px,
        ):
            exposed.add(revolution)
    return exposed


def _score_one_track(
    track: ParticleTrack,
    *,
    all_points: Sequence[_BeltPoint],
    phase_px_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    frame_height_px: float,
    map_height_px: float,
    config: BeltRevolutionRecurrenceConfig,
) -> BeltRevolutionTrackScore:
    center_y, center_x, belt_y_rms, belt_x_std = _track_center(
        track,
        phase_px_by_frame=phase_px_by_frame,
        map_height_px=map_height_px,
    )
    self_revolutions = _track_revolutions(track, revolution_by_frame)
    if center_y is None or center_x is None:
        return BeltRevolutionTrackScore(
            track_id=int(track.track_id),
            n_detections=track.n_detections,
            frame_start=track.frame_start,
            frame_end=track.frame_end,
            belt_y_center_px=center_y,
            belt_x_center_px=center_x,
            belt_y_rms_px=belt_y_rms,
            belt_x_std_px=belt_x_std,
            self_revolution_count=len(self_revolutions),
            self_revolutions_json=json.dumps(sorted(self_revolutions)),
            other_exposed_revolutions=0,
            other_hit_revolutions=0,
            other_hit_revolutions_json="[]",
            other_hit_detections=0,
            recurrence_fraction=0.0,
            recurrence_score=0.0,
            runtime_recurrence_rejected=False,
            causal_read="track has no belt-coordinate center",
        )

    exposed = _exposed_other_revolutions(
        belt_y_center=center_y,
        self_revolutions=self_revolutions,
        phase_px_by_frame=phase_px_by_frame,
        revolution_by_frame=revolution_by_frame,
        frame_height_px=frame_height_px,
        map_height_px=map_height_px,
    )
    hit_revolutions: set[int] = set()
    hit_detections = 0
    for point in all_points:
        if point.track_id == track.track_id:
            continue
        if point.revolution in self_revolutions or point.revolution not in exposed:
            continue
        if circular_distance(point.belt_y, center_y, map_height_px) > config.radius_y_px:
            continue
        if abs(point.x - center_x) > config.radius_x_px:
            continue
        hit_revolutions.add(point.revolution)
        hit_detections += 1

    recurrence_fraction = (
        0.0 if not exposed else float(len(hit_revolutions) / len(exposed))
    )
    required_fraction = max(config.min_recurrence_fraction, 1e-12)
    recurrence_score = min(
        1.0,
        min(
            len(hit_revolutions) / max(1, config.min_other_revolutions),
            hit_detections / max(1, config.min_other_detections),
            recurrence_fraction / required_fraction,
        ),
    )
    rejected = (
        track.n_detections >= config.min_track_detections
        and len(hit_revolutions) >= config.min_other_revolutions
        and hit_detections >= config.min_other_detections
        and recurrence_fraction >= config.min_recurrence_fraction
    )
    if rejected:
        read = "other detections recur at this belt coordinate in exposed revolutions"
    elif not exposed:
        read = "no other belt revolution exposed this coordinate"
    elif len(hit_revolutions) == 0:
        read = "belt-fixed track, but no leave-track-out recurrence"
    else:
        read = "weak recurrence evidence below runtime filter threshold"

    return BeltRevolutionTrackScore(
        track_id=int(track.track_id),
        n_detections=track.n_detections,
        frame_start=track.frame_start,
        frame_end=track.frame_end,
        belt_y_center_px=center_y,
        belt_x_center_px=center_x,
        belt_y_rms_px=belt_y_rms,
        belt_x_std_px=belt_x_std,
        self_revolution_count=len(self_revolutions),
        self_revolutions_json=json.dumps(sorted(self_revolutions)),
        other_exposed_revolutions=len(exposed),
        other_hit_revolutions=len(hit_revolutions),
        other_hit_revolutions_json=json.dumps(sorted(hit_revolutions)),
        other_hit_detections=hit_detections,
        recurrence_fraction=recurrence_fraction,
        recurrence_score=recurrence_score,
        runtime_recurrence_rejected=rejected,
        causal_read=read,
    )


def score_belt_revolution_track_recurrence(
    tracks: Sequence[ParticleTrack],
    *,
    phase_px_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    frame_height_px: float,
    map_height_px: float,
    config: BeltRevolutionRecurrenceConfig | None = None,
) -> list[BeltRevolutionTrackScore]:
    """Score tracks by leave-current-track-out recurrence across belt revolutions."""

    cfg = config or BeltRevolutionRecurrenceConfig()
    validate_recurrence_config(cfg)
    if len(phase_px_by_frame) != len(revolution_by_frame):
        raise ValueError("phase_px_by_frame and revolution_by_frame must have equal length")
    if frame_height_px <= 0 or map_height_px <= 0:
        raise ValueError("frame_height_px and map_height_px must be positive")

    points = _belt_points(
        tracks,
        phase_px_by_frame=phase_px_by_frame,
        revolution_by_frame=revolution_by_frame,
        map_height_px=map_height_px,
    )
    return [
        _score_one_track(
            track,
            all_points=points,
            phase_px_by_frame=phase_px_by_frame,
            revolution_by_frame=revolution_by_frame,
            frame_height_px=frame_height_px,
            map_height_px=map_height_px,
            config=cfg,
        )
        for track in tracks
    ]

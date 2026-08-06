"""Require recurrent-artifact tracks to align with scored velocity rows."""

from __future__ import annotations

import sys
from collections import Counter
from typing import Any, Sequence

from . import tracking as _tracking

_PATCHED_ATTR = "_beltmap_recurrent_gate_alignment_patched"
_ORIGINAL_ATTR = "_beltmap_original_score_particle_velocities"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original scorer behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_score_particle_velocities = _unwrap_patched_callable(
    _tracking.score_particle_velocities
)


def _format_track_ids(values: Sequence[Any]) -> str:
    """Format track identifiers deterministically for validation errors."""

    return ", ".join(repr(value) for value in sorted(values, key=repr))


def aligned_score_particle_velocities(
    velocities: Sequence[_tracking.ParticleVelocity],
    *,
    config: _tracking.TrackFilterConfig | None = None,
    tracks: Sequence[_tracking.ParticleTrack] | None = None,
) -> list[_tracking.ParticleTrackScore]:
    """Score velocities only when recurrent evidence is unambiguously aligned.

    Recurrent-artifact gating is meaningful only when every velocity row has
    exactly one corresponding track. The historical scorer built a dictionary
    from the supplied tracks, silently overwriting duplicate identifiers and
    treating missing tracks as empty recurrent evidence. Both cases could let a
    velocity pass an explicitly enabled recurrent-artifact gate.
    """

    materialized_velocities = list(velocities)
    materialized_tracks = None if tracks is None else list(tracks)
    cfg = config or _tracking.TrackFilterConfig()

    if (
        cfg.max_recurrent_artifact_track_score is not None
        and materialized_tracks is not None
    ):
        track_ids = [track.track_id for track in materialized_tracks]
        duplicate_ids = [
            track_id
            for track_id, count in Counter(track_ids).items()
            if count > 1
        ]
        if duplicate_ids:
            raise ValueError(
                "tracks must have unique track_id values when recurrent-artifact "
                f"gating is enabled; duplicates: {_format_track_ids(duplicate_ids)}"
            )

        available_ids = set(track_ids)
        missing_ids = {
            velocity.track_id
            for velocity in materialized_velocities
            if velocity.track_id not in available_ids
        }
        if missing_ids:
            raise ValueError(
                "tracks must contain one matching track for every velocity when "
                "recurrent-artifact gating is enabled; missing track_id values: "
                f"{_format_track_ids(tuple(missing_ids))}"
            )

    return _original_score_particle_velocities(
        materialized_velocities,
        config=cfg,
        tracks=materialized_tracks,
    )


setattr(aligned_score_particle_velocities, _PATCHED_ATTR, True)
setattr(
    aligned_score_particle_velocities,
    _ORIGINAL_ATTR,
    _original_score_particle_velocities,
)
_tracking.score_particle_velocities = aligned_score_particle_velocities

_package = sys.modules.get(__package__)
if _package is not None:
    setattr(_package, "score_particle_velocities", aligned_score_particle_velocities)

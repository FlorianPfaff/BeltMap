"""Filter particle-velocity rows by their corresponding score.

The original helper collected accepted track identifiers and then returned every
velocity row whose identifier appeared in that set. Duplicate identifiers can
therefore leak a rejected row into the result whenever another row with the same
identifier passes the gates. This compatibility patch preserves input order and
keeps each row only when its own score is accepted.
"""

from __future__ import annotations

from typing import Any, Sequence

from . import tracking as _tracking

_PATCHED_ATTR = "_beltmap_rowwise_velocity_filter_patched"
_ORIGINAL_ATTR = "_beltmap_original_filter_particle_velocities"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original helper behind this patch, if already installed."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_filter_particle_velocities = _unwrap_patched_callable(
    _tracking.filter_particle_velocities
)


def rowwise_filter_particle_velocities(
    velocities: Sequence[_tracking.ParticleVelocity],
    *,
    config: _tracking.TrackFilterConfig | None = None,
    tracks: Sequence[_tracking.ParticleTrack] | None = None,
) -> list[_tracking.ParticleVelocity]:
    """Return only rows whose corresponding velocity score is accepted."""

    velocity_rows = list(velocities)
    scores = _tracking.score_particle_velocities(
        velocity_rows,
        config=config,
        tracks=tracks,
    )
    return [
        velocity
        for velocity, score in zip(velocity_rows, scores, strict=True)
        if score.accepted
    ]


setattr(rowwise_filter_particle_velocities, _PATCHED_ATTR, True)
setattr(
    rowwise_filter_particle_velocities,
    _ORIGINAL_ATTR,
    _original_filter_particle_velocities,
)
_tracking.filter_particle_velocities = rowwise_filter_particle_velocities

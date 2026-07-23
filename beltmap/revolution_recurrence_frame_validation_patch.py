"""Reject invalid detection frame indices in revolution-recurrence scoring.

Revolution recurrence indexes dense phase and revolution arrays. Fractional frame
indices previously passed through ``round`` and were silently assigned to a nearby
frame, while negative or non-finite indices could be ignored or fail indirectly.
Validate the frame contract before recurrence geometry is computed.
"""

from __future__ import annotations

import math
from functools import wraps
from typing import Any, Sequence

import numpy as np

from . import revolution_recurrence as _revolution_recurrence

_PATCHED_ATTR = "_beltmap_revolution_recurrence_frame_validation_patched"
_ORIGINAL_ATTR = "_beltmap_original_score_belt_revolution_track_recurrence"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the scorer behind this compatibility patch, if already installed."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_score_belt_revolution_track_recurrence = _unwrap_patched_callable(
    _revolution_recurrence.score_belt_revolution_track_recurrence
)


def _validated_frame_index(value: Any, *, track_id: Any, position: int) -> int:
    """Return an exact non-negative frame index or raise a contextual error."""

    if isinstance(value, (bool, np.bool_)):
        parsed = None
    else:
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            parsed = None
    if (
        parsed is None
        or not math.isfinite(parsed)
        or parsed < 0.0
        or not parsed.is_integer()
    ):
        raise ValueError(
            "detection frame_index must be a finite non-negative integer; "
            f"track {track_id!r}, detection {position} has {value!r}"
        )
    return int(parsed)


@wraps(_original_score_belt_revolution_track_recurrence)
def validate_detection_frame_indices(
    tracks: Sequence[_revolution_recurrence.ParticleTrack],
    *,
    phase_px_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    frame_height_px: float,
    map_height_px: float,
    config: _revolution_recurrence.BeltRevolutionRecurrenceConfig | None = None,
) -> list[_revolution_recurrence.BeltRevolutionTrackScore]:
    """Score recurrence only when every detection references an exact frame."""

    for track in tracks:
        for position, detection in enumerate(track.detections):
            _validated_frame_index(
                detection.frame_index,
                track_id=track.track_id,
                position=position,
            )
    return _original_score_belt_revolution_track_recurrence(
        tracks,
        phase_px_by_frame=phase_px_by_frame,
        revolution_by_frame=revolution_by_frame,
        frame_height_px=frame_height_px,
        map_height_px=map_height_px,
        config=config,
    )


setattr(validate_detection_frame_indices, _PATCHED_ATTR, True)
setattr(
    validate_detection_frame_indices,
    _ORIGINAL_ATTR,
    _original_score_belt_revolution_track_recurrence,
)
_revolution_recurrence.score_belt_revolution_track_recurrence = (
    validate_detection_frame_indices
)

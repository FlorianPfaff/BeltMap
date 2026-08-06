"""Compatibility fix for texture-stress velocity subset assignment."""

from __future__ import annotations

from typing import Any, Iterable

from . import texture_stress as _texture_stress


PATCH_MARKER = "texture-stress-velocity-single-anchor-v1"


def velocity_rows_in_frames(
    rows: Iterable[dict[str, Any]],
    frames: set[int],
) -> list[dict[str, Any]]:
    """Select velocity rows by one representative frame per track.

    Track-level velocity estimates must belong to at most one mutually exclusive
    texture-stress subset.  Use the track midpoint when both endpoints are
    available, then fall back to the available endpoint for legacy rows.
    """

    selected: list[dict[str, Any]] = []
    for row in rows:
        start = _texture_stress.finite_int(row.get("frame_start"))
        end = _texture_stress.finite_int(row.get("frame_end"))
        if start is not None and end is not None:
            representative_frame = int(round((start + end) / 2.0))
        elif start is not None:
            representative_frame = start
        elif end is not None:
            representative_frame = end
        else:
            continue
        if representative_frame in frames:
            selected.append(row)
    return selected


_texture_stress.velocity_rows_in_frames = velocity_rows_in_frames

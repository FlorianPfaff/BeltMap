"""Reject duplicate phase rows in runtime revolution-recurrence inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beltmap.cli import filter_revolution_recurrence as _filter_revolution_recurrence

_PATCHED_ATTR = "_beltmap_revolution_recurrence_unique_phase_rows_patched"
_ORIGINAL_ATTR = "_beltmap_revolution_recurrence_original_load_phase_px_by_frame"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original phase loader if this patch is imported repeatedly."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_load_phase_px_by_frame = _unwrap_patched_callable(
    _filter_revolution_recurrence.load_phase_px_by_frame
)


def load_unique_phase_px_by_frame(path: Path, frame_count: int) -> list[float]:
    """Reject ambiguous duplicate phase estimates before loading the dense series."""

    seen_frames: set[int] = set()
    for row_number, row in enumerate(
        _filter_revolution_recurrence.read_csv(path),
        start=2,
    ):
        frame_index = int(float(row["frame_index"]))
        if not 0 <= frame_index < frame_count:
            continue
        if frame_index in seen_frames:
            raise ValueError(
                "phase_estimates.csv contains duplicate phase estimate "
                f"for frame {frame_index} at row {row_number}"
            )
        seen_frames.add(frame_index)

    return _original_load_phase_px_by_frame(path, frame_count)


setattr(load_unique_phase_px_by_frame, _PATCHED_ATTR, True)
setattr(
    load_unique_phase_px_by_frame,
    _ORIGINAL_ATTR,
    _original_load_phase_px_by_frame,
)
_filter_revolution_recurrence.load_phase_px_by_frame = load_unique_phase_px_by_frame

"""Validate phase rows in runtime revolution-recurrence inputs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from beltmap.cli import filter_revolution_recurrence as _filter_revolution_recurrence

_PATCHED_ATTR = "_beltmap_revolution_recurrence_unique_phase_rows_patched"
_ORIGINAL_ATTR = "_beltmap_revolution_recurrence_original_load_phase_px_by_frame"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_load_phase_px_by_frame = _unwrap_patched_callable(
    _filter_revolution_recurrence.load_phase_px_by_frame
)


def _exact_frame_index(value: Any, *, row_number: int) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "phase_estimates.csv contains invalid frame_index "
            f"{value!r} at row {row_number}; expected a finite integer"
        ) from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(
            "phase_estimates.csv contains invalid frame_index "
            f"{value!r} at row {row_number}; expected a finite integer"
        )
    return int(parsed)


def _finite_phase_px(value: Any, *, frame_index: int, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "phase_estimates.csv contains invalid phase_px "
            f"{value!r} for frame {frame_index} at row {row_number}; "
            "expected a finite number"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            "phase_estimates.csv contains invalid phase_px "
            f"{value!r} for frame {frame_index} at row {row_number}; "
            "expected a finite number"
        )
    return parsed


def load_unique_phase_px_by_frame(path: Path, frame_count: int) -> list[float]:
    seen_frames: set[int] = set()
    for row_number, row in enumerate(
        _filter_revolution_recurrence.read_csv(path),
        start=2,
    ):
        frame_index = _exact_frame_index(
            row.get("frame_index", ""),
            row_number=row_number,
        )
        if not 0 <= frame_index < frame_count:
            continue
        _finite_phase_px(
            row.get("phase_px", ""),
            frame_index=frame_index,
            row_number=row_number,
        )
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

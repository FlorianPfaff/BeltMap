from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from beltmap import yolo_recurrence as _yolo_recurrence

_PATCHED_ATTR = "_beltmap_yolo_recurrence_frame_validated"
_ORIGINAL_ATTR = "_beltmap_yolo_recurrence_original_score_detection_recurrence"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_score_detection_recurrence = _unwrap_patched_callable(
    _yolo_recurrence.score_detection_recurrence
)


def _validate_detection_frame(
    row: Mapping[str, Any],
    *,
    phase_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    source_images: Mapping[int, Any],
) -> int:
    if "frame_index" not in row:
        raise ValueError("YOLO recurrence detection row is missing frame_index")
    frame_index = _yolo_recurrence.int_value(row["frame_index"], name="frame_index")

    phase_count = len(phase_by_frame)
    if frame_index < 0 or frame_index >= phase_count:
        raise ValueError(
            "YOLO recurrence detection frame_index "
            f"{frame_index} is outside phase_estimates range [0, {phase_count})"
        )

    revolution_count = len(revolution_by_frame)
    if frame_index >= revolution_count:
        raise ValueError(
            "YOLO recurrence detection frame_index "
            f"{frame_index} is outside revolution-index range [0, {revolution_count})"
        )

    try:
        phase = float(phase_by_frame[frame_index])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "YOLO recurrence detection frame_index "
            f"{frame_index} has no usable phase estimate"
        ) from exc
    if not np.isfinite(phase):
        raise ValueError(
            "YOLO recurrence detection frame_index "
            f"{frame_index} has a non-finite phase estimate"
        )

    if frame_index not in source_images:
        raise ValueError(
            "YOLO recurrence detection frame_index "
            f"{frame_index} has no matching source image"
        )

    return frame_index


def frame_validating_score_detection_recurrence(
    row: Mapping[str, Any],
    *,
    phase_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    source_images: Mapping[int, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Reject detection rows that would otherwise index the wrong/missing frame."""

    _validate_detection_frame(
        row,
        phase_by_frame=phase_by_frame,
        revolution_by_frame=revolution_by_frame,
        source_images=source_images,
    )
    return _original_score_detection_recurrence(
        row,
        phase_by_frame=phase_by_frame,
        revolution_by_frame=revolution_by_frame,
        source_images=source_images,
        **kwargs,
    )


setattr(frame_validating_score_detection_recurrence, _PATCHED_ATTR, True)
setattr(
    frame_validating_score_detection_recurrence,
    _ORIGINAL_ATTR,
    _original_score_detection_recurrence,
)
_yolo_recurrence.score_detection_recurrence = frame_validating_score_detection_recurrence

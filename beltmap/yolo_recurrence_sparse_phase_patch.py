from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from beltmap import yolo_recurrence as _yolo_recurrence

_PATCHED_ATTR = "_beltmap_yolo_recurrence_sparse_phase_patched"
_ORIGINAL_LOAD_ATTR = "_beltmap_yolo_recurrence_original_load_phase_px_by_frame"
_ORIGINAL_FIND_ATTR = "_beltmap_yolo_recurrence_original_find_revisit"


def _unwrap_patched_callable(func: Any, original_attr: str) -> Any:
    """Return the original callable behind our wrapper, if already patched."""

    return getattr(func, original_attr, func)


_original_load_phase_px_by_frame = _unwrap_patched_callable(
    _yolo_recurrence.load_phase_px_by_frame,
    _ORIGINAL_LOAD_ATTR,
)
_original_find_revisit = _unwrap_patched_callable(
    _yolo_recurrence.find_revisit,
    _ORIGINAL_FIND_ATTR,
)


def sparse_load_phase_px_by_frame(path: Path, *, frame_count: int) -> list[float]:
    """Load phase rows without requiring a dense zero-based frame prefix.

    YOLO recurrence can be run on subsets whose source frame names keep their
    original absolute frame indices.  In that case ``phase_estimates.csv`` may
    contain only frames such as 1000..1499, while the scorer still needs an
    indexable ``phase_by_frame`` list up to ``max(frame_index)``.  Represent
    missing frames as NaN and let detection-frame validation reject only rows
    that are actually scored without a usable phase estimate.
    """

    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    phases = [float("nan") for _ in range(frame_count)]
    observed_in_range = 0
    for row in _yolo_recurrence.read_csv_rows(path):
        frame = _yolo_recurrence.int_value(row.get("frame_index"), name="frame_index")
        phase = _yolo_recurrence.float_value(row.get("phase_px"), name="phase_px")
        if 0 <= frame < frame_count:
            if not math.isfinite(phases[frame]):
                observed_in_range += 1
            phases[frame] = phase

    if frame_count > 0 and observed_in_range == 0:
        raise ValueError(
            "phase_estimates.csv contains no usable frame in range "
            f"0..{frame_count - 1}"
        )
    return phases


def sparse_phase_find_revisit(
    *,
    frame_index: int,
    belt_y: float,
    x: float,
    patch_shape: tuple[int, int],
    revolution_offset: int,
    phase_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    source_images: Mapping[int, Path],
    image_shape: tuple[int, int],
    map_height: int,
) -> tuple[int, float, _yolo_recurrence.PatchBox] | None:
    """Find a revisit frame while skipping sparse/missing phase entries."""

    source_revolution = int(revolution_by_frame[frame_index])
    target_revolution = source_revolution + revolution_offset
    candidates: list[tuple[float, int, float, _yolo_recurrence.PatchBox]] = []
    for candidate_frame, phase in enumerate(phase_by_frame):
        if candidate_frame == frame_index:
            continue
        if candidate_frame not in source_images:
            continue
        if candidate_frame >= len(revolution_by_frame):
            continue
        try:
            candidate_phase = float(phase)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(candidate_phase):
            continue
        if int(revolution_by_frame[candidate_frame]) != target_revolution:
            continue
        projected_y = (float(belt_y) - candidate_phase) % map_height
        if not math.isfinite(projected_y):
            continue
        if projected_y < 0.0 or projected_y >= image_shape[0]:
            continue
        patch = _yolo_recurrence.centered_patch_box(
            y=projected_y,
            x=x,
            height=patch_shape[0],
            width=patch_shape[1],
            image_shape=image_shape,
        )
        if patch is None:
            continue
        candidates.append(
            (abs(projected_y - image_shape[0] / 2.0), candidate_frame, projected_y, patch)
        )

    if not candidates:
        return None
    _distance, selected_frame, projected_y, selected_patch = min(
        candidates,
        key=lambda item: (abs(item[2] - image_shape[0] / 2.0), abs(item[1] - frame_index)),
    )
    return selected_frame, projected_y, selected_patch


setattr(sparse_load_phase_px_by_frame, _PATCHED_ATTR, True)
setattr(sparse_load_phase_px_by_frame, _ORIGINAL_LOAD_ATTR, _original_load_phase_px_by_frame)
setattr(sparse_phase_find_revisit, _PATCHED_ATTR, True)
setattr(sparse_phase_find_revisit, _ORIGINAL_FIND_ATTR, _original_find_revisit)

_yolo_recurrence.load_phase_px_by_frame = sparse_load_phase_px_by_frame
_yolo_recurrence.find_revisit = sparse_phase_find_revisit

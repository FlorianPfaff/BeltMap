"""Prefer the most complete per-detection score field for FROC sweeps.

BeltMap outputs can contain several candidate confidence columns.  Selecting the
first column with any finite value can choose a sparsely populated diagnostic
field even when a later field covers every detection.  The resulting threshold
sweep then treats most detections as permanently unscored and cannot express the
available confidence ordering.
"""

from __future__ import annotations

from typing import Any, Iterable

from . import compare_runs as _compare_runs

_PATCHED_ATTR = "_beltmap_froc_score_field_coverage_patched"
_ORIGINAL_ATTR = "_beltmap_original_detection_score_field"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original selector behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_detection_score_field = _unwrap_patched_callable(
    _compare_runs.detection_score_field
)


def most_complete_detection_score_field(
    rows: Iterable[dict[str, Any]],
    *,
    score_fields: tuple[str, ...] = _compare_runs.FROC_SCORE_FIELDS,
) -> str | None:
    """Return the available score field with the greatest finite-row coverage.

    Candidate order remains the deterministic tie-breaker, preserving the
    historical preference for ``peak_signal`` when fields cover equally many
    detections.  Rows still missing the selected field are handled conservatively
    by :func:`beltmap.compare_runs.detection_froc_curve` and remain present at
    every threshold.
    """

    buffered_rows = list(rows)
    best_field: str | None = None
    best_finite_count = 0
    for field in score_fields:
        finite_count = sum(
            _compare_runs.finite_float(row.get(field)) is not None
            for row in buffered_rows
        )
        if finite_count > best_finite_count:
            best_field = field
            best_finite_count = finite_count
    return best_field


setattr(most_complete_detection_score_field, _PATCHED_ATTR, True)
setattr(
    most_complete_detection_score_field,
    _ORIGINAL_ATTR,
    _original_detection_score_field,
)
_compare_runs.detection_score_field = most_complete_detection_score_field

# Import for side effect: validate every comparison-report destination before
# it can overwrite truth labels, run artifacts, previews, or a sibling output.
from . import compare_report_path_collision_patch as _compare_report_path_collision_patch  # noqa: E402,F401

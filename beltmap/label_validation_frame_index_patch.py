"""Reject negative frame indices during truth-label validation.

BeltMap frame identifiers address a finite image sequence and must therefore be
non-negative.  The label validator historically accepted any finite integer,
including ``-1``, so a self-consistent reviewed file could be marked metric-ready
while referring to a frame that cannot exist in the source sequence.
"""

from __future__ import annotations

from typing import Any

from . import label_validation as _label_validation

_PATCHED_ATTR = "_beltmap_nonnegative_label_frame_indices_patched"
_ORIGINAL_ATTR = "_beltmap_original_label_validation_finite_int"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the integer parser behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_finite_int = _unwrap_patched_callable(_label_validation.finite_int)


def finite_nonnegative_int(value: Any) -> int | None:
    """Return a finite non-negative integer, otherwise ``None``."""

    parsed = _original_finite_int(value)
    if parsed is None or parsed < 0:
        return None
    return parsed


setattr(finite_nonnegative_int, _PATCHED_ATTR, True)
setattr(finite_nonnegative_int, _ORIGINAL_ATTR, _original_finite_int)
_label_validation.finite_int = finite_nonnegative_int

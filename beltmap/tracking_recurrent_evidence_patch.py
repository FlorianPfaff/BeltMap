"""Treat malformed recurrent-artifact track evidence as unavailable.

Track-level recurrent-artifact scoring consumes overlap fractions and probabilities
stored on ``ParticleDetection`` objects.  Those values are optional, but when
present they must be finite numbers in ``[0, 1]``.  The historical helper
silently clamped out-of-range values into that interval and attempted to coerce
booleans and arbitrary objects with ``float()``.  Corrupt metadata could
therefore become strong evidence or abort scoring instead of being ignored.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import tracking as _tracking

_PATCHED_ATTR = "_beltmap_recurrent_evidence_validation_patched"
_ORIGINAL_ATTR = "_beltmap_original_finite_unit_interval_value"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original parser behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_finite_unit_interval_value = _unwrap_patched_callable(
    _tracking._finite_unit_interval_value
)


def finite_unit_interval_or_none(value: Any) -> float | None:
    """Return a valid unit-interval value, otherwise treat it as unavailable."""

    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        return None
    return parsed


setattr(finite_unit_interval_or_none, _PATCHED_ATTR, True)
setattr(
    finite_unit_interval_or_none,
    _ORIGINAL_ATTR,
    _original_finite_unit_interval_value,
)
_tracking._finite_unit_interval_value = finite_unit_interval_or_none

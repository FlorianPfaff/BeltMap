"""Treat circular means with vanishing resultants as undefined.

An exactly balanced set of belt coordinates, such as two points separated by half
the belt period, has no mean direction. Floating-point roundoff leaves a tiny
nonzero complex resultant, so ``numpy.angle`` otherwise returns an arbitrary belt
coordinate. Recurrence scoring must treat that case as missing rather than using a
numerically unstable center.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import revolution_recurrence as _revolution_recurrence

_PATCHED_ATTR = "_beltmap_undefined_circular_mean_patched"
_ORIGINAL_ATTR = "_beltmap_original_circular_mean"
_RESULTANT_ATOL = 1e-12


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the circular-mean implementation behind this patch, if present."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_circular_mean = _unwrap_patched_callable(
    _revolution_recurrence.circular_mean
)


def stable_circular_mean(
    values: Sequence[float],
    period: float,
) -> float | None:
    """Return the circular mean, or ``None`` when its direction is undefined."""

    if not values:
        return None
    angles = 2.0 * np.pi * np.asarray(values, dtype=np.float64) / period
    vector = np.mean(np.exp(1j * angles))
    if abs(vector) <= _RESULTANT_ATOL:
        return None
    return float((np.angle(vector) % (2.0 * np.pi)) * period / (2.0 * np.pi))


setattr(stable_circular_mean, _PATCHED_ATTR, True)
setattr(stable_circular_mean, _ORIGINAL_ATTR, _original_circular_mean)
_revolution_recurrence.circular_mean = stable_circular_mean

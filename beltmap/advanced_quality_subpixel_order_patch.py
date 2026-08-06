"""Make quadratic subpixel refinement independent of offset input order.

``quadratic_subpixel_minimum`` fits the best sample and its two neighboring
samples.  Those neighbors are geometric neighbors on the offset axis, not the
entries adjacent to the best sample in an arbitrary caller-provided sequence.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import advanced_quality as _advanced_quality

_PATCHED_ATTR = "_beltmap_subpixel_offset_order_patched"
_ORIGINAL_ATTR = "_beltmap_original_quadratic_subpixel_minimum"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the estimator behind this compatibility patch, if present."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_quadratic_subpixel_minimum = _unwrap_patched_callable(
    _advanced_quality.quadratic_subpixel_minimum
)


def order_invariant_quadratic_subpixel_minimum(
    offsets: Sequence[float],
    losses: Sequence[float],
) -> float:
    """Fit the local quadratic after sorting finite samples by offset."""

    x = np.asarray(offsets, dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size == 0:
        return _original_quadratic_subpixel_minimum(offsets, losses)

    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return _original_quadratic_subpixel_minimum(offsets, losses)

    finite_x = x[finite]
    finite_y = y[finite]
    order = np.argsort(finite_x, kind="stable")
    return _original_quadratic_subpixel_minimum(
        finite_x[order],
        finite_y[order],
    )


setattr(order_invariant_quadratic_subpixel_minimum, _PATCHED_ATTR, True)
setattr(
    order_invariant_quadratic_subpixel_minimum,
    _ORIGINAL_ATTR,
    _original_quadratic_subpixel_minimum,
)
_advanced_quality.quadratic_subpixel_minimum = (
    order_invariant_quadratic_subpixel_minimum
)

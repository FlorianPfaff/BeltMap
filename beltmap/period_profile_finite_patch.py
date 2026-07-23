"""Reject non-finite samples before estimating belt periods.

The period estimator reports lags in pixels.  Removing ``NaN`` or infinite
profile entries before autocorrelation compresses the row coordinate and makes
those lags refer to a different signal geometry.  Fail explicitly instead of
silently changing the physical pixel spacing.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from . import operational_improvements as _operational

_PATCHED_ATTR = "_beltmap_finite_period_profile_patched"
_ORIGINAL_ATTR = "_beltmap_original_estimate_period_from_profile"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the estimator behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_estimate_period_from_profile = _unwrap_patched_callable(
    _operational.estimate_period_from_profile
)


@wraps(_original_estimate_period_from_profile)
def finite_estimate_period_from_profile(
    profile,
    *,
    min_period_px: int = 8,
    max_period_px: int | None = None,
    top_k: int = 5,
):
    """Estimate a period only when every profile position is finite."""

    values = np.asarray(profile, dtype=np.float64).ravel()
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "profile must contain only finite values; removing invalid samples "
            "would change pixel-lag spacing"
        )
    return _original_estimate_period_from_profile(
        profile,
        min_period_px=min_period_px,
        max_period_px=max_period_px,
        top_k=top_k,
    )


setattr(finite_estimate_period_from_profile, _PATCHED_ATTR, True)
setattr(
    finite_estimate_period_from_profile,
    _ORIGINAL_ATTR,
    _original_estimate_period_from_profile,
)
_operational.estimate_period_from_profile = finite_estimate_period_from_profile

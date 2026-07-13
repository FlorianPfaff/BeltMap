"""Preserve profile coordinates when estimating periods with missing rows.

The original estimator removed non-finite profile samples before computing its
lagged correlations.  Removing a missing row closes the spatial gap, so every
subsequent sample is compared at the wrong physical lag.  This patch keeps the
original profile axis and evaluates each candidate lag on the finite pairs that
actually overlap at that lag.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import operational_improvements as _operational

_PATCHED_ATTR = "_beltmap_period_estimation_finite_overlap_patched"
_ORIGINAL_ATTR = "_beltmap_original_estimate_period_from_profile"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the estimator behind this compatibility patch, if present."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_estimate_period_from_profile = _unwrap_patched_callable(
    _operational.estimate_period_from_profile
)


def finite_overlap_estimate_period_from_profile(
    profile,
    *,
    min_period_px: int = 8,
    max_period_px: int | None = None,
    top_k: int = 5,
):
    """Estimate profile periodicity without compressing non-finite samples.

    Candidate correlations retain the original pixel coordinates and use only
    pairs for which both samples are finite.  Unsupported lags are omitted rather
    than receiving a potentially misleading score.
    """

    values = np.asarray(profile, dtype=np.float64).ravel()
    min_period_px = _operational._positive_integer_value(
        min_period_px,
        "min_period_px",
    )
    top_k = _operational._positive_integer_value(top_k, "top_k")
    max_period_px = (
        None
        if max_period_px is None
        else _operational._positive_integer_value(max_period_px, "max_period_px")
    )

    if values.size < 2 * min_period_px:
        raise ValueError("profile is too short for the requested minimum period")
    max_period = (
        values.size // 2
        if max_period_px is None
        else min(int(max_period_px), values.size - 1)
    )
    if max_period < min_period_px:
        raise ValueError("invalid period search range")

    finite = np.isfinite(values)
    if int(np.count_nonzero(finite)) < 2:
        raise ValueError("profile has no variation")
    centered = values - float(np.mean(values[finite]))
    std = float(np.std(centered[finite]))
    if std <= 0:
        raise ValueError("profile has no variation")
    centered /= std

    candidates: list[tuple[int, float]] = []
    for period in range(int(min_period_px), int(max_period) + 1):
        finite_pairs = finite[:-period] & finite[period:]
        if int(np.count_nonzero(finite_pairs)) < 2:
            continue
        first = centered[:-period][finite_pairs]
        second = centered[period:][finite_pairs]
        denominator = float(
            np.sqrt(np.sum(first * first) * np.sum(second * second))
        )
        score = (
            0.0
            if denominator <= 0
            else float(np.sum(first * second) / denominator)
        )
        candidates.append((period, score))

    if not candidates:
        raise ValueError(
            "profile has insufficient finite overlap for the requested period range"
        )
    candidates.sort(key=lambda item: item[1], reverse=True)
    best_period, best_score = candidates[0]
    return _operational.PeriodEstimate(
        period_px=int(best_period),
        score=float(best_score),
        candidates=tuple(
            (int(period), float(score))
            for period, score in candidates[:top_k]
        ),
    )


setattr(finite_overlap_estimate_period_from_profile, _PATCHED_ATTR, True)
setattr(
    finite_overlap_estimate_period_from_profile,
    _ORIGINAL_ATTR,
    _original_estimate_period_from_profile,
)
_operational.estimate_period_from_profile = finite_overlap_estimate_period_from_profile

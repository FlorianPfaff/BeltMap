"""Apply the Benjamini-Hochberg step-up rejection cutoff correctly."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import operational_improvements as _operational

_PATCHED_ATTR = "_beltmap_benjamini_hochberg_step_up_patched"
_ORIGINAL_ATTR = "_beltmap_original_fdr_threshold_from_p_values"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original helper behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_fdr_threshold_from_p_values = _unwrap_patched_callable(
    _operational.fdr_threshold_from_p_values
)


def benjamini_hochberg_step_up_threshold(
    p_values,
    scores,
    *,
    alpha: float = 0.01,
) -> float | None:
    """Return the minimum score in the Benjamini-Hochberg rejection set.

    Benjamini-Hochberg is a step-up procedure: after finding the largest sorted
    rank ``k`` satisfying ``p_(k) <= alpha * k / m``, every hypothesis with a
    p-value at most ``p_(k)`` is rejected.  Testing each rank independently can
    omit earlier hypotheses when an early critical value fails but a later one
    passes.
    """

    p = np.asarray(p_values, dtype=np.float64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    valid = np.isfinite(p) & np.isfinite(s)
    p = p[valid]
    s = s[valid]
    if p.size == 0:
        return None

    order = np.argsort(p)
    sorted_p = p[order]
    critical_values = alpha * (np.arange(1, p.size + 1) / p.size)
    passing_ranks = np.flatnonzero(sorted_p <= critical_values)
    if passing_ranks.size == 0:
        return None

    p_cutoff = sorted_p[passing_ranks[-1]]
    accepted_scores = s[p <= p_cutoff]
    return float(np.min(accepted_scores))


setattr(benjamini_hochberg_step_up_threshold, _PATCHED_ATTR, True)
setattr(
    benjamini_hochberg_step_up_threshold,
    _ORIGINAL_ATTR,
    _original_fdr_threshold_from_p_values,
)
_operational.fdr_threshold_from_p_values = benjamini_hochberg_step_up_threshold

# Import for side effect: keep period-estimation lags tied to the original belt
# row coordinate instead of compressing away non-finite profile positions.
from . import period_profile_finite_patch as _period_profile_finite_patch  # noqa: E402,F401

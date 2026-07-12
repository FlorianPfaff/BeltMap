"""Keep robust photometric coefficients aligned with the retained pixels.

The original iterative fitter updates its retained-pixel mask after every fit.  If
that update happens on the final configured iteration, the returned coefficients
still belong to the previous mask even though ``n_pixels`` and ``rmse_gray`` are
computed from the newly trimmed mask.  In particular, ``max_iterations=1`` can
report that an outlier was trimmed while returning the fully outlier-biased fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import advanced_quality as _advanced_quality

_PATCHED_ATTR = "_beltmap_final_gain_offset_refit_patched"
_ORIGINAL_ATTR = "_beltmap_original_robust_gain_offset"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the fitter behind this compatibility patch, if present."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_robust_gain_offset = _unwrap_patched_callable(
    _advanced_quality.robust_gain_offset
)


def refitting_robust_gain_offset(
    observed,
    expected,
    *,
    mask=None,
    trim_fraction: float = 0.05,
    max_iterations: int = 3,
    min_pixels: int = 128,
):
    """Fit gain/offset after every accepted trimming update.

    The final least-squares coefficients, reported pixel count, and RMSE always
    refer to the same retained-pixel set.  A trimming update that would leave too
    few pixels or make the gain unidentifiable is ignored.
    """

    trim_fraction = _advanced_quality._finite_real(
        trim_fraction,
        "trim_fraction",
    )
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    max_iterations = _advanced_quality._integer_value(
        max_iterations,
        "max_iterations",
    )
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    min_pixels = _advanced_quality._integer_value(min_pixels, "min_pixels")
    if min_pixels < 1:
        raise ValueError("min_pixels must be positive")

    obs = _advanced_quality.as_float_image(observed, name="observed")
    exp = _advanced_quality.as_float_image(expected, name="expected")
    if obs.shape != exp.shape:
        raise ValueError("observed and expected must have the same shape")
    valid = _advanced_quality.finite_mask(obs, exp)
    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != obs.shape:
            raise ValueError("mask must have the same shape as observed")
        valid &= user_mask

    x = exp[valid].ravel()
    y = obs[valid].ravel()
    if x.size < min_pixels:
        raise ValueError(
            f"not enough valid pixels for photometric fit: {x.size} < {min_pixels}"
        )
    if np.unique(x).size < 2:
        raise ValueError("expected must contain at least two distinct finite values")

    def fit_retained(retained: np.ndarray) -> tuple[float, float]:
        kept_x = x[retained]
        kept_y = y[retained]
        design = np.column_stack([kept_x, np.ones(kept_x.size)])
        gain_value, offset_value = np.linalg.lstsq(
            design,
            kept_y,
            rcond=None,
        )[0]
        return float(gain_value), float(offset_value)

    keep = np.ones(x.size, dtype=bool)
    for _iteration in range(max_iterations):
        gain, offset = fit_retained(keep)
        if trim_fraction <= 0:
            break
        residual = y[keep] - (gain * x[keep] + offset)
        cutoff = np.quantile(np.abs(residual), 1.0 - trim_fraction)
        retained_within_current = np.abs(residual) <= cutoff
        new_keep = np.zeros_like(keep)
        new_keep[np.flatnonzero(keep)[retained_within_current]] = True
        if np.array_equal(new_keep, keep):
            break
        if int(np.count_nonzero(new_keep)) < min_pixels:
            break
        if np.unique(x[new_keep]).size < 2:
            break
        keep = new_keep

    # A mask update may have occurred on the final loop iteration.  Refit once on
    # that exact mask so the coefficients and diagnostics describe one data set.
    gain, offset = fit_retained(keep)
    fitted = gain * x[keep] + offset
    rmse = float(np.sqrt(np.mean(np.square(y[keep] - fitted))))
    return _advanced_quality.GainOffsetFit(
        gain=gain,
        offset=offset,
        n_pixels=int(np.count_nonzero(keep)),
        rmse_gray=rmse,
        trimmed_fraction=float(1.0 - np.count_nonzero(keep) / x.size),
    )


setattr(refitting_robust_gain_offset, _PATCHED_ATTR, True)
setattr(
    refitting_robust_gain_offset,
    _ORIGINAL_ATTR,
    _original_robust_gain_offset,
)
_advanced_quality.robust_gain_offset = refitting_robust_gain_offset

"""Prevent circular edge wrap in integer image-shift diagnostics.

The advanced-quality shift estimator models ordinary camera/crop translation, not
periodic image motion.  ``numpy.roll`` therefore must not let pixels leaving one
edge re-enter at the opposite edge when candidate shifts are scored.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import advanced_quality as _advanced_quality

_PATCHED_ATTR = "_beltmap_nonwrapping_integer_shift_patched"
_ORIGINAL_ATTR = "_beltmap_original_estimate_integer_xy_shift"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the estimator behind this compatibility patch, if present."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_estimate_integer_xy_shift = _unwrap_patched_callable(
    _advanced_quality.estimate_integer_xy_shift
)


def nonwrapping_estimate_integer_xy_shift(
    observed,
    expected,
    *,
    mask=None,
    max_shift_y_px: int = 4,
    max_shift_x_px: int = 4,
    trim_fraction: float = 0.08,
):
    """Estimate translation using only geometrically overlapping image support.

    Positive shifts move ``expected`` down/right into ``observed``.  Candidate
    losses exclude rows and columns that leave the image instead of circularly
    wrapping them to the opposite boundary.
    """

    max_shift_y_px = _advanced_quality._integer_value(
        max_shift_y_px,
        "max_shift_y_px",
    )
    max_shift_x_px = _advanced_quality._integer_value(
        max_shift_x_px,
        "max_shift_x_px",
    )
    trim_fraction = _advanced_quality._finite_real(trim_fraction, "trim_fraction")
    if not 0.0 <= trim_fraction < 1.0:
        raise ValueError("trim_fraction must be in [0, 1)")
    if max_shift_y_px < 0 or max_shift_x_px < 0:
        raise ValueError("max shifts must be non-negative")

    obs = _advanced_quality.as_float_image(observed, name="observed")
    exp = _advanced_quality.as_float_image(expected, name="expected")
    if obs.ndim != 2 or exp.ndim != 2:
        raise ValueError("observed and expected must be 2-D arrays")
    if obs.shape != exp.shape:
        raise ValueError("observed and expected must have the same shape")

    observed_valid = np.isfinite(obs)
    expected_valid = np.isfinite(exp)
    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != obs.shape:
            raise ValueError("mask must have the same shape as observed")
        observed_valid &= user_mask
        expected_valid &= user_mask

    height, width = obs.shape
    losses: list[tuple[float, int, int]] = []
    for dy in range(-max_shift_y_px, max_shift_y_px + 1):
        observed_y_start = max(0, dy)
        observed_y_stop = min(height, height + dy)
        expected_y_start = max(0, -dy)
        expected_y_stop = min(height, height - dy)

        for dx in range(-max_shift_x_px, max_shift_x_px + 1):
            observed_x_start = max(0, dx)
            observed_x_stop = min(width, width + dx)
            expected_x_start = max(0, -dx)
            expected_x_stop = min(width, width - dx)

            if (
                observed_y_start >= observed_y_stop
                or observed_x_start >= observed_x_stop
            ):
                loss = float("inf")
            else:
                observed_slice = (
                    slice(observed_y_start, observed_y_stop),
                    slice(observed_x_start, observed_x_stop),
                )
                expected_slice = (
                    slice(expected_y_start, expected_y_stop),
                    slice(expected_x_start, expected_x_stop),
                )
                overlap_valid = (
                    observed_valid[observed_slice]
                    & expected_valid[expected_slice]
                )
                residual = obs[observed_slice] - exp[expected_slice]
                loss = _advanced_quality._trimmed_loss(
                    residual[overlap_valid],
                    trim_fraction=trim_fraction,
                )
            losses.append((loss, dy, dx))

    finite_losses = [loss for loss, _dy, _dx in losses if np.isfinite(loss)]
    if not finite_losses:
        raise ValueError("no finite overlap for integer shift estimation")
    best_loss, best_y, best_x = min(losses, key=lambda item: item[0])
    median_loss = float(np.median(finite_losses))
    score = 0.0 if median_loss <= 0 else max(0.0, 1.0 - best_loss / median_loss)
    return _advanced_quality.ShiftEstimate(best_y, best_x, best_loss, score)


setattr(nonwrapping_estimate_integer_xy_shift, _PATCHED_ATTR, True)
setattr(
    nonwrapping_estimate_integer_xy_shift,
    _ORIGINAL_ATTR,
    _original_estimate_integer_xy_shift,
)
_advanced_quality.estimate_integer_xy_shift = nonwrapping_estimate_integer_xy_shift

# Import for side effect: sort finite offset samples before local subpixel fitting.
from . import advanced_quality_subpixel_order_patch as _advanced_quality_subpixel_order_patch  # noqa: E402,F401

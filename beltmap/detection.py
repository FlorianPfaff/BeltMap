"""Particle detection from normalized belt residual images."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .residual import ResidualImage


DETECTION_MODES = {"positive", "negative", "absolute"}
FloatArray = NDArray[np.floating]


def detection_signal_from_residual(
    residual: ArrayLike | ResidualImage,
    *,
    mode: str = "positive",
) -> FloatArray:
    """Return the oriented detection score implied by ``mode``.

    ``positive`` keeps bright residuals, ``negative`` flips dark residuals, and
    ``absolute`` scores both polarities by residual magnitude. Higher values are
    more particle-like for the selected polarity. Invalid pixels in
    ``ResidualImage.mask`` and non-finite residuals are returned as ``nan`` so
    downstream centroid/peak computations ignore them naturally.
    """

    values, valid = _residual_values_and_valid_mask(residual)
    signal = _oriented_detection_signal(values, mode=mode)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    result[valid] = signal[valid]
    return result


def detect_particles_from_residual(
    residual: ArrayLike | ResidualImage,
    *,
    threshold: float,
    mask: ArrayLike | None = None,
    mode: str = "positive",
    low_threshold: float | None = None,
) -> NDArray[np.bool_]:
    """Detect particles by thresholding an oriented normalized residual image.

    Parameters
    ----------
    residual:
        Normalized residual image or ``ResidualImage``.
    threshold:
        High threshold on the detection signal. For backward compatibility, the
        default ``mode='positive'`` is equivalent to ``residual > threshold``.
    mask:
        Optional boolean mask of pixels allowed to become detections.
    mode:
        ``positive`` detects bright particles, ``negative`` detects dark
        particles, and ``absolute`` detects either polarity.
    low_threshold:
        Optional lower hysteresis threshold. When set, pixels above
        ``low_threshold`` are retained only if they are 8-connected to a seed
        pixel above ``threshold``. This grows weak particle shoulders while
        suppressing isolated low-score noise.

    Invalid pixels, non-finite residuals, and pixels excluded by ``mask`` are
    returned as ``False``.
    """

    _validate_threshold("threshold", threshold)
    if low_threshold is not None:
        _validate_threshold("low_threshold", low_threshold)
        if low_threshold > threshold:
            raise ValueError("low_threshold must be less than or equal to threshold")

    values, valid = _residual_values_and_valid_mask(residual)
    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != values.shape:
            raise ValueError("mask must have the same shape as residual")
        valid &= user_mask

    signal = _oriented_detection_signal(values, mode=mode)
    seeds = valid & (signal > threshold)
    if low_threshold is None:
        return seeds

    candidates = valid & (signal > low_threshold)
    return _hysteresis_mask(seeds, candidates)


def _residual_values_and_valid_mask(
    residual: ArrayLike | ResidualImage,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    if isinstance(residual, ResidualImage):
        values = np.asarray(residual.normalized, dtype=np.float64)
        valid = np.asarray(residual.mask, dtype=bool).copy()
    else:
        values = np.asarray(residual, dtype=np.float64)
        if values.size == 0:
            raise ValueError("residual must not be empty")
        valid = np.ones(values.shape, dtype=bool)

    valid &= np.isfinite(values)
    return values, valid


def _oriented_detection_signal(values: FloatArray, *, mode: str) -> FloatArray:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "positive":
        return values
    if normalized_mode == "negative":
        return -values
    if normalized_mode == "absolute":
        return np.abs(values)
    choices = ", ".join(sorted(DETECTION_MODES))
    raise ValueError(f"mode must be one of {choices}")


def _validate_threshold(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _hysteresis_mask(
    seeds: NDArray[np.bool_],
    candidates: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    if candidates.ndim != 2:
        raise ValueError("hysteresis detection requires a 2-D residual")

    accepted = np.zeros(candidates.shape, dtype=bool)
    seed_rows, seed_cols = np.nonzero(seeds & candidates)
    if seed_rows.size == 0:
        return accepted

    stack = [(int(row), int(col)) for row, col in zip(seed_rows, seed_cols)]
    accepted[seed_rows, seed_cols] = True
    height, width = candidates.shape
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    while stack:
        row, col = stack.pop()
        for row_offset, col_offset in offsets:
            next_row = row + row_offset
            next_col = col + col_offset
            if (
                0 <= next_row < height
                and 0 <= next_col < width
                and candidates[next_row, next_col]
                and not accepted[next_row, next_col]
            ):
                accepted[next_row, next_col] = True
                stack.append((next_row, next_col))
    return accepted

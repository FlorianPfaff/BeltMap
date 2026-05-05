"""Particle detection from normalized belt residual images."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .residual import ResidualImage


def detect_particles_from_residual(
    residual: ArrayLike | ResidualImage,
    *,
    threshold: float,
    mask: ArrayLike | None = None,
) -> NDArray[np.bool_]:
    """Detect bright particles by thresholding a normalized residual image.

    For bright particles on a darker belt this is intentionally just:

    ``particle_mask = residual > threshold``

    Invalid pixels, non-finite residuals, and any pixels excluded by ``mask`` are
    returned as ``False``.
    """

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    if isinstance(residual, ResidualImage):
        values = np.asarray(residual.normalized, dtype=np.float64)
        valid = np.asarray(residual.mask, dtype=bool).copy()
    else:
        values = np.asarray(residual, dtype=np.float64)
        if values.size == 0:
            raise ValueError("residual must not be empty")
        valid = np.ones(values.shape, dtype=bool)

    valid &= np.isfinite(values)
    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != values.shape:
            raise ValueError("mask must have the same shape as residual")
        valid &= user_mask

    return valid & (values > threshold)

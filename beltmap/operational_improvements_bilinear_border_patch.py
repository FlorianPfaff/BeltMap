"""Preserve valid image-border samples in bilinear perspective warps.

The original sampler required both the floor and the next pixel index to lie
inside the image.  Samples exactly on the bottom or right border therefore used
the fill value even though the requested coordinate itself was valid.  Clamp
the interpolation neighbour at the border and reject only coordinates outside
the image support.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import operational_improvements as _operational

_PATCHED_ATTR = "_beltmap_bilinear_border_sampling_patched"
_ORIGINAL_ATTR = "_beltmap_original_sample_bilinear"
_BORDER_TOLERANCE_PX = 1e-9


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the sampler behind this compatibility patch, if already loaded."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_sample_bilinear = _unwrap_patched_callable(
    _operational._sample_bilinear
)


def border_safe_sample_bilinear(
    image: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    *,
    fill_value: float,
) -> np.ndarray:
    """Sample inclusively on valid borders without reading outside the image."""

    out = np.full(y.shape, fill_value, dtype=np.float64)
    if image.shape[0] == 0 or image.shape[1] == 0:
        return out

    max_y = float(image.shape[0] - 1)
    max_x = float(image.shape[1] - 1)
    finite = np.isfinite(y) & np.isfinite(x)
    valid = (
        finite
        & (y >= -_BORDER_TOLERANCE_PX)
        & (y <= max_y + _BORDER_TOLERANCE_PX)
        & (x >= -_BORDER_TOLERANCE_PX)
        & (x <= max_x + _BORDER_TOLERANCE_PX)
    )

    safe_y = np.where(valid, np.clip(y, 0.0, max_y), 0.0)
    safe_x = np.where(valid, np.clip(x, 0.0, max_x), 0.0)
    y0 = np.floor(safe_y).astype(np.int64)
    x0 = np.floor(safe_x).astype(np.int64)
    y1 = np.minimum(y0 + 1, image.shape[0] - 1)
    x1 = np.minimum(x0 + 1, image.shape[1] - 1)
    wy = safe_y - y0
    wx = safe_x - x0

    out[valid] = (
        (1 - wy[valid]) * (1 - wx[valid]) * image[y0[valid], x0[valid]]
        + (1 - wy[valid]) * wx[valid] * image[y0[valid], x1[valid]]
        + wy[valid] * (1 - wx[valid]) * image[y1[valid], x0[valid]]
        + wy[valid] * wx[valid] * image[y1[valid], x1[valid]]
    )
    return out


setattr(border_safe_sample_bilinear, _PATCHED_ATTR, True)
setattr(
    border_safe_sample_bilinear,
    _ORIGINAL_ATTR,
    _original_sample_bilinear,
)
_operational._sample_bilinear = border_safe_sample_bilinear

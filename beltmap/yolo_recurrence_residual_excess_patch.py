from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from beltmap import yolo_recurrence as _yolo_recurrence

_PATCHED_ATTR = "_beltmap_yolo_recurrence_residual_excess_patched"
_ORIGINAL_ATTR = "_beltmap_yolo_recurrence_original_patch_excess"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_patch_excess = _unwrap_patched_callable(_yolo_recurrence.patch_excess)


def residual_patch_excess(
    raw_patch: NDArray[np.floating],
    background_patch: NDArray[np.floating],
) -> float:
    """Return the maximum positive evidence after subtracting BeltMap background.

    The recurrence filter is meant to compare transient residual evidence at the
    same belt coordinate across revolutions.  The previous implementation used
    ``max(raw_patch) - percentile99(background_patch)``, which can score static
    belt texture as particle evidence when the raw patch and the rendered belt
    background share a bright mark.  Use the pixelwise residual instead so belt
    texture explained by the map contributes zero recurrence evidence.
    """

    raw = np.asarray(raw_patch, dtype=np.float64)
    background = np.asarray(background_patch, dtype=np.float64)
    if raw.shape != background.shape:
        raise ValueError(
            "raw_patch and background_patch must have the same shape for recurrence excess"
        )
    mask = np.isfinite(raw) & np.isfinite(background)
    if not mask.any():
        return 0.0
    residual = raw[mask] - background[mask]
    if residual.size == 0:
        return 0.0
    maximum = float(np.max(residual))
    if not np.isfinite(maximum):
        return 0.0
    return max(0.0, maximum)


setattr(residual_patch_excess, _PATCHED_ATTR, True)
setattr(residual_patch_excess, _ORIGINAL_ATTR, _original_patch_excess)
_yolo_recurrence.patch_excess = residual_patch_excess

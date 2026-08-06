"""Keep per-frame false-pixel budgets invariant to residual stack length."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from . import operational_improvements as _operational

_PATCHED_ATTR = "_beltmap_recommend_threshold_per_frame_patched"
_ORIGINAL_ATTR = "_beltmap_original_recommend_threshold"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original threshold helper if this patch is reloaded."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_recommend_threshold = _unwrap_patched_callable(
    _operational.recommend_threshold
)


def recommend_threshold_per_frame(
    residual: ArrayLike,
    *,
    expected_false_pixels_per_frame: float = 1.0,
    polarity: str = "bright",
    mask: ArrayLike | None = None,
) -> float:
    """Recommend a pooled threshold using a per-frame false-pixel budget.

    Residual stacks use their final two dimensions as image dimensions and all
    leading dimensions as frame dimensions.  The expected false-pixel count is
    therefore multiplied by the number of frames before converting it to a tail
    probability over the pooled valid residual values.  A 2-D residual remains
    a single-frame input, preserving the original behavior.
    """

    values = np.asarray(residual, dtype=np.float64)
    valid = np.isfinite(values)
    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != values.shape:
            raise ValueError("mask must have the same shape as residual")
        valid &= user_mask

    signal = _operational._polarity_signal(values, polarity)[valid]
    if signal.size == 0:
        raise ValueError("no valid residual values")

    frame_count = 1
    if values.ndim > 2:
        frame_count = int(np.prod(values.shape[:-2], dtype=np.int64))

    tail_probability = min(
        max(
            expected_false_pixels_per_frame * frame_count / signal.size,
            0.0,
        ),
        1.0,
    )
    quantile = 1.0 - tail_probability
    return float(np.quantile(signal, quantile))


setattr(recommend_threshold_per_frame, _PATCHED_ATTR, True)
setattr(
    recommend_threshold_per_frame,
    _ORIGINAL_ATTR,
    _original_recommend_threshold,
)
_operational.recommend_threshold = recommend_threshold_per_frame

# Import for side effect: keep comparison-report named preview discovery
# restricted to regular files before preview paths can reach Pillow.
from . import compare_named_preview_file_patch as _compare_named_preview_file_patch  # noqa: E402,F401

# Import for side effect: reject duplicate frame IDs and non-finite values in
# irregular-frame timestamp CSV files before they reach timing calculations.
from . import timestamp_csv_validation_patch as _timestamp_csv_validation_patch  # noqa: E402,F401

# Import for side effect: prevent evaluation reports from overwriting one another
# or the standard artifacts belonging to an evaluated BeltMap run.
from . import evaluation_path_collision_patch as _evaluation_path_collision_patch  # noqa: E402,F401

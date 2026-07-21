from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from beltmap import phase as _phase

_REFINE_ORIGINAL_ATTR = "_beltmap_phase_original_refine_quadratic_offset"
_SCORE_ORIGINAL_ATTR = "_beltmap_phase_original_loss_to_score"


def _unwrap_patched_callable(func: Any, original_attr: str) -> Any:
    """Return the original callable behind our wrapper, if already patched."""

    return getattr(func, original_attr, func)


_original_refine_quadratic_offset = _unwrap_patched_callable(
    _phase._refine_quadratic_offset,
    _REFINE_ORIGINAL_ATTR,
)
_original_loss_to_score = _unwrap_patched_callable(
    _phase._loss_to_score,
    _SCORE_ORIGINAL_ATTR,
)


def nonnegative_refine_quadratic_offset(
    losses: Sequence[tuple[float, float]],
    best_index: int,
) -> tuple[float, float]:
    """Refine a grid-search loss minimum without returning negative MSE values.

    The quadratic interpolant is only a local approximation to a mean-square loss.
    With asymmetric neighboring losses its vertex can dip slightly below zero, but
    downstream code treats the value as a physical loss for diagnostics and score
    conversion.  Clamp the approximation at zero instead of allowing impossible
    negative losses to produce scores larger than one.
    """

    refined_loss, refined_offset = _original_refine_quadratic_offset(losses, best_index)
    if np.isfinite(refined_loss) and refined_loss < 0.0:
        refined_loss = 0.0
    return float(refined_loss), float(refined_offset)


def bounded_loss_to_score(best_loss: float, all_losses: Iterable[float]) -> float:
    """Convert registration losses to a score in the documented [0, 1] range."""

    clipped_best_loss = float(best_loss)
    if np.isfinite(clipped_best_loss) and clipped_best_loss < 0.0:
        clipped_best_loss = 0.0
    score = _original_loss_to_score(clipped_best_loss, all_losses)
    if not np.isfinite(score):
        return 0.0
    return float(np.clip(score, 0.0, 1.0))


setattr(
    nonnegative_refine_quadratic_offset,
    _REFINE_ORIGINAL_ATTR,
    _original_refine_quadratic_offset,
)
setattr(
    bounded_loss_to_score,
    _SCORE_ORIGINAL_ATTR,
    _original_loss_to_score,
)

_phase._refine_quadratic_offset = nonnegative_refine_quadratic_offset
_phase._loss_to_score = bounded_loss_to_score

# Import for side effect: preserve valid bottom/right border pixels in bilinear
# perspective warps instead of replacing them with the configured fill value.
from . import (  # noqa: E402,F401
    operational_improvements_bilinear_border_patch as _operational_improvements_bilinear_border_patch,
)

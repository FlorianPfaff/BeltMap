"""Keep recurrent-artifact revolution indices monotonic across reference frames."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import recurrent_artifacts as _recurrent_artifacts
from .phase import BeltMotionModel

_PATCHED_ATTR = "_beltmap_elapsed_revolution_indices_patched"
_ORIGINAL_ATTR = "_beltmap_original_belt_revolution_indices"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original helper if this compatibility patch is reloaded."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_belt_revolution_indices = _unwrap_patched_callable(
    _recurrent_artifacts.belt_revolution_indices
)


def elapsed_belt_revolution_indices(
    frame_count: int,
    motion_model: BeltMotionModel,
) -> NDArray[np.integer]:
    """Return revolution bins from elapsed travel since the first processed frame.

    ``reference_frame`` defines the phase-model origin, not a turning point in belt
    motion. Measuring absolute distance to that origin creates a V-shaped sequence
    whenever it lies inside the processed interval and merges temporally distinct
    passes into the same recurrent-artifact revolution. Elapsed travel is monotonic
    and remains independent of the arbitrary model reference.
    """

    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    period_px = motion_model.period_px
    if period_px is None or not np.isfinite(period_px) or period_px <= 0:
        raise ValueError("motion_model period must be finite and positive")
    velocity = motion_model.image_velocity_px_per_frame
    if isinstance(velocity, (bool, np.bool_)):
        raise ValueError("motion_model velocity must be finite")
    try:
        velocity_value = float(velocity)
    except (TypeError, ValueError) as exc:
        raise ValueError("motion_model velocity must be finite") from exc
    if not np.isfinite(velocity_value):
        raise ValueError("motion_model velocity must be finite")

    frames = np.arange(frame_count, dtype=np.float64)
    displacement = abs(velocity_value) * frames
    return np.floor(displacement / float(period_px)).astype(np.int64)


setattr(elapsed_belt_revolution_indices, _PATCHED_ATTR, True)
setattr(
    elapsed_belt_revolution_indices,
    _ORIGINAL_ATTR,
    _original_belt_revolution_indices,
)
_recurrent_artifacts.belt_revolution_indices = elapsed_belt_revolution_indices

_package = sys.modules.get(__package__)
if _package is not None:
    setattr(
        _package,
        "belt_revolution_indices",
        elapsed_belt_revolution_indices,
    )

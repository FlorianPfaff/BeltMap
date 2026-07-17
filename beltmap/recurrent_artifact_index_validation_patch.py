"""Validate recurrent-artifact revolution-index inputs before NumPy conversion.

The legacy helper read motion-model fields directly instead of using
``BeltMotionModel.phase_at()``, which is where those fields are normally
validated.  Non-finite values could therefore reach ``np.floor(...).astype``
and become platform integer sentinels rather than producing a useful error.
"""

from __future__ import annotations

from numbers import Real
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import recurrent_artifacts as _recurrent_artifacts
from .phase import BeltMotionModel

_PATCHED_ATTR = "_beltmap_revolution_index_validation_patched"
_ORIGINAL_ATTR = "_beltmap_original_belt_revolution_indices"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the implementation behind this compatibility patch, if present."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_belt_revolution_indices = _unwrap_patched_callable(
    _recurrent_artifacts.belt_revolution_indices
)


def _finite_real(value: Any, name: str) -> float:
    """Return ``value`` as a finite float without accepting booleans."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _nonnegative_integer(value: Any, name: str) -> int:
    """Return a finite non-negative integral value."""

    parsed = _finite_real(value, name)
    if not parsed.is_integer() or parsed < 0:
        raise ValueError(f"{name} must be a finite non-negative integer")
    return int(parsed)


def validated_belt_revolution_indices(
    frame_count: int,
    motion_model: BeltMotionModel,
) -> NDArray[np.integer]:
    """Return revolution labels after validating all numeric model inputs."""

    normalized_frame_count = _nonnegative_integer(frame_count, "frame_count")
    _finite_real(
        motion_model.image_velocity_px_per_frame,
        "image_velocity_px_per_frame",
    )
    period_px = motion_model.period_px
    if period_px is None or _finite_real(period_px, "period_px") <= 0:
        raise ValueError("motion_model period must be finite and positive")
    _finite_real(motion_model.reference_frame, "reference_frame")
    return _original_belt_revolution_indices(normalized_frame_count, motion_model)


setattr(validated_belt_revolution_indices, _PATCHED_ATTR, True)
setattr(
    validated_belt_revolution_indices,
    _ORIGINAL_ATTR,
    _original_belt_revolution_indices,
)
_recurrent_artifacts.belt_revolution_indices = validated_belt_revolution_indices

# ``beltmap.__init__`` binds the public symbol before loading compatibility
# patches.  Keep that package-level alias synchronized with the patched module.
_package = sys.modules.get(__package__)
if _package is not None:
    setattr(_package, "belt_revolution_indices", validated_belt_revolution_indices)

"""Keep recurrent-artifact revolution labels independent of the phase anchor."""

from __future__ import annotations

import sys
from numbers import Integral, Real
from typing import Any

import numpy as np

from . import recurrent_artifacts as _recurrent

_PATCHED_ATTR = "_beltmap_revolution_reference_frame_patched"


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and numeric")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite and numeric")
    return parsed


def _frame_count(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("frame_count must be a finite non-negative integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError("frame_count must be a finite non-negative integer")
    return parsed


def reference_invariant_belt_revolution_indices(
    frame_count: int,
    motion_model: _recurrent.BeltMotionModel,
) -> np.ndarray:
    """Return chronological revolution labels starting at the first frame.

    ``reference_frame`` anchors the phase equation but must not change how many
    complete belt periods elapsed inside the processed sequence. Count traveled
    distance from selected frame zero so the labels remain monotonic and invariant
    to an equivalent change of phase anchor.
    """

    count = _frame_count(frame_count)
    velocity = _finite_real(
        motion_model.image_velocity_px_per_frame,
        "motion_model image velocity",
    )
    _finite_real(motion_model.reference_frame, "motion_model reference frame")
    if motion_model.period_px is None:
        raise ValueError("motion_model period must be finite and positive")
    period = _finite_real(motion_model.period_px, "motion_model period")
    if period <= 0:
        raise ValueError("motion_model period must be finite and positive")

    frames_from_sequence_start = np.arange(count, dtype=np.float64)
    displacement = abs(velocity) * frames_from_sequence_start
    return np.floor(displacement / period).astype(np.int64)


setattr(reference_invariant_belt_revolution_indices, _PATCHED_ATTR, True)
_recurrent.belt_revolution_indices = reference_invariant_belt_revolution_indices

_package = sys.modules.get(__package__)
if _package is not None:
    setattr(
        _package,
        "belt_revolution_indices",
        reference_invariant_belt_revolution_indices,
    )

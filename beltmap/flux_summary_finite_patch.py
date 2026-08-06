"""Reject non-finite metadata and derived values in particle-flux summaries."""

from __future__ import annotations

from functools import wraps
import math
from typing import Any, Mapping, Sequence

import numpy as np

from . import operational_improvements as _operational

_PATCHED_ATTR = "_beltmap_finite_flux_summary_patched"
_ORIGINAL_ATTR = "_beltmap_original_summarize_flux"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_summarize_flux = _unwrap_patched_callable(_operational.summarize_flux)


def _finite_optional_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _finite_frame_count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("frame_count must be a finite non-negative integer")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "frame_count must be a finite non-negative integer"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0.0 or not parsed.is_integer():
        raise ValueError("frame_count must be a finite non-negative integer")
    return int(parsed)


@wraps(_original_summarize_flux)
def summarize_flux_with_finite_outputs(
    velocity_rows: Sequence[Mapping[str, Any]],
    *,
    frame_count: int | None = None,
    frame_rate_hz: float | None = None,
    duration_s: float | None = None,
    belt_velocity_px_per_s: float | None = None,
    accepted_only: bool = False,
) -> _operational.FluxSummary:
    summary = _original_summarize_flux(
        velocity_rows,
        frame_count=_finite_frame_count(frame_count),
        frame_rate_hz=_finite_optional_float(
            frame_rate_hz,
            name="frame_rate_hz",
        ),
        duration_s=_finite_optional_float(duration_s, name="duration_s"),
        belt_velocity_px_per_s=_finite_optional_float(
            belt_velocity_px_per_s,
            name="belt_velocity_px_per_s",
        ),
        accepted_only=accepted_only,
    )
    for name in (
        "duration_s",
        "frame_rate_hz",
        "particle_flux_per_s",
        "median_velocity_ratio_y",
        "q25_velocity_ratio_y",
        "q75_velocity_ratio_y",
        "mean_velocity_y_px_per_s",
        "belt_velocity_px_per_s",
    ):
        value = getattr(summary, name)
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    return summary


setattr(summarize_flux_with_finite_outputs, _PATCHED_ATTR, True)
setattr(
    summarize_flux_with_finite_outputs,
    _ORIGINAL_ATTR,
    _original_summarize_flux,
)
_operational.summarize_flux = summarize_flux_with_finite_outputs

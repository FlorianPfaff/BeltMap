"""Keep flux velocity summaries honest about their time units."""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any, Mapping, Sequence

from . import operational_improvements as _operational

_PATCHED_ATTR = "_beltmap_flux_velocity_units_patched"
_ORIGINAL_ATTR = "_beltmap_flux_velocity_units_original"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_summarize_flux = _unwrap_patched_callable(_operational.summarize_flux)


@wraps(_original_summarize_flux)
def unit_safe_summarize_flux(
    velocity_rows: Sequence[Mapping[str, Any]],
    *,
    frame_count: int | None = None,
    frame_rate_hz: float | None = None,
    duration_s: float | None = None,
    belt_velocity_px_per_s: float | None = None,
    accepted_only: bool = False,
):
    parsed_rate = _operational._finite_float(frame_rate_hz)
    valid_rate = parsed_rate if parsed_rate is not None and parsed_rate > 0.0 else None
    summary = _original_summarize_flux(
        velocity_rows,
        frame_count=frame_count,
        frame_rate_hz=valid_rate,
        duration_s=duration_s,
        belt_velocity_px_per_s=belt_velocity_px_per_s,
        accepted_only=accepted_only,
    )
    if valid_rate is None and summary.mean_velocity_y_px_per_s is not None:
        summary = replace(summary, mean_velocity_y_px_per_s=None)
    return summary


setattr(unit_safe_summarize_flux, _PATCHED_ATTR, True)
setattr(unit_safe_summarize_flux, _ORIGINAL_ATTR, _original_summarize_flux)
_operational.summarize_flux = unit_safe_summarize_flux

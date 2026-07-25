"""Keep adaptive phase-coverage gains bounded by the number of bins.

A crop can span more than one complete belt-map period. The original adaptive
sampler converted that height to more phase bins than exist and then wrapped the
indices modulo the bin count. Repeated bin indices were counted repeatedly when
computing ``coverage_gain``, so a single frame could claim more newly covered bins
than the entire phase grid contains and dominate the selection objective.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Sequence

import numpy as np

from . import operational_improvements as _operational

_PATCHED_ATTR = "_beltmap_adaptive_sampling_coverage_patched"
_ORIGINAL_ATTR = "_beltmap_original_select_adaptive_map_frames"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original adaptive sampler behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_select_adaptive_map_frames = _unwrap_patched_callable(
    _operational.select_adaptive_map_frames
)


@wraps(_original_select_adaptive_map_frames)
def select_adaptive_map_frames_with_bounded_coverage(
    phases_px: Sequence[float],
    *,
    map_height_px: int,
    sample_count: int,
    crop_height_px: int = 1,
    quality_scores: Sequence[float] | None = None,
    bin_count: int | None = None,
) -> list[_operational.AdaptiveSample]:
    """Select frames without counting periodic phase bins more than once.

    Once a crop spans one complete map period, it already covers every phase bin.
    Additional wrapped periods cannot increase unique phase coverage, so cap the
    effective crop height at one map height before delegating to the sampler.
    """

    phases = np.asarray(phases_px, dtype=np.float64)
    if phases.size == 0 or not np.all(np.isfinite(phases)):
        return _original_select_adaptive_map_frames(
            phases_px,
            map_height_px=map_height_px,
            sample_count=sample_count,
            crop_height_px=crop_height_px,
            quality_scores=quality_scores,
            bin_count=bin_count,
        )

    validated_map_height = _operational._positive_integer_value(
        map_height_px,
        "map_height_px",
    )
    validated_sample_count = _operational._positive_integer_value(
        sample_count,
        "sample_count",
    )
    validated_crop_height = _operational._positive_integer_value(
        crop_height_px,
        "crop_height_px",
    )
    validated_bin_count = (
        None
        if bin_count is None
        else _operational._positive_integer_value(bin_count, "bin_count")
    )
    return _original_select_adaptive_map_frames(
        phases_px,
        map_height_px=validated_map_height,
        sample_count=validated_sample_count,
        crop_height_px=min(validated_crop_height, validated_map_height),
        quality_scores=quality_scores,
        bin_count=validated_bin_count,
    )


setattr(select_adaptive_map_frames_with_bounded_coverage, _PATCHED_ATTR, True)
setattr(
    select_adaptive_map_frames_with_bounded_coverage,
    _ORIGINAL_ATTR,
    _original_select_adaptive_map_frames,
)
_operational.select_adaptive_map_frames = (
    select_adaptive_map_frames_with_bounded_coverage
)

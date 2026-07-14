"""Keep finite-strip visual-QC coverage from wrapping across map boundaries.

The nominal coverage diagnostic historically applied modulo arithmetic to every
phase-shifted crop row.  That is correct for a known periodic belt map, but it
makes an inferred finite strip appear observed at the opposite edge whenever a
frame extends beyond reconstructed support.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import visual_qc as _visual_qc

_PATCHED_ATTR = "_beltmap_visual_qc_finite_strip_coverage_patched"
_ORIGINAL_ATTR = "_beltmap_original_estimate_belt_map_coverage"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original coverage estimator behind this patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_estimate_belt_map_coverage = _unwrap_patched_callable(
    _visual_qc.estimate_belt_map_coverage
)


def period_aware_estimate_belt_map_coverage(
    phase_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> np.ndarray | None:
    """Estimate coverage without wrapping inferred finite-strip support.

    Legacy metadata without an explicit ``belt_map_periodic`` field retains the
    historical cyclic behavior.  When the field is explicitly false, linear
    interpolation weight outside ``[0, belt_map_height_px)`` is discarded rather
    than reassigned to the opposite map boundary.
    """

    height = _visual_qc.finite_int(metadata.get("belt_map_height_px"))
    shape = _visual_qc.belt_region_shape(metadata)
    if height is None or height <= 0 or shape is None:
        return None

    crop_height, _crop_width = shape
    row_counts = np.zeros(height, dtype=np.float64)
    periodic = metadata.get("belt_map_periodic") is not False

    for row in phase_rows:
        phase = _visual_qc.finite_float(row.get("phase_px"))
        if phase is None:
            continue
        positions = np.arange(crop_height, dtype=np.float64) + phase
        y0 = np.floor(positions).astype(np.int64)
        frac = positions - y0
        y1 = y0 + 1

        if periodic:
            np.add.at(row_counts, y0 % height, 1.0 - frac)
            np.add.at(row_counts, y1 % height, frac)
            continue

        valid_y0 = (y0 >= 0) & (y0 < height)
        valid_y1 = (y1 >= 0) & (y1 < height)
        np.add.at(row_counts, y0[valid_y0], (1.0 - frac)[valid_y0])
        np.add.at(row_counts, y1[valid_y1], frac[valid_y1])

    return row_counts


setattr(period_aware_estimate_belt_map_coverage, _PATCHED_ATTR, True)
setattr(
    period_aware_estimate_belt_map_coverage,
    _ORIGINAL_ATTR,
    _original_estimate_belt_map_coverage,
)
_visual_qc.estimate_belt_map_coverage = period_aware_estimate_belt_map_coverage

"""Use translation-equivariant nearest-row projection for recurrent artifacts.

NumPy's ``rint`` rounds exact half-integers to the nearest even integer.  Applied
independently to consecutive image rows, that can collapse two rows onto one belt
row and skip the row between them.  Registration commonly produces half-pixel
phases, so recurrent-artifact masks and exposure maps need a tie rule that
preserves unit row spacing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import recurrent_artifacts as _recurrent_artifacts

_PATCHED_ATTR = "_beltmap_recurrent_artifact_rounding_patched"


def nearest_belt_rows_for_image_rows(
    image_rows: range,
    *,
    phase_px: float,
    map_height: int,
    image_height: int | None = None,
) -> NDArray[np.integer]:
    """Map image rows to nearest belt rows without round-to-even collisions."""

    rows = np.fromiter(image_rows, dtype=np.float64)
    if rows.size == 0:
        return np.array([], dtype=np.int64)
    if image_height is None:
        rows = rows[rows >= 0]
    else:
        rows = rows[(rows >= 0) & (rows < image_height)]
    if rows.size == 0:
        return np.array([], dtype=np.int64)

    # floor(x + 0.5) is translation-equivariant for integer row shifts.  Unlike
    # np.rint, it maps n + 0.5 to n + 1 for every integer n, so consecutive image
    # rows remain consecutive belt rows at half-pixel phases.
    belt_rows = np.floor(rows + phase_px + 0.5).astype(np.int64) % map_height
    return np.unique(belt_rows)


setattr(nearest_belt_rows_for_image_rows, _PATCHED_ATTR, True)
_recurrent_artifacts._belt_rows_for_image_rows = nearest_belt_rows_for_image_rows

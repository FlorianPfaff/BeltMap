"""Keep map-uncertainty coverage finite for inferred belt-map strips.

The post-run uncertainty helper historically treated every belt-map height as a
cyclic period. Modern driver metadata distinguishes a trusted physical period
from finite inferred support, so row-exposure counts must honor that state too.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import postrun_improvements as _postrun

_PATCHED_ATTR = "_beltmap_map_uncertainty_period_state_patched"
_ORIGINAL_ATTR = "_beltmap_map_uncertainty_period_state_original"
_PERIODIC_UNSET = object()
_PERIODIC_CONTEXT: ContextVar[object | bool] = ContextVar(
    "beltmap_map_uncertainty_periodic",
    default=_PERIODIC_UNSET,
)


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original callable behind this patch, if already installed."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_compute_phase_row_counts = _unwrap_patched_callable(
    _postrun.compute_phase_row_counts
)
_original_write_map_uncertainty_outputs = _unwrap_patched_callable(
    _postrun.write_map_uncertainty_outputs
)


def _metadata_is_periodic(metadata: Mapping[str, Any]) -> bool:
    """Resolve modern period-state metadata while preserving legacy behavior."""

    periodic = metadata.get("belt_map_periodic")
    if isinstance(periodic, (bool, np.bool_)):
        return bool(periodic)

    period_known = metadata.get("belt_period_known")
    if isinstance(period_known, (bool, np.bool_)):
        return bool(period_known)

    if "model_period_px" in metadata:
        return metadata.get("model_period_px") not in (None, "")

    # Old output directories predate explicit period-state metadata and were
    # historically interpreted periodically by this diagnostic.
    return True


def period_aware_compute_phase_row_counts(
    phases_px,
    *,
    map_height,
    crop_height,
    periodic: bool | None = None,
):
    """Count row exposure without wrapping finite inferred map support.

    Direct callers retain the historical periodic default. The output writer
    supplies the metadata-derived state through a context variable, and callers
    may also request finite-strip semantics explicitly with ``periodic=False``.
    """

    if periodic is None:
        context_periodic = _PERIODIC_CONTEXT.get()
        periodic_value = (
            True if context_periodic is _PERIODIC_UNSET else bool(context_periodic)
        )
    else:
        if not isinstance(periodic, (bool, np.bool_)):
            raise ValueError("periodic must be boolean when set")
        periodic_value = bool(periodic)

    if periodic_value:
        return _original_compute_phase_row_counts(
            phases_px,
            map_height=map_height,
            crop_height=crop_height,
        )

    map_height_value = _postrun._positive_integer_value(map_height, "map_height")
    crop_height_value = _postrun._positive_integer_value(crop_height, "crop_height")
    counts = np.zeros(map_height_value, dtype=np.uint64)
    image_rows = np.arange(crop_height_value, dtype=np.float64)
    for phase in phases_px:
        if not np.isfinite(phase):
            continue
        rows = np.floor(image_rows + float(phase)).astype(np.int64)
        valid = (rows >= 0) & (rows < map_height_value)
        if np.any(valid):
            counts += np.bincount(
                rows[valid],
                minlength=map_height_value,
            ).astype(np.uint64)
    return counts


def period_aware_write_map_uncertainty_outputs(*args, **kwargs):
    """Run uncertainty export with row wrapping selected from run metadata."""

    if "output_dir" in kwargs:
        output_dir = Path(kwargs["output_dir"])
    elif args:
        output_dir = Path(args[0])
    else:
        raise TypeError("write_map_uncertainty_outputs requires output_dir")

    periodic = _metadata_is_periodic(_postrun.load_metadata(output_dir))
    token = _PERIODIC_CONTEXT.set(periodic)
    try:
        return _original_write_map_uncertainty_outputs(*args, **kwargs)
    finally:
        _PERIODIC_CONTEXT.reset(token)


for patched, original in (
    (period_aware_compute_phase_row_counts, _original_compute_phase_row_counts),
    (
        period_aware_write_map_uncertainty_outputs,
        _original_write_map_uncertainty_outputs,
    ),
):
    setattr(patched, _PATCHED_ATTR, True)
    setattr(patched, _ORIGINAL_ATTR, original)

_postrun.compute_phase_row_counts = period_aware_compute_phase_row_counts
_postrun.write_map_uncertainty_outputs = period_aware_write_map_uncertainty_outputs

# Import for side effect: reject negative metadata counts before post-run quality
# diagnostics use them as physical event totals.
from . import postrun_count_metadata_patch as _postrun_count_metadata_patch  # noqa: E402,F401

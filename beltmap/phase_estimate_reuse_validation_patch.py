"""Reject non-finite numeric values in reused phase-estimate CSVs.

The reuse loader historically accepted ``NaN`` and infinite phase, correction,
drift, and registration-quality values because Python's ``float`` constructor
parses them and :class:`~beltmap.phase.PhaseEstimate` does not validate its
fields. Those values can then propagate into rendering, residual generation,
and diagnostics.
"""

from __future__ import annotations

import math
from functools import wraps
from pathlib import Path
from typing import Any

from . import driver as _driver

_PATCHED_ATTR = "_beltmap_finite_reused_phase_estimates_patched"
_ORIGINAL_ATTR = "_beltmap_original_load_phase_estimates"

_REQUIRED_FIELDS = {
    "phase_px": "phase_px",
    "predicted_phase_px": "predicted_phase_px",
    "correction_px": "correction_px",
    "phase_drift_px": "drift_px",
}
_OPTIONAL_FIELDS = {
    "loss": "loss",
    "score": "score",
    "second_best_loss": "second_best_loss",
    "loss_gap": "loss_gap",
    "loss_gap_ratio": "loss_gap_ratio",
    "loss_curvature": "loss_curvature",
    "uncertainty_px": "uncertainty_px",
}


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the loader behind this compatibility patch, if already installed."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_load_phase_estimates = _unwrap_patched_callable(
    _driver.load_phase_estimates
)


def _is_finite_number(value: Any) -> bool:
    """Return whether ``value`` represents a finite, non-boolean number."""

    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(parsed)


def _validate_finite_phase_estimates(
    path: Path,
    estimates: dict[int, Any],
) -> None:
    """Reject non-finite values before reused estimates reach the pipeline."""

    for frame_index, estimate in estimates.items():
        for csv_field, attribute in _REQUIRED_FIELDS.items():
            value = getattr(estimate, attribute)
            if not _is_finite_number(value):
                raise ValueError(
                    f"{path} contains non-finite {csv_field} "
                    f"for frame {frame_index}"
                )
        for csv_field, attribute in _OPTIONAL_FIELDS.items():
            value = getattr(estimate, attribute)
            if value is not None and not _is_finite_number(value):
                raise ValueError(
                    f"{path} contains non-finite {csv_field} "
                    f"for frame {frame_index}"
                )


@wraps(_original_load_phase_estimates)
def finite_load_phase_estimates(
    path: Path,
    *,
    expected_image_paths: list[Path] | None = None,
    data_dir: Path | None = None,
):
    """Load reused phases and require every populated numeric field to be finite."""

    estimates = _original_load_phase_estimates(
        path,
        expected_image_paths=expected_image_paths,
        data_dir=data_dir,
    )
    _validate_finite_phase_estimates(Path(path), estimates)
    return estimates


setattr(finite_load_phase_estimates, _PATCHED_ATTR, True)
setattr(
    finite_load_phase_estimates,
    _ORIGINAL_ATTR,
    _original_load_phase_estimates,
)
_driver.load_phase_estimates = finite_load_phase_estimates

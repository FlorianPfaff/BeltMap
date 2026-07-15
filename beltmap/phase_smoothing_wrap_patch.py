"""Unwrap periodic phase corrections before trajectory smoothing.

Phase corrections are represented canonically in ``[-period / 2, period / 2)``.
A smooth trajectory crossing that boundary therefore appears to jump by almost a
full period.  Local linear smoothing must operate on an unwrapped trajectory and
only canonicalize the correction again when producing each output estimate.
"""

from __future__ import annotations

import sys
from typing import Any, Sequence

import numpy as np

from . import phase as _phase

_PATCHED_ATTR = "_beltmap_wrap_aware_phase_smoothing_patched"
_ORIGINAL_ATTR = "_beltmap_original_smooth_phase_estimates"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the smoother behind this compatibility patch, if present."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_smooth_phase_estimates = _unwrap_patched_callable(
    _phase.smooth_phase_estimates
)


def _unwrap_valid_corrections(
    frame_indices: _phase.FloatArray,
    corrections: _phase.FloatArray,
    valid: np.ndarray,
    period_px: float,
) -> _phase.FloatArray:
    """Unwrap valid corrections in chronological frame order."""

    unwrapped = corrections.copy()
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 2:
        return unwrapped

    order = valid_indices[
        np.argsort(frame_indices[valid_indices], kind="stable")
    ]
    radians = corrections[order] * (2.0 * np.pi / period_px)
    unwrapped[order] = np.unwrap(radians) * (period_px / (2.0 * np.pi))
    return unwrapped


def wrap_aware_smooth_phase_estimates(
    estimates: Sequence[_phase.PhaseEstimate],
    *,
    period_px: float | None = None,
    config: _phase.PhaseTrajectorySmoothingConfig | None = None,
) -> list[_phase.PhaseEstimate]:
    """Smooth periodic corrections without fitting across the wrap discontinuity."""

    if period_px is None:
        return _original_smooth_phase_estimates(
            estimates,
            period_px=None,
            config=config,
        )

    cfg = (config or _phase.PhaseTrajectorySmoothingConfig()).normalized()
    if not estimates:
        return []
    if cfg.window_radius_frames == 0:
        return list(estimates)

    period = _phase._finite_float_value(period_px, "period_px")
    if period <= 0:
        raise ValueError("period_px must be positive")

    frame_indices = np.asarray(
        [estimate.frame_index for estimate in estimates],
        dtype=np.float64,
    )
    corrections = np.asarray(
        [
            _phase._phase_correction_from_estimate(estimate, period)
            for estimate in estimates
        ],
        dtype=np.float64,
    )
    scores = np.asarray(
        [np.nan if estimate.score is None else estimate.score for estimate in estimates],
        dtype=np.float64,
    )

    valid = np.isfinite(frame_indices) & np.isfinite(corrections)
    if cfg.min_score is not None:
        valid &= np.isfinite(scores) & (scores >= cfg.min_score)
    if cfg.max_abs_correction_px is not None:
        valid &= np.abs(corrections) <= cfg.max_abs_correction_px
    if not np.any(valid):
        return list(estimates)

    unwrapped_corrections = _unwrap_valid_corrections(
        frame_indices,
        corrections,
        valid,
        period,
    )
    weights = np.ones(len(estimates), dtype=np.float64)
    scored = np.isfinite(scores)
    weights[scored] = np.maximum(scores[scored], 1e-6)
    weights[~valid] = 0.0

    smoothed: list[_phase.PhaseEstimate] = []
    smoothed_unwrapped = _phase._smooth_corrections(
        frame_indices,
        unwrapped_corrections,
        weights,
        valid,
        cfg,
    )
    for estimate, correction in zip(estimates, smoothed_unwrapped, strict=True):
        if np.isfinite(correction):
            canonical_correction = _phase._cyclic_difference(
                float(correction),
                0.0,
                period,
            )
        else:
            canonical_correction = _phase._phase_correction_from_estimate(
                estimate,
                period,
            )
        phase_px = _phase.wrap_phase(
            estimate.predicted_phase_px + canonical_correction,
            period,
        )
        smoothed.append(
            _phase.PhaseEstimate(
                phase_px=phase_px,
                frame_index=estimate.frame_index,
                predicted_phase_px=estimate.predicted_phase_px,
                correction_px=canonical_correction,
                loss=estimate.loss,
                score=estimate.score,
                second_best_loss=estimate.second_best_loss,
                loss_gap=estimate.loss_gap,
                loss_gap_ratio=estimate.loss_gap_ratio,
                loss_curvature=estimate.loss_curvature,
                uncertainty_px=estimate.uncertainty_px,
                method=_phase._smoothed_method_name(estimate.method),
                drift_px=estimate.drift_px,
            )
        )
    return smoothed


setattr(wrap_aware_smooth_phase_estimates, _PATCHED_ATTR, True)
setattr(
    wrap_aware_smooth_phase_estimates,
    _ORIGINAL_ATTR,
    _original_smooth_phase_estimates,
)
_phase.smooth_phase_estimates = wrap_aware_smooth_phase_estimates

_package = sys.modules.get(__package__)
if _package is not None:
    setattr(_package, "smooth_phase_estimates", wrap_aware_smooth_phase_estimates)

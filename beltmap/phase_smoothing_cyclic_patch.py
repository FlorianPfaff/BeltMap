"""Make periodic phase smoothing robust to the cyclic branch cut."""

from __future__ import annotations

import contextvars
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

from . import phase as _phase

_SMOOTH_ORIGINAL_ATTR = "_beltmap_phase_original_smooth_phase_estimates"
_SMOOTH_CORRECTIONS_ORIGINAL_ATTR = "_beltmap_phase_original_smooth_corrections"
_ACTIVE_SMOOTHING_PERIOD: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "beltmap_active_phase_smoothing_period",
    default=None,
)


def _unwrap_patched_callable(func: Any, original_attr: str) -> Any:
    """Return the original callable behind our wrapper, if already patched."""

    return getattr(func, original_attr, func)


_original_smooth_phase_estimates = _unwrap_patched_callable(
    _phase.smooth_phase_estimates,
    _SMOOTH_ORIGINAL_ATTR,
)
_original_smooth_corrections = _unwrap_patched_callable(
    _phase._smooth_corrections,
    _SMOOTH_CORRECTIONS_ORIGINAL_ATTR,
)


def _unwrap_cyclic_values(
    values: _phase.FloatArray,
    weights: _phase.FloatArray,
    *,
    period_px: float,
) -> _phase.FloatArray:
    """Place cyclic corrections on one local branch before linear smoothing."""

    angles = (2.0 * np.pi / period_px) * values
    weighted_cos = float(np.sum(weights * np.cos(angles)))
    weighted_sin = float(np.sum(weights * np.sin(angles)))
    weight_sum = float(np.sum(weights))
    resultant = float(np.hypot(weighted_cos, weighted_sin))
    if resultant <= np.finfo(np.float64).eps * max(1.0, weight_sum):
        reference = float(values[int(np.argmax(weights))])
    else:
        reference = float(
            np.arctan2(weighted_sin, weighted_cos)
            * period_px
            / (2.0 * np.pi)
        )

    half_period = 0.5 * period_px
    return reference + (values - reference + half_period) % period_px - half_period


def cyclic_safe_smooth_corrections(
    frame_indices: _phase.FloatArray,
    corrections: _phase.FloatArray,
    weights: _phase.FloatArray,
    valid: np.ndarray,
    config: _phase.PhaseTrajectorySmoothingConfig,
) -> _phase.FloatArray:
    """Smooth periodic corrections without averaging across the branch cut."""

    active_period = _ACTIVE_SMOOTHING_PERIOD.get()
    if active_period is None:
        return _original_smooth_corrections(
            frame_indices,
            corrections,
            weights,
            valid,
            config,
        )

    period_px = _phase._finite_float_value(active_period, "period_px")
    if period_px <= 0:
        raise ValueError("period_px must be positive")

    result = np.full(corrections.shape, np.nan, dtype=np.float64)
    valid_indices = np.flatnonzero(valid)
    for output_index, center_frame in enumerate(frame_indices):
        in_window = valid & (
            np.abs(frame_indices - center_frame) <= config.window_radius_frames
        )
        selected = np.flatnonzero(in_window)
        if selected.size < min(config.min_support, valid_indices.size):
            order = np.argsort(np.abs(frame_indices[valid_indices] - center_frame))
            selected = valid_indices[
                order[: min(config.min_support, valid_indices.size)]
            ]
        if selected.size == 0:
            continue
        local_kernel = np.maximum(
            1e-6,
            1.0
            - np.abs(frame_indices[selected] - center_frame)
            / (config.window_radius_frames + 1.0),
        )
        local_weights = weights[selected] * local_kernel
        local_corrections = _unwrap_cyclic_values(
            corrections[selected],
            local_weights,
            period_px=period_px,
        )
        prediction = _phase._robust_local_linear_prediction(
            frame_indices[selected],
            local_corrections,
            local_weights,
            center_frame,
            robust_sigma=config.robust_sigma,
            min_support=min(config.min_support, selected.size),
        )
        if np.isfinite(prediction):
            prediction = _phase._cyclic_difference(prediction, 0.0, period_px)
        result[output_index] = prediction
    return result


def cyclic_safe_smooth_phase_estimates(
    estimates: Sequence[_phase.PhaseEstimate],
    *,
    period_px: float | None = None,
    config: _phase.PhaseTrajectorySmoothingConfig | None = None,
) -> list[_phase.PhaseEstimate]:
    """Smooth periodic corrections on a locally unwrapped branch."""

    token = _ACTIVE_SMOOTHING_PERIOD.set(period_px)
    try:
        return _original_smooth_phase_estimates(
            estimates,
            period_px=period_px,
            config=config,
        )
    finally:
        _ACTIVE_SMOOTHING_PERIOD.reset(token)


setattr(
    cyclic_safe_smooth_phase_estimates,
    _SMOOTH_ORIGINAL_ATTR,
    _original_smooth_phase_estimates,
)
setattr(
    cyclic_safe_smooth_corrections,
    _SMOOTH_CORRECTIONS_ORIGINAL_ATTR,
    _original_smooth_corrections,
)

_phase._smooth_corrections = cyclic_safe_smooth_corrections
_phase.smooth_phase_estimates = cyclic_safe_smooth_phase_estimates

_package = sys.modules.get(__package__)
if _package is not None:
    setattr(_package, "smooth_phase_estimates", cyclic_safe_smooth_phase_estimates)

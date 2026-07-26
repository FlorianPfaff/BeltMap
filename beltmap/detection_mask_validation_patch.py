"""Reject ambiguous non-boolean masks before BeltMap image analysis."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from . import advanced_quality_shift_patch as _advanced_quality_shift_patch  # noqa: F401
from . import advanced_quality as _advanced_quality
from . import detection as _detection
from . import tracking as _tracking
from .residual import ResidualImage

_PATCHED_ATTR = "_beltmap_detection_mask_validation_patched"
_ORIGINAL_DETECT_ATTR = "_beltmap_original_detect_particles_from_residual"
_ORIGINAL_VALUES_ATTR = "_beltmap_original_residual_values_and_valid_mask"
_ORIGINAL_EXTRACT_ATTR = "_beltmap_original_extract_particle_detections"
_ORIGINAL_GAIN_OFFSET_ATTR = "_beltmap_original_robust_gain_offset"
_ORIGINAL_SHIFT_ATTR = "_beltmap_original_estimate_integer_xy_shift_before_mask_validation"
_NONWRAPPING_SHIFT_PATCHED_ATTR = "_beltmap_nonwrapping_integer_shift_patched"
_NONWRAPPING_SHIFT_ORIGINAL_ATTR = "_beltmap_original_estimate_integer_xy_shift"


def _unwrap_patched_callable(func: Any, original_attr: str) -> Any:
    """Return the original callable behind a previous patch reload."""

    return getattr(func, original_attr, func)


_original_detect_particles_from_residual = _unwrap_patched_callable(
    _detection.detect_particles_from_residual,
    _ORIGINAL_DETECT_ATTR,
)
_original_residual_values_and_valid_mask = _unwrap_patched_callable(
    _detection._residual_values_and_valid_mask,
    _ORIGINAL_VALUES_ATTR,
)
_original_extract_particle_detections = _unwrap_patched_callable(
    _tracking.extract_particle_detections,
    _ORIGINAL_EXTRACT_ATTR,
)
_original_robust_gain_offset = _unwrap_patched_callable(
    _advanced_quality.robust_gain_offset,
    _ORIGINAL_GAIN_OFFSET_ATTR,
)
_original_estimate_integer_xy_shift = _unwrap_patched_callable(
    _advanced_quality.estimate_integer_xy_shift,
    _ORIGINAL_SHIFT_ATTR,
)


def _validate_binary_values(mask: ArrayLike, *, name: str) -> np.ndarray:
    """Return a mask array after rejecting ambiguous non-binary values."""

    raw = np.asarray(mask)
    if raw.dtype == np.bool_:
        return raw
    try:
        numeric = np.issubdtype(raw.dtype, np.number)
    except TypeError:
        numeric = False
    if not numeric:
        raise ValueError(f"{name} must be a boolean or binary 0/1 array")
    if not np.all(np.isfinite(raw)) or not np.all((raw == 0) | (raw == 1)):
        raise ValueError(f"{name} must be a boolean or binary 0/1 array")
    return raw


def _validate_binary_mask(
    mask: ArrayLike,
    *,
    expected_shape: tuple[int, ...],
    name: str,
    shape_error: str,
) -> np.ndarray:
    """Accept boolean or legacy binary masks, but reject ambiguous values."""

    raw = _validate_binary_values(mask, name=name)
    if raw.shape != expected_shape:
        raise ValueError(shape_error)
    return raw


def validating_residual_values_and_valid_mask(
    residual: ArrayLike | ResidualImage,
):
    """Validate ``ResidualImage.mask`` before the detector coerces its dtype."""

    if isinstance(residual, ResidualImage):
        normalized_shape = np.asarray(residual.normalized).shape
        _validate_binary_mask(
            residual.mask,
            expected_shape=normalized_shape,
            name="ResidualImage mask",
            shape_error="ResidualImage mask must have the same shape as normalized",
        )
    return _original_residual_values_and_valid_mask(residual)


def validating_detect_particles_from_residual(
    residual: ArrayLike | ResidualImage,
    *,
    threshold: float,
    mask: ArrayLike | None = None,
    mode: str = "positive",
    low_threshold: float | None = None,
):
    """Reject invalid optional masks instead of treating non-zero values as true."""

    if mask is not None:
        residual_values = residual.normalized if isinstance(residual, ResidualImage) else residual
        mask = _validate_binary_mask(
            mask,
            expected_shape=np.asarray(residual_values).shape,
            name="mask",
            shape_error="mask must have the same shape as residual",
        )
    return _original_detect_particles_from_residual(
        residual,
        threshold=threshold,
        mask=mask,
        mode=mode,
        low_threshold=low_threshold,
    )


def validating_extract_particle_detections(
    particle_mask: ArrayLike,
    *,
    residual: ArrayLike | ResidualImage | None = None,
    frame_index: float = 0.0,
    config: _tracking.ParticleComponentConfig | None = None,
    signal_mode: str | None = None,
):
    """Reject invalid component masks before NumPy turns nonzero values true."""

    particle_mask = _validate_binary_values(particle_mask, name="particle_mask")
    return _original_extract_particle_detections(
        particle_mask,
        residual=residual,
        frame_index=frame_index,
        config=config,
        signal_mode=signal_mode,
    )


def validating_robust_gain_offset(
    observed: ArrayLike,
    expected: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    trim_fraction: float = 0.05,
    max_iterations: int = 3,
    min_pixels: int = 128,
):
    """Reject ambiguous photometric-fit masks before boolean coercion."""

    if mask is not None:
        mask = _validate_binary_mask(
            mask,
            expected_shape=np.asarray(observed).shape,
            name="mask",
            shape_error="mask must have the same shape as observed",
        )
    return _original_robust_gain_offset(
        observed,
        expected,
        mask=mask,
        trim_fraction=trim_fraction,
        max_iterations=max_iterations,
        min_pixels=min_pixels,
    )


def validating_estimate_integer_xy_shift(
    observed: ArrayLike,
    expected: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    max_shift_y_px: int = 4,
    max_shift_x_px: int = 4,
    trim_fraction: float = 0.08,
):
    """Reject ambiguous registration masks before boolean coercion."""

    if mask is not None:
        mask = _validate_binary_mask(
            mask,
            expected_shape=np.asarray(observed).shape,
            name="mask",
            shape_error="mask must have the same shape as observed",
        )
    return _original_estimate_integer_xy_shift(
        observed,
        expected,
        mask=mask,
        max_shift_y_px=max_shift_y_px,
        max_shift_x_px=max_shift_x_px,
        trim_fraction=trim_fraction,
    )


def _mark_wrapper(func: Any, original_attr: str, original: Any) -> None:
    setattr(func, _PATCHED_ATTR, True)
    setattr(func, original_attr, original)


_mark_wrapper(
    validating_residual_values_and_valid_mask,
    _ORIGINAL_VALUES_ATTR,
    _original_residual_values_and_valid_mask,
)
_mark_wrapper(
    validating_detect_particles_from_residual,
    _ORIGINAL_DETECT_ATTR,
    _original_detect_particles_from_residual,
)
_mark_wrapper(
    validating_extract_particle_detections,
    _ORIGINAL_EXTRACT_ATTR,
    _original_extract_particle_detections,
)
_mark_wrapper(
    validating_robust_gain_offset,
    _ORIGINAL_GAIN_OFFSET_ATTR,
    _original_robust_gain_offset,
)
_mark_wrapper(
    validating_estimate_integer_xy_shift,
    _ORIGINAL_SHIFT_ATTR,
    _original_estimate_integer_xy_shift,
)
for _attribute in (
    _NONWRAPPING_SHIFT_PATCHED_ATTR,
    _NONWRAPPING_SHIFT_ORIGINAL_ATTR,
):
    if hasattr(_original_estimate_integer_xy_shift, _attribute):
        setattr(
            validating_estimate_integer_xy_shift,
            _attribute,
            getattr(_original_estimate_integer_xy_shift, _attribute),
        )

_detection._residual_values_and_valid_mask = validating_residual_values_and_valid_mask
_detection.detect_particles_from_residual = validating_detect_particles_from_residual
_tracking.extract_particle_detections = validating_extract_particle_detections
_advanced_quality.robust_gain_offset = validating_robust_gain_offset
_advanced_quality.estimate_integer_xy_shift = validating_estimate_integer_xy_shift

_package = sys.modules.get(__package__)
if _package is not None:
    setattr(
        _package,
        "detect_particles_from_residual",
        validating_detect_particles_from_residual,
    )
    setattr(
        _package,
        "extract_particle_detections",
        validating_extract_particle_detections,
    )

"""Residual images for particle localization on a reconstructed belt."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .phase import BeltMotionModel, PhaseEstimate, PhaseRegistrationConfig
from .rendering import BeltRegion, CleanBeltRender, render_expected_clean_belt

FloatArray = NDArray[np.floating]
NOISE_EXCLUSION_MODES = {"positive", "negative", "absolute"}


@dataclass(frozen=True)
class ResidualConfig:
    """Settings for local residual normalization."""

    noise_radius_px: int = 15
    clip_sigma: float | None = 5.0
    noise_exclusion_sigma: float | None = 4.0
    noise_exclusion_radius_px: int = 2
    min_noise: float = 1e-6
    noise_exclusion_mode: str = "positive"
    fill_value: float = np.nan


@dataclass(frozen=True)
class ResidualImage:
    """Raw and locally normalized residuals for one frame."""

    raw: FloatArray
    local_noise: FloatArray
    normalized: FloatArray
    mask: NDArray[np.bool_]
    expected_background: FloatArray
    clean_render: CleanBeltRender | None = None


def generate_residual_image(
    image: ArrayLike,
    expected_background: ArrayLike | CleanBeltRender,
    *,
    mask: ArrayLike | None = None,
    config: ResidualConfig | None = None,
) -> ResidualImage:
    """Return ``(image - expected_background) / local_noise``."""

    cfg = config or ResidualConfig()
    fill_value = _finite_or_nan_config_value(cfg.fill_value, "fill_value")
    observed = _as_float_image(image, name="image")
    clean_render = (
        expected_background
        if isinstance(expected_background, CleanBeltRender)
        else None
    )
    expected = (
        clean_render.image
        if clean_render is not None
        else _as_float_image(expected_background, name="expected_background")
    )
    if observed.shape != expected.shape:
        raise ValueError("image and expected_background must have the same shape")
    if clean_render is not None and clean_render.mask.shape != observed.shape:
        raise ValueError("CleanBeltRender mask must have the same shape as image")

    valid = np.isfinite(observed) & np.isfinite(expected)
    if clean_render is not None:
        valid &= clean_render.mask
    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != observed.shape:
            raise ValueError("mask must have the same shape as image")
        valid &= user_mask

    raw_values = observed - expected
    local_noise = estimate_local_noise(raw_values, mask=valid, config=cfg)
    raw = np.full(observed.shape, fill_value, dtype=np.float64)
    normalized = np.full(observed.shape, fill_value, dtype=np.float64)
    raw[valid] = raw_values[valid]
    normalized[valid] = raw_values[valid] / local_noise[valid]
    return ResidualImage(
        raw=raw,
        local_noise=local_noise,
        normalized=normalized,
        mask=valid,
        expected_background=expected,
        clean_render=clean_render,
    )


def render_clean_belt_residual(
    *,
    image: ArrayLike,
    belt_map: ArrayLike,
    frame_index: float,
    motion_model: BeltMotionModel | None = None,
    belt_region: BeltRegion | tuple[int, int, int, int] | None = None,
    phase_estimate: PhaseEstimate | None = None,
    registration_config: PhaseRegistrationConfig | None = None,
    registration_mask: ArrayLike | None = None,
    residual_mask: ArrayLike | None = None,
    residual_config: ResidualConfig | None = None,
    periodic: bool | None = None,
) -> ResidualImage:
    """Render the clean belt for ``image`` and return its normalized residual."""

    observed = _as_float_image(image, name="image")
    clean = render_expected_clean_belt(
        belt_map=belt_map,
        frame_index=frame_index,
        motion_model=motion_model,
        observed_frame=observed,
        belt_region=belt_region,
        output_shape=observed.shape,
        phase_estimate=phase_estimate,
        registration_config=registration_config,
        registration_mask=registration_mask,
        periodic=periodic,
    )
    residual_mask = _expand_mask_to_image(
        residual_mask,
        image_shape=observed.shape,
        belt_region=clean.belt_region,
    )
    return generate_residual_image(
        observed,
        clean,
        mask=residual_mask,
        config=residual_config,
    )


def estimate_local_noise(
    residual: ArrayLike,
    *,
    mask: ArrayLike | None = None,
    config: ResidualConfig | None = None,
) -> FloatArray:
    """Estimate robust local noise scale for a residual image."""

    cfg = config or ResidualConfig()
    noise_radius_px = _nonnegative_integer_config_value(
        cfg.noise_radius_px,
        "noise_radius_px",
    )
    min_noise = _positive_config_value(cfg.min_noise, "min_noise")
    clip_sigma = _optional_positive_config_value(cfg.clip_sigma, "clip_sigma")
    noise_exclusion_sigma = _optional_positive_config_value(
        cfg.noise_exclusion_sigma,
        "noise_exclusion_sigma",
    )
    noise_exclusion_radius_px = _nonnegative_integer_config_value(
        cfg.noise_exclusion_radius_px,
        "noise_exclusion_radius_px",
    )
    noise_exclusion_mode = _validate_noise_exclusion_mode(cfg.noise_exclusion_mode)
    fill_value = _finite_or_nan_config_value(cfg.fill_value, "fill_value")

    values = _as_float_image(residual, name="residual")
    valid = np.isfinite(values)
    if mask is not None:
        user_mask = np.asarray(mask, dtype=bool)
        if user_mask.shape != values.shape:
            raise ValueError("mask must have the same shape as residual")
        valid &= user_mask
    if not valid.any():
        raise ValueError("mask excludes all residual pixels")

    sample = values[valid]
    center = float(np.median(sample))
    global_sigma = _robust_sigma(sample, center=center, min_noise=min_noise)
    noise_valid = valid.copy()
    particle_noise_mask = _particle_noise_exclusion_mask(
        values,
        valid=valid,
        center=center,
        global_sigma=global_sigma,
        noise_exclusion_sigma=noise_exclusion_sigma,
        noise_exclusion_radius_px=noise_exclusion_radius_px,
        mode=noise_exclusion_mode,
    )
    if particle_noise_mask.any():
        noise_valid &= ~particle_noise_mask
        if not noise_valid.any():
            noise_valid = valid.copy()
    centered = np.zeros(values.shape, dtype=np.float64)
    centered[valid] = values[valid] - center
    if clip_sigma is not None:
        centered = np.clip(
            centered,
            -clip_sigma * global_sigma,
            clip_sigma * global_sigma,
        )

    local_var = _masked_box_mean(
        np.square(centered),
        noise_valid,
        radius=noise_radius_px,
    )
    local_var = np.where(
        np.isfinite(local_var),
        local_var,
        global_sigma * global_sigma,
    )
    local_noise = np.sqrt(np.maximum(local_var, min_noise * min_noise))
    local_noise[~valid] = fill_value
    return local_noise


def _as_float_image(image: ArrayLike, *, name: str) -> FloatArray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    return arr


def _robust_sigma(values: FloatArray, *, center: float, min_noise: float) -> float:
    mad = float(np.median(np.abs(values - center)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma < min_noise:
        sigma = min_noise
    return sigma


def _validate_noise_exclusion_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise ValueError("ResidualConfig.noise_exclusion_mode must be a string")
    normalized = mode.strip().lower()
    if normalized in NOISE_EXCLUSION_MODES:
        return normalized
    choices = ", ".join(sorted(NOISE_EXCLUSION_MODES))
    raise ValueError(
        "ResidualConfig.noise_exclusion_mode must be one of "
        f"{choices}; got {mode!r}"
    )


def _positive_config_value(value: float, name: str) -> float:
    parsed = _finite_config_value(value, name)
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _optional_positive_config_value(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return _positive_config_value(value, name)


def _nonnegative_integer_config_value(value: int, name: str) -> int:
    parsed = _finite_config_value(value, name)
    if not np.isfinite(parsed) or parsed < 0 or not parsed.is_integer():
        raise ValueError(f"{name} must be a finite non-negative integer")
    return int(parsed)


def _finite_config_value(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _finite_or_nan_config_value(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric or NaN, not boolean")
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric or NaN")
    parsed = float(value)
    if not np.isfinite(parsed) and not np.isnan(parsed):
        raise ValueError(f"{name} must be finite or NaN")
    return parsed


def _particle_noise_exclusion_mask(
    values: FloatArray,
    *,
    valid: NDArray[np.bool_],
    center: float,
    global_sigma: float,
    noise_exclusion_sigma: float | None,
    noise_exclusion_radius_px: int,
    mode: str,
) -> NDArray[np.bool_]:
    if noise_exclusion_sigma is None:
        return np.zeros(values.shape, dtype=bool)
    threshold = noise_exclusion_sigma * global_sigma
    centered = values - center
    if mode == "positive":
        particle_like = valid & (centered > threshold)
    elif mode == "negative":
        particle_like = valid & (centered < -threshold)
    else:
        particle_like = valid & (np.abs(centered) > threshold)
    if not particle_like.any():
        return particle_like
    if noise_exclusion_radius_px > 0:
        particle_like = (
            _dilate_mask(particle_like, radius=noise_exclusion_radius_px)
            & valid
        )
    return particle_like


def _dilate_mask(mask: NDArray[np.bool_], *, radius: int) -> NDArray[np.bool_]:
    if radius == 0:
        return mask.copy()
    return _box_sum(mask.astype(np.float64), radius=radius) > 0


def _expand_mask_to_image(
    mask: ArrayLike | None,
    *,
    image_shape: tuple[int, int],
    belt_region: BeltRegion,
) -> NDArray[np.bool_] | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.shape == image_shape:
        return arr
    if arr.shape == belt_region.shape:
        expanded = np.zeros(image_shape, dtype=bool)
        expanded[belt_region.y_slice, belt_region.x_slice] = arr
        return expanded
    raise ValueError("residual_mask must match image shape or belt_region shape")


def _masked_box_mean(
    values: FloatArray,
    mask: NDArray[np.bool_],
    *,
    radius: int,
) -> FloatArray:
    weights = mask.astype(np.float64)
    numerator = _box_sum(values * weights, radius=radius)
    denominator = _box_sum(weights, radius=radius)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _box_sum(image: FloatArray, *, radius: int) -> FloatArray:
    if radius == 0:
        return image.astype(np.float64, copy=True)
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="constant")
    integral = np.pad(
        np.cumsum(
            np.cumsum(padded, axis=0, dtype=np.float64),
            axis=1,
            dtype=np.float64,
        ),
        ((1, 0), (1, 0)),
        mode="constant",
    )
    window = 2 * radius + 1
    return (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )

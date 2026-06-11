"""Render clean belt backgrounds from a belt map and phase estimate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .phase import (
    BeltMotionModel,
    PhaseEstimate,
    PhaseRegistrationConfig,
    estimate_phase,
    render_belt_view,
)


FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class BeltRegion:
    """Rectangular belt crop in full-frame image coordinates."""

    top: int
    left: int
    height: int
    width: int

    @property
    def y_slice(self) -> slice:
        return slice(self.top, self.top + self.height)

    @property
    def x_slice(self) -> slice:
        return slice(self.left, self.left + self.width)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


@dataclass(frozen=True)
class CleanBeltRender:
    """Rendered clean belt image and the pixels where that render is valid."""

    image: FloatArray
    mask: NDArray[np.bool_]
    phase_estimate: PhaseEstimate
    belt_region: BeltRegion

    @property
    def belt_crop(self) -> FloatArray:
        """Return the rendered belt crop without surrounding background pixels."""

        return self.image[self.belt_region.y_slice, self.belt_region.x_slice]


def render_expected_clean_belt(
    *,
    belt_map: ArrayLike,
    frame_index: float,
    motion_model: BeltMotionModel | None = None,
    observed_frame: ArrayLike | None = None,
    belt_region: BeltRegion | tuple[int, int, int, int] | None = None,
    output_shape: tuple[int, int] | None = None,
    phase_estimate: PhaseEstimate | None = None,
    registration_config: PhaseRegistrationConfig | None = None,
    registration_mask: ArrayLike | None = None,
    periodic: bool | None = None,
    fill_value: float = np.nan,
) -> CleanBeltRender:
    """Render the expected particle-free belt for one frame.

    The returned ``image`` is a full-frame-sized float array. Pixels outside the
    belt region are set to ``fill_value`` and marked invalid in ``mask`` so later
    subtraction can ignore camera background.

    If ``observed_frame`` is supplied and ``phase_estimate`` is not, the phase is
    refined by registering the observed belt crop against ``belt_map``.
    Otherwise the constant-speed ``motion_model`` supplies the phase. When a
    ``phase_estimate`` is passed explicitly, ``motion_model`` is not required.

    ``periodic`` controls whether belt-map rows wrap cyclically. When omitted,
    the value is inferred from ``motion_model.period_px`` when a motion model is
    available; otherwise it defaults to the historical cyclic behavior.
    """

    belt = _as_float_image(belt_map, name="belt_map")
    if belt.ndim != 2:
        raise ValueError("belt_map must be a 2-D array")

    observed = None
    if observed_frame is not None:
        observed = _as_float_image(observed_frame, name="observed_frame")
        if observed.ndim != 2:
            raise ValueError("observed_frame must be a 2-D array")
        if output_shape is not None and _resolve_shape(output_shape) != observed.shape:
            raise ValueError(
                "output_shape must match observed_frame.shape when observed_frame "
                "is supplied"
            )
        if output_shape is None:
            output_shape = observed.shape

    region = _resolve_belt_region(
        belt_region=belt_region,
        belt_width=belt.shape[1],
        output_shape=output_shape,
        fallback_shape=belt.shape,
    )
    output_shape = _resolve_output_shape(output_shape, observed, region)
    _validate_region(region, output_shape)

    if region.width != belt.shape[1]:
        raise ValueError("belt_region width must match belt_map width")

    render_periodic = _resolve_periodic_rendering(periodic, motion_model)
    phase = phase_estimate
    if phase is None:
        if motion_model is None:
            raise ValueError("motion_model is required when phase_estimate is not supplied")
        frame_crop = observed[region.y_slice, region.x_slice] if observed is not None else None
        mask_crop = _resolve_registration_mask(registration_mask, output_shape, region)
        phase = estimate_phase(
            frame_index,
            motion_model,
            frame=frame_crop,
            belt_map=belt if frame_crop is not None else None,
            config=registration_config,
            mask=mask_crop,
        )

    clean_crop = render_belt_view(
        belt,
        phase.phase_px,
        region.height,
        periodic=render_periodic,
    )
    clean = np.full(output_shape, fill_value, dtype=np.float64)
    valid = np.zeros(output_shape, dtype=bool)
    clean[region.y_slice, region.x_slice] = clean_crop
    valid[region.y_slice, region.x_slice] = np.isfinite(clean_crop)

    return CleanBeltRender(
        image=clean,
        mask=valid,
        phase_estimate=phase,
        belt_region=region,
    )


def _resolve_periodic_rendering(
    periodic: bool | None,
    motion_model: BeltMotionModel | None,
) -> bool:
    if periodic is not None:
        return bool(periodic)
    if motion_model is not None:
        return motion_model.period_px is not None
    return True


def _as_float_image(image: ArrayLike, *, name: str) -> FloatArray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _resolve_belt_region(
    *,
    belt_region: BeltRegion | tuple[int, int, int, int] | None,
    belt_width: int,
    output_shape: tuple[int, int] | None,
    fallback_shape: tuple[int, int],
) -> BeltRegion:
    if belt_region is None:
        height, width = output_shape if output_shape is not None else fallback_shape
        return BeltRegion(top=0, left=0, height=height, width=width)
    if isinstance(belt_region, BeltRegion):
        return BeltRegion(
            top=_integer_config_value(belt_region.top, "belt_region top"),
            left=_integer_config_value(belt_region.left, "belt_region left"),
            height=_integer_config_value(belt_region.height, "belt_region height"),
            width=_integer_config_value(belt_region.width, "belt_region width"),
        )
    if len(belt_region) != 4:
        raise ValueError("belt_region tuple must be (top, left, height, width)")
    top, left, height, width = belt_region
    region = BeltRegion(
        _integer_config_value(top, "belt_region top"),
        _integer_config_value(left, "belt_region left"),
        _integer_config_value(height, "belt_region height"),
        _integer_config_value(width, "belt_region width"),
    )
    if output_shape is None and region.width != belt_width:
        raise ValueError("belt_region width must match belt_map width")
    return region


def _resolve_output_shape(
    output_shape: tuple[int, int] | None,
    observed: FloatArray | None,
    region: BeltRegion,
) -> tuple[int, int]:
    if output_shape is not None:
        return _resolve_shape(output_shape)
    if observed is not None:
        return observed.shape
    return (region.top + region.height, region.left + region.width)


def _resolve_shape(output_shape: tuple[int, int]) -> tuple[int, int]:
    if len(output_shape) != 2:
        raise ValueError("output_shape must be (height, width)")
    return (
        _integer_config_value(output_shape[0], "output_shape height"),
        _integer_config_value(output_shape[1], "output_shape width"),
    )


def _integer_config_value(value: int, name: str) -> int:
    parsed = float(value)
    if not np.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(f"{name} must be a finite integer")
    return int(parsed)


def _validate_region(region: BeltRegion, output_shape: tuple[int, int]) -> None:
    if region.top < 0 or region.left < 0:
        raise ValueError("belt_region top and left must be non-negative")
    if region.height <= 0 or region.width <= 0:
        raise ValueError("belt_region height and width must be positive")
    if output_shape[0] <= 0 or output_shape[1] <= 0:
        raise ValueError("output_shape values must be positive")
    if region.top + region.height > output_shape[0]:
        raise ValueError("belt_region extends below output_shape")
    if region.left + region.width > output_shape[1]:
        raise ValueError("belt_region extends beyond output_shape width")


def _resolve_registration_mask(
    registration_mask: ArrayLike | None,
    output_shape: tuple[int, int],
    region: BeltRegion,
) -> NDArray[np.bool_] | None:
    if registration_mask is None:
        return None
    mask = np.asarray(registration_mask, dtype=bool)
    if mask.shape == output_shape:
        return mask[region.y_slice, region.x_slice]
    if mask.shape == region.shape:
        return mask
    raise ValueError("registration_mask must match output_shape or belt_region shape")

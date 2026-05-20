"""Belt-motion helpers for the packaged image driver."""

from __future__ import annotations

import math
import os

import numpy as np

from ._driver_runtime import crop, emit, env_bool, env_float, env_int, read_gray

SELECTED_FRAME_VELOCITY_UNIT = "selected_frame"
SOURCE_FRAME_VELOCITY_UNIT = "source_frame"
VELOCITY_FRAME_UNITS = {
    SELECTED_FRAME_VELOCITY_UNIT,
    SOURCE_FRAME_VELOCITY_UNIT,
}


def normalize_velocity_frame_unit(value: str) -> str:
    """Return the canonical frame unit for a manually supplied belt velocity."""

    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        SELECTED_FRAME_VELOCITY_UNIT: SELECTED_FRAME_VELOCITY_UNIT,
        "selected": SELECTED_FRAME_VELOCITY_UNIT,
        "processed": SELECTED_FRAME_VELOCITY_UNIT,
        "processed_frame": SELECTED_FRAME_VELOCITY_UNIT,
        "strided_frame": SELECTED_FRAME_VELOCITY_UNIT,
        "output_frame": SELECTED_FRAME_VELOCITY_UNIT,
        SOURCE_FRAME_VELOCITY_UNIT: SOURCE_FRAME_VELOCITY_UNIT,
        "source": SOURCE_FRAME_VELOCITY_UNIT,
        "original": SOURCE_FRAME_VELOCITY_UNIT,
        "original_frame": SOURCE_FRAME_VELOCITY_UNIT,
        "input_frame": SOURCE_FRAME_VELOCITY_UNIT,
        "raw_frame": SOURCE_FRAME_VELOCITY_UNIT,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(VELOCITY_FRAME_UNITS))
        raise ValueError(
            f"BELT_VELOCITY_FRAME_UNIT must be one of {choices}; got {value!r}"
        ) from exc


def resolve_velocity_frame_unit(frame_stride: int) -> str:
    """Resolve the frame unit for a manually supplied velocity.

    Numeric BELT_VELOCITY_PX_PER_FRAME values are ambiguous when FRAME_STRIDE > 1:
    they may refer either to adjacent original input frames or to adjacent selected
    frames after striding. Refuse the ambiguous case instead of silently applying
    the wrong phase increment.
    """

    if frame_stride < 1:
        raise ValueError("FRAME_STRIDE must be at least 1")
    value = os.getenv("BELT_VELOCITY_FRAME_UNIT", "").strip()
    if value:
        return normalize_velocity_frame_unit(value)
    if frame_stride == 1:
        return SELECTED_FRAME_VELOCITY_UNIT
    raise ValueError(
        f"BELT_VELOCITY_PX_PER_FRAME was supplied with FRAME_STRIDE={frame_stride}. "
        "Set BELT_VELOCITY_FRAME_UNIT=selected_frame if the supplied velocity is "
        "already in pixels per processed/selected frame, or set "
        "BELT_VELOCITY_FRAME_UNIT=source_frame if it is in pixels per adjacent "
        "original input frame. Source-frame velocities are multiplied by FRAME_STRIDE."
    )


def resolve_supplied_velocity(velocity_spec: str, frame_stride: int) -> tuple[float, str, float]:
    """Return effective selected-frame velocity, frame unit, and raw supplied value."""

    raw_velocity = float(velocity_spec)
    frame_unit = resolve_velocity_frame_unit(frame_stride)
    if frame_unit == SOURCE_FRAME_VELOCITY_UNIT:
        return raw_velocity * frame_stride, frame_unit, raw_velocity
    return raw_velocity, frame_unit, raw_velocity


def parse_region(first_frame: np.ndarray) -> tuple[int, int, int, int]:
    value = os.getenv("BELT_REGION", "").strip()
    height, width = first_frame.shape
    if not value:
        return 0, 0, height, width
    top, left, crop_height, crop_width = [int(x.strip()) for x in value.split(",")]
    if (
        top < 0 or left < 0 or crop_height <= 0 or crop_width <= 0
        or top + crop_height > height or left + crop_width > width
    ):
        raise ValueError(f"Invalid BELT_REGION={value!r} for image shape {(height, width)}")
    return top, left, crop_height, crop_width


def is_full_frame_region(
    region: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> bool:
    top, left, height, width = region
    return top == 0 and left == 0 and (height, width) == frame_shape


def validate_auto_velocity_region(
    region: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> None:
    if is_full_frame_region(region, frame_shape) and not env_bool("ALLOW_FULL_FRAME_AUTO_VELOCITY"):
        raise ValueError(
            "BELT_VELOCITY_PX_PER_FRAME=auto is unsafe with a full-frame BELT_REGION. "
            "Set BELT_REGION to the belt crop, supply BELT_VELOCITY_PX_PER_FRAME explicitly, "
            "or set ALLOW_FULL_FRAME_AUTO_VELOCITY=1 if the full frame truly contains only belt texture."
        )


def validate_auto_velocity_estimate(
    velocity: float,
    shifts: list[float],
    *,
    max_shift: int,
) -> None:
    min_abs_velocity = env_float("AUTO_VELOCITY_MIN_ABS_PX_PER_FRAME", 0.25, minimum=0.0)
    if abs(velocity) < min_abs_velocity:
        raise ValueError(
            f"Auto-estimated belt velocity {velocity:.6g} px/frame is below "
            f"AUTO_VELOCITY_MIN_ABS_PX_PER_FRAME={min_abs_velocity}. Supply BELT_REGION "
            "and/or BELT_VELOCITY_PX_PER_FRAME explicitly."
        )
    max_edge_fraction = env_float("AUTO_VELOCITY_MAX_EDGE_FRACTION", 0.2, minimum=0.0)
    if max_edge_fraction > 1.0:
        raise ValueError("AUTO_VELOCITY_MAX_EDGE_FRACTION must be in [0, 1]")
    if shifts:
        edge_fraction = float(np.mean(np.abs(np.asarray(shifts)) >= 0.9 * max_shift))
        if edge_fraction > max_edge_fraction:
            raise ValueError(
                f"Auto velocity search often hit the search edge: edge_fraction={edge_fraction:.3f}, "
                f"max_shift={max_shift}. Increase VELOCITY_SEARCH_RADIUS_PX or supply velocity explicitly."
            )


def correlation_shift(previous: np.ndarray, current: np.ndarray, max_shift: int) -> float:
    if previous.shape != current.shape:
        raise ValueError(
            "previous and current frames must have the same shape for velocity estimation"
        )
    if previous.ndim == 0:
        raise ValueError("previous and current frames must be non-scalar arrays")
    if max_shift >= previous.shape[0]:
        raise ValueError(
            "max_shift must be smaller than the image height for velocity estimation; "
            f"got max_shift={max_shift}, height={previous.shape[0]}"
        )

    def score(shift: int) -> float:
        if shift > 0:
            a, b = previous[:-shift], current[shift:]
        elif shift < 0:
            a, b = previous[-shift:], current[:shift]
        else:
            a, b = previous, current
        a = a.astype(np.float64, copy=False) - float(np.mean(a))
        b = b.astype(np.float64, copy=False) - float(np.mean(b))
        denominator = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
        return -np.inf if denominator <= 0 else float(np.sum(a * b) / denominator)

    shifts = np.arange(-max_shift, max_shift + 1)
    scores = np.array([score(int(s)) for s in shifts])
    best_index = int(np.argmax(scores))
    best_shift = float(shifts[best_index])
    if 0 < best_index < len(scores) - 1:
        y0, y1, y2 = scores[best_index - 1], scores[best_index], scores[best_index + 1]
        denominator = y0 - 2 * y1 + y2
        if abs(float(denominator)) > 1e-12:
            delta = 0.5 * (y0 - y2) / denominator
            if np.isfinite(delta) and -1 <= delta <= 1:
                best_shift += float(delta)
    return best_shift


def estimate_velocity(paths: list, region: tuple[int, int, int, int]) -> tuple[float, list[float]]:
    max_shift = env_int("VELOCITY_SEARCH_RADIUS_PX", 50, minimum=1)
    _, _, crop_height, _crop_width = region
    if max_shift >= crop_height:
        raise ValueError(
            "VELOCITY_SEARCH_RADIUS_PX must be smaller than the BELT_REGION height; "
            f"got VELOCITY_SEARCH_RADIUS_PX={max_shift}, BELT_REGION height={crop_height}"
        )
    pair_count = min(len(paths) - 1, env_int("VELOCITY_ESTIMATION_PAIRS", 100, minimum=1))
    progress_interval = env_int("PROGRESS_INTERVAL_FRAMES", 25, minimum=1)
    if pair_count < 1:
        raise ValueError("Automatic velocity estimation requires at least two frames")
    emit("velocity", "estimating belt velocity", pair_count=pair_count, max_shift_px=max_shift)
    shifts: list[float] = []
    previous = crop(read_gray(paths[0]), region)
    for index in range(1, pair_count + 1):
        current = crop(read_gray(paths[index]), region)
        shifts.append(correlation_shift(previous, current, max_shift))
        previous = current
        if index == 1 or index == pair_count or index % progress_interval == 0:
            emit("velocity", f"estimated {index}/{pair_count} shifts", current_shift_px=shifts[-1], median_shift_px=float(np.median(shifts)))
    velocity = float(np.median(shifts))
    validate_auto_velocity_estimate(velocity, shifts, max_shift=max_shift)
    return velocity, shifts

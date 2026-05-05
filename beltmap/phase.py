"""Belt phase prediction and registration.

The phase convention used here is:

    belt_coordinate_y = image_y + phase_px

for pixels inside the cropped belt image. Positive ``image_velocity_px_per_frame``
means that belt texture moves downward in image coordinates. Therefore the phase
decreases over time for a downward-moving belt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class BeltMotionModel:
    """Constant-speed belt phase model.

    Parameters
    ----------
    image_velocity_px_per_frame:
        Signed belt texture velocity in image coordinates. Positive means the
        belt texture moves downward from one frame to the next.
    period_px:
        Belt circumference in pixels in the same coordinate system as the belt
        map rows. If set, phases are wrapped modulo this period.
    reference_frame:
        Frame index at which ``reference_phase_px`` is defined.
    reference_phase_px:
        Belt-map row corresponding to image row 0 at ``reference_frame``.
    """

    image_velocity_px_per_frame: float
    period_px: float | None = None
    reference_frame: float = 0.0
    reference_phase_px: float = 0.0

    def phase_at(self, frame_index: float) -> float:
        """Return the predicted phase for ``frame_index``."""

        phase = self.reference_phase_px - self.image_velocity_px_per_frame * (
            frame_index - self.reference_frame
        )
        return wrap_phase(phase, self.period_px)

    def coordinate_rows(self, frame_index: float, height: int) -> FloatArray:
        """Return belt-coordinate rows for image rows ``0..height-1``."""

        rows = np.arange(height, dtype=np.float64) + self.phase_at(frame_index)
        if self.period_px is not None:
            rows = np.mod(rows, self.period_px)
        return rows


@dataclass(frozen=True)
class PhaseRegistrationConfig:
    """Settings for local phase refinement by registration."""

    search_radius_px: float = 8.0
    search_step_px: float = 0.5
    trim_fraction: float = 0.08
    highpass_radius_px: int = 15

    def candidate_offsets(self) -> FloatArray:
        """Return the tested phase offsets, including zero when possible."""

        if self.search_step_px <= 0:
            raise ValueError("search_step_px must be positive")
        count = int(np.floor(2 * self.search_radius_px / self.search_step_px)) + 1
        offsets = -self.search_radius_px + self.search_step_px * np.arange(
            count, dtype=np.float64
        )
        if not np.any(np.isclose(offsets, 0.0)):
            offsets = np.sort(np.append(offsets, 0.0))
        return offsets


@dataclass(frozen=True)
class PhaseEstimate:
    """A belt phase estimate for one frame."""

    phase_px: float
    frame_index: float
    predicted_phase_px: float
    correction_px: float = 0.0
    loss: float | None = None
    score: float | None = None
    method: str = "motion_model"


def wrap_phase(phase_px: float, period_px: float | None) -> float:
    """Wrap ``phase_px`` modulo ``period_px`` when a period is known."""

    if period_px is None:
        return float(phase_px)
    if period_px <= 0:
        raise ValueError("period_px must be positive")
    return float(phase_px % period_px)


def render_belt_view(
    belt_map: ArrayLike,
    phase_px: float,
    height: int,
    *,
    x_slice: slice | None = None,
) -> FloatArray:
    """Render the clean belt background expected in an image crop.

    ``belt_map`` is indexed by belt-coordinate row and image x coordinate.
    Fractional phases are rendered with linear interpolation along the belt
    coordinate axis and cyclic wrapping.
    """

    belt = _as_float_image(belt_map, name="belt_map")
    if belt.ndim != 2:
        raise ValueError("belt_map must be a 2-D array")
    if height <= 0:
        raise ValueError("height must be positive")

    if x_slice is not None:
        belt = belt[:, x_slice]

    period = belt.shape[0]
    rows = (np.arange(height, dtype=np.float64) + phase_px) % period
    row0 = np.floor(rows).astype(np.int64)
    row1 = (row0 + 1) % period
    weight = (rows - row0)[:, None]
    return (1.0 - weight) * belt[row0] + weight * belt[row1]


def estimate_phase(
    frame_index: float,
    motion_model: BeltMotionModel,
    *,
    frame: ArrayLike | None = None,
    belt_map: ArrayLike | None = None,
    config: PhaseRegistrationConfig | None = None,
    mask: ArrayLike | None = None,
) -> PhaseEstimate:
    """Estimate belt phase from a motion model, optionally refined by registration."""

    predicted = motion_model.phase_at(frame_index)
    if frame is None or belt_map is None:
        return PhaseEstimate(
            phase_px=predicted,
            frame_index=frame_index,
            predicted_phase_px=predicted,
        )

    return refine_phase_by_registration(
        frame=frame,
        belt_map=belt_map,
        predicted_phase_px=predicted,
        frame_index=frame_index,
        period_px=motion_model.period_px,
        config=config,
        mask=mask,
    )


def refine_phase_by_registration(
    *,
    frame: ArrayLike,
    belt_map: ArrayLike,
    predicted_phase_px: float,
    frame_index: float = 0.0,
    period_px: float | None = None,
    config: PhaseRegistrationConfig | None = None,
    mask: ArrayLike | None = None,
) -> PhaseEstimate:
    """Refine a predicted phase by robust local registration.

    The search is one-dimensional in belt phase. This is deliberate: the camera
    crop and x alignment are assumed to be known, while the belt phase changes
    frame-to-frame.
    """

    cfg = config or PhaseRegistrationConfig()
    observed = _as_float_image(frame, name="frame")
    belt = _as_float_image(belt_map, name="belt_map")
    if observed.ndim != 2 or belt.ndim != 2:
        raise ValueError("frame and belt_map must be 2-D arrays")
    if observed.shape[1] != belt.shape[1]:
        raise ValueError(
            "frame and belt_map must have the same width; crop or x-slice before registration"
        )

    valid_mask = _prepare_mask(mask, observed.shape)
    observed_prepared = _prepare_for_registration(observed, cfg.highpass_radius_px)

    losses: list[tuple[float, float]] = []
    for offset in cfg.candidate_offsets():
        phase = wrap_phase(predicted_phase_px + float(offset), period_px)
        expected = render_belt_view(belt, phase, observed.shape[0])
        expected_prepared = _prepare_for_registration(expected, cfg.highpass_radius_px)
        loss = _trimmed_mean_square(
            observed_prepared - expected_prepared,
            trim_fraction=cfg.trim_fraction,
            mask=valid_mask,
        )
        losses.append((loss, float(offset)))

    best_loss, best_offset = min(losses, key=lambda item: item[0])
    phase = wrap_phase(predicted_phase_px + best_offset, period_px)
    score = _loss_to_score(best_loss, (loss for loss, _offset in losses))
    return PhaseEstimate(
        phase_px=phase,
        frame_index=frame_index,
        predicted_phase_px=predicted_phase_px,
        correction_px=best_offset,
        loss=best_loss,
        score=score,
        method="registration",
    )


def _as_float_image(image: ArrayLike, *, name: str) -> FloatArray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _prepare_mask(mask: ArrayLike | None, shape: tuple[int, int]) -> NDArray[np.bool_] | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != shape:
        raise ValueError("mask must have the same shape as frame")
    return arr


def _prepare_for_registration(image: FloatArray, highpass_radius_px: int) -> FloatArray:
    if highpass_radius_px <= 0:
        prepared = image.copy()
    else:
        prepared = image - _box_blur(image, radius=highpass_radius_px)
    std = float(np.std(prepared))
    if std > 0:
        prepared = prepared / std
    return prepared


def _box_blur(image: FloatArray, radius: int) -> FloatArray:
    """Fast separable box blur implemented with NumPy only."""

    if radius <= 0:
        return image.copy()
    blurred = _uniform_filter_axis(image, radius=radius, axis=0)
    return _uniform_filter_axis(blurred, radius=radius, axis=1)


def _uniform_filter_axis(image: FloatArray, radius: int, axis: int) -> FloatArray:
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (radius, radius)
    padded = np.pad(image, pad_width, mode="edge")
    moved = np.moveaxis(padded, axis, 0)
    csum = np.cumsum(moved, axis=0, dtype=np.float64)
    window = 2 * radius + 1
    summed = csum[window:] - csum[:-window]
    first = np.take(moved, [0], axis=0) * window
    summed = np.concatenate([first, summed], axis=0)
    result = summed / window
    trim = [slice(None), slice(None)]
    trim[axis] = slice(0, image.shape[axis])
    return np.moveaxis(result, 0, axis)[tuple(trim)]


def _trimmed_mean_square(
    residual: FloatArray,
    *,
    trim_fraction: float,
    mask: NDArray[np.bool_] | None,
) -> float:
    if not 0 <= trim_fraction < 1:
        raise ValueError("trim_fraction must be in [0, 1)")
    values = residual[mask] if mask is not None else residual.ravel()
    if values.size == 0:
        raise ValueError("registration mask excludes all pixels")
    squared = np.square(values)
    if trim_fraction > 0 and squared.size > 1:
        cutoff = np.quantile(squared, 1.0 - trim_fraction)
        squared = squared[squared <= cutoff]
    return float(np.mean(squared))


def _loss_to_score(best_loss: float, all_losses: Iterable[float]) -> float:
    losses = np.fromiter(all_losses, dtype=np.float64)
    if losses.size == 0:
        return 0.0
    median_loss = float(np.median(losses))
    if median_loss <= 0:
        return 1.0
    return float(max(0.0, 1.0 - best_loss / median_loss))

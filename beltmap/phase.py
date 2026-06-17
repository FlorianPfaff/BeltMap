"""Belt phase prediction and registration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class BeltMotionModel:
    """Constant-speed belt phase model."""

    image_velocity_px_per_frame: float
    period_px: float | None = None
    reference_frame: float = 0.0
    reference_phase_px: float = 0.0

    def phase_at(self, frame_index: float) -> float:
        phase = self.reference_phase_px - self.image_velocity_px_per_frame * (
            frame_index - self.reference_frame
        )
        return wrap_phase(phase, self.period_px)

    def coordinate_rows(self, frame_index: float, height: int) -> FloatArray:
        height = _positive_integer_value(height, "height")
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
    subpixel_refinement: bool = False
    robust_normalization: bool = False

    def candidate_offsets(self) -> FloatArray:
        cfg = self.normalized()
        radius = cfg.search_radius_px
        if radius < 0:
            raise ValueError("search_radius_px must be non-negative")
        step = cfg.search_step_px
        if step <= 0:
            raise ValueError("search_step_px must be positive")
        if radius == 0.0:
            return np.asarray([0.0], dtype=np.float64)
        positive = step * np.arange(int(np.floor(radius / step)) + 1, dtype=np.float64)
        if not np.any(np.isclose(positive, radius)):
            positive = np.append(positive, radius)
        return np.r_[-positive[:0:-1], positive].astype(np.float64, copy=False)

    def normalized(self) -> PhaseRegistrationConfig:
        search_radius_px = _finite_float_value(
            self.search_radius_px,
            "search_radius_px",
        )
        search_step_px = _finite_float_value(
            self.search_step_px,
            "search_step_px",
        )
        trim_fraction = _finite_float_value(self.trim_fraction, "trim_fraction")
        highpass_radius_px = _nonnegative_integer_value(
            self.highpass_radius_px,
            "highpass_radius_px",
        )
        if not 0 <= trim_fraction < 1:
            raise ValueError("trim_fraction must be in [0, 1)")
        return replace(
            self,
            search_radius_px=search_radius_px,
            search_step_px=search_step_px,
            trim_fraction=trim_fraction,
            highpass_radius_px=highpass_radius_px,
        )


@dataclass(frozen=True)
class PhaseTrajectorySmoothingConfig:
    """Settings for smoothing registration corrections over a frame sequence."""

    window_radius_frames: int = 0
    min_score: float | None = None
    max_abs_correction_px: float | None = None
    robust_sigma: float = 3.0
    min_support: int = 3

    def validate(self) -> None:
        _nonnegative_integer_value(
            self.window_radius_frames,
            "window_radius_frames",
        )
        _positive_integer_value(self.min_support, "min_support")
        robust_sigma = _finite_float_value(self.robust_sigma, "robust_sigma")
        if robust_sigma <= 0:
            raise ValueError("robust_sigma must be positive")
        min_score = _optional_finite_float_value(self.min_score, "min_score")
        if min_score is not None and min_score < 0:
            raise ValueError("min_score must be non-negative when set")
        max_abs_correction_px = _optional_finite_float_value(
            self.max_abs_correction_px,
            "max_abs_correction_px",
        )
        if max_abs_correction_px is not None and max_abs_correction_px < 0:
            raise ValueError("max_abs_correction_px must be non-negative when set")


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
    drift_px: float = 0.0
    second_best_loss: float | None = None
    loss_gap: float | None = None
    loss_gap_ratio: float | None = None
    loss_curvature: float | None = None
    uncertainty_px: float | None = None


@dataclass(frozen=True)
class PhaseDriftConfig:
    """Settings for online, slowly varying phase-drift compensation."""

    enabled: bool = False
    smoothing_alpha: float = 0.15
    min_score: float = 0.05
    max_abs_residual_correction_px: float | None = None
    max_abs_drift_px: float | None = None


class PhaseDriftFilter:
    """Stateful online filter for residual phase drift."""

    def __init__(
        self,
        config: PhaseDriftConfig | None = None,
        *,
        initial_drift_px: float = 0.0,
        period_px: float | None = None,
    ) -> None:
        raw_config = config or PhaseDriftConfig()
        smoothing_alpha = _finite_float_value(
            raw_config.smoothing_alpha,
            "PhaseDriftConfig.smoothing_alpha",
        )
        if not 0.0 <= smoothing_alpha <= 1.0:
            raise ValueError("PhaseDriftConfig.smoothing_alpha must be in [0, 1]")
        min_score = _finite_float_value(
            raw_config.min_score,
            "PhaseDriftConfig.min_score",
        )
        if min_score < 0:
            raise ValueError("PhaseDriftConfig.min_score must be non-negative")
        max_abs_residual_correction_px = _optional_finite_float_value(
            raw_config.max_abs_residual_correction_px,
            "max_abs_residual_correction_px",
        )
        if (
            max_abs_residual_correction_px is not None
            and max_abs_residual_correction_px < 0
        ):
            raise ValueError("max_abs_residual_correction_px must be non-negative")
        max_abs_drift_px = _optional_finite_float_value(
            raw_config.max_abs_drift_px,
            "max_abs_drift_px",
        )
        if max_abs_drift_px is not None and max_abs_drift_px < 0:
            raise ValueError("max_abs_drift_px must be non-negative")
        self.config = replace(
            raw_config,
            smoothing_alpha=smoothing_alpha,
            min_score=min_score,
            max_abs_residual_correction_px=max_abs_residual_correction_px,
            max_abs_drift_px=max_abs_drift_px,
        )
        if period_px is None:
            self.period_px = None
        else:
            period = _finite_float_value(period_px, "period_px")
            if period <= 0:
                raise ValueError("period_px must be positive")
            self.period_px = period
        self.drift_px = _finite_float_value(initial_drift_px, "initial_drift_px")
        self.accepted_updates = 0
        self.rejected_updates = 0

    def predict(self, nominal_phase_px: float) -> float:
        return wrap_phase(float(nominal_phase_px) + self.drift_px, self.period_px)

    def observe(self, estimate: PhaseEstimate) -> PhaseEstimate:
        method = estimate.method
        if self.config.enabled and method != "motion_model":
            method = f"{method}+drift"
        updated = replace(estimate, drift_px=self.drift_px, method=method)
        if not self.config.enabled:
            return updated
        if not self._accepts(estimate):
            self.rejected_updates += 1
            return updated
        proposed = self.drift_px + float(estimate.correction_px)
        if self.config.max_abs_drift_px is not None:
            proposed = float(
                np.clip(
                    proposed,
                    -self.config.max_abs_drift_px,
                    self.config.max_abs_drift_px,
                )
            )
        alpha = self.config.smoothing_alpha
        self.drift_px = (1.0 - alpha) * self.drift_px + alpha * proposed
        self.accepted_updates += 1
        return updated

    def _accepts(self, estimate: PhaseEstimate) -> bool:
        score = 0.0 if estimate.score is None else float(estimate.score)
        if score < self.config.min_score:
            return False
        max_abs = self.config.max_abs_residual_correction_px
        return max_abs is None or abs(float(estimate.correction_px)) <= max_abs


def wrap_phase(phase_px: float, period_px: float | None) -> float:
    phase = _finite_float_value(phase_px, "phase_px")
    if period_px is None:
        return phase
    period = _finite_float_value(period_px, "period_px")
    if period <= 0:
        raise ValueError("period_px must be positive")
    return float(phase % period)


def render_belt_view(
    belt_map: ArrayLike,
    phase_px: float,
    height: int,
    *,
    x_slice: slice | None = None,
    periodic: bool = True,
) -> FloatArray:
    """Render the expected belt crop at a phase.

    With ``periodic=False``, rows outside the finite map support are returned as
    ``nan`` instead of wrapping to the opposite edge.
    """

    belt = _as_float_image(belt_map, name="belt_map")
    if belt.ndim != 2:
        raise ValueError("belt_map must be a 2-D array")
    phase = _finite_float_value(phase_px, "phase_px")
    height = _positive_integer_value(height, "height")
    if x_slice is not None:
        belt = belt[:, x_slice]

    map_height = belt.shape[0]
    rows = np.arange(height, dtype=np.float64) + phase
    if periodic:
        rows = rows % map_height
        row0 = np.floor(rows).astype(np.int64)
        row1 = (row0 + 1) % map_height
        weight = (rows - row0)[:, None]
        return (1.0 - weight) * belt[row0] + weight * belt[row1]

    valid = (rows >= 0.0) & (rows <= float(map_height - 1))
    clipped_rows = np.clip(rows, 0.0, float(map_height - 1))
    row0 = np.floor(clipped_rows).astype(np.int64)
    row1 = np.minimum(row0 + 1, map_height - 1)
    weight = (clipped_rows - row0)[:, None]
    rendered = (1.0 - weight) * belt[row0] + weight * belt[row1]
    rendered[~valid, :] = np.nan
    return rendered


def estimate_phase(
    frame_index: float,
    motion_model: BeltMotionModel,
    *,
    frame: ArrayLike | None = None,
    belt_map: ArrayLike | None = None,
    config: PhaseRegistrationConfig | None = None,
    mask: ArrayLike | None = None,
) -> PhaseEstimate:
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
    cfg = (config or PhaseRegistrationConfig()).normalized()
    observed = _as_float_image(frame, name="frame")
    belt = _as_float_image(belt_map, name="belt_map")
    if observed.ndim != 2 or belt.ndim != 2:
        raise ValueError("frame and belt_map must be 2-D arrays")
    if observed.shape[1] != belt.shape[1]:
        raise ValueError(
            "frame and belt_map must have the same width; crop or x-slice before registration"
        )

    user_mask = _prepare_mask(mask, observed.shape)
    valid_mask = _registration_valid_mask(observed, user_mask)
    observed_prepared = _prepare_for_registration(
        observed,
        cfg.highpass_radius_px,
        robust_normalization=cfg.robust_normalization,
        mask=valid_mask,
    )

    losses: list[tuple[float, float]] = []
    periodic = period_px is not None
    for offset in cfg.candidate_offsets():
        phase = wrap_phase(predicted_phase_px + float(offset), period_px)
        expected = render_belt_view(
            belt,
            phase,
            observed.shape[0],
            periodic=periodic,
        )
        candidate_mask = valid_mask & np.isfinite(expected)
        if not np.any(candidate_mask):
            losses.append((float("inf"), float(offset)))
            continue
        expected_prepared = _prepare_for_registration(
            expected,
            cfg.highpass_radius_px,
            robust_normalization=cfg.robust_normalization,
            mask=candidate_mask,
        )
        loss = _trimmed_mean_square(
            observed_prepared - expected_prepared,
            trim_fraction=cfg.trim_fraction,
            mask=candidate_mask,
        )
        losses.append((loss, float(offset)))

    if not any(np.isfinite(loss) for loss, _offset in losses):
        raise ValueError(
            "registration search has no valid overlap with finite belt-map support"
        )
    best_index = min(range(len(losses)), key=lambda index: losses[index][0])
    best_loss, best_offset = losses[best_index]
    if cfg.subpixel_refinement:
        best_loss, best_offset = _refine_quadratic_offset(losses, best_index)
    phase = wrap_phase(predicted_phase_px + best_offset, period_px)
    score = _loss_to_score(best_loss, (loss for loss, _offset in losses))
    diagnostics = _registration_loss_diagnostics(
        losses,
        best_index=best_index,
        best_loss=best_loss,
    )
    return PhaseEstimate(
        phase_px=phase,
        frame_index=frame_index,
        predicted_phase_px=predicted_phase_px,
        correction_px=best_offset,
        loss=best_loss,
        score=score,
        **diagnostics,
        method="registration",
    )


def smooth_phase_estimates(
    estimates: Sequence[PhaseEstimate],
    *,
    period_px: float | None = None,
    config: PhaseTrajectorySmoothingConfig | None = None,
) -> list[PhaseEstimate]:
    cfg = config or PhaseTrajectorySmoothingConfig()
    cfg.validate()
    if not estimates:
        return []
    if cfg.window_radius_frames == 0:
        return list(estimates)

    frame_indices = np.asarray(
        [estimate.frame_index for estimate in estimates], dtype=np.float64
    )
    corrections = np.asarray(
        [
            _phase_correction_from_estimate(estimate, period_px)
            for estimate in estimates
        ],
        dtype=np.float64,
    )
    scores = np.asarray(
        [
            np.nan if estimate.score is None else estimate.score
            for estimate in estimates
        ],
        dtype=np.float64,
    )

    valid = np.isfinite(frame_indices) & np.isfinite(corrections)
    if cfg.min_score is not None:
        valid &= np.isfinite(scores) & (scores >= cfg.min_score)
    if cfg.max_abs_correction_px is not None:
        valid &= np.abs(corrections) <= cfg.max_abs_correction_px
    if not np.any(valid):
        return list(estimates)

    weights = np.ones(len(estimates), dtype=np.float64)
    scored = np.isfinite(scores)
    weights[scored] = np.maximum(scores[scored], 1e-6)
    weights[~valid] = 0.0

    smoothed = []
    for estimate, correction in zip(
        estimates,
        _smooth_corrections(frame_indices, corrections, weights, valid, cfg),
        strict=True,
    ):
        if not np.isfinite(correction):
            correction = _phase_correction_from_estimate(estimate, period_px)
        phase = wrap_phase(estimate.predicted_phase_px + float(correction), period_px)
        smoothed.append(
            PhaseEstimate(
                phase_px=phase,
                frame_index=estimate.frame_index,
                predicted_phase_px=estimate.predicted_phase_px,
                correction_px=float(correction),
                loss=estimate.loss,
                score=estimate.score,
                second_best_loss=estimate.second_best_loss,
                loss_gap=estimate.loss_gap,
                loss_gap_ratio=estimate.loss_gap_ratio,
                loss_curvature=estimate.loss_curvature,
                uncertainty_px=estimate.uncertainty_px,
                method=_smoothed_method_name(estimate.method),
                drift_px=estimate.drift_px,
            )
        )
    return smoothed


def _phase_correction_from_estimate(
    estimate: PhaseEstimate, period_px: float | None
) -> float:
    if period_px is None:
        return float(estimate.correction_px)
    return _cyclic_difference(estimate.phase_px, estimate.predicted_phase_px, period_px)


def _cyclic_difference(phase_px: float, reference_px: float, period_px: float) -> float:
    if period_px <= 0:
        raise ValueError("period_px must be positive")
    return float(
        (phase_px - reference_px + 0.5 * period_px) % period_px - 0.5 * period_px
    )


def _smooth_corrections(
    frame_indices: FloatArray,
    corrections: FloatArray,
    weights: FloatArray,
    valid: NDArray[np.bool_],
    config: PhaseTrajectorySmoothingConfig,
) -> FloatArray:
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
        result[output_index] = _robust_local_linear_prediction(
            frame_indices[selected],
            corrections[selected],
            weights[selected] * local_kernel,
            center_frame,
            robust_sigma=config.robust_sigma,
            min_support=min(config.min_support, selected.size),
        )
    return result


def _robust_local_linear_prediction(
    x: FloatArray,
    y: FloatArray,
    weights: FloatArray,
    center_x: float,
    *,
    robust_sigma: float,
    min_support: int,
) -> float:
    x, y, weights = _finite_weighted_samples(x, y, weights)
    if x.size == 0:
        return float("nan")
    keep = _median_outlier_mask(y, robust_sigma=robust_sigma)
    if int(np.count_nonzero(keep)) >= min_support and not np.all(keep):
        x, y, weights = x[keep], y[keep], weights[keep]
    prediction, fitted = _weighted_linear_fit(x, y, weights, center_x)
    if x.size < max(3, min_support):
        return prediction
    keep = _median_outlier_mask(y - fitted, robust_sigma=robust_sigma)
    if int(np.count_nonzero(keep)) < min_support or np.all(keep):
        return prediction
    refined_prediction, _ = _weighted_linear_fit(
        x[keep], y[keep], weights[keep], center_x
    )
    return refined_prediction


def _finite_weighted_samples(
    x: FloatArray,
    y: FloatArray,
    weights: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    return x[finite], y[finite], weights[finite]


def _median_outlier_mask(
    values: FloatArray, *, robust_sigma: float
) -> NDArray[np.bool_]:
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    center = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.ones(values.shape, dtype=bool)
    return np.abs(values - center) <= robust_sigma * scale


def _weighted_linear_fit(
    x: FloatArray,
    y: FloatArray,
    weights: FloatArray,
    center_x: float,
) -> tuple[float, FloatArray]:
    x, y, weights = _finite_weighted_samples(x, y, weights)
    if x.size == 0:
        return float("nan"), np.full(0, np.nan, dtype=np.float64)
    dx = x - center_x
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        mean_y = float(np.mean(y))
        return mean_y, np.full_like(y, mean_y, dtype=np.float64)
    sum_x = float(np.sum(weights * dx))
    sum_y = float(np.sum(weights * y))
    sum_xx = float(np.sum(weights * dx * dx))
    sum_xy = float(np.sum(weights * dx * y))
    denominator = weight_sum * sum_xx - sum_x * sum_x
    if abs(denominator) <= 1e-12:
        mean_y = sum_y / weight_sum
        return float(mean_y), np.full_like(y, mean_y, dtype=np.float64)
    intercept = (sum_y * sum_xx - sum_x * sum_xy) / denominator
    slope = (weight_sum * sum_xy - sum_x * sum_y) / denominator
    return float(intercept), intercept + slope * dx


def _smoothed_method_name(method: str) -> str:
    return method if method.endswith("_smoothed") else f"{method}_smoothed"


def _as_float_image(image: ArrayLike, *, name: str) -> FloatArray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _finite_float_value(value: float, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _optional_finite_float_value(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float_value(value, name)


def _nonnegative_integer_value(value: int, name: str) -> int:
    parsed = _finite_float_value(value, name)
    if parsed < 0 or not parsed.is_integer():
        raise ValueError(f"{name} must be a finite non-negative integer")
    return int(parsed)


def _positive_integer_value(value: int, name: str) -> int:
    parsed = _finite_float_value(value, name)
    if parsed < 1 or not parsed.is_integer():
        raise ValueError(f"{name} must be a finite positive integer")
    return int(parsed)


def _prepare_mask(
    mask: ArrayLike | None, shape: tuple[int, int]
) -> NDArray[np.bool_] | None:
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != shape:
        raise ValueError("mask must have the same shape as frame")
    return arr


def _registration_valid_mask(
    image: ArrayLike,
    mask: NDArray[np.bool_] | None,
) -> NDArray[np.bool_]:
    values = np.asarray(image, dtype=np.float64)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        raise ValueError("registration mask excludes all finite pixels")
    return valid


def _prepare_for_registration(
    image: FloatArray,
    highpass_radius_px: int,
    *,
    robust_normalization: bool = False,
    mask: NDArray[np.bool_] | None = None,
) -> FloatArray:
    values = np.asarray(image, dtype=np.float64)
    valid = np.isfinite(values)
    if mask is not None:
        if mask.shape != values.shape:
            raise ValueError("mask must have the same shape as image")
        valid &= mask
    if not np.any(valid):
        raise ValueError("registration mask excludes all finite pixels")
    fill_value = float(np.median(values[valid]))
    sanitized = np.where(valid, values, fill_value)
    prepared = (
        sanitized.copy()
        if highpass_radius_px <= 0
        else sanitized - _box_blur(sanitized, radius=highpass_radius_px)
    )
    scale_values = prepared[valid]
    scale = (
        _robust_scale(scale_values)
        if robust_normalization
        else float(np.std(scale_values))
    )
    if scale > 0:
        prepared = prepared / scale
    prepared[~valid] = np.nan
    return prepared


def _robust_scale(values: FloatArray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad if mad > 0 else float(np.std(values))


def _refine_quadratic_offset(
    losses: list[tuple[float, float]], best_index: int
) -> tuple[float, float]:
    if best_index <= 0 or best_index >= len(losses) - 1:
        return losses[best_index]
    window = losses[best_index - 1 : best_index + 2]
    offsets = np.array([offset for _loss, offset in window], dtype=np.float64)
    values = np.array([loss for loss, _offset in window], dtype=np.float64)
    if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(values)):
        return losses[best_index]
    if np.unique(offsets).size != offsets.size:
        return losses[best_index]
    quadratic, linear, constant = np.polyfit(offsets, values, deg=2)
    if not np.isfinite(quadratic) or quadratic <= 0:
        return losses[best_index]
    refined_offset = float(-linear / (2.0 * quadratic))
    if not np.isfinite(refined_offset) or not float(
        np.min(offsets)
    ) <= refined_offset <= float(np.max(offsets)):
        return losses[best_index]
    refined_loss = float(
        quadratic * refined_offset * refined_offset + linear * refined_offset + constant
    )
    best_loss, best_offset = losses[best_index]
    if np.isfinite(refined_loss) and refined_loss <= best_loss:
        return refined_loss, refined_offset
    return best_loss, best_offset


def _box_blur(image: FloatArray, radius: int) -> FloatArray:
    if radius <= 0:
        return image.copy()
    return _uniform_filter_axis(
        _uniform_filter_axis(image, radius=radius, axis=0), radius=radius, axis=1
    )


def _uniform_filter_axis(image: FloatArray, radius: int, axis: int) -> FloatArray:
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (radius, radius)
    moved = np.moveaxis(np.pad(image, pad_width, mode="edge"), axis, 0)
    csum = np.concatenate(
        [
            np.zeros((1,) + moved.shape[1:], dtype=np.float64),
            np.cumsum(moved, axis=0, dtype=np.float64),
        ],
        axis=0,
    )
    window = 2 * radius + 1
    return np.moveaxis((csum[window:] - csum[:-window]) / window, 0, axis)


def _trimmed_mean_square(
    residual: FloatArray,
    *,
    trim_fraction: float,
    mask: NDArray[np.bool_] | None,
) -> float:
    if not 0 <= trim_fraction < 1:
        raise ValueError("trim_fraction must be in [0, 1)")
    values = residual[mask] if mask is not None else residual.ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("registration mask excludes all pixels")
    squared = np.square(values)
    if trim_fraction > 0 and squared.size > 1:
        squared = squared[squared <= np.quantile(squared, 1.0 - trim_fraction)]
    return float(np.mean(squared))


def _loss_to_score(best_loss: float, all_losses: Iterable[float]) -> float:
    losses = np.fromiter(all_losses, dtype=np.float64)
    losses = losses[np.isfinite(losses)]
    if losses.size == 0:
        return 0.0
    median_loss = float(np.median(losses))
    if median_loss <= 0:
        return 1.0
    return float(max(0.0, 1.0 - best_loss / median_loss))


def _registration_loss_diagnostics(
    losses: Sequence[tuple[float, float]],
    *,
    best_index: int,
    best_loss: float,
) -> dict[str, float | None]:
    """Return ambiguity and local-curvature diagnostics for phase registration."""

    finite_losses = [
        (float(loss), index)
        for index, (loss, _offset) in enumerate(losses)
        if np.isfinite(loss)
    ]
    other_losses = [loss for loss, index in finite_losses if index != best_index]
    second_best_loss = float(min(other_losses)) if other_losses else None

    loss_gap: float | None = None
    loss_gap_ratio: float | None = None
    if second_best_loss is not None and np.isfinite(best_loss):
        loss_gap = float(second_best_loss - best_loss)
        denominator = max(abs(float(best_loss)), 1e-12)
        loss_gap_ratio = float(loss_gap / denominator)

    loss_curvature = _registration_loss_curvature(losses, best_index)
    uncertainty_px: float | None = None
    if loss_curvature is not None and np.isfinite(best_loss):
        uncertainty_px = float(np.sqrt(max(float(best_loss), 1e-12) / loss_curvature))

    return {
        "second_best_loss": second_best_loss,
        "loss_gap": loss_gap,
        "loss_gap_ratio": loss_gap_ratio,
        "loss_curvature": loss_curvature,
        "uncertainty_px": uncertainty_px,
    }


def _registration_loss_curvature(
    losses: Sequence[tuple[float, float]],
    best_index: int,
) -> float | None:
    """Estimate local loss curvature around the grid-search winner."""

    if best_index <= 0 or best_index >= len(losses) - 1:
        return None
    window = losses[best_index - 1 : best_index + 2]
    offsets = np.array([offset for _loss, offset in window], dtype=np.float64)
    values = np.array([loss for loss, _offset in window], dtype=np.float64)
    if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(values)):
        return None
    if np.unique(offsets).size != offsets.size:
        return None
    quadratic, _linear, _constant = np.polyfit(offsets, values, deg=2)
    if not np.isfinite(quadratic) or quadratic <= 0:
        return None
    return float(2.0 * quadratic)

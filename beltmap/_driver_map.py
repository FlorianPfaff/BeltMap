"""Belt-map reconstruction helpers for the packaged image driver."""

from __future__ import annotations

import inspect
import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from . import _driver_runtime as rt
from ._driver_runtime import crop, emit, env_float, env_int, read_gray
from .detection import detect_particles_from_residual
from .operational_improvements import select_adaptive_map_frames
from .phase import PhaseEstimate, PhaseRegistrationConfig, refine_phase_by_registration, render_belt_view
from .residual import ResidualConfig, ResidualImage, generate_residual_image
from .tracking import ParticleComponentConfig, extract_particle_detections

MAP_PARTICLE_MASK_MODES = {"positive", "negative", "absolute", "hysteresis_abs"}
MAP_AGGREGATION_METHODS = {"mean", "huber"}
MAP_SAMPLING_STRATEGIES = {"uniform", "adaptive_phase_coverage"}
MAP_SAMPLING_STRATEGY_ENV = "MAP_SAMPLING_STRATEGY"
MAP_RECONSTRUCTION_TRIM_FRACTION_ENV = "MAP_RECONSTRUCTION_TRIM_FRACTION"
PHASE_REFINEMENT_FIELDS = [
    "iteration", "frame_index", "predicted_phase_px", "raw_correction_px",
    "smoothed_correction_px", "refined_phase_px", "loss", "score",
    "used_for_refinement", "rejection_reason",
]
_IMPORT_UNCHECKED = object()
_IMPORT_MISSING = object()
_SCIPY_NDIMAGE: Any = _IMPORT_UNCHECKED
_SKIMAGE_MEASURE: Any = _IMPORT_UNCHECKED
_SKIMAGE_MORPHOLOGY: Any = _IMPORT_UNCHECKED


@dataclass(frozen=True)
class PhaseFeedbackConfig:
    """Settings for rebuilding the belt map from registered phase corrections."""

    iterations: int = 0
    min_score: float = 0.0
    max_abs_correction_px: float | None = None
    smoothing_window_frames: int = 25
    registration_config: PhaseRegistrationConfig = field(default_factory=PhaseRegistrationConfig)


@dataclass(frozen=True)
class BeltMapBuildResult:
    """Result and diagnostics from a belt-map build."""

    belt_map: np.ndarray
    reference_phase: float
    map_height: int
    phase_refinement_rows: list[dict]
    phase_by_frame: np.ndarray | None = None


def belt_phase(frame_index: int, velocity: float, reference_phase: float, period: float | None) -> float:
    phase = reference_phase - velocity * frame_index
    return phase % period if period else phase


def map_geometry(frame_count: int, crop_height: int, velocity: float, supplied_period: int | None) -> tuple[int, float, float | None]:
    if supplied_period is not None and supplied_period <= 0:
        raise ValueError("supplied_period must be positive when set")
    if supplied_period:
        return supplied_period, 0.0, float(supplied_period)
    phases = -velocity * np.arange(frame_count, dtype=np.float64)
    reference_phase = -float(np.min(phases))
    map_height = int(math.ceil(float(np.max(phases) - np.min(phases)) + crop_height + 2))
    return max(map_height, crop_height), reference_phase, None


def sample_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    sample_count = max(1, min(frame_count, sample_count))
    return sorted(set(int(i) for i in np.linspace(0, frame_count - 1, sample_count)))


def validate_map_sampling_strategy(strategy: str) -> str:
    """Return a normalized map-frame sampling strategy."""

    normalized = strategy.strip().lower().replace("-", "_")
    aliases = {
        "adaptive": "adaptive_phase_coverage",
        "adaptive_phase": "adaptive_phase_coverage",
        "phase_coverage": "adaptive_phase_coverage",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in MAP_SAMPLING_STRATEGIES:
        choices = ", ".join(sorted(MAP_SAMPLING_STRATEGIES))
        raise ValueError(f"{MAP_SAMPLING_STRATEGY_ENV} must be one of {choices}, got {strategy!r}")
    return normalized


def validate_map_particle_mask_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in MAP_PARTICLE_MASK_MODES:
        choices = ", ".join(sorted(MAP_PARTICLE_MASK_MODES))
        raise ValueError(f"MAP_PARTICLE_MASK_MODE must be one of {choices}, got {mode!r}")
    return normalized


def validate_map_aggregation(method: str) -> str:
    normalized = method.strip().lower()
    if normalized not in MAP_AGGREGATION_METHODS:
        choices = ", ".join(sorted(MAP_AGGREGATION_METHODS))
        raise ValueError(f"MAP_AGGREGATION must be one of {choices}, got {method!r}")
    return normalized


def validate_map_trim_fraction(trim_fraction: float) -> float:
    """Validate the symmetric trim fraction used for robust belt-map means."""

    value = float(trim_fraction)
    if not np.isfinite(value) or not 0.0 <= value < 0.5:
        raise ValueError(f"{MAP_RECONSTRUCTION_TRIM_FRACTION_ENV} must be in [0, 0.5), got {trim_fraction!r}")
    return value


def expanded_detection_mask(detections: list, shape: tuple[int, int], *, margin_px: int) -> np.ndarray:
    if margin_px < 0:
        raise ValueError("margin_px must be non-negative")
    mask = np.zeros(shape, dtype=bool)
    height, width = shape
    for detection in detections:
        top = max(0, detection.bbox_top - margin_px)
        left = max(0, detection.bbox_left - margin_px)
        bottom = min(height, detection.bbox_bottom + margin_px)
        right = min(width, detection.bbox_right + margin_px)
        mask[top:bottom, left:right] = True
    return mask


def build_belt_map(
    paths: list,
    region: tuple[int, int, int, int],
    velocity: float,
    supplied_period: int | None,
    *,
    mask_iterations: int = 0,
    mask_threshold: float = 5.0,
    mask_mode: str = "positive",
    mask_grow_threshold: float = 2.0,
    mask_dilation_px: int = 0,
    mask_margin_px: int = 8,
    mask_min_area_px: int = 4,
    aggregation: str = "mean",
    robust_iterations: int = 1,
    robust_huber_delta: float = 3.0,
    robust_min_scale: float = 1.0,
    sampling_strategy: str = "uniform",
    map_trim_fraction: float | None = None,
    fractional_splat: bool = True,
    phase_feedback_config: PhaseFeedbackConfig | None = None,
) -> tuple[np.ndarray, float, int]:
    result = build_belt_map_result(
        paths=paths,
        region=region,
        velocity=velocity,
        supplied_period=supplied_period,
        mask_iterations=mask_iterations,
        mask_threshold=mask_threshold,
        mask_mode=mask_mode,
        mask_grow_threshold=mask_grow_threshold,
        mask_dilation_px=mask_dilation_px,
        mask_margin_px=mask_margin_px,
        mask_min_area_px=mask_min_area_px,
        aggregation=aggregation,
        robust_iterations=robust_iterations,
        robust_huber_delta=robust_huber_delta,
        robust_min_scale=robust_min_scale,
        sampling_strategy=sampling_strategy,
        map_trim_fraction=map_trim_fraction,
        fractional_splat=fractional_splat,
        phase_feedback_config=phase_feedback_config,
    )
    if result.phase_refinement_rows:
        rt.write_csv(rt.OUT / "phase_refinement.csv", result.phase_refinement_rows, PHASE_REFINEMENT_FIELDS)
        emit(
            "belt_map",
            "wrote phase-feedback refinement diagnostics",
            phase_refinement_csv=rt.OUT / "phase_refinement.csv",
            phase_refinement_rows=len(result.phase_refinement_rows),
        )
    return result.belt_map, result.reference_phase, result.map_height


def build_belt_map_result(
    *,
    paths: list,
    region: tuple[int, int, int, int],
    velocity: float,
    supplied_period: int | None,
    mask_iterations: int = 0,
    mask_threshold: float = 5.0,
    mask_mode: str = "positive",
    mask_grow_threshold: float = 2.0,
    mask_dilation_px: int = 0,
    mask_margin_px: int = 8,
    mask_min_area_px: int = 4,
    aggregation: str = "mean",
    robust_iterations: int = 1,
    robust_huber_delta: float = 3.0,
    robust_min_scale: float = 1.0,
    sampling_strategy: str | None = None,
    map_trim_fraction: float | None = None,
    fractional_splat: bool = True,
    phase_feedback_config: PhaseFeedbackConfig | None = None,
) -> BeltMapBuildResult:
    if not paths:
        raise ValueError("paths must contain at least one image")
    mask_mode = validate_map_particle_mask_mode(mask_mode)
    aggregation = validate_map_aggregation(aggregation)
    sampling_strategy = validate_map_sampling_strategy(
        os.getenv(MAP_SAMPLING_STRATEGY_ENV, "uniform")
        if sampling_strategy is None else sampling_strategy
    )
    if mask_grow_threshold < 0:
        raise ValueError("mask_grow_threshold must be non-negative")
    if mask_dilation_px < 0:
        raise ValueError("mask_dilation_px must be non-negative")
    if robust_iterations < 0:
        raise ValueError("robust_iterations must be non-negative")
    if robust_huber_delta <= 0:
        raise ValueError("robust_huber_delta must be positive")
    if robust_min_scale <= 0:
        raise ValueError("robust_min_scale must be positive")
    cfg = _validate_phase_feedback_config(
        phase_feedback_config if phase_feedback_config is not None else _env_phase_feedback_config()
    )
    map_trim_fraction = validate_map_trim_fraction(
        env_float(MAP_RECONSTRUCTION_TRIM_FRACTION_ENV, 0.0, minimum=0.0)
        if map_trim_fraction is None
        else map_trim_fraction
    )
    _, _, crop_height, crop_width = region
    max_samples = env_int("MAP_SAMPLE_FRAMES", 120, minimum=1)
    map_height, reference_phase, model_period = map_geometry(len(paths), crop_height, velocity, supplied_period)
    samples = select_map_sample_indices(
        frame_count=len(paths),
        sample_count=max_samples,
        velocity=velocity,
        reference_phase=reference_phase,
        model_period=model_period,
        map_height=map_height,
        crop_height=crop_height,
        sampling_strategy=sampling_strategy,
    )
    emit(
        "belt_map",
        "building clean belt map",
        sampled_frames=len(samples),
        selected_frames=len(paths),
        crop_height=crop_height,
        crop_width=crop_width,
        sampling_strategy=sampling_strategy,
        map_height=map_height,
        mask_iterations=mask_iterations,
        mask_threshold=mask_threshold,
        mask_mode=mask_mode,
        mask_grow_threshold=mask_grow_threshold,
        mask_dilation_px=mask_dilation_px,
        mask_margin_px=mask_margin_px,
        mask_min_area_px=mask_min_area_px,
        aggregation=aggregation,
        robust_iterations=robust_iterations if aggregation == "huber" else 0,
        robust_huber_delta=robust_huber_delta,
        robust_min_scale=robust_min_scale,
        fractional_splat=fractional_splat,
        phase_refinement_iterations=cfg.iterations,
        phase_refinement_smoothing_window_frames=cfg.smoothing_window_frames,
    )
    belt_map, _coverage = accumulate_belt_map(
        paths=paths,
        samples=samples,
        region=region,
        velocity=velocity,
        reference_phase=reference_phase,
        model_period=model_period,
        map_height=map_height,
        previous_belt_map=None,
        mask_threshold=mask_threshold,
        mask_mode=mask_mode,
        mask_grow_threshold=mask_grow_threshold,
        mask_dilation_px=mask_dilation_px,
        mask_margin_px=mask_margin_px,
        mask_min_area_px=mask_min_area_px,
        map_trim_fraction=map_trim_fraction,
        fractional_splat=fractional_splat,
        pass_label="initial",
    )
    phase_by_frame: np.ndarray | None = None
    phase_rows: list[dict] = []
    if cfg.iterations > 0:
        phase_by_frame = _constant_model_phases(
            frame_count=len(paths),
            velocity=velocity,
            reference_phase=reference_phase,
            model_period=model_period,
        )
        for iteration in range(1, cfg.iterations + 1):
            phase_by_frame, rows = refine_phase_feedback(
                paths=paths,
                samples=samples,
                region=region,
                belt_map=belt_map,
                predicted_phase_by_frame=phase_by_frame,
                model_period=model_period,
                config=cfg,
                iteration=iteration,
            )
            phase_rows.extend(rows)
            belt_map, coverage = accumulate_belt_map(
                paths=paths,
                samples=samples,
                region=region,
                velocity=velocity,
                reference_phase=reference_phase,
                model_period=model_period,
                map_height=map_height,
                previous_belt_map=None,
                mask_threshold=mask_threshold,
                mask_mode=mask_mode,
                mask_grow_threshold=mask_grow_threshold,
                mask_dilation_px=mask_dilation_px,
                mask_margin_px=mask_margin_px,
                mask_min_area_px=mask_min_area_px,
                map_trim_fraction=map_trim_fraction,
                fractional_splat=fractional_splat,
                pass_label=f"phase-refined-{iteration}",
                phase_by_frame=phase_by_frame,
            )
            used = sum(1 for row in rows if row["used_for_refinement"])
            emit(
                "belt_map",
                f"completed phase-feedback map refinement iteration {iteration}/{cfg.iterations}",
                used_phase_corrections=used,
                candidate_phase_corrections=len(rows),
                observed_pixels=coverage["observed_pixels"],
                total_pixels=coverage["total_pixels"],
            )
    for iteration in range(1, mask_iterations + 1):
        belt_map, coverage = accumulate_belt_map(
            paths=paths,
            samples=samples,
            region=region,
            velocity=velocity,
            reference_phase=reference_phase,
            model_period=model_period,
            map_height=map_height,
            previous_belt_map=belt_map,
            mask_threshold=mask_threshold,
            mask_mode=mask_mode,
            mask_grow_threshold=mask_grow_threshold,
            mask_dilation_px=mask_dilation_px,
            mask_margin_px=mask_margin_px,
            mask_min_area_px=mask_min_area_px,
            map_trim_fraction=map_trim_fraction,
            fractional_splat=fractional_splat,
            pass_label=f"masked-{iteration}",
            phase_by_frame=phase_by_frame,
        )
        emit(
            "belt_map",
            f"completed particle-masked map iteration {iteration}/{mask_iterations}",
            masked_pixels=coverage["masked_pixels"],
            contributed_pixels=coverage["contributed_pixels"],
            observed_pixels=coverage["observed_pixels"],
            total_pixels=coverage["total_pixels"],
        )
    if aggregation == "huber" and robust_iterations > 0:
        for iteration in range(1, robust_iterations + 1):
            belt_map, coverage = accumulate_belt_map(
                paths=paths,
                samples=samples,
                region=region,
                velocity=velocity,
                reference_phase=reference_phase,
                model_period=model_period,
                map_height=map_height,
                previous_belt_map=belt_map,
                mask_threshold=mask_threshold,
                mask_mode=mask_mode,
                mask_grow_threshold=mask_grow_threshold,
                mask_dilation_px=mask_dilation_px,
                mask_margin_px=mask_margin_px,
                mask_min_area_px=mask_min_area_px,
                pass_label=f"huber-{iteration}",
                map_trim_fraction=map_trim_fraction,
                fractional_splat=fractional_splat,
                phase_by_frame=phase_by_frame,
                robust_reference_belt_map=belt_map,
                robust_huber_delta=robust_huber_delta,
                robust_min_scale=robust_min_scale,
            )
            emit(
                "belt_map",
                f"completed robust Huber map refinement {iteration}/{robust_iterations}",
                masked_pixels=coverage["masked_pixels"],
                contributed_pixels=coverage["contributed_pixels"],
                observed_pixels=coverage["observed_pixels"],
                total_pixels=coverage["total_pixels"],
                robust_huber_delta=robust_huber_delta,
                robust_min_scale=robust_min_scale,
            )
    return BeltMapBuildResult(
        belt_map=belt_map,
        reference_phase=reference_phase,
        map_height=map_height,
        phase_refinement_rows=phase_rows,
        phase_by_frame=phase_by_frame,
    )


def accumulate_belt_map(
    *,
    paths: list,
    samples: list[int],
    region: tuple[int, int, int, int],
    velocity: float,
    reference_phase: float,
    model_period: float | None,
    map_height: int,
    previous_belt_map: np.ndarray | None,
    mask_threshold: float,
    mask_mode: str,
    mask_grow_threshold: float,
    mask_dilation_px: int,
    mask_margin_px: int,
    mask_min_area_px: int,
    pass_label: str,
    map_trim_fraction: float = 0.0,
    fractional_splat: bool = True,
    phase_by_frame: Sequence[float] | Mapping[int, float] | None = None,
    robust_reference_belt_map: np.ndarray | None = None,
    robust_huber_delta: float = 3.0,
    robust_min_scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, int]]:
    _, _, crop_height, crop_width = region
    progress_interval = env_int("PROGRESS_INTERVAL_FRAMES", 25, minimum=1)
    use_particle_mask = previous_belt_map is not None
    residual_config = ResidualConfig(
        noise_exclusion_mode=(
            "absolute" if mask_mode in {"absolute", "hysteresis_abs"} else mask_mode
        ),
    )
    use_huber_weights = robust_reference_belt_map is not None
    if use_huber_weights and robust_huber_delta <= 0:
        raise ValueError("robust_huber_delta must be positive")
    if use_huber_weights and robust_min_scale <= 0:
        raise ValueError("robust_min_scale must be positive")
    if not 0.0 <= map_trim_fraction < 0.5:
        raise ValueError(
            f"{MAP_RECONSTRUCTION_TRIM_FRACTION_ENV} must be in [0, 0.5), "
            f"got {map_trim_fraction!r}"
        )
    use_trimmed_mean = map_trim_fraction > 0.0 and not use_huber_weights
    sums = np.zeros((map_height, crop_width), dtype=np.float64)
    weights = np.zeros((map_height, crop_width), dtype=np.float64)
    stacked_values: list[np.ndarray] = []
    stacked_weights: list[np.ndarray] = []
    masked_pixels = 0
    contributed_pixels = 0
    for sample_number, index in enumerate(samples, start=1):
        frame = crop(read_gray(paths[index]), region).astype(np.float64, copy=False)
        phase = _phase_for_frame(
            index,
            velocity=velocity,
            reference_phase=reference_phase,
            model_period=model_period,
            phase_by_frame=phase_by_frame,
        )
        valid = np.ones(frame.shape, dtype=bool)
        expected = None
        if use_particle_mask:
            expected = render_belt_view(previous_belt_map, phase, crop_height)
            residual = generate_residual_image(frame, expected, config=residual_config)
            particle_mask = detect_map_particle_mask(
                residual,
                mode=mask_mode,
                threshold=mask_threshold,
                grow_threshold=mask_grow_threshold,
                dilation_px=mask_dilation_px,
                margin_px=mask_margin_px,
                min_area_px=mask_min_area_px,
            )
            valid &= ~particle_mask
            masked_pixels += int(np.count_nonzero(particle_mask))
        pixel_weights = None
        if use_huber_weights:
            if expected is None:
                expected = render_belt_view(
                    robust_reference_belt_map,
                    phase,
                    crop_height,
                )
            raw_residual = frame - expected
            finite_valid = valid & np.isfinite(raw_residual)
            pixel_weights = np.zeros(frame.shape, dtype=np.float64)
            if np.any(finite_valid):
                center = float(np.median(raw_residual[finite_valid]))
                scale = robust_residual_scale(
                    raw_residual,
                    finite_valid,
                    min_scale=robust_min_scale,
                )
                cutoff = robust_huber_delta * scale
                centered_abs = np.abs(raw_residual - center)
                pixel_weights[finite_valid] = 1.0
                outliers = finite_valid & (centered_abs > cutoff)
                pixel_weights[outliers] = cutoff / centered_abs[outliers]
                valid &= pixel_weights > 0
        frame_sums = sums if not use_trimmed_mean else np.zeros_like(sums)
        frame_weights = weights if not use_trimmed_mean else np.zeros_like(weights)
        if fractional_splat:
            contributed_pixels += _accumulate_frame_linear(
                sums=frame_sums,
                weights=frame_weights,
                frame=frame,
                valid=valid,
                phase=phase,
                map_height=map_height,
                model_period=model_period,
                pixel_weights=pixel_weights,
            )
        else:
            contributed_pixels += _accumulate_frame_nearest(
                sums=frame_sums,
                weights=frame_weights,
                frame=frame,
                valid=valid,
                phase=phase,
                map_height=map_height,
                model_period=model_period,
                pixel_weights=pixel_weights,
            )
        if use_trimmed_mean:
            stacked_values.append(frame_sums)
            stacked_weights.append(frame_weights)
            sums += frame_sums
            weights += frame_weights
        if sample_number == 1 or sample_number == len(samples) or sample_number % progress_interval == 0:
            emit(
                "belt_map",
                f"accumulated {sample_number}/{len(samples)} sampled frames",
                pass_label=pass_label,
                source_frame_index=index,
                observed_pixels=int(np.count_nonzero(weights)),
                masked_pixels=masked_pixels,
            )
    known_pixels = weights > 0
    if not np.any(known_pixels):
        raise RuntimeError("No pixels contributed to the belt map")
    emit(
        "belt_map",
        "interpolating unobserved belt-map pixels",
        pass_label=pass_label,
        observed_pixels=int(np.count_nonzero(known_pixels)),
        total_pixels=int(weights.size),
        masked_pixels=masked_pixels,
    )
    belt_map = np.empty_like(sums, dtype=np.float32)
    x = np.arange(map_height, dtype=np.float64)
    total_weight = float(np.sum(weights))
    if total_weight <= 0:
        raise RuntimeError("No pixels contributed to the belt map")
    if use_trimmed_mean:
        values_by_sample = np.stack(stacked_values, axis=0)
        weights_by_sample = np.stack(stacked_weights, axis=0)
        sums, weights = _trim_belt_map_accumulators(
            values_by_sample,
            weights_by_sample,
            trim_fraction=map_trim_fraction,
        )
        known_pixels = weights > 0
        total_weight = float(np.sum(weights))
        if total_weight <= 0:
            raise RuntimeError("No pixels contributed to the belt map after trimming")
    global_mean = float(np.sum(sums) / total_weight)
    for col in range(crop_width):
        known = np.flatnonzero(known_pixels[:, col])
        if known.size == 0:
            belt_map[:, col] = global_mean
            continue
        values = sums[known, col] / weights[known, col]
        if model_period and known.size > 1:
            xp = np.r_[known - map_height, known, known + map_height].astype(np.float64)
            belt_map[:, col] = np.interp(x, xp, np.r_[values, values, values]).astype(np.float32)
        elif known.size == 1:
            belt_map[:, col] = float(values[0])
        else:
            belt_map[:, col] = np.interp(x, known.astype(np.float64), values).astype(np.float32)
    return belt_map, {
        "masked_pixels": masked_pixels,
        "contributed_pixels": contributed_pixels,
        "observed_pixels": int(np.count_nonzero(known_pixels)),
        "total_pixels": int(weights.size),
    }


def _trim_belt_map_accumulators(
    values_by_sample: np.ndarray,
    weights_by_sample: np.ndarray,
    *,
    trim_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    _sample_count, map_height, map_width = values_by_sample.shape
    sums = np.zeros((map_height, map_width), dtype=np.float64)
    weights = np.zeros((map_height, map_width), dtype=np.float64)
    for row in range(map_height):
        for col in range(map_width):
            positive = weights_by_sample[:, row, col] > 0
            if not np.any(positive):
                continue
            values = (
                values_by_sample[positive, row, col]
                / weights_by_sample[positive, row, col]
            )
            sample_weights = weights_by_sample[positive, row, col]
            keep = _trim_sample_mask(values, trim_fraction=trim_fraction)
            if not np.any(keep):
                continue
            sums[row, col] = float(np.sum(values[keep] * sample_weights[keep]))
            weights[row, col] = float(np.sum(sample_weights[keep]))
    return sums, weights


def _trim_sample_mask(values: np.ndarray, *, trim_fraction: float) -> np.ndarray:
    if trim_fraction <= 0.0 or values.size <= 1:
        return np.ones(values.shape, dtype=bool)
    sorted_order = np.argsort(values)
    trim_count = int(np.floor(trim_fraction * values.size))
    if trim_count <= 0:
        return np.ones(values.shape, dtype=bool)
    keep_sorted = sorted_order[trim_count : values.size - trim_count]
    keep = np.zeros(values.shape, dtype=bool)
    keep[keep_sorted] = True
    return keep


def _accumulate_frame_linear(
    *,
    sums: np.ndarray,
    weights: np.ndarray,
    frame: np.ndarray,
    valid: np.ndarray,
    phase: float,
    map_height: int,
    model_period: float | None,
    pixel_weights: np.ndarray | None = None,
) -> int:
    """Accumulate one frame with the same linear row model used for rendering."""

    if sums.shape != weights.shape:
        raise ValueError("sums and weights must have the same shape")
    if sums.shape[0] != map_height:
        raise ValueError("map_height must match the accumulator height")
    if frame.ndim != 2:
        raise ValueError("frame must be a 2-D array")
    if valid.shape != frame.shape:
        raise ValueError("valid must have the same shape as frame")
    if frame.shape[1] != sums.shape[1]:
        raise ValueError("frame width must match the accumulator width")
    if pixel_weights is not None and pixel_weights.shape != frame.shape:
        raise ValueError("pixel_weights must have the same shape as frame")

    rows = np.arange(frame.shape[0], dtype=np.float64) + float(phase)
    if model_period:
        rows = np.mod(rows, map_height)
    else:
        rows = np.clip(rows, 0.0, float(map_height - 1))
    row0 = np.floor(rows).astype(np.int64)
    if model_period:
        row1 = (row0 + 1) % map_height
    else:
        row1 = np.minimum(row0 + 1, map_height - 1)
    row1_weight = rows - row0
    row0_weight = 1.0 - row1_weight

    contributed_pixels = 0
    for y in range(frame.shape[0]):
        valid_cols = valid[y]
        pixel_count = int(np.count_nonzero(valid_cols))
        if pixel_count == 0:
            continue
        values = frame[y, valid_cols]
        sample_weights = (
            np.ones(values.shape, dtype=np.float64)
            if pixel_weights is None
            else pixel_weights[y, valid_cols].astype(np.float64, copy=False)
        )
        positive = sample_weights > 0
        if not np.any(positive):
            continue
        values = values[positive]
        sample_weights = sample_weights[positive]
        target_cols = np.flatnonzero(valid_cols)[positive]
        weight0 = float(row0_weight[y])
        weight1 = float(row1_weight[y])
        if weight0 > 0.0:
            sums[row0[y], target_cols] += weight0 * sample_weights * values
            weights[row0[y], target_cols] += weight0 * sample_weights
        if weight1 > 0.0:
            sums[row1[y], target_cols] += weight1 * sample_weights * values
            weights[row1[y], target_cols] += weight1 * sample_weights
        contributed_pixels += int(np.count_nonzero(positive))
    return contributed_pixels


def _accumulate_frame_nearest(
    *,
    sums: np.ndarray,
    weights: np.ndarray,
    frame: np.ndarray,
    valid: np.ndarray,
    phase: float,
    map_height: int,
    model_period: float | None,
    pixel_weights: np.ndarray | None = None,
) -> int:
    """Accumulate one frame by assigning each image row to the nearest map row."""

    if sums.shape != weights.shape:
        raise ValueError("sums and weights must have the same shape")
    if sums.shape[0] != map_height:
        raise ValueError("map_height must match the accumulator height")
    if frame.ndim != 2:
        raise ValueError("frame must be a 2-D array")
    if valid.shape != frame.shape:
        raise ValueError("valid must have the same shape as frame")
    if frame.shape[1] != sums.shape[1]:
        raise ValueError("frame width must match the accumulator width")
    if pixel_weights is not None and pixel_weights.shape != frame.shape:
        raise ValueError("pixel_weights must have the same shape as frame")

    rows = np.arange(frame.shape[0], dtype=np.float64) + float(phase)
    if model_period:
        rows = np.mod(rows, map_height)
    else:
        rows = np.clip(rows, 0.0, float(map_height - 1))
    target_rows = np.floor(rows + 0.5).astype(np.int64)
    if model_period:
        target_rows %= map_height
    else:
        target_rows = np.clip(target_rows, 0, map_height - 1)

    contributed_pixels = 0
    for y in range(frame.shape[0]):
        valid_cols = valid[y]
        if not np.any(valid_cols):
            continue
        values = frame[y, valid_cols]
        sample_weights = (
            np.ones(values.shape, dtype=np.float64)
            if pixel_weights is None
            else pixel_weights[y, valid_cols].astype(np.float64, copy=False)
        )
        positive = sample_weights > 0
        if not np.any(positive):
            continue
        target_cols = np.flatnonzero(valid_cols)[positive]
        sums[target_rows[y], target_cols] += sample_weights[positive] * values[positive]
        weights[target_rows[y], target_cols] += sample_weights[positive]
        contributed_pixels += int(np.count_nonzero(positive))
    return contributed_pixels


def robust_residual_scale(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    min_scale: float,
) -> float:
    values = np.asarray(residual, dtype=np.float64)[valid]
    values = values[np.isfinite(values)]
    if values.size < 2:
        return min_scale
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    if not math.isfinite(mad) or mad <= 0:
        return min_scale
    return max(1.4826 * mad, min_scale)


def detect_map_particle_mask(
    residual: ResidualImage,
    *,
    mode: str,
    threshold: float,
    grow_threshold: float,
    dilation_px: int,
    margin_px: int,
    min_area_px: int,
) -> np.ndarray:
    mode = validate_map_particle_mask_mode(mode)
    if mode in {"positive", "negative"}:
        raw_mask = detect_particles_from_residual(
            residual,
            threshold=threshold,
            mode=mode,
        )
        return _component_mask_with_optional_dilation(
            raw_mask,
            residual=residual,
            min_area_px=min_area_px,
            margin_px=margin_px,
            dilation_px=dilation_px,
        )
    values = np.asarray(residual.normalized, dtype=np.float64)
    valid = np.asarray(residual.mask, dtype=bool) & np.isfinite(values)
    abs_values = np.abs(values)
    if mode == "absolute":
        raw_mask = detect_particles_from_residual(
            residual,
            threshold=threshold,
            mode="absolute",
        )
        return _component_mask_with_optional_dilation(
            raw_mask,
            residual=residual,
            min_area_px=min_area_px,
            margin_px=margin_px,
            dilation_px=dilation_px,
        )
    if grow_threshold > threshold:
        raise ValueError(
            "grow_threshold must be less than or equal to threshold for hysteresis_abs map masks"
        )
    seed_mask = valid & (abs_values >= threshold)
    grow_mask = valid & (abs_values >= grow_threshold)
    if not np.any(seed_mask) or not np.any(grow_mask):
        return np.zeros(values.shape, dtype=bool)
    labels, component_count = _label_components(grow_mask)
    if component_count == 0:
        return np.zeros(values.shape, dtype=bool)
    seed_labels = np.unique(labels[seed_mask])
    seed_labels = seed_labels[seed_labels != 0]
    if seed_labels.size == 0:
        return np.zeros(values.shape, dtype=bool)
    particle_mask = np.isin(labels, seed_labels)
    particle_mask = _morphological_cleanup(particle_mask, min_area_px=min_area_px, dilation_px=dilation_px)
    if margin_px > 0:
        particle_mask = _component_bbox_mask(particle_mask, residual=residual, min_area_px=1, margin_px=margin_px)
    return particle_mask


def select_map_sample_indices(
    *,
    frame_count: int,
    sample_count: int,
    velocity: float,
    reference_phase: float,
    model_period: float | None,
    map_height: int,
    crop_height: int,
    sampling_strategy: str,
) -> list[int]:
    """Select source frames for map reconstruction.

    ``uniform`` preserves the original linspace sampling.  The adaptive strategy
    uses nominal belt phases to spread samples across belt-coordinate coverage,
    which is useful when a periodic run overrepresents some belt phases or when
    the requested sample budget is smaller than the available sequence.
    """

    strategy = validate_map_sampling_strategy(sampling_strategy)
    if strategy == "uniform":
        return sample_indices(frame_count, sample_count)

    phases = _constant_model_phases(
        frame_count=frame_count,
        velocity=velocity,
        reference_phase=reference_phase,
        model_period=model_period,
    )
    selected = select_adaptive_map_frames(
        phases,
        map_height_px=map_height,
        sample_count=max(1, min(frame_count, sample_count)),
        crop_height_px=max(1, crop_height),
    )
    return sorted({sample.frame_index for sample in selected})


def refine_phase_feedback(
    *,
    paths: list,
    samples: list[int],
    region: tuple[int, int, int, int],
    belt_map: np.ndarray,
    predicted_phase_by_frame: Sequence[float],
    model_period: float | None,
    config: PhaseFeedbackConfig,
    iteration: int,
) -> tuple[np.ndarray, list[dict]]:
    used_corrections = np.full(len(paths), np.nan, dtype=np.float64)
    rows: list[dict] = []
    progress_interval = env_int("PROGRESS_INTERVAL_FRAMES", 25, minimum=1)
    emit(
        "belt_map",
        "estimating phase-feedback corrections",
        iteration=iteration,
        sampled_frames=len(samples),
        min_score=config.min_score,
        max_abs_correction_px=config.max_abs_correction_px,
        smoothing_window_frames=config.smoothing_window_frames,
    )
    for sample_number, index in enumerate(samples, start=1):
        frame = crop(read_gray(paths[index]), region).astype(np.float64, copy=False)
        predicted_phase = float(predicted_phase_by_frame[index])
        estimate = refine_phase_by_registration(
            frame=frame,
            belt_map=belt_map,
            predicted_phase_px=predicted_phase,
            frame_index=float(index),
            period_px=model_period,
            config=config.registration_config,
        )
        correction = float(estimate.correction_px)
        used, reason = _use_phase_feedback_estimate(estimate, config)
        if used:
            used_corrections[index] = correction
        rows.append(
            {
                "iteration": iteration,
                "frame_index": index,
                "predicted_phase_px": predicted_phase,
                "raw_correction_px": correction,
                "smoothed_correction_px": 0.0,
                "refined_phase_px": predicted_phase,
                "loss": "" if estimate.loss is None else estimate.loss,
                "score": "" if estimate.score is None else estimate.score,
                "used_for_refinement": used,
                "rejection_reason": reason,
            }
        )
        if sample_number == 1 or sample_number == len(samples) or sample_number % progress_interval == 0:
            emit(
                "belt_map",
                f"registered {sample_number}/{len(samples)} phase-feedback frames",
                iteration=iteration,
                source_frame_index=index,
                current_correction_px=correction,
                used_for_refinement=used,
            )
    smoothed_corrections = smooth_phase_corrections(
        frame_count=len(paths),
        correction_by_frame=used_corrections,
        smoothing_window_frames=config.smoothing_window_frames,
    )
    refined = np.asarray(predicted_phase_by_frame, dtype=np.float64) + smoothed_corrections
    if model_period is not None:
        refined = np.mod(refined, model_period)
    for row in rows:
        index = int(row["frame_index"])
        row["smoothed_correction_px"] = float(smoothed_corrections[index])
        row["refined_phase_px"] = float(refined[index])
    return refined, rows


def smooth_phase_corrections(*, frame_count: int, correction_by_frame: Sequence[float], smoothing_window_frames: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    corrections = np.asarray(correction_by_frame, dtype=np.float64)
    if corrections.shape[0] != frame_count:
        raise ValueError("correction_by_frame must have one value per frame")
    valid = np.isfinite(corrections)
    if not np.any(valid):
        return np.zeros(frame_count, dtype=np.float64)
    frames = np.arange(frame_count, dtype=np.float64)
    interpolated = np.interp(frames, frames[valid], corrections[valid])
    if smoothing_window_frames <= 1:
        return interpolated.astype(np.float64, copy=False)
    half_window = max(1, int(smoothing_window_frames) // 2)
    smoothed = np.empty(frame_count, dtype=np.float64)
    for index in range(frame_count):
        start = max(0, index - half_window)
        stop = min(frame_count, index + half_window + 1)
        smoothed[index] = float(np.median(interpolated[start:stop]))
    return smoothed


def _component_mask_with_optional_dilation(
    raw_mask: np.ndarray,
    *,
    residual: ResidualImage,
    min_area_px: int,
    margin_px: int,
    dilation_px: int,
) -> np.ndarray:
    if dilation_px > 0:
        cleaned = _morphological_cleanup(
            raw_mask,
            min_area_px=min_area_px,
            dilation_px=dilation_px,
        )
        return _component_bbox_mask(cleaned, residual=residual, min_area_px=1, margin_px=margin_px)
    return _component_bbox_mask(raw_mask, residual=residual, min_area_px=min_area_px, margin_px=margin_px)


def _component_bbox_mask(raw_mask: np.ndarray, *, residual: ResidualImage, min_area_px: int, margin_px: int) -> np.ndarray:
    if not np.any(raw_mask):
        return np.zeros(raw_mask.shape, dtype=bool)
    component_config = ParticleComponentConfig(min_area_px=min_area_px, weighted_centroid=False)
    detections = extract_particle_detections(raw_mask, residual=residual, frame_index=0.0, config=component_config)
    return expanded_detection_mask(detections, raw_mask.shape, margin_px=margin_px)


def _morphological_cleanup(mask: np.ndarray, *, min_area_px: int, dilation_px: int) -> np.ndarray:
    cleaned = np.asarray(mask, dtype=bool)
    if not np.any(cleaned):
        return cleaned
    morphology = _load_skimage_morphology()
    if morphology is not None:
        min_area = max(1, int(min_area_px))
        if min_area > 1:
            cleaned = _remove_small_objects_skimage(morphology, cleaned, min_area=min_area)
            cleaned = _remove_small_holes_skimage(morphology, cleaned, min_area=min_area)
        if dilation_px > 0:
            dilation = getattr(morphology, "dilation", morphology.binary_dilation)
            cleaned = dilation(cleaned, morphology.disk(int(dilation_px)))
        return np.asarray(cleaned, dtype=bool)
    cleaned = _remove_small_components(cleaned, min_area_px=min_area_px)
    ndimage = _load_scipy_ndimage()
    if ndimage is not None:
        cleaned = ndimage.binary_fill_holes(cleaned)
        if dilation_px > 0:
            cleaned = ndimage.binary_dilation(cleaned, structure=np.ones((3, 3), dtype=bool), iterations=int(dilation_px))
        return np.asarray(cleaned, dtype=bool)
    return _binary_dilation_numpy(cleaned, iterations=int(dilation_px))


def _remove_small_objects_skimage(morphology: Any, mask: np.ndarray, *, min_area: int) -> np.ndarray:
    parameters = inspect.signature(morphology.remove_small_objects).parameters
    if "max_size" in parameters:
        return morphology.remove_small_objects(mask, max_size=min_area - 1)
    return morphology.remove_small_objects(mask, min_size=min_area)


def _remove_small_holes_skimage(morphology: Any, mask: np.ndarray, *, min_area: int) -> np.ndarray:
    parameters = inspect.signature(morphology.remove_small_holes).parameters
    if "max_size" in parameters:
        return morphology.remove_small_holes(mask, max_size=min_area - 1)
    return morphology.remove_small_holes(mask, area_threshold=min_area)


def _remove_small_components(mask: np.ndarray, *, min_area_px: int) -> np.ndarray:
    labels, component_count = _label_components(mask)
    if component_count == 0:
        return np.zeros(mask.shape, dtype=bool)
    counts = np.bincount(labels.ravel(), minlength=component_count + 1)
    keep = counts >= max(1, int(min_area_px))
    keep[0] = False
    return keep[labels]


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    ndimage = _load_scipy_ndimage()
    if ndimage is not None:
        labels, component_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
        return np.asarray(labels, dtype=np.int64), int(component_count)
    measure = _load_skimage_measure()
    if measure is not None:
        labels, component_count = measure.label(mask, connectivity=2, background=0, return_num=True)
        return np.asarray(labels, dtype=np.int64), int(component_count)
    return _label_components_numpy(mask)


def _label_components_numpy(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=np.int64)
    height, width = mask.shape
    component_count = 0
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for start_row, start_col in np.argwhere(mask):
        row = int(start_row)
        col = int(start_col)
        if labels[row, col] != 0:
            continue
        component_count += 1
        stack = [(row, col)]
        labels[row, col] = component_count
        while stack:
            current_row, current_col = stack.pop()
            for row_offset, col_offset in offsets:
                next_row = current_row + row_offset
                next_col = current_col + col_offset
                if 0 <= next_row < height and 0 <= next_col < width and mask[next_row, next_col] and labels[next_row, next_col] == 0:
                    labels[next_row, next_col] = component_count
                    stack.append((next_row, next_col))
    return labels, component_count


def _binary_dilation_numpy(mask: np.ndarray, *, iterations: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _iteration in range(max(0, iterations)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = (
            padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:] |
            padded[1:-1, :-2] | padded[1:-1, 1:-1] | padded[1:-1, 2:] |
            padded[2:, :-2] | padded[2:, 1:-1] | padded[2:, 2:]
        )
    return result


def _constant_model_phases(*, frame_count: int, velocity: float, reference_phase: float, model_period: float | None) -> np.ndarray:
    return np.asarray([belt_phase(index, velocity, reference_phase, model_period) for index in range(frame_count)], dtype=np.float64)


def _phase_for_frame(
    index: int,
    *,
    velocity: float,
    reference_phase: float,
    model_period: float | None,
    phase_by_frame: Sequence[float] | Mapping[int, float] | None,
) -> float:
    if phase_by_frame is not None:
        try:
            return float(phase_by_frame[index])
        except (KeyError, IndexError):
            pass
    return belt_phase(index, velocity, reference_phase, model_period)


def _env_phase_feedback_config() -> PhaseFeedbackConfig:
    max_abs = env_float("PHASE_REFINEMENT_MAX_ABS_CORRECTION_PX", 0.0, minimum=0.0)
    return PhaseFeedbackConfig(
        iterations=env_int("PHASE_REFINEMENT_ITERATIONS", 0, minimum=0),
        min_score=env_float("PHASE_REFINEMENT_MIN_SCORE", 0.0, minimum=0.0),
        max_abs_correction_px=None if max_abs <= 0 else max_abs,
        smoothing_window_frames=env_int("PHASE_REFINEMENT_SMOOTHING_WINDOW_FRAMES", 25, minimum=0),
        registration_config=PhaseRegistrationConfig(
            search_radius_px=env_float("REGISTRATION_SEARCH_RADIUS_PX", 8.0, minimum=0.0),
            search_step_px=env_float("REGISTRATION_SEARCH_STEP_PX", 0.5, minimum=1e-9),
        ),
    )


def _validate_phase_feedback_config(config: PhaseFeedbackConfig) -> PhaseFeedbackConfig:
    if config.iterations < 0:
        raise ValueError("phase-feedback iterations must be non-negative")
    if config.min_score < 0:
        raise ValueError("phase-feedback min_score must be non-negative")
    if config.max_abs_correction_px is not None and config.max_abs_correction_px < 0:
        raise ValueError("phase-feedback max_abs_correction_px must be non-negative")
    if config.smoothing_window_frames < 0:
        raise ValueError("phase-feedback smoothing_window_frames must be non-negative")
    if config.registration_config.search_radius_px < 0:
        raise ValueError("phase-feedback registration search_radius_px must be non-negative")
    return config


def _use_phase_feedback_estimate(estimate: PhaseEstimate, config: PhaseFeedbackConfig) -> tuple[bool, str]:
    correction = float(estimate.correction_px)
    if not np.isfinite(correction):
        return False, "non-finite-correction"
    if estimate.loss is not None and not np.isfinite(float(estimate.loss)):
        return False, "non-finite-loss"
    if config.min_score > 0:
        if estimate.score is None or not np.isfinite(float(estimate.score)):
            return False, "missing-score"
        if float(estimate.score) < config.min_score:
            return False, "low-score"
    if config.max_abs_correction_px is not None and abs(correction) > config.max_abs_correction_px:
        return False, "large-correction"
    if _is_search_boundary_correction(correction, config.registration_config):
        return False, "search-boundary"
    return True, ""


def _is_search_boundary_correction(correction: float, config: PhaseRegistrationConfig) -> bool:
    if config.search_radius_px <= 0:
        return False
    tolerance = max(1e-9, 0.5 * config.search_step_px)
    return abs(abs(correction) - config.search_radius_px) <= tolerance


def _load_scipy_ndimage() -> Any | None:
    global _SCIPY_NDIMAGE
    if _SCIPY_NDIMAGE is _IMPORT_UNCHECKED:
        try:
            from scipy import ndimage
        except ImportError:
            _SCIPY_NDIMAGE = _IMPORT_MISSING
        else:
            _SCIPY_NDIMAGE = ndimage
    return None if _SCIPY_NDIMAGE is _IMPORT_MISSING else _SCIPY_NDIMAGE


def _load_skimage_measure() -> Any | None:
    global _SKIMAGE_MEASURE
    if _SKIMAGE_MEASURE is _IMPORT_UNCHECKED:
        try:
            from skimage import measure
        except ImportError:
            _SKIMAGE_MEASURE = _IMPORT_MISSING
        else:
            _SKIMAGE_MEASURE = measure
    return None if _SKIMAGE_MEASURE is _IMPORT_MISSING else _SKIMAGE_MEASURE


def _load_skimage_morphology() -> Any | None:
    global _SKIMAGE_MORPHOLOGY
    if _SKIMAGE_MORPHOLOGY is _IMPORT_UNCHECKED:
        try:
            from skimage import morphology
        except ImportError:
            _SKIMAGE_MORPHOLOGY = _IMPORT_MISSING
        else:
            _SKIMAGE_MORPHOLOGY = morphology
    return None if _SKIMAGE_MORPHOLOGY is _IMPORT_MISSING else _SKIMAGE_MORPHOLOGY

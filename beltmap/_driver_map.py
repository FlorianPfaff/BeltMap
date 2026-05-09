"""Belt-map reconstruction helpers for the packaged image driver."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from . import _driver_runtime as rt
from ._driver_runtime import crop, emit, env_float, env_int, read_gray
from .detection import detect_particles_from_residual
from .phase import PhaseEstimate, PhaseRegistrationConfig, refine_phase_by_registration, render_belt_view
from .residual import ResidualConfig, ResidualImage, generate_residual_image
from .tracking import ParticleComponentConfig, extract_particle_detections

MAP_PARTICLE_MASK_MODES = {"positive", "absolute", "hysteresis_abs"}
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
    if supplied_period:
        return supplied_period, 0.0, float(supplied_period)
    phases = -velocity * np.arange(frame_count, dtype=np.float64)
    reference_phase = -float(np.min(phases))
    map_height = int(math.ceil(float(np.max(phases) - np.min(phases)) + crop_height + 2))
    return max(map_height, crop_height), reference_phase, None


def sample_indices(frame_count: int, sample_count: int) -> list[int]:
    sample_count = max(1, min(frame_count, sample_count))
    return sorted(set(int(i) for i in np.linspace(0, frame_count - 1, sample_count)))


def validate_map_particle_mask_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized not in MAP_PARTICLE_MASK_MODES:
        choices = ", ".join(sorted(MAP_PARTICLE_MASK_MODES))
        raise ValueError(f"MAP_PARTICLE_MASK_MODE must be one of {choices}, got {mode!r}")
    return normalized


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
    phase_feedback_config: PhaseFeedbackConfig | None = None,
) -> BeltMapBuildResult:
    if not paths:
        raise ValueError("paths must contain at least one image")
    mask_mode = validate_map_particle_mask_mode(mask_mode)
    if mask_grow_threshold < 0:
        raise ValueError("mask_grow_threshold must be non-negative")
    if mask_dilation_px < 0:
        raise ValueError("mask_dilation_px must be non-negative")
    cfg = _validate_phase_feedback_config(
        phase_feedback_config if phase_feedback_config is not None else _env_phase_feedback_config()
    )
    _, _, crop_height, crop_width = region
    max_samples = env_int("MAP_SAMPLE_FRAMES", 120, minimum=1)
    map_height, reference_phase, model_period = map_geometry(len(paths), crop_height, velocity, supplied_period)
    samples = sample_indices(len(paths), max_samples)
    emit(
        "belt_map",
        "building clean belt map",
        sampled_frames=len(samples),
        selected_frames=len(paths),
        crop_height=crop_height,
        crop_width=crop_width,
        map_height=map_height,
        mask_iterations=mask_iterations,
        mask_threshold=mask_threshold,
        mask_mode=mask_mode,
        mask_grow_threshold=mask_grow_threshold,
        mask_dilation_px=mask_dilation_px,
        mask_margin_px=mask_margin_px,
        mask_min_area_px=mask_min_area_px,
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
    phase_by_frame: Sequence[float] | Mapping[int, float] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    _, _, crop_height, crop_width = region
    progress_interval = env_int("PROGRESS_INTERVAL_FRAMES", 25, minimum=1)
    use_particle_mask = previous_belt_map is not None
    residual_config = ResidualConfig()
    sums = np.zeros((map_height, crop_width), dtype=np.float64)
    counts = np.zeros((map_height, crop_width), dtype=np.uint16)
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
        coordinates = np.rint(np.arange(crop_height) + phase).astype(np.int64)
        coordinates = coordinates % map_height if model_period else np.clip(coordinates, 0, map_height - 1)
        for y, row in enumerate(coordinates):
            valid_cols = valid[y]
            sums[row, valid_cols] += frame[y, valid_cols]
            counts[row, valid_cols] += 1
            contributed_pixels += int(np.count_nonzero(valid_cols))
        if sample_number == 1 or sample_number == len(samples) or sample_number % progress_interval == 0:
            emit(
                "belt_map",
                f"accumulated {sample_number}/{len(samples)} sampled frames",
                pass_label=pass_label,
                source_frame_index=index,
                observed_pixels=int(np.count_nonzero(counts)),
                masked_pixels=masked_pixels,
            )
    known_pixels = counts > 0
    if not np.any(known_pixels):
        raise RuntimeError("No pixels contributed to the belt map")
    emit(
        "belt_map",
        "interpolating unobserved belt-map pixels",
        pass_label=pass_label,
        observed_pixels=int(np.count_nonzero(known_pixels)),
        total_pixels=int(counts.size),
        masked_pixels=masked_pixels,
    )
    belt_map = np.empty_like(sums, dtype=np.float32)
    x = np.arange(map_height, dtype=np.float64)
    global_mean = float(np.sum(sums) / np.sum(counts))
    for col in range(crop_width):
        known = np.flatnonzero(known_pixels[:, col])
        if known.size == 0:
            belt_map[:, col] = global_mean
            continue
        values = sums[known, col] / counts[known, col].astype(np.float64)
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
        "total_pixels": int(counts.size),
    }


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
    if mode == "positive":
        raw_mask = detect_particles_from_residual(residual, threshold=threshold)
        return _component_bbox_mask(raw_mask, residual=residual, min_area_px=min_area_px, margin_px=margin_px)
    values = np.asarray(residual.normalized, dtype=np.float64)
    valid = np.asarray(residual.mask, dtype=bool) & np.isfinite(values)
    abs_values = np.abs(values)
    if mode == "absolute":
        raw_mask = valid & (abs_values > threshold)
        return _component_bbox_mask(raw_mask, residual=residual, min_area_px=min_area_px, margin_px=margin_px)
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

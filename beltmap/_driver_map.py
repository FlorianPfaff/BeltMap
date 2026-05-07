"""Belt-map reconstruction helpers for the packaged image driver."""

from __future__ import annotations

import math

import numpy as np

from ._driver_runtime import crop, emit, env_int, read_gray
from .detection import detect_particles_from_residual
from .phase import render_belt_view
from .residual import ResidualConfig, generate_residual_image
from .tracking import ParticleComponentConfig, extract_particle_detections


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
    mask_margin_px: int = 8,
    mask_min_area_px: int = 4,
) -> tuple[np.ndarray, float, int]:
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
        mask_margin_px=mask_margin_px,
        mask_min_area_px=mask_min_area_px,
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
        mask_margin_px=mask_margin_px,
        mask_min_area_px=mask_min_area_px,
        pass_label="initial",
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
            mask_margin_px=mask_margin_px,
            mask_min_area_px=mask_min_area_px,
            pass_label=f"masked-{iteration}",
        )
        emit(
            "belt_map",
            f"completed particle-masked map iteration {iteration}/{mask_iterations}",
            masked_pixels=coverage["masked_pixels"],
            contributed_pixels=coverage["contributed_pixels"],
            observed_pixels=coverage["observed_pixels"],
            total_pixels=coverage["total_pixels"],
        )
    return belt_map, reference_phase, map_height


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
    mask_margin_px: int,
    mask_min_area_px: int,
    pass_label: str,
) -> tuple[np.ndarray, dict[str, int]]:
    _, _, crop_height, crop_width = region
    progress_interval = env_int("PROGRESS_INTERVAL_FRAMES", 25, minimum=1)
    use_particle_mask = previous_belt_map is not None
    component_config = ParticleComponentConfig(min_area_px=mask_min_area_px)
    residual_config = ResidualConfig()
    sums = np.zeros((map_height, crop_width), dtype=np.float64)
    counts = np.zeros((map_height, crop_width), dtype=np.uint16)
    masked_pixels = 0
    contributed_pixels = 0
    for sample_number, index in enumerate(samples, start=1):
        frame = crop(read_gray(paths[index]), region).astype(np.float64, copy=False)
        phase = belt_phase(index, velocity, reference_phase, model_period)
        valid = np.ones(frame.shape, dtype=bool)
        if use_particle_mask:
            expected = render_belt_view(previous_belt_map, phase, crop_height)
            residual = generate_residual_image(frame, expected, config=residual_config)
            raw_mask = detect_particles_from_residual(residual, threshold=mask_threshold)
            detections = extract_particle_detections(raw_mask, residual=residual, frame_index=float(index), config=component_config)
            particle_mask = expanded_detection_mask(detections, frame.shape, margin_px=mask_margin_px)
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
            emit("belt_map", f"accumulated {sample_number}/{len(samples)} sampled frames", pass_label=pass_label, source_frame_index=index, observed_pixels=int(np.count_nonzero(counts)), masked_pixels=masked_pixels)
    known_pixels = counts > 0
    if not np.any(known_pixels):
        raise RuntimeError("No pixels contributed to the belt map")
    emit("belt_map", "interpolating unobserved belt-map pixels", pass_label=pass_label, observed_pixels=int(np.count_nonzero(known_pixels)), total_pixels=int(counts.size), masked_pixels=masked_pixels)
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

"""Packaged image-sequence driver entry point for BeltMap."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import (
    BeltMotionModel,
    PhaseEstimate,
    ParticleComponentConfig,
    ParticleTrackingConfig,
    PhaseRegistrationConfig,
    ResidualConfig,
    ResidualImage,
    RecurrentArtifactConfig,
    TrackFilterConfig,
    detect_particles_from_residual,
    estimate_particle_velocities_vs_belt,
    estimate_local_noise,
    extract_particle_detections,
    render_clean_belt_residual,
    score_particle_velocities,
    track_particle_detections,
)
from . import _driver_runtime as rt
from ._driver_map import (
    PHASE_REFINEMENT_FIELDS,
    PhaseFeedbackConfig,
    build_belt_map_result,
    expanded_detection_mask,
    sample_indices,
)
from ._driver_motion import (
    estimate_velocity,
    parse_region,
    resolve_supplied_velocity,
    validate_auto_velocity_region,
)
from .recurrent_artifacts import (
    RECURRENT_ARTIFACT_MODES,
    belt_revolution_indices,
    build_recurrent_artifact_map,
    score_recurrent_artifact_detections,
    score_recurrent_artifact_detections_excluding_current_revolution,
)

DETECTION_FIELDS = [
    "frame_index", "image", "label", "y", "x", "area_px",
    "bbox_top", "bbox_left", "bbox_bottom", "bbox_right",
    "mean_signal", "peak_signal",
    "recurrent_artifact_overlap_fraction",
    "recurrent_artifact_probability",
    "recurrent_artifact_required_peak_signal",
]
PHASE_FIELDS = [
    "frame_index", "image", "phase_px", "phase_fraction", "phase_rad",
    "predicted_phase_px", "correction_px", "loss", "score", "method",
]
VELOCITY_FIELDS = [
    "track_id", "n_detections", "frame_start", "frame_end",
    "velocity_y_px_per_frame", "velocity_x_px_per_frame", "speed_px_per_frame",
    "belt_velocity_y_px_per_frame", "velocity_ratio_y",
    "belt_minus_particle_velocity_y_px_per_frame",
]
TRACK_DETECTION_FIELDS = [
    "track_id", "track_detection_index",
    "frame_index", "image", "label", "y", "x", "area_px",
    "bbox_top", "bbox_left", "bbox_bottom", "bbox_right",
    "mean_signal", "peak_signal",
    "recurrent_artifact_overlap_fraction",
    "recurrent_artifact_probability",
    "recurrent_artifact_required_peak_signal",
]
RECURRENT_ARTIFACT_DETECTION_FIELDS = [
    *DETECTION_FIELDS,
    "recurrent_artifact_rejected",
]
TRACK_SCORE_FIELDS = [
    "track_id", "n_detections", "frame_start", "frame_end",
    "velocity_y_px_per_frame", "velocity_x_px_per_frame",
    "velocity_ratio_y", "abs_x_velocity_px_per_frame",
    "passes_min_track_length", "passes_velocity_ratio",
    "passes_lateral_velocity", "accepted", "plausibility_score",
]


def optional_positive_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def optional_positive_float(name: str, default: float = 0.0) -> float | None:
    value = rt.env_float(name, default, minimum=0.0)
    return None if value <= 0 else value


def optional_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value) if value else None


def optional_csv_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    return None if value == "" else float(value)


def load_reuse_metadata(belt_map_path: Path) -> tuple[dict, Path | None]:
    metadata_path = belt_map_path.with_name("metadata.json")
    if not metadata_path.exists():
        return {}, None
    return json.loads(metadata_path.read_text(encoding="utf-8")), metadata_path


def _relative_image_name(path: Path, *, data_dir: Path) -> str:
    try:
        return str(path.relative_to(data_dir))
    except ValueError:
        try:
            return str(path.resolve().relative_to(data_dir.resolve()))
        except ValueError:
            return str(path)


def _normalize_phase_image_name(image: str) -> str:
    normalized = image.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def load_phase_estimates(
    path: Path,
    *,
    expected_image_paths: list[Path] | None = None,
    data_dir: Path | None = None,
) -> dict[int, PhaseEstimate]:
    estimates: dict[int, PhaseEstimate] = {}
    image_names: dict[int, str] = {}
    require_image_names = expected_image_paths is not None

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if require_image_names and (
            reader.fieldnames is None or "image" not in reader.fieldnames
        ):
            raise ValueError("phase estimates used for reuse must include an image column")
        for row in reader:
            frame_index = int(row["frame_index"])
            if frame_index in estimates:
                raise ValueError(f"duplicate phase estimate for frame {frame_index}")
            if require_image_names:
                image_name = row.get("image", "").strip()
                if not image_name:
                    raise ValueError(
                        f"phase estimate for frame {frame_index} has an empty image column"
                    )
                image_names[frame_index] = image_name
            estimates[frame_index] = PhaseEstimate(
                phase_px=float(row["phase_px"]),
                frame_index=float(row["frame_index"]),
                predicted_phase_px=float(row["predicted_phase_px"]),
                correction_px=float(row["correction_px"]),
                loss=optional_csv_float(row, "loss"),
                score=optional_csv_float(row, "score"),
                method=row.get("method", "loaded_phase_estimate") or "loaded_phase_estimate",
            )
    if not estimates:
        raise ValueError(f"no phase estimates found in {path}")
    if expected_image_paths is not None:
        validate_reused_phase_estimates(
            estimates,
            frame_count=len(expected_image_paths),
            image_names=image_names,
            paths=expected_image_paths,
            data_dir=data_dir if data_dir is not None else rt.DATA,
        )
    return estimates


def load_recurrent_artifact_map(
    path: Path,
    *,
    map_shape: tuple[int, int],
) -> np.ndarray:
    artifact_map = np.load(path)
    if artifact_map.ndim != 2:
        raise ValueError(
            "REUSE_RECURRENT_ARTIFACT_MAP_PATH must point to a 2-D recurrent "
            "artifact mask or probability .npy"
        )
    if artifact_map.shape != map_shape:
        raise ValueError(
            "reused recurrent artifact map shape does not match belt map and crop width: "
            f"{artifact_map.shape} != {map_shape}"
        )
    if artifact_map.dtype == np.bool_ or np.issubdtype(artifact_map.dtype, np.bool_):
        return np.asarray(artifact_map, dtype=bool)
    artifact_map = np.asarray(artifact_map, dtype=np.float32)
    if not np.all(np.isfinite(artifact_map)):
        raise ValueError("reused recurrent artifact probability map must be finite")
    if np.any((artifact_map < 0.0) | (artifact_map > 1.0)):
        raise ValueError(
            "reused recurrent artifact probability map values must be in [0, 1]"
        )
    return artifact_map


def validate_reused_phase_estimates(
    estimates: dict[int, PhaseEstimate],
    *,
    frame_count: int,
    image_names: dict[int, str] | None = None,
    paths: list[Path] | None = None,
    data_dir: Path | None = None,
) -> None:
    missing = [index for index in range(frame_count) if index not in estimates]
    if missing:
        preview = ", ".join(str(index) for index in missing[:8])
        raise ValueError(
            f"phase estimates are missing {len(missing)} selected frames; first missing: {preview}"
        )
    if image_names is None and paths is None:
        return
    if image_names is None or paths is None:
        raise ValueError("image_names and paths must be provided together")
    if len(paths) != frame_count:
        raise ValueError(
            "frame_count must match number of selected image paths when validating "
            "phase image names"
        )

    root = data_dir if data_dir is not None else rt.DATA
    mismatches: list[tuple[int, str, str]] = []
    for index, path in enumerate(paths):
        actual = image_names.get(index, "")
        expected = _relative_image_name(path, data_dir=root)
        if _normalize_phase_image_name(actual) != _normalize_phase_image_name(expected):
            mismatches.append((index, actual, expected))

    if mismatches:
        preview = ", ".join(
            f"{index}: {actual!r} != {expected!r}"
            for index, actual, expected in mismatches[:3]
        )
        raise ValueError(
            "phase estimates image column does not match selected image sequence; "
            f"first mismatches: {preview}"
        )


def phase_estimate_row(frame_index: int, path, residual, period_px: float) -> dict:
    if residual.clean_render is None:
        raise ValueError("phase estimates require residuals with a clean belt render")
    estimate = residual.clean_render.phase_estimate
    phase_fraction = estimate.phase_px / period_px
    return {
        "frame_index": frame_index,
        "image": str(path.relative_to(rt.DATA)),
        "phase_px": estimate.phase_px,
        "phase_fraction": phase_fraction,
        "phase_rad": phase_fraction * 2.0 * np.pi,
        "predicted_phase_px": estimate.predicted_phase_px,
        "correction_px": estimate.correction_px,
        "loss": "" if estimate.loss is None else estimate.loss,
        "score": "" if estimate.score is None else estimate.score,
        "method": estimate.method,
    }


def write_detection_outputs(detections_by_frame: list, detection_rows: list[dict]) -> None:
    rt.write_csv(rt.OUT / "detections.csv", detection_rows, DETECTION_FIELDS)
    rt.write_csv(
        rt.OUT / "detections_per_frame.csv",
        [{"frame_index": i, "n_detections": len(dets)} for i, dets in enumerate(detections_by_frame)],
        ["frame_index", "n_detections"],
    )


def write_phase_outputs(phase_rows: list[dict]) -> None:
    rt.write_csv(rt.OUT / "phase_estimates.csv", phase_rows, PHASE_FIELDS)


def write_phase_refinement_outputs(phase_refinement_rows: list[dict]) -> None:
    rt.write_csv(rt.OUT / "phase_refinement.csv", phase_refinement_rows, PHASE_REFINEMENT_FIELDS)


def detection_rows_for_frame(detections: list, path: Path, frame_index: int) -> list[dict]:
    rows: list[dict] = []
    for detection in detections:
        row = {
            field: getattr(detection, field)
            for field in DETECTION_FIELDS
            if field != "image"
        }
        row["frame_index"] = frame_index
        row["image"] = str(path.relative_to(rt.DATA))
        rows.append(row)
    return rows


def detection_rows_from_frames(detections_by_frame: list, paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for frame_index, detections in enumerate(detections_by_frame):
        rows.extend(detection_rows_for_frame(detections, paths[frame_index], frame_index))
    return rows


def recurrent_artifact_rows_from_scores(
    scored_by_frame: list,
    paths: list[Path],
) -> list[dict]:
    rows: list[dict] = []
    for frame_index, scores in enumerate(scored_by_frame):
        for score in scores:
            row = detection_rows_for_frame(
                [score.detection],
                paths[frame_index],
                frame_index,
            )[0]
            row["recurrent_artifact_rejected"] = score.rejected
            rows.append(row)
    return rows


def track_detection_rows(tracks: list, paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for track in tracks:
        for detection_index, detection in enumerate(track.detections):
            frame_index = int(detection.frame_index)
            row = {
                field: getattr(detection, field)
                for field in DETECTION_FIELDS
                if field != "image"
            }
            row["track_id"] = track.track_id
            row["track_detection_index"] = detection_index
            row["frame_index"] = frame_index
            row["image"] = str(paths[frame_index].relative_to(rt.DATA))
            rows.append(row)
    return rows


def should_save_residual_preview(frame_index: int, preview_frames: int, preview_interval: int) -> bool:
    return frame_index < preview_frames or (preview_interval > 0 and frame_index % preview_interval == 0)


def apply_static_noise_floor(residual: ResidualImage, static_noise: np.ndarray | None) -> ResidualImage:
    """Normalize a residual with an image-fixed noise floor when available."""

    if static_noise is None:
        return residual
    noise_floor = np.asarray(static_noise, dtype=np.float64)
    if noise_floor.shape != residual.local_noise.shape:
        raise ValueError(
            "static noise map shape must match residual shape: "
            f"{noise_floor.shape} != {residual.local_noise.shape}"
        )
    noise_floor = np.where(np.isfinite(noise_floor) & (noise_floor > 0), noise_floor, 0.0)
    local_noise = np.maximum(residual.local_noise, noise_floor)
    valid = residual.mask & np.isfinite(residual.raw) & np.isfinite(local_noise) & (local_noise > 0)
    normalized = np.full(residual.normalized.shape, np.nan, dtype=np.float64)
    normalized[valid] = residual.raw[valid] / local_noise[valid]
    return ResidualImage(
        raw=residual.raw,
        local_noise=local_noise,
        normalized=normalized,
        mask=valid,
        expected_background=residual.expected_background,
        clean_render=residual.clean_render,
    )


def subtract_static_background(
    residual: ResidualImage,
    static_background: np.ndarray | None,
    *,
    residual_config: ResidualConfig,
) -> ResidualImage:
    """Subtract an image-fixed additive background from a belt residual.

    The learned map lives in crop/image coordinates, while the belt map lives in
    belt coordinates. After subtracting the static component, local noise is
    recomputed from the corrected residual so fixed illumination structures no
    longer inflate the normalization.
    """

    if static_background is None:
        return residual
    background = np.asarray(static_background, dtype=np.float64)
    if background.shape != residual.raw.shape:
        raise ValueError(
            "static background map shape must match residual shape: "
            f"{background.shape} != {residual.raw.shape}"
        )
    background = np.where(np.isfinite(background), background, 0.0)
    valid = residual.mask & np.isfinite(residual.raw)
    corrected_raw_values = residual.raw - background
    local_noise = estimate_local_noise(
        corrected_raw_values,
        mask=valid,
        config=residual_config,
    )
    normalized = np.full(
        residual.normalized.shape,
        residual_config.fill_value,
        dtype=np.float64,
    )
    norm_valid = valid & np.isfinite(local_noise) & (local_noise > 0)
    normalized[norm_valid] = corrected_raw_values[norm_valid] / local_noise[norm_valid]
    raw = np.full(residual.raw.shape, residual_config.fill_value, dtype=np.float64)
    raw[valid] = corrected_raw_values[valid]
    expected = residual.expected_background + background
    return ResidualImage(
        raw=raw,
        local_noise=local_noise,
        normalized=normalized,
        mask=norm_valid,
        expected_background=expected,
        clean_render=residual.clean_render,
    )


def _nanmedian(values: np.ndarray, *, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmedian(values, axis=axis)


def learn_static_residual_noise_map(
    *,
    paths: list[Path],
    belt_map: np.ndarray,
    motion_model: BeltMotionModel,
    region: tuple[int, int, int, int],
    phase_estimates: dict[int, PhaseEstimate] | None,
    registration_config: PhaseRegistrationConfig,
    residual_config: ResidualConfig,
    sample_frames: int,
    min_scale: float,
    mask_threshold: float | None = None,
    mask_margin_px: int = 0,
    mask_min_area_px: int = 1,
    chunk_rows: int = 48,
) -> np.ndarray:
    """Estimate per-pixel residual MAD from belt-subtracted sampled frames."""

    if sample_frames <= 0:
        raise ValueError("sample_frames must be positive")
    if min_scale < 0:
        raise ValueError("min_scale must be non-negative")
    if mask_threshold is not None and mask_threshold <= 0:
        mask_threshold = None
    if mask_margin_px < 0:
        raise ValueError("mask_margin_px must be non-negative")
    if mask_min_area_px < 1:
        raise ValueError("mask_min_area_px must be at least 1")

    _, _, crop_height, crop_width = region
    samples = sample_indices(len(paths), sample_frames)
    rt.emit(
        "static_noise",
        "learning static residual-noise map",
        sampled_frames=len(samples),
        selected_frames=len(paths),
        mask_threshold=mask_threshold,
        mask_margin_px=mask_margin_px,
        mask_min_area_px=mask_min_area_px,
        min_scale=min_scale,
    )
    component_config = ParticleComponentConfig(
        min_area_px=mask_min_area_px,
        weighted_centroid=False,
    )
    progress_interval = rt.env_int("PROGRESS_INTERVAL_FRAMES", 25, minimum=1)
    with tempfile.TemporaryDirectory(prefix="static_noise_", dir=rt.OUT) as temp_dir:
        stack_path = Path(temp_dir) / "residual_stack.npy"
        residual_stack = np.lib.format.open_memmap(
            stack_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(samples), crop_height, crop_width),
        )
        masked_pixels = 0
        for sample_number, frame_index in enumerate(samples, start=1):
            frame = rt.crop(rt.read_gray(paths[frame_index]), region)
            phase_estimate = (
                phase_estimates[frame_index]
                if phase_estimates is not None
                else None
            )
            residual = render_clean_belt_residual(
                image=frame,
                belt_map=belt_map,
                frame_index=float(frame_index),
                motion_model=motion_model,
                belt_region=None,
                phase_estimate=phase_estimate,
                registration_config=registration_config,
                residual_config=residual_config,
            )
            raw = np.asarray(residual.raw, dtype=np.float32).copy()
            if mask_threshold is not None:
                mask = detect_particles_from_residual(residual, threshold=mask_threshold)
                detections = extract_particle_detections(
                    mask,
                    residual=residual,
                    frame_index=float(frame_index),
                    config=component_config,
                )
                particle_mask = expanded_detection_mask(
                    detections,
                    raw.shape,
                    margin_px=mask_margin_px,
                )
                raw[particle_mask] = np.nan
                masked_pixels += int(np.count_nonzero(particle_mask))
            residual_stack[sample_number - 1] = raw
            if sample_number == 1 or sample_number == len(samples) or sample_number % progress_interval == 0:
                rt.emit(
                    "static_noise",
                    f"sampled {sample_number}/{len(samples)} residual frames",
                    source_frame_index=frame_index,
                    masked_pixels=masked_pixels,
                )

        static_noise = np.empty((crop_height, crop_width), dtype=np.float32)
        for row_start in range(0, crop_height, chunk_rows):
            row_stop = min(crop_height, row_start + chunk_rows)
            block = np.asarray(residual_stack[:, row_start:row_stop, :], dtype=np.float32)
            center = _nanmedian(block, axis=0)
            deviations = np.abs(block - center[None, :, :])
            mad = _nanmedian(deviations, axis=0)
            noise = (1.4826 * mad).astype(np.float32)
            noise[~np.isfinite(noise)] = min_scale
            if min_scale > 0:
                noise = np.maximum(noise, min_scale)
            static_noise[row_start:row_stop] = noise
            rt.emit(
                "static_noise",
                "computed static-noise row chunk",
                row_start=row_start,
                row_stop=row_stop,
                crop_height=crop_height,
            )
            del block, center, deviations, mad, noise
        residual_stack.flush()
        del residual_stack

    finite = static_noise[np.isfinite(static_noise)]
    rt.emit(
        "static_noise",
        "finished static residual-noise map",
        median_noise=float(np.median(finite)) if finite.size else None,
        p95_noise=float(np.percentile(finite, 95)) if finite.size else None,
        max_noise=float(np.max(finite)) if finite.size else None,
    )
    return static_noise


def learn_static_residual_background_map(
    *,
    paths: list[Path],
    belt_map: np.ndarray,
    motion_model: BeltMotionModel,
    region: tuple[int, int, int, int],
    phase_estimates: dict[int, PhaseEstimate] | None,
    registration_config: PhaseRegistrationConfig,
    residual_config: ResidualConfig,
    sample_frames: int,
    mask_threshold: float | None = None,
    mask_margin_px: int = 0,
    mask_min_area_px: int = 1,
    chunk_rows: int = 48,
) -> np.ndarray:
    """Estimate an additive image-fixed background from belt-subtracted residuals."""

    if sample_frames <= 0:
        raise ValueError("sample_frames must be positive")
    if mask_threshold is not None and mask_threshold <= 0:
        mask_threshold = None
    if mask_margin_px < 0:
        raise ValueError("mask_margin_px must be non-negative")
    if mask_min_area_px < 1:
        raise ValueError("mask_min_area_px must be at least 1")

    _, _, crop_height, crop_width = region
    samples = sample_indices(len(paths), sample_frames)
    rt.emit(
        "static_background",
        "learning additive static residual-background map",
        sampled_frames=len(samples),
        selected_frames=len(paths),
        mask_threshold=mask_threshold,
        mask_margin_px=mask_margin_px,
        mask_min_area_px=mask_min_area_px,
    )
    component_config = ParticleComponentConfig(
        min_area_px=mask_min_area_px,
        weighted_centroid=False,
    )
    progress_interval = rt.env_int("PROGRESS_INTERVAL_FRAMES", 25, minimum=1)
    with tempfile.TemporaryDirectory(prefix="static_background_", dir=rt.OUT) as temp_dir:
        stack_path = Path(temp_dir) / "residual_stack.npy"
        residual_stack = np.lib.format.open_memmap(
            stack_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(samples), crop_height, crop_width),
        )
        masked_pixels = 0
        for sample_number, frame_index in enumerate(samples, start=1):
            frame = rt.crop(rt.read_gray(paths[frame_index]), region)
            phase_estimate = (
                phase_estimates[frame_index]
                if phase_estimates is not None
                else None
            )
            residual = render_clean_belt_residual(
                image=frame,
                belt_map=belt_map,
                frame_index=float(frame_index),
                motion_model=motion_model,
                belt_region=None,
                phase_estimate=phase_estimate,
                registration_config=registration_config,
                residual_config=residual_config,
            )
            raw = np.asarray(residual.raw, dtype=np.float32).copy()
            if mask_threshold is not None:
                mask = detect_particles_from_residual(residual, threshold=mask_threshold)
                detections = extract_particle_detections(
                    mask,
                    residual=residual,
                    frame_index=float(frame_index),
                    config=component_config,
                )
                particle_mask = expanded_detection_mask(
                    detections,
                    raw.shape,
                    margin_px=mask_margin_px,
                )
                raw[particle_mask] = np.nan
                masked_pixels += int(np.count_nonzero(particle_mask))
            residual_stack[sample_number - 1] = raw
            if (
                sample_number == 1
                or sample_number == len(samples)
                or sample_number % progress_interval == 0
            ):
                rt.emit(
                    "static_background",
                    f"sampled {sample_number}/{len(samples)} residual frames",
                    source_frame_index=frame_index,
                    masked_pixels=masked_pixels,
                )

        static_background = np.empty((crop_height, crop_width), dtype=np.float32)
        for row_start in range(0, crop_height, chunk_rows):
            row_stop = min(crop_height, row_start + chunk_rows)
            block = np.asarray(residual_stack[:, row_start:row_stop, :], dtype=np.float32)
            center = _nanmedian(block, axis=0).astype(np.float32)
            center[~np.isfinite(center)] = 0.0
            static_background[row_start:row_stop] = center
            rt.emit(
                "static_background",
                "computed static-background row chunk",
                row_start=row_start,
                row_stop=row_stop,
                crop_height=crop_height,
            )
            del block, center
        residual_stack.flush()
        del residual_stack

    finite = static_background[np.isfinite(static_background)]
    rt.emit(
        "static_background",
        "finished additive static residual-background map",
        median_background=float(np.median(finite)) if finite.size else None,
        p05_background=float(np.percentile(finite, 5)) if finite.size else None,
        p95_background=float(np.percentile(finite, 95)) if finite.size else None,
        max_abs_background=float(np.max(np.abs(finite))) if finite.size else None,
    )
    return static_background


def main() -> None:
    """Run the BeltMap image-sequence driver."""

    rt.refresh_runtime_paths()
    rt.OUT.mkdir(parents=True, exist_ok=True)
    rt.emit("startup", "starting BeltMap image driver", data_dir=rt.DATA, output_dir=rt.OUT)
    paths, discovered_frame_count, frame_stride = rt.image_paths()
    rt.emit(
        "images",
        "selected image sequence",
        discovered_frames=discovered_frame_count,
        selected_frames=len(paths),
        frame_stride=frame_stride,
        first_image=paths[0],
        last_image=paths[-1],
    )

    first = rt.read_gray(paths[0])
    region = parse_region(first)
    rt.emit(
        "images",
        "loaded first frame and parsed crop region",
        first_image_shape=list(first.shape),
        belt_region={"top": region[0], "left": region[1], "height": region[2], "width": region[3]},
    )

    velocity_spec = os.getenv("BELT_VELOCITY_PX_PER_FRAME", "auto").strip().lower()
    belt_velocity_source = "auto"
    belt_velocity_frame_unit = "selected_frame"
    supplied_belt_velocity_px_per_frame: float | None = None
    if velocity_spec == "auto":
        validate_auto_velocity_region(region, first.shape)
        belt_velocity, pair_shifts = estimate_velocity(paths, region)
    else:
        (
            belt_velocity,
            belt_velocity_frame_unit,
            supplied_belt_velocity_px_per_frame,
        ) = resolve_supplied_velocity(velocity_spec, frame_stride)
        belt_velocity_source = "supplied"
        pair_shifts = []
        rt.emit(
            "velocity",
            "using supplied belt velocity",
            supplied_belt_velocity_px_per_frame=supplied_belt_velocity_px_per_frame,
            belt_velocity_frame_unit=belt_velocity_frame_unit,
            belt_velocity_px_per_selected_frame=belt_velocity,
            frame_stride=frame_stride,
        )

    period_px = optional_positive_int("BELT_PERIOD_PX")
    detection_threshold = rt.env_float("DETECTION_THRESHOLD", 5.0)
    min_area_px = rt.env_int("MIN_AREA_PX", 4, minimum=1)
    detection_max_area_px = optional_positive_int("DETECTION_MAX_AREA_PX")
    detection_min_bbox_width_px = optional_positive_int("DETECTION_MIN_BBOX_WIDTH_PX")
    detection_min_bbox_height_px = optional_positive_int("DETECTION_MIN_BBOX_HEIGHT_PX")
    detection_max_bbox_aspect_ratio = optional_positive_float(
        "DETECTION_MAX_BBOX_ASPECT_RATIO",
        0.0,
    )
    detection_min_bbox_extent = optional_positive_float("DETECTION_MIN_BBOX_EXTENT", 0.0)
    residual_noise_radius_px = rt.env_int("RESIDUAL_NOISE_RADIUS_PX", 15, minimum=0)
    residual_clip_sigma = optional_positive_float("RESIDUAL_CLIP_SIGMA", 5.0)
    residual_min_noise = rt.env_float("RESIDUAL_MIN_NOISE", 1e-6, minimum=0.0)
    if residual_min_noise <= 0:
        raise ValueError("RESIDUAL_MIN_NOISE must be positive")
    residual_noise_exclusion_sigma = optional_positive_float("RESIDUAL_NOISE_EXCLUSION_SIGMA", 4.0)
    residual_noise_exclusion_radius_px = rt.env_int(
        "RESIDUAL_NOISE_EXCLUSION_RADIUS_PX", 2, minimum=0
    )
    min_track_length = rt.env_int("MIN_TRACK_LENGTH", 2, minimum=1)
    tracking_assignment_method = os.getenv("TRACKING_ASSIGNMENT_METHOD", "global").strip().lower()
    tracking_area_cost_weight_px = rt.env_float(
        "TRACKING_AREA_COST_WEIGHT_PX", 0.0, minimum=0.0
    )
    tracking_signal_cost_weight_px = rt.env_float(
        "TRACKING_SIGNAL_COST_WEIGHT_PX", 0.0, minimum=0.0
    )
    tracking_lateral_cost_weight = rt.env_float(
        "TRACKING_LATERAL_COST_WEIGHT", 0.0, minimum=0.0
    )
    tracking_max_area_ratio = optional_positive_float("TRACKING_MAX_AREA_RATIO", 0.0)
    map_mask_iterations = rt.env_int("MAP_MASK_ITERATIONS", 1, minimum=0)
    map_particle_mask_threshold = rt.env_float("MAP_PARTICLE_MASK_THRESHOLD", detection_threshold, minimum=0.0)
    map_particle_mask_mode = os.getenv("MAP_PARTICLE_MASK_MODE", "positive").strip().lower()
    map_particle_mask_grow_threshold = rt.env_float("MAP_PARTICLE_MASK_GROW_THRESHOLD", 2.0, minimum=0.0)
    map_particle_mask_dilation_px = rt.env_int("MAP_PARTICLE_MASK_DILATION_PX", 0, minimum=0)
    map_particle_mask_margin_px = rt.env_int("MAP_PARTICLE_MASK_MARGIN_PX", 8, minimum=0)
    map_particle_mask_min_area_px = rt.env_int("MAP_PARTICLE_MASK_MIN_AREA_PX", min_area_px, minimum=1)
    map_aggregation = os.getenv("MAP_AGGREGATION", "mean").strip().lower()
    map_robust_iterations = rt.env_int("MAP_ROBUST_ITERATIONS", 1, minimum=0)
    map_robust_huber_delta = rt.env_float(
        "MAP_ROBUST_HUBER_DELTA", 3.0, minimum=1e-9
    )
    map_robust_min_scale = rt.env_float(
        "MAP_ROBUST_MIN_SCALE", 1.0, minimum=1e-9
    )
    reuse_belt_map_path = optional_path("REUSE_BELT_MAP_PATH")
    reuse_phase_estimates_path = optional_path("REUSE_PHASE_ESTIMATES_PATH")
    reuse_static_noise_path = optional_path("REUSE_STATIC_NOISE_PATH")
    reuse_static_background_path = optional_path("REUSE_STATIC_BACKGROUND_PATH")
    reuse_recurrent_artifact_map_path = optional_path("REUSE_RECURRENT_ARTIFACT_MAP_PATH")
    if reuse_phase_estimates_path is not None and reuse_belt_map_path is None:
        raise ValueError("REUSE_PHASE_ESTIMATES_PATH requires REUSE_BELT_MAP_PATH")
    static_noise_sample_frames = rt.env_int("STATIC_NOISE_SAMPLE_FRAMES", 0, minimum=0)
    static_noise_min_scale = rt.env_float("STATIC_NOISE_MIN_SCALE", 0.0, minimum=0.0)
    static_noise_mask_threshold = optional_positive_float("STATIC_NOISE_MASK_THRESHOLD", 0.0)
    static_noise_mask_margin_px = rt.env_int("STATIC_NOISE_MASK_MARGIN_PX", 8, minimum=0)
    static_noise_mask_min_area_px = rt.env_int("STATIC_NOISE_MASK_MIN_AREA_PX", min_area_px, minimum=1)
    static_background_sample_frames = rt.env_int("STATIC_BACKGROUND_SAMPLE_FRAMES", 0, minimum=0)
    static_background_mask_threshold = optional_positive_float("STATIC_BACKGROUND_MASK_THRESHOLD", 0.0)
    static_background_mask_margin_px = rt.env_int("STATIC_BACKGROUND_MASK_MARGIN_PX", 8, minimum=0)
    static_background_mask_min_area_px = rt.env_int("STATIC_BACKGROUND_MASK_MIN_AREA_PX", min_area_px, minimum=1)
    recurrent_artifact_config = RecurrentArtifactConfig(
        min_revolutions=rt.env_int("RECURRENT_ARTIFACT_MIN_REVOLUTIONS", 0, minimum=0),
        margin_px=rt.env_int("RECURRENT_ARTIFACT_MARGIN_PX", 2, minimum=0),
        max_overlap_fraction=rt.env_float(
            "RECURRENT_ARTIFACT_MAX_OVERLAP_FRACTION",
            0.3,
            minimum=0.0,
        ),
        min_recurrence_probability=rt.env_float(
            "RECURRENT_ARTIFACT_MIN_RECURRENCE_PROBABILITY",
            0.0,
            minimum=0.0,
        ),
        mode=os.getenv("RECURRENT_ARTIFACT_MODE", "hard").strip().lower(),
        soft_penalty_weight=rt.env_float(
            "RECURRENT_ARTIFACT_SOFT_PENALTY_WEIGHT",
            1.0,
            minimum=0.0,
        ),
    )
    if recurrent_artifact_config.max_overlap_fraction > 1:
        raise ValueError("RECURRENT_ARTIFACT_MAX_OVERLAP_FRACTION must be in [0, 1]")
    if recurrent_artifact_config.min_recurrence_probability > 1:
        raise ValueError(
            "RECURRENT_ARTIFACT_MIN_RECURRENCE_PROBABILITY must be in [0, 1]"
        )
    if recurrent_artifact_config.mode not in RECURRENT_ARTIFACT_MODES:
        choices = ", ".join(sorted(RECURRENT_ARTIFACT_MODES))
        raise ValueError(f"RECURRENT_ARTIFACT_MODE must be one of {choices}")
    registration_config = PhaseRegistrationConfig(
        search_radius_px=rt.env_float("REGISTRATION_SEARCH_RADIUS_PX", 8.0, minimum=0.0),
        search_step_px=rt.env_float("REGISTRATION_SEARCH_STEP_PX", 0.5, minimum=1e-9),
    )
    phase_refinement_iterations = rt.env_int("PHASE_REFINEMENT_ITERATIONS", 0, minimum=0)
    phase_refinement_min_score = rt.env_float("PHASE_REFINEMENT_MIN_SCORE", 0.0, minimum=0.0)
    phase_refinement_max_abs_correction_px = optional_positive_float(
        "PHASE_REFINEMENT_MAX_ABS_CORRECTION_PX",
        0.0,
    )
    phase_refinement_smoothing_window_frames = rt.env_int(
        "PHASE_REFINEMENT_SMOOTHING_WINDOW_FRAMES",
        25,
        minimum=0,
    )
    rt.emit(
        "config",
        "runtime parameters",
        belt_velocity_px_per_frame=belt_velocity,
        belt_velocity_source=belt_velocity_source,
        belt_velocity_frame_unit=belt_velocity_frame_unit,
        supplied_belt_velocity_px_per_frame=supplied_belt_velocity_px_per_frame,
        belt_period_px=period_px,
        detection_threshold=detection_threshold,
        min_area_px=min_area_px,
        detection_max_area_px=detection_max_area_px,
        detection_min_bbox_width_px=detection_min_bbox_width_px,
        detection_min_bbox_height_px=detection_min_bbox_height_px,
        detection_max_bbox_aspect_ratio=detection_max_bbox_aspect_ratio,
        detection_min_bbox_extent=detection_min_bbox_extent,
        residual_noise_radius_px=residual_noise_radius_px,
        residual_clip_sigma=residual_clip_sigma,
        residual_min_noise=residual_min_noise,
        residual_noise_exclusion_sigma=residual_noise_exclusion_sigma,
        residual_noise_exclusion_radius_px=residual_noise_exclusion_radius_px,
        min_track_length=min_track_length,
        tracking_assignment_method=tracking_assignment_method,
        tracking_area_cost_weight_px=tracking_area_cost_weight_px,
        tracking_signal_cost_weight_px=tracking_signal_cost_weight_px,
        tracking_lateral_cost_weight=tracking_lateral_cost_weight,
        tracking_max_area_ratio=tracking_max_area_ratio,
        map_mask_iterations=map_mask_iterations,
        map_particle_mask_threshold=map_particle_mask_threshold,
        map_particle_mask_mode=map_particle_mask_mode,
        map_particle_mask_grow_threshold=map_particle_mask_grow_threshold,
        map_particle_mask_dilation_px=map_particle_mask_dilation_px,
        map_particle_mask_margin_px=map_particle_mask_margin_px,
        map_particle_mask_min_area_px=map_particle_mask_min_area_px,
        map_aggregation=map_aggregation,
        map_robust_iterations=map_robust_iterations,
        map_robust_huber_delta=map_robust_huber_delta,
        map_robust_min_scale=map_robust_min_scale,
        reuse_belt_map_path=reuse_belt_map_path,
        reuse_phase_estimates_path=reuse_phase_estimates_path,
        reuse_static_noise_path=reuse_static_noise_path,
        reuse_static_background_path=reuse_static_background_path,
        reuse_recurrent_artifact_map_path=reuse_recurrent_artifact_map_path,
        static_noise_sample_frames=static_noise_sample_frames,
        static_noise_min_scale=static_noise_min_scale,
        static_noise_mask_threshold=static_noise_mask_threshold,
        static_noise_mask_margin_px=static_noise_mask_margin_px,
        static_noise_mask_min_area_px=static_noise_mask_min_area_px,
        static_background_sample_frames=static_background_sample_frames,
        static_background_mask_threshold=static_background_mask_threshold,
        static_background_mask_margin_px=static_background_mask_margin_px,
        static_background_mask_min_area_px=static_background_mask_min_area_px,
        recurrent_artifact_min_revolutions=recurrent_artifact_config.min_revolutions,
        recurrent_artifact_margin_px=recurrent_artifact_config.margin_px,
        recurrent_artifact_max_overlap_fraction=recurrent_artifact_config.max_overlap_fraction,
        recurrent_artifact_min_recurrence_probability=recurrent_artifact_config.min_recurrence_probability,
        recurrent_artifact_mode=recurrent_artifact_config.mode,
        recurrent_artifact_soft_penalty_weight=recurrent_artifact_config.soft_penalty_weight,
        registration_search_radius_px=registration_config.search_radius_px,
        registration_search_step_px=registration_config.search_step_px,
        phase_refinement_iterations=phase_refinement_iterations,
        phase_refinement_min_score=phase_refinement_min_score,
        phase_refinement_max_abs_correction_px=phase_refinement_max_abs_correction_px,
        phase_refinement_smoothing_window_frames=phase_refinement_smoothing_window_frames,
    )

    reuse_metadata: dict = {}
    reuse_metadata_path: Path | None = None
    phase_refinement_rows: list[dict] = []
    if reuse_belt_map_path is not None:
        belt_map = np.load(reuse_belt_map_path)
        if belt_map.ndim != 2:
            raise ValueError("REUSE_BELT_MAP_PATH must point to a 2-D belt_map.npy")
        if belt_map.shape[1] != region[3]:
            raise ValueError(
                "reused belt map width does not match BELT_REGION width: "
                f"{belt_map.shape[1]} != {region[3]}"
            )
        map_height = int(belt_map.shape[0])
        reuse_metadata, reuse_metadata_path = load_reuse_metadata(reuse_belt_map_path)
        reference_phase = float(reuse_metadata.get("reference_phase_px", 0.0))
        if period_px is not None and period_px != map_height:
            rt.emit(
                "belt_map",
                "reused belt-map height differs from supplied BELT_PERIOD_PX; using loaded map height",
                supplied_period_px=period_px,
                belt_map_height_px=map_height,
            )
        write_phase_refinement_outputs(phase_refinement_rows)
        rt.emit(
            "belt_map",
            "loaded reused belt-map outputs",
            source_belt_map_npy=reuse_belt_map_path,
            source_metadata_json=reuse_metadata_path,
            belt_map_shape=list(belt_map.shape),
            reference_phase_px=reference_phase,
        )
    else:
        build_result = build_belt_map_result(
            paths=paths,
            region=region,
            velocity=belt_velocity,
            supplied_period=period_px,
            mask_iterations=map_mask_iterations,
            mask_threshold=map_particle_mask_threshold,
            mask_mode=map_particle_mask_mode,
            mask_grow_threshold=map_particle_mask_grow_threshold,
            mask_dilation_px=map_particle_mask_dilation_px,
            mask_margin_px=map_particle_mask_margin_px,
            mask_min_area_px=map_particle_mask_min_area_px,
            aggregation=map_aggregation,
            robust_iterations=map_robust_iterations,
            robust_huber_delta=map_robust_huber_delta,
            robust_min_scale=map_robust_min_scale,
            phase_feedback_config=PhaseFeedbackConfig(
                iterations=phase_refinement_iterations,
                min_score=phase_refinement_min_score,
                max_abs_correction_px=phase_refinement_max_abs_correction_px,
                smoothing_window_frames=phase_refinement_smoothing_window_frames,
                registration_config=registration_config,
            ),
        )
        belt_map = build_result.belt_map
        reference_phase = build_result.reference_phase
        map_height = build_result.map_height
        phase_refinement_rows = build_result.phase_refinement_rows
        write_phase_refinement_outputs(phase_refinement_rows)
    np.save(rt.OUT / "belt_map.npy", belt_map)
    rt.save_png(belt_map, rt.OUT / "belt_map.png")
    rt.emit(
        "belt_map",
        "saved belt-map outputs",
        belt_map_shape=list(belt_map.shape),
        belt_map_npy=rt.OUT / "belt_map.npy",
        belt_map_png=rt.OUT / "belt_map.png",
        phase_refinement_csv=rt.OUT / "phase_refinement.csv",
        phase_refinement_rows=len(phase_refinement_rows),
    )

    motion_model = BeltMotionModel(
        image_velocity_px_per_frame=belt_velocity,
        period_px=float(map_height),
        reference_frame=0.0,
        reference_phase_px=reference_phase,
    )
    component_config = ParticleComponentConfig(
        min_area_px=min_area_px,
        max_area_px=detection_max_area_px,
        min_bbox_width_px=detection_min_bbox_width_px,
        min_bbox_height_px=detection_min_bbox_height_px,
        max_bbox_aspect_ratio=detection_max_bbox_aspect_ratio,
        min_bbox_extent=detection_min_bbox_extent,
    )
    residual_config = ResidualConfig(
        noise_radius_px=residual_noise_radius_px,
        clip_sigma=residual_clip_sigma,
        noise_exclusion_sigma=residual_noise_exclusion_sigma,
        noise_exclusion_radius_px=residual_noise_exclusion_radius_px,
        min_noise=residual_min_noise,
    )
    reused_phase_estimates = (
        load_phase_estimates(
            reuse_phase_estimates_path,
            expected_image_paths=paths,
            data_dir=rt.DATA,
        )
        if reuse_phase_estimates_path is not None
        else None
    )
    if reused_phase_estimates is not None:
        rt.emit(
            "detect",
            "loaded reused phase estimates",
            source_phase_estimates_csv=reuse_phase_estimates_path,
            phase_estimates=len(reused_phase_estimates),
        )
    static_background_map: np.ndarray | None = None
    if reuse_static_background_path is not None:
        static_background_map = np.load(reuse_static_background_path)
        if static_background_map.ndim != 2:
            raise ValueError("REUSE_STATIC_BACKGROUND_PATH must point to a 2-D static_background.npy")
        if static_background_map.shape != (region[2], region[3]):
            raise ValueError(
                "reused static background map shape does not match BELT_REGION: "
                f"{static_background_map.shape} != {(region[2], region[3])}"
            )
        static_background_map = np.asarray(static_background_map, dtype=np.float32)
        static_background_map = np.where(
            np.isfinite(static_background_map),
            static_background_map,
            0.0,
        ).astype(np.float32, copy=False)
        rt.emit(
            "static_background",
            "loaded reused additive static residual-background map",
            source_static_background_npy=reuse_static_background_path,
            static_background_shape=list(static_background_map.shape),
        )
    elif static_background_sample_frames > 0:
        static_background_map = learn_static_residual_background_map(
            paths=paths,
            belt_map=belt_map,
            motion_model=motion_model,
            region=region,
            phase_estimates=reused_phase_estimates,
            registration_config=registration_config,
            residual_config=residual_config,
            sample_frames=static_background_sample_frames,
            mask_threshold=static_background_mask_threshold,
            mask_margin_px=static_background_mask_margin_px,
            mask_min_area_px=static_background_mask_min_area_px,
        )
        np.save(rt.OUT / "static_background.npy", static_background_map)
        rt.save_png(static_background_map, rt.OUT / "static_background.png")
        rt.emit(
            "static_background",
            "saved additive static residual-background map",
            static_background_npy=rt.OUT / "static_background.npy",
            static_background_png=rt.OUT / "static_background.png",
        )
    if static_background_map is not None and reuse_static_background_path is not None:
        np.save(rt.OUT / "static_background.npy", static_background_map)
        rt.save_png(static_background_map, rt.OUT / "static_background.png")

    static_noise_map: np.ndarray | None = None
    if reuse_static_noise_path is not None:
        static_noise_map = np.load(reuse_static_noise_path)
        if static_noise_map.ndim != 2:
            raise ValueError("REUSE_STATIC_NOISE_PATH must point to a 2-D static_noise.npy")
        if static_noise_map.shape != (region[2], region[3]):
            raise ValueError(
                "reused static noise map shape does not match BELT_REGION: "
                f"{static_noise_map.shape} != {(region[2], region[3])}"
            )
        static_noise_map = np.asarray(static_noise_map, dtype=np.float32)
        static_noise_map = np.where(
            np.isfinite(static_noise_map) & (static_noise_map > 0),
            static_noise_map,
            0.0,
        ).astype(np.float32, copy=False)
        rt.emit(
            "static_noise",
            "loaded reused static residual-noise map",
            source_static_noise_npy=reuse_static_noise_path,
            static_noise_shape=list(static_noise_map.shape),
        )
    elif static_noise_sample_frames > 0:
        static_noise_map = learn_static_residual_noise_map(
            paths=paths,
            belt_map=belt_map,
            motion_model=motion_model,
            region=region,
            phase_estimates=reused_phase_estimates,
            registration_config=registration_config,
            residual_config=residual_config,
            sample_frames=static_noise_sample_frames,
            min_scale=static_noise_min_scale,
            mask_threshold=static_noise_mask_threshold,
            mask_margin_px=static_noise_mask_margin_px,
            mask_min_area_px=static_noise_mask_min_area_px,
        )
        np.save(rt.OUT / "static_noise.npy", static_noise_map)
        rt.save_png(static_noise_map, rt.OUT / "static_noise.png")
        rt.emit(
            "static_noise",
            "saved static residual-noise map",
            static_noise_npy=rt.OUT / "static_noise.npy",
            static_noise_png=rt.OUT / "static_noise.png",
        )
    if static_noise_map is not None and reuse_static_noise_path is not None:
        np.save(rt.OUT / "static_noise.npy", static_noise_map)
        rt.save_png(static_noise_map, rt.OUT / "static_noise.png")

    reused_recurrent_artifact_map: np.ndarray | None = None
    if reuse_recurrent_artifact_map_path is not None:
        reused_recurrent_artifact_map = load_recurrent_artifact_map(
            reuse_recurrent_artifact_map_path,
            map_shape=(map_height, region[3]),
        )
        rt.emit(
            "recurrent_artifact",
            "loaded reused recurrent belt-coordinate artifact map",
            source_recurrent_artifact_map_npy=reuse_recurrent_artifact_map_path,
            artifact_pixels=int(np.count_nonzero(reused_recurrent_artifact_map)),
            artifact_map_shape=list(reused_recurrent_artifact_map.shape),
        )

    progress_interval = rt.env_int("PROGRESS_INTERVAL_FRAMES", 25, minimum=1)
    partial_output_interval = rt.env_int("PARTIAL_OUTPUT_INTERVAL_FRAMES", 250, minimum=0)
    residual_preview_frames = rt.env_int("DEBUG_RESIDUAL_PREVIEW_FRAMES", 3, minimum=0)
    residual_preview_interval = rt.env_int("DEBUG_RESIDUAL_PREVIEW_INTERVAL_FRAMES", 0, minimum=0)
    recurrent_artifact_enabled = (
        recurrent_artifact_config.min_revolutions > 0
        or reuse_recurrent_artifact_map_path is not None
    )
    rt.emit(
        "detect",
        "starting residual rendering and particle detection",
        selected_frames=len(paths),
        progress_interval_frames=progress_interval,
        partial_output_interval_frames=partial_output_interval,
        residual_preview_frames=residual_preview_frames,
        residual_preview_interval_frames=residual_preview_interval,
        recurrent_artifact_filter_enabled=recurrent_artifact_enabled,
    )

    detections_by_frame = []
    detection_rows: list[dict] = []
    phase_rows: list[dict] = []
    phase_px_by_frame: list[float] = []
    detection_start = rt.time.perf_counter()
    for frame_index, path in enumerate(paths):
        frame = rt.crop(rt.read_gray(path), region)
        phase_estimate = (
            reused_phase_estimates[frame_index]
            if reused_phase_estimates is not None
            else None
        )
        residual = render_clean_belt_residual(
            image=frame,
            belt_map=belt_map,
            frame_index=float(frame_index),
            motion_model=motion_model,
            belt_region=None,
            phase_estimate=phase_estimate,
            registration_config=registration_config,
            residual_config=residual_config,
        )
        residual = subtract_static_background(
            residual,
            static_background_map,
            residual_config=residual_config,
        )
        residual = apply_static_noise_floor(residual, static_noise_map)
        phase_row = phase_estimate_row(frame_index, path, residual, float(map_height))
        phase_rows.append(phase_row)
        phase_px_by_frame.append(float(phase_row["phase_px"]))
        if should_save_residual_preview(frame_index, residual_preview_frames, residual_preview_interval):
            rt.save_png(residual, rt.OUT / f"residual_frame_{frame_index:06d}.png")
        mask = detect_particles_from_residual(residual, threshold=detection_threshold)
        detections = extract_particle_detections(mask, residual=residual, frame_index=float(frame_index), config=component_config)
        detections_by_frame.append(detections)
        detection_rows.extend(detection_rows_for_frame(detections, path, frame_index))
        processed = frame_index + 1
        if (
            not recurrent_artifact_enabled
            and partial_output_interval > 0
            and (processed == 1 or processed % partial_output_interval == 0)
        ):
            write_detection_outputs(detections_by_frame, detection_rows)
            write_phase_outputs(phase_rows)
            rt.emit("detect", "wrote partial detection and phase outputs", processed_frames=processed, total_detections=len(detection_rows), phase_estimates=len(phase_rows))
        if processed == 1 or processed == len(paths) or processed % progress_interval == 0:
            dt = rt.time.perf_counter() - detection_start
            fps = processed / dt if dt > 0 else float("inf")
            remaining = len(paths) - processed
            eta = remaining / fps if fps > 0 else float("inf")
            rt.emit("detect", f"processed {processed}/{len(paths)} frames", processed_frames=processed, remaining_frames=remaining, detections_this_frame=len(detections), total_detections=len(detection_rows), frames_per_second=round(fps, 4), eta_s=round(eta, 1) if np.isfinite(eta) else None, current_image=path)

    recurrent_artifact_pixels = 0
    recurrent_artifact_rejected = 0
    recurrent_artifact_revolutions = 0
    recurrent_artifact_source = "none"
    if recurrent_artifact_enabled:
        map_shape = (map_height, region[3])
        recurrent_artifact_candidate_detections: int | None = None
        if reuse_recurrent_artifact_map_path is not None:
            recurrent_artifact_source = "loaded"
            assert reused_recurrent_artifact_map is not None
            recurrent_artifact_mask = reused_recurrent_artifact_map
            recurrent_artifact_pixels = int(np.count_nonzero(recurrent_artifact_mask))
            np.save(rt.OUT / "recurrent_artifact_map.npy", recurrent_artifact_mask)
            rt.save_png(
                recurrent_artifact_mask.astype(np.float32),
                rt.OUT / "recurrent_artifact_map.png",
            )
            rt.emit(
                "recurrent_artifact",
                "saved reused recurrent belt-coordinate artifact map",
                source_recurrent_artifact_map_npy=reuse_recurrent_artifact_map_path,
                artifact_pixels=recurrent_artifact_pixels,
                recurrent_artifact_map_npy=rt.OUT / "recurrent_artifact_map.npy",
            )
            recurrent_artifact_scores = score_recurrent_artifact_detections(
                detections_by_frame,
                phase_px_by_frame,
                recurrent_artifact_mask,
                config=recurrent_artifact_config,
                detection_threshold=detection_threshold,
            )
        else:
            recurrent_artifact_source = "built"
            rt.emit(
                "recurrent_artifact",
                "building recurrent belt-coordinate artifact map",
                min_revolutions=recurrent_artifact_config.min_revolutions,
                margin_px=recurrent_artifact_config.margin_px,
                max_overlap_fraction=recurrent_artifact_config.max_overlap_fraction,
                min_recurrence_probability=(
                    recurrent_artifact_config.min_recurrence_probability
                ),
                mode=recurrent_artifact_config.mode,
                soft_penalty_weight=recurrent_artifact_config.soft_penalty_weight,
            )
            revolution_by_frame = belt_revolution_indices(len(paths), motion_model)
            recurrent_result = build_recurrent_artifact_map(
                detections_by_frame,
                phase_px_by_frame,
                revolution_by_frame,
                map_shape=map_shape,
                config=recurrent_artifact_config,
                frame_shape=(region[2], region[3]),
            )
            recurrent_artifact_mask = recurrent_result.mask
            recurrent_artifact_pixels = recurrent_result.artifact_pixels
            recurrent_artifact_revolutions = recurrent_result.revolution_count
            recurrent_artifact_candidate_detections = recurrent_result.candidate_detections
            np.save(rt.OUT / "recurrent_artifact_map.npy", recurrent_result.mask)
            np.save(rt.OUT / "recurrent_artifact_counts.npy", recurrent_result.counts)
            np.save(
                rt.OUT / "recurrent_artifact_exposure_counts.npy",
                recurrent_result.exposure_counts,
            )
            np.save(
                rt.OUT / "recurrent_artifact_probability.npy",
                recurrent_result.probability,
            )
            rt.save_png(
                recurrent_result.mask.astype(np.float32),
                rt.OUT / "recurrent_artifact_map.png",
            )
            rt.save_png(recurrent_result.counts, rt.OUT / "recurrent_artifact_counts.png")
            rt.save_png(
                recurrent_result.probability,
                rt.OUT / "recurrent_artifact_probability.png",
            )
            recurrent_artifact_scores = (
                score_recurrent_artifact_detections_excluding_current_revolution(
                    detections_by_frame,
                    phase_px_by_frame,
                    revolution_by_frame,
                    recurrent_result,
                    config=recurrent_artifact_config,
                    detection_threshold=detection_threshold,
                )
            )
        recurrent_artifact_rows = recurrent_artifact_rows_from_scores(
            recurrent_artifact_scores,
            paths,
        )
        rt.write_csv(
            rt.OUT / "recurrent_artifact_detections.csv",
            recurrent_artifact_rows,
            RECURRENT_ARTIFACT_DETECTION_FIELDS,
        )
        recurrent_artifact_rejected = sum(
            1
            for frame_scores in recurrent_artifact_scores
            for score in frame_scores
            if score.rejected
        )
        detections_by_frame = [
            [score.detection for score in frame_scores if not score.rejected]
            for frame_scores in recurrent_artifact_scores
        ]
        detection_rows = detection_rows_from_frames(detections_by_frame, paths)
        rt.emit(
            "recurrent_artifact",
            "filtered recurrent belt-coordinate artifacts",
            source=recurrent_artifact_source,
            revolutions=recurrent_artifact_revolutions,
            candidate_detections=recurrent_artifact_candidate_detections,
            artifact_pixels=recurrent_artifact_pixels,
            rejected_detections=recurrent_artifact_rejected,
            remaining_detections=len(detection_rows),
            recurrent_artifact_detections_csv=rt.OUT / "recurrent_artifact_detections.csv",
            recurrent_artifact_map_npy=rt.OUT / "recurrent_artifact_map.npy",
            recurrent_artifact_counts_npy=(
                rt.OUT / "recurrent_artifact_counts.npy"
                if recurrent_artifact_source == "built"
                else None
            ),
            recurrent_artifact_exposure_counts_npy=(
                rt.OUT / "recurrent_artifact_exposure_counts.npy"
                if recurrent_artifact_source == "built"
                else None
            ),
            recurrent_artifact_probability_npy=(
                rt.OUT / "recurrent_artifact_probability.npy"
                if recurrent_artifact_source == "built"
                else None
            ),
        )

    write_detection_outputs(detections_by_frame, detection_rows)
    write_phase_outputs(phase_rows)
    rt.emit("detect", "finished residual rendering, phase estimation, and detection", processed_frames=len(paths), total_detections=len(detection_rows), phase_estimates=len(phase_rows))

    max_match = os.getenv("MAX_MATCH_DISTANCE_PX", "").strip()
    tracking_config = ParticleTrackingConfig(
        max_match_distance_px=float(max_match) if max_match else max(5.0, 1.5 * abs(belt_velocity)),
        velocity_prior_y_px_per_frame=0.8 * belt_velocity,
        assignment_method=tracking_assignment_method,
        area_cost_weight_px=tracking_area_cost_weight_px,
        signal_cost_weight_px=tracking_signal_cost_weight_px,
        lateral_cost_weight=tracking_lateral_cost_weight,
        max_area_ratio=tracking_max_area_ratio,
    )
    rt.emit(
        "track",
        "starting particle tracking",
        frames=len(detections_by_frame),
        max_match_distance_px=tracking_config.max_match_distance_px,
        velocity_prior_y_px_per_frame=tracking_config.velocity_prior_y_px_per_frame,
        velocity_prior_x_px_per_frame=tracking_config.velocity_prior_x_px_per_frame,
        assignment_method=tracking_config.assignment_method,
        area_cost_weight_px=tracking_config.area_cost_weight_px,
        signal_cost_weight_px=tracking_config.signal_cost_weight_px,
        lateral_cost_weight=tracking_config.lateral_cost_weight,
        max_area_ratio=tracking_config.max_area_ratio,
    )
    tracks = track_particle_detections(detections_by_frame, config=tracking_config, frame_indices=[float(i) for i in range(len(paths))])
    rt.emit("track", "finished particle tracking", tracks=len(tracks))
    track_rows = track_detection_rows(tracks, paths)
    rt.write_csv(rt.OUT / "tracks.csv", track_rows, TRACK_DETECTION_FIELDS)
    rt.emit("track", "wrote track detection assignments", track_detection_rows=len(track_rows))

    velocity_rows = []
    velocity_objects = []
    if abs(belt_velocity) > 1e-9:
        rt.emit("velocity", "estimating particle velocities relative to belt", min_track_length=min_track_length)
        for velocity in estimate_particle_velocities_vs_belt(tracks, belt_image_velocity_px_per_frame=belt_velocity, min_track_length=min_track_length):
            velocity_objects.append(velocity)
            velocity_rows.append(asdict(velocity))
    else:
        rt.emit("velocity", "skipped particle velocity estimation because belt velocity is near zero")
    rt.write_csv(rt.OUT / "velocities.csv", velocity_rows, VELOCITY_FIELDS)
    rt.emit("velocity", "wrote velocity estimates", velocity_estimates=len(velocity_rows))
    track_filter_config = TrackFilterConfig(
        min_track_length=rt.env_int("TRACK_FILTER_MIN_LENGTH", max(5, min_track_length), minimum=1),
        min_velocity_ratio_y=rt.env_float("TRACK_FILTER_MIN_VELOCITY_RATIO_Y", 0.0),
        max_velocity_ratio_y=rt.env_float("TRACK_FILTER_MAX_VELOCITY_RATIO_Y", 1.1),
        max_abs_x_velocity_px_per_frame=optional_positive_float(
            "TRACK_FILTER_MAX_ABS_X_VELOCITY_PX_PER_FRAME",
            0.0,
        ),
    )
    track_scores = score_particle_velocities(
        velocity_objects,
        config=track_filter_config,
    )
    accepted_track_ids = {score.track_id for score in track_scores if score.accepted}
    filtered_velocity_rows = [
        asdict(velocity)
        for velocity in velocity_objects
        if velocity.track_id in accepted_track_ids
    ]
    rt.write_csv(
        rt.OUT / "track_scores.csv",
        [asdict(score) for score in track_scores],
        TRACK_SCORE_FIELDS,
    )
    rt.write_csv(rt.OUT / "filtered_velocities.csv", filtered_velocity_rows, VELOCITY_FIELDS)
    filtered_track_rows = [
        row for row in track_rows if row["track_id"] in accepted_track_ids
    ]
    rt.write_csv(rt.OUT / "filtered_tracks.csv", filtered_track_rows, TRACK_DETECTION_FIELDS)
    rt.emit(
        "velocity",
        "wrote track-filter outputs",
        track_scores=len(track_scores),
        filtered_velocity_estimates=len(filtered_velocity_rows),
        filtered_track_detection_rows=len(filtered_track_rows),
        track_filter_min_length=track_filter_config.min_track_length,
        track_filter_min_velocity_ratio_y=track_filter_config.min_velocity_ratio_y,
        track_filter_max_velocity_ratio_y=track_filter_config.max_velocity_ratio_y,
        track_filter_max_abs_x_velocity_px_per_frame=track_filter_config.max_abs_x_velocity_px_per_frame,
    )

    metadata = {
        "n_images": len(paths),
        "discovered_frame_count": discovered_frame_count,
        "frame_stride": frame_stride,
        "first_image_shape": list(first.shape),
        "belt_region": {"top": region[0], "left": region[1], "height": region[2], "width": region[3]},
        "belt_velocity_source": belt_velocity_source,
        "belt_velocity_frame_unit": belt_velocity_frame_unit,
        "supplied_belt_velocity_px_per_frame": supplied_belt_velocity_px_per_frame,
        "belt_velocity_px_per_frame": belt_velocity,
        "belt_period_px_input": period_px,
        "belt_map_height_px": map_height,
        "reference_phase_px": reference_phase,
        "detection_threshold": detection_threshold,
        "min_area_px": min_area_px,
        "detection_max_area_px": detection_max_area_px,
        "detection_min_bbox_width_px": detection_min_bbox_width_px,
        "detection_min_bbox_height_px": detection_min_bbox_height_px,
        "detection_max_bbox_aspect_ratio": detection_max_bbox_aspect_ratio,
        "detection_min_bbox_extent": detection_min_bbox_extent,
        "tracking_assignment_method": tracking_config.assignment_method,
        "tracking_area_cost_weight_px": tracking_config.area_cost_weight_px,
        "tracking_signal_cost_weight_px": tracking_config.signal_cost_weight_px,
        "tracking_lateral_cost_weight": tracking_config.lateral_cost_weight,
        "tracking_max_area_ratio": tracking_config.max_area_ratio,
        "map_mask_iterations": map_mask_iterations,
        "map_particle_mask_threshold": map_particle_mask_threshold,
        "map_particle_mask_mode": map_particle_mask_mode,
        "map_particle_mask_grow_threshold": map_particle_mask_grow_threshold,
        "map_particle_mask_dilation_px": map_particle_mask_dilation_px,
        "map_particle_mask_margin_px": map_particle_mask_margin_px,
        "map_particle_mask_min_area_px": map_particle_mask_min_area_px,
        "map_aggregation": map_aggregation,
        "map_robust_iterations": map_robust_iterations,
        "map_robust_huber_delta": map_robust_huber_delta,
        "map_robust_min_scale": map_robust_min_scale,
        "phase_refinement_iterations": phase_refinement_iterations,
        "phase_refinement_min_score": phase_refinement_min_score,
        "phase_refinement_max_abs_correction_px": phase_refinement_max_abs_correction_px,
        "phase_refinement_smoothing_window_frames": phase_refinement_smoothing_window_frames,
        "reused_belt_map": reuse_belt_map_path is not None,
        "reuse_belt_map_path": "" if reuse_belt_map_path is None else str(reuse_belt_map_path),
        "reuse_phase_estimates_path": "" if reuse_phase_estimates_path is None else str(reuse_phase_estimates_path),
        "reuse_static_noise_path": "" if reuse_static_noise_path is None else str(reuse_static_noise_path),
        "reuse_static_background_path": "" if reuse_static_background_path is None else str(reuse_static_background_path),
        "reuse_recurrent_artifact_map_path": "" if reuse_recurrent_artifact_map_path is None else str(reuse_recurrent_artifact_map_path),
        "static_noise_sample_frames": static_noise_sample_frames,
        "static_noise_min_scale": static_noise_min_scale,
        "static_noise_mask_threshold": static_noise_mask_threshold,
        "static_noise_mask_margin_px": static_noise_mask_margin_px,
        "static_noise_mask_min_area_px": static_noise_mask_min_area_px,
        "static_noise_map_used": static_noise_map is not None,
        "static_background_sample_frames": static_background_sample_frames,
        "static_background_mask_threshold": static_background_mask_threshold,
        "static_background_mask_margin_px": static_background_mask_margin_px,
        "static_background_mask_min_area_px": static_background_mask_min_area_px,
        "static_background_map_used": static_background_map is not None,
        "recurrent_artifact_min_revolutions": recurrent_artifact_config.min_revolutions,
        "recurrent_artifact_margin_px": recurrent_artifact_config.margin_px,
        "recurrent_artifact_max_overlap_fraction": recurrent_artifact_config.max_overlap_fraction,
        "recurrent_artifact_min_recurrence_probability": recurrent_artifact_config.min_recurrence_probability,
        "recurrent_artifact_mode": recurrent_artifact_config.mode,
        "recurrent_artifact_soft_penalty_weight": recurrent_artifact_config.soft_penalty_weight,
        "recurrent_artifact_filter_used": recurrent_artifact_enabled,
        "recurrent_artifact_source": recurrent_artifact_source,
        "recurrent_artifact_revolutions": recurrent_artifact_revolutions,
        "recurrent_artifact_pixels": recurrent_artifact_pixels,
        "n_recurrent_artifact_rejected": recurrent_artifact_rejected,
        "reuse_metadata_path": "" if reuse_metadata_path is None else str(reuse_metadata_path),
        "phase_estimate_source": "loaded" if reused_phase_estimates is not None else "registration",
        "n_phase_refinement_rows": len(phase_refinement_rows),
        "n_phase_refinement_used": sum(1 for row in phase_refinement_rows if row.get("used_for_refinement")),
        "n_phase_estimates": len(phase_rows),
        "n_detections": len(detection_rows),
        "n_tracks": len(tracks),
        "n_velocity_estimates": len(velocity_rows),
        "n_filtered_velocity_estimates": len(filtered_velocity_rows),
        "track_filter_min_length": track_filter_config.min_track_length,
        "track_filter_min_velocity_ratio_y": track_filter_config.min_velocity_ratio_y,
        "track_filter_max_velocity_ratio_y": track_filter_config.max_velocity_ratio_y,
        "track_filter_max_abs_x_velocity_px_per_frame": track_filter_config.max_abs_x_velocity_px_per_frame,
        "auto_velocity_pair_shifts": pair_shifts,
        "elapsed_s": rt.elapsed_s(),
    }
    metadata_path = rt.OUT / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rt.emit("done", "finished BeltMap image driver", metadata_json=metadata_path)


if __name__ == "__main__":  # pragma: no cover
    main()

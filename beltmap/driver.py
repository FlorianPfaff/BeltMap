"""Packaged image-sequence driver entry point for BeltMap."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import warnings
from dataclasses import asdict, dataclass, replace
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
from .advanced_quality import (
    apply_gain_offset,
    robust_gain_offset,
    smooth_phase_velocity,
    theil_sen_slope,
    unwrap_periodic,
)
from .cross_map_agreement import (
    CrossMapAgreementConfig,
    CrossMapAgreementScore,
    filter_detections_by_agreement,
    score_cross_map_agreement,
)
from .detection import detect_particles_from_residual_hysteresis, normalize_detection_mode
from ._driver_map import (
    PHASE_REFINEMENT_FIELDS,
    PhaseFeedbackConfig,
    build_belt_map_result,
    detect_map_particle_mask,
    estimate_local_illumination_field,
    expanded_detection_mask,
    map_sampling_strategy_from_env,
    sample_indices,
)
from ._driver_motion import (
    estimate_velocity,
    parse_region,
    resolve_supplied_velocity,
    validate_auto_velocity_region,
)
from .map_risk import (
    BeltMapRiskMaps,
    compute_belt_map_risk_maps,
    load_belt_map_support,
    score_map_risk_detections,
)
from .phase import (
    PhaseDriftConfig,
    PhaseDriftFilter,
    PhaseTrajectorySmoothingConfig,
    refine_phase_by_registration,
    smooth_phase_estimates,
)
from .recurrent_artifacts import (
    RECURRENT_ARTIFACT_MODES,
    belt_revolution_indices,
    build_recurrent_artifact_map,
    score_recurrent_artifact_detections,
    score_recurrent_artifact_detections_excluding_current_revolution,
)
from .revolution_split import (
    REVOLUTION_SPLIT_DETECTION_SUMMARY_FIELDS,
    REVOLUTION_SPLIT_FRAME_FIELDS,
    REVOLUTION_SPLIT_REVOLUTION_FIELDS,
    REVOLUTION_SPLIT_SCORE_SUMMARY_FIELDS,
    RevolutionSplit,
    build_revolution_split,
    parse_revolution_indices,
    revolution_split_detection_summary_rows,
    revolution_split_frame_rows,
    revolution_split_revolution_rows,
    revolution_split_score_summary_rows,
)
from .residual import generate_residual_image

DETECTION_FIELDS = [
    "frame_index", "image", "label", "y", "x", "area_px",
    "bbox_top", "bbox_left", "bbox_bottom", "bbox_right",
    "mean_signal", "peak_signal",
    "map_support_min",
    "map_support_mean",
    "map_risk_mean",
    "map_risk_max",
    "map_interpolated_fraction",
    "map_low_support_fraction",
    "recurrent_artifact_overlap_fraction",
    "recurrent_artifact_probability",
    "recurrent_artifact_required_peak_signal",
]
PHASE_FIELDS = [
    "frame_index", "image", "phase_px", "phase_fraction", "phase_rad",
    "predicted_phase_px", "correction_px", "phase_drift_px", "loss", "score", "method",
]
PHASE_ESTIMATION_MODES = {
    "motion_model",
    "registration",
    "smoothed_registration",
}
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
    "map_support_min",
    "map_support_mean",
    "map_risk_mean",
    "map_risk_max",
    "map_interpolated_fraction",
    "map_low_support_fraction",
    "recurrent_artifact_overlap_fraction",
    "recurrent_artifact_probability",
    "recurrent_artifact_required_peak_signal",
]
RECURRENT_ARTIFACT_DETECTION_FIELDS = [
    *DETECTION_FIELDS,
    "recurrent_artifact_rejected",
]
MAP_RISK_DETECTION_FIELDS = [
    *DETECTION_FIELDS,
    "map_risk_rejected",
]
TRACK_SCORE_FIELDS = [
    "track_id", "n_detections", "frame_start", "frame_end",
    "velocity_y_px_per_frame", "velocity_x_px_per_frame",
    "velocity_ratio_y", "abs_x_velocity_px_per_frame",
    "passes_min_track_length", "passes_velocity_ratio",
    "passes_lateral_velocity",
    "n_recurrent_artifact_scored_detections",
    "mean_recurrent_artifact_overlap_fraction",
    "max_recurrent_artifact_overlap_fraction",
    "mean_recurrent_artifact_probability",
    "max_recurrent_artifact_probability",
    "recurrent_artifact_hit_fraction",
    "recurrent_artifact_track_score",
    "passes_recurrent_artifact",
    "accepted", "plausibility_score",
]
PHOTOMETRIC_FIELDS = [
    "frame_index", "image", "gain", "offset", "n_pixels", "rmse_gray", "trimmed_fraction", "status",
]
REVOLUTION_SPLIT_GHOST_DETECTION_FIELDS = [
    *DETECTION_FIELDS,
    "revolution_index", "split",
    "train_artifact_rejected",
]
LOCAL_ILLUMINATION_FIELDS = [
    "frame_index", "image", "tile_px",
    "mask_threshold", "mask_mode", "mask_grow_threshold",
    "mask_dilation_px", "mask_margin_px", "mask_min_area_px",
    "fit_pixels", "masked_pixels",
    "field_median_gray", "field_p05_gray", "field_p95_gray", "field_max_abs_gray",
    "residual_median_before_gray", "residual_median_after_gray",
    "residual_rmse_before_gray", "residual_rmse_after_gray",
    "status",
]


@dataclass(frozen=True)
class ReusedPhaseEstimate:
    """Loaded phase estimate plus the source image recorded in phase_estimates.csv."""

    estimate: PhaseEstimate
    image: str | None = None


def optional_positive_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def optional_positive_float(name: str, default: float = 0.0) -> float | None:
    value = rt.env_float(name, default, minimum=0.0)
    return None if value <= 0 else value


def probability_env_float(name: str, default: float) -> float:
    value = rt.env_float(name, default, minimum=0.0)
    if value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def optional_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value) if value else None


def cross_map_agreement_sample_folds(
    samples: tuple[int, ...] | list[int],
    *,
    min_samples_per_map: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    unique = tuple(sorted({int(index) for index in samples}))
    if len(unique) < 2 * min_samples_per_map:
        raise ValueError(
            "CROSS_MAP_AGREEMENT_ENABLED requires at least "
            f"{2 * min_samples_per_map} sampled map frames; got {len(unique)}"
        )
    folds = (unique[0::2], unique[1::2])
    if min(len(fold) for fold in folds) < min_samples_per_map:
        raise ValueError(
            "cross-fitted map split produced too few samples per map: "
            f"{[len(fold) for fold in folds]} < {min_samples_per_map}"
        )
    return folds


def static_residual_sample_frames(name: str, *, frame_count: int) -> int:
    value = os.getenv(name, "").strip().lower()
    if value == "auto":
        return max(1, min(frame_count, 120))
    return rt.env_int(name, 0, minimum=0)


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
                drift_px=optional_csv_float(row, "phase_drift_px") or 0.0,
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
    if np.issubdtype(artifact_map.dtype, np.integer):
        if np.any((artifact_map < 0) | (artifact_map > 1)):
            raise ValueError(
                "reused recurrent artifact integer map values must be 0 or 1"
            )
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
        "image": _relative_image_name(path, data_dir=rt.DATA),
        "phase_px": estimate.phase_px,
        "phase_fraction": phase_fraction,
        "phase_rad": phase_fraction * 2.0 * np.pi,
        "predicted_phase_px": estimate.predicted_phase_px,
        "correction_px": estimate.correction_px,
        "phase_drift_px": estimate.drift_px,
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


def normalize_phase_estimation_mode(value: str) -> str:
    """Return the phase source used during residual rendering."""

    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "nominal": "motion_model",
        "fixed": "motion_model",
        "fixed_nominal_velocity": "motion_model",
        "motion_model": "motion_model",
        "registration": "registration",
        "texture_registration": "registration",
        "smoothed": "smoothed_registration",
        "smoothed_registration": "smoothed_registration",
        "texture_smoothed": "smoothed_registration",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(PHASE_ESTIMATION_MODES))
        raise ValueError(
            f"PHASE_ESTIMATION_MODE must be one of {choices}; got {value!r}"
        ) from exc


def motion_model_phase_estimate(
    motion_model: BeltMotionModel,
    *,
    frame_index: float,
) -> PhaseEstimate:
    """Build an explicit nominal phase estimate to bypass registration."""

    phase = motion_model.phase_at(frame_index)
    return PhaseEstimate(
        phase_px=phase,
        frame_index=frame_index,
        predicted_phase_px=phase,
        method="motion_model",
    )


def nominal_phase_estimates(
    paths: list[Path],
    motion_model: BeltMotionModel,
) -> dict[int, PhaseEstimate]:
    return {
        index: motion_model_phase_estimate(motion_model, frame_index=float(index))
        for index, _path in enumerate(paths)
    }


def texture_phase_velocity_summary(
    phase_rows: list[dict],
    *,
    period_px: float,
    nominal_velocity_px_per_frame: float,
) -> dict[str, float | str | int | None]:
    """Estimate belt velocity from registered phase rows for diagnostics."""

    frames: list[float] = []
    phases: list[float] = []
    scores: list[float] = []
    methods: list[str] = []
    for row in phase_rows:
        try:
            frame = float(row["frame_index"])
            phase = float(row["phase_px"])
        except (TypeError, ValueError, KeyError):
            continue
        frames.append(frame)
        phases.append(phase)
        try:
            score = float(row.get("score", ""))
        except (TypeError, ValueError):
            score = float("nan")
        scores.append(score)
        methods.append(str(row.get("method", "")))
    if len(frames) < 2:
        return {
            "texture_phase_velocity_status": "insufficient_phase_rows",
            "texture_phase_velocity_samples": len(frames),
        }
    if not any("registration" in method for method in methods):
        return {
            "texture_phase_velocity_status": "not_texture_registered",
            "texture_phase_velocity_samples": len(frames),
        }
    frame_arr = np.asarray(frames, dtype=np.float64)
    phase_arr = np.asarray(phases, dtype=np.float64)
    score_arr = np.asarray(scores, dtype=np.float64)
    unwrapped = unwrap_periodic(phase_arr, period_px)
    velocity = -theil_sen_slope(frame_arr, unwrapped)
    smoothed = smooth_phase_velocity(
        phase_arr,
        period_px=period_px,
        scores=score_arr if np.isfinite(score_arr).any() else None,
    )
    smoothed_velocity = np.asarray(
        smoothed["velocity_px_per_frame"],
        dtype=np.float64,
    )
    finite_smoothed = smoothed_velocity[np.isfinite(smoothed_velocity)]
    median_smoothed: float | None = (
        float(np.median(finite_smoothed))
        if finite_smoothed.size
        else None
    )
    return {
        "texture_phase_velocity_status": "ok",
        "texture_phase_velocity_samples": len(frames),
        "texture_phase_velocity_px_per_frame": float(velocity),
        "texture_phase_velocity_error_px_per_frame": float(
            velocity - nominal_velocity_px_per_frame
        ),
        "texture_phase_smoothed_velocity_px_per_frame": median_smoothed,
        "texture_phase_smoothed_velocity_error_px_per_frame": (
            float(median_smoothed - nominal_velocity_px_per_frame)
            if median_smoothed is not None
            else None
        ),
    }


def write_phase_refinement_outputs(phase_refinement_rows: list[dict]) -> None:
    rt.write_csv(rt.OUT / "phase_refinement.csv", phase_refinement_rows, PHASE_REFINEMENT_FIELDS)


def write_photometric_outputs(photometric_rows: list[dict]) -> None:
    rt.write_csv(rt.OUT / "photometric_fits.csv", photometric_rows, PHOTOMETRIC_FIELDS)


def write_local_illumination_outputs(local_illumination_rows: list[dict]) -> None:
    rt.write_csv(rt.OUT / "local_illumination_fits.csv", local_illumination_rows, LOCAL_ILLUMINATION_FIELDS)


def detection_rows_for_frame(detections: list, path: Path, frame_index: int) -> list[dict]:
    rows: list[dict] = []
    for detection in detections:
        row = {
            field: getattr(detection, field)
            for field in DETECTION_FIELDS
            if field != "image"
        }
        row["frame_index"] = frame_index
        row["image"] = _relative_image_name(path, data_dir=rt.DATA)
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


def revolution_split_ghost_rows_from_scores(
    scored_by_frame: list,
    paths: list[Path],
    split: RevolutionSplit,
) -> list[dict]:
    rows: list[dict] = []
    for frame_index, scores in enumerate(scored_by_frame):
        for score in scores:
            row = detection_rows_for_frame(
                [score.detection],
                paths[frame_index],
                frame_index,
            )[0]
            row["revolution_index"] = split.revolution_by_frame[frame_index]
            row["split"] = split.frame_split[frame_index]
            row["train_artifact_rejected"] = score.rejected
            rows.append(row)
    return rows


def map_risk_rows_from_scores(
    scored_detections: list,
    path: Path,
    frame_index: int,
) -> list[dict]:
    rows: list[dict] = []
    for score in scored_detections:
        row = detection_rows_for_frame(
            [score.detection],
            path,
            frame_index,
        )[0]
        row["map_risk_rejected"] = score.rejected
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
            row["image"] = _relative_image_name(paths[frame_index], data_dir=rt.DATA)
            rows.append(row)
    return rows


def should_save_residual_preview(frame_index: int, preview_frames: int, preview_interval: int) -> bool:
    return frame_index < preview_frames or (preview_interval > 0 and frame_index % preview_interval == 0)


def detect_final_particle_mask(
    residual: ResidualImage,
    *,
    method: str,
    threshold: float,
    grow_threshold: float,
) -> np.ndarray:
    """Return the final particle mask for the configured detector."""

    if method == "threshold":
        return detect_particles_from_residual(residual, threshold=threshold)
    if method == "hysteresis":
        return detect_particles_from_residual_hysteresis(
            residual,
            threshold=threshold,
            grow_threshold=grow_threshold,
        )
    if method == "hysteresis_abs":
        return detect_particles_from_residual_hysteresis(
            residual,
            threshold=threshold,
            grow_threshold=grow_threshold,
            absolute=True,
        )
    raise ValueError(f"unknown detection method {method!r}")


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


def apply_photometric_correction(
    *,
    frame: np.ndarray,
    residual: ResidualImage,
    residual_config: ResidualConfig,
    frame_index: int,
    path: Path,
    enabled: bool,
    trim_fraction: float,
    max_iterations: int,
    min_pixels: int,
) -> tuple[ResidualImage, dict | None]:
    """Fit and apply a per-frame gain/offset correction to the clean render."""

    if not enabled:
        return residual, None

    row: dict = {
        "frame_index": frame_index,
        "image": _relative_image_name(path, data_dir=rt.DATA),
        "gain": "",
        "offset": "",
        "n_pixels": "",
        "rmse_gray": "",
        "trimmed_fraction": "",
        "status": "ok",
    }
    try:
        fit = robust_gain_offset(
            observed=frame,
            expected=residual.expected_background,
            mask=residual.mask,
            trim_fraction=trim_fraction,
            max_iterations=max_iterations,
            min_pixels=min_pixels,
        )
    except ValueError as exc:
        row["status"] = f"skipped:{exc}"
        return residual, row

    corrected_expected = apply_gain_offset(residual.expected_background, fit)
    corrected = generate_residual_image(
        frame,
        corrected_expected,
        mask=residual.mask,
        config=residual_config,
    )
    corrected = ResidualImage(
        raw=corrected.raw,
        local_noise=corrected.local_noise,
        normalized=corrected.normalized,
        mask=corrected.mask,
        expected_background=corrected.expected_background,
        clean_render=residual.clean_render,
    )
    row.update(
        gain=fit.gain,
        offset=fit.offset,
        n_pixels=fit.n_pixels,
        rmse_gray=fit.rmse_gray,
        trimmed_fraction=fit.trimmed_fraction,
    )
    return corrected, row


def apply_local_illumination_correction(
    *,
    frame: np.ndarray,
    residual: ResidualImage,
    residual_config: ResidualConfig,
    enabled: bool,
    tile_px: int,
    mask_threshold: float,
    mask_mode: str,
    mask_grow_threshold: float,
    mask_dilation_px: int,
    mask_margin_px: int,
    mask_min_area_px: int,
) -> ResidualImage:
    """Subtract a low-frequency additive residual field before detection."""

    if not enabled:
        return residual
    if tile_px < 1:
        raise ValueError("local illumination tile size must be positive")

    particle_mask = detect_map_particle_mask(
        residual,
        mode=mask_mode,
        threshold=mask_threshold,
        grow_threshold=mask_grow_threshold,
        dilation_px=mask_dilation_px,
        margin_px=mask_margin_px,
        min_area_px=mask_min_area_px,
    )
    field_valid = np.asarray(residual.mask, dtype=bool) & ~particle_mask
    illumination_field = estimate_local_illumination_field(
        residual.raw,
        field_valid,
        tile_px=tile_px,
    )
    corrected_expected = residual.expected_background + illumination_field
    corrected = generate_residual_image(
        frame,
        corrected_expected,
        mask=residual.mask,
        config=residual_config,
    )
    return ResidualImage(
        raw=corrected.raw,
        local_noise=corrected.local_noise,
        normalized=corrected.normalized,
        mask=corrected.mask,
        expected_background=corrected.expected_background,
        clean_render=residual.clean_render,
    )


def estimate_smoothed_phase_sequence(
    *,
    paths: list[Path],
    region: tuple[int, int, int, int],
    belt_map: np.ndarray,
    motion_model: BeltMotionModel,
    registration_config: PhaseRegistrationConfig,
    phase_drift_config: PhaseDriftConfig,
    window_radius_frames: int,
    min_score: float | None,
    max_abs_correction_px: float | None,
    min_support: int,
) -> dict[int, PhaseEstimate] | None:
    """Estimate all phases once and smooth the registration correction trajectory."""

    if window_radius_frames <= 0:
        return None
    rt.emit(
        "phase_smoothing",
        "estimating all frame phases for offline smoothing",
        selected_frames=len(paths),
        window_radius_frames=window_radius_frames,
        min_score=min_score,
        max_abs_correction_px=max_abs_correction_px,
        min_support=min_support,
    )
    drift_filter = PhaseDriftFilter(
        phase_drift_config,
        period_px=motion_model.period_px,
    )
    estimates: list[PhaseEstimate] = []
    progress_interval = rt.env_int("PROGRESS_INTERVAL_FRAMES", 25, minimum=1)
    start = rt.time.perf_counter()
    for frame_index, path in enumerate(paths):
        frame = rt.crop(rt.read_gray(path), region)
        nominal_phase = motion_model.phase_at(float(frame_index))
        predicted_phase = (
            drift_filter.predict(nominal_phase)
            if phase_drift_config.enabled
            else nominal_phase
        )
        estimate = refine_phase_by_registration(
            frame=frame,
            belt_map=belt_map,
            predicted_phase_px=predicted_phase,
            frame_index=float(frame_index),
            period_px=motion_model.period_px,
            config=registration_config,
        )
        if phase_drift_config.enabled:
            estimate = drift_filter.observe(estimate)
        estimates.append(estimate)
        processed = frame_index + 1
        if processed == 1 or processed == len(paths) or processed % progress_interval == 0:
            dt = rt.time.perf_counter() - start
            rt.emit(
                "phase_smoothing",
                f"estimated {processed}/{len(paths)} frame phases",
                processed_frames=processed,
                frames_per_second=round(processed / dt, 4) if dt > 0 else None,
            )

    smoothed = smooth_phase_estimates(
        estimates,
        period_px=motion_model.period_px,
        config=PhaseTrajectorySmoothingConfig(
            window_radius_frames=window_radius_frames,
            min_score=min_score,
            max_abs_correction_px=max_abs_correction_px,
            min_support=min_support,
        ),
    )
    rt.emit("phase_smoothing", "finished offline phase smoothing", phase_estimates=len(smoothed))
    return {index: estimate for index, estimate in enumerate(smoothed)}


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
    mask_mode: str = "positive",
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
        mask_mode=mask_mode,
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
                mask = detect_particles_from_residual(
                    residual,
                    threshold=mask_threshold,
                    mode=mask_mode,
                )
                detections = extract_particle_detections(
                    mask,
                    residual=residual,
                    signal_mode=mask_mode,
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
    mask_mode: str = "positive",
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
        mask_mode=mask_mode,
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
                mask = detect_particles_from_residual(
                    residual,
                    threshold=mask_threshold,
                    mode=mask_mode,
                )
                detections = extract_particle_detections(
                    mask,
                    residual=residual,
                    signal_mode=mask_mode,
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
    reuse_belt_map_input_path = optional_path("REUSE_BELT_MAP_PATH")
    reuse_map_support_input_path = optional_path("REUSE_MAP_SUPPORT_PATH")
    if reuse_map_support_input_path is None and reuse_belt_map_input_path is not None:
        sibling_support = reuse_belt_map_input_path.with_name("belt_map_support.npy")
        if sibling_support.exists():
            reuse_map_support_input_path = sibling_support
    protected_output_inputs = [
        path for path in (
            reuse_belt_map_input_path,
            reuse_map_support_input_path,
            optional_path("REUSE_STATIC_NOISE_PATH"),
            optional_path("REUSE_STATIC_BACKGROUND_PATH"),
            optional_path("REUSE_RECURRENT_ARTIFACT_MAP_PATH"),
            optional_path("REUSE_PHASE_ESTIMATES_PATH"),
        )
        if path is not None
    ]
    if reuse_belt_map_input_path is not None:
        protected_output_inputs.append(
            reuse_belt_map_input_path.with_name("metadata.json")
        )
    rt.clear_generated_outputs(protected_paths=protected_output_inputs)
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
    detection_mode = normalize_detection_mode(os.getenv("DETECTION_MODE", "positive"))
    detection_low_threshold = optional_positive_float("DETECTION_LOW_THRESHOLD", 0.0)
    min_area_px = rt.env_int("MIN_AREA_PX", 4, minimum=1)
    detection_max_area_px = optional_positive_int("DETECTION_MAX_AREA_PX")
    detection_min_bbox_width_px = optional_positive_int("DETECTION_MIN_BBOX_WIDTH_PX")
    detection_min_bbox_height_px = optional_positive_int("DETECTION_MIN_BBOX_HEIGHT_PX")
    detection_max_bbox_aspect_ratio = optional_positive_float(
        "DETECTION_MAX_BBOX_ASPECT_RATIO",
        0.0,
    )
    detection_min_bbox_extent = optional_positive_float("DETECTION_MIN_BBOX_EXTENT", 0.0)
    detection_split_merged_components = env_bool("DETECTION_SPLIT_MERGED_COMPONENTS", False)
    detection_split_min_projection_gap_px = rt.env_int(
        "DETECTION_SPLIT_MIN_PROJECTION_GAP_PX", 2, minimum=1
    )
    detection_split_min_component_area_px = optional_positive_int(
        "DETECTION_SPLIT_MIN_COMPONENT_AREA_PX"
    )
    residual_noise_radius_px = rt.env_int("RESIDUAL_NOISE_RADIUS_PX", 15, minimum=0)
    residual_clip_sigma = optional_positive_float("RESIDUAL_CLIP_SIGMA", 5.0)
    residual_min_noise = rt.env_float("RESIDUAL_MIN_NOISE", 1e-6, minimum=0.0)
    if residual_min_noise <= 0:
        raise ValueError("RESIDUAL_MIN_NOISE must be positive")
    residual_noise_exclusion_sigma = optional_positive_float("RESIDUAL_NOISE_EXCLUSION_SIGMA", 4.0)
    residual_noise_exclusion_radius_px = rt.env_int(
        "RESIDUAL_NOISE_EXCLUSION_RADIUS_PX", 2, minimum=0
    )
    photometric_enabled = env_bool("PHOTOMETRIC_ENABLED", False)
    photometric_trim_fraction = rt.env_float("PHOTOMETRIC_TRIM_FRACTION", 0.05, minimum=0.0)
    if photometric_trim_fraction >= 0.5:
        raise ValueError("PHOTOMETRIC_TRIM_FRACTION must be in [0, 0.5)")
    photometric_max_iterations = rt.env_int("PHOTOMETRIC_MAX_ITERATIONS", 3, minimum=1)
    photometric_min_pixels = rt.env_int("PHOTOMETRIC_MIN_PIXELS", 128, minimum=1)
    detection_local_illumination_correction = env_bool("DETECTION_LOCAL_ILLUMINATION_CORRECTION", False)
    detection_local_illumination_tile_px = rt.env_int("DETECTION_LOCAL_ILLUMINATION_TILE_PX", 64, minimum=1)
    detection_local_illumination_min_pixels = rt.env_int("DETECTION_LOCAL_ILLUMINATION_MIN_PIXELS", 128, minimum=1)
    min_track_length = rt.env_int("MIN_TRACK_LENGTH", 2, minimum=2)
    tracking_max_frame_gap = rt.env_float("TRACKING_MAX_FRAME_GAP", 1.0, minimum=1e-9)
    tracking_velocity_fit_method = os.getenv("TRACKING_VELOCITY_FIT_METHOD", "linear").strip().lower()
    track_filter_max_recurrent_artifact_track_score = optional_positive_float(
        "TRACK_FILTER_MAX_RECURRENT_ARTIFACT_SCORE",
        0.0,
    )
    if (
        track_filter_max_recurrent_artifact_track_score is not None
        and track_filter_max_recurrent_artifact_track_score > 1
    ):
        raise ValueError(
            "TRACK_FILTER_MAX_RECURRENT_ARTIFACT_SCORE must be in [0, 1] or 0 to disable"
        )
    track_filter_recurrent_artifact_detection_threshold = rt.env_float(
        "TRACK_FILTER_RECURRENT_ARTIFACT_DETECTION_THRESHOLD",
        0.3,
        minimum=0.0,
    )
    if track_filter_recurrent_artifact_detection_threshold > 1:
        raise ValueError("TRACK_FILTER_RECURRENT_ARTIFACT_DETECTION_THRESHOLD must be in [0, 1]")
    map_mask_iterations = rt.env_int("MAP_MASK_ITERATIONS", 1, minimum=0)
    map_sampling_strategy = map_sampling_strategy_from_env()
    map_particle_mask_threshold = rt.env_float("MAP_PARTICLE_MASK_THRESHOLD", detection_threshold, minimum=0.0)
    map_particle_mask_mode_value = os.getenv("MAP_PARTICLE_MASK_MODE", "").strip().lower()
    if map_particle_mask_mode_value:
        map_particle_mask_mode = map_particle_mask_mode_value
    elif detection_mode in {"negative", "absolute"}:
        map_particle_mask_mode = detection_mode
    else:
        map_particle_mask_mode = "positive"
    map_particle_mask_grow_threshold = rt.env_float("MAP_PARTICLE_MASK_GROW_THRESHOLD", 2.0, minimum=0.0)
    map_particle_mask_dilation_px = rt.env_int("MAP_PARTICLE_MASK_DILATION_PX", 0, minimum=0)
    map_fractional_splat = env_bool("MAP_FRACTIONAL_SPLAT", True)
    map_frame_median_offset_correction = env_bool("MAP_FRAME_MEDIAN_OFFSET_CORRECTION", False)
    map_local_illumination_correction = env_bool("MAP_LOCAL_ILLUMINATION_CORRECTION", False)
    map_local_illumination_tile_px = rt.env_int("MAP_LOCAL_ILLUMINATION_TILE_PX", 64, minimum=1)
    map_particle_mask_margin_px = rt.env_int("MAP_PARTICLE_MASK_MARGIN_PX", 8, minimum=0)
    map_particle_mask_min_area_px = rt.env_int("MAP_PARTICLE_MASK_MIN_AREA_PX", min_area_px, minimum=1)
    detection_local_illumination_mask_threshold = rt.env_float(
        "DETECTION_LOCAL_ILLUMINATION_MASK_THRESHOLD",
        map_particle_mask_threshold,
        minimum=0.0,
    )
    detection_local_illumination_mask_mode_value = os.getenv(
        "DETECTION_LOCAL_ILLUMINATION_MASK_MODE", ""
    ).strip().lower()
    detection_local_illumination_mask_mode = (
        detection_local_illumination_mask_mode_value or map_particle_mask_mode
    )
    detection_local_illumination_mask_grow_threshold = rt.env_float(
        "DETECTION_LOCAL_ILLUMINATION_MASK_GROW_THRESHOLD",
        map_particle_mask_grow_threshold,
        minimum=0.0,
    )
    detection_local_illumination_mask_dilation_px = rt.env_int("DETECTION_LOCAL_ILLUMINATION_MASK_DILATION_PX", map_particle_mask_dilation_px, minimum=0)
    detection_local_illumination_mask_margin_px = rt.env_int("DETECTION_LOCAL_ILLUMINATION_MASK_MARGIN_PX", map_particle_mask_margin_px, minimum=0)
    detection_local_illumination_mask_min_area_px = rt.env_int("DETECTION_LOCAL_ILLUMINATION_MASK_MIN_AREA_PX", map_particle_mask_min_area_px, minimum=1)
    map_aggregation = os.getenv("MAP_AGGREGATION", "mean").strip().lower()
    map_robust_iterations = rt.env_int("MAP_ROBUST_ITERATIONS", 1, minimum=0)
    map_robust_huber_delta = rt.env_float(
        "MAP_ROBUST_HUBER_DELTA", 3.0, minimum=1e-9
    )
    map_robust_min_scale = rt.env_float(
        "MAP_ROBUST_MIN_SCALE", 1.0, minimum=1e-9
    )
    revolution_split_enabled = env_bool("REVOLUTION_SPLIT_ENABLED", False)
    revolution_split_eval_every = rt.env_int("REVOLUTION_SPLIT_EVAL_EVERY", 3, minimum=1)
    revolution_split_eval_offset = rt.env_int("REVOLUTION_SPLIT_EVAL_OFFSET", 0, minimum=0)
    revolution_split_eval_revolutions_spec = os.getenv("REVOLUTION_SPLIT_EVAL_REVOLUTIONS", "")
    revolution_split_min_train_revolutions = rt.env_int("REVOLUTION_SPLIT_MIN_TRAIN_REVOLUTIONS", 1, minimum=1)
    revolution_split_min_eval_revolutions = rt.env_int("REVOLUTION_SPLIT_MIN_EVAL_REVOLUTIONS", 1, minimum=1)
    revolution_split_ghost_min_revolutions = rt.env_int("REVOLUTION_SPLIT_GHOST_MIN_REVOLUTIONS", 2, minimum=1)
    reuse_belt_map_path = optional_path("REUSE_BELT_MAP_PATH")
    reuse_phase_estimates_path = optional_path("REUSE_PHASE_ESTIMATES_PATH")
    reuse_static_noise_path = optional_path("REUSE_STATIC_NOISE_PATH")
    reuse_static_background_path = optional_path("REUSE_STATIC_BACKGROUND_PATH")
    reuse_recurrent_artifact_map_path = optional_path("REUSE_RECURRENT_ARTIFACT_MAP_PATH")
    if reuse_phase_estimates_path is not None and reuse_belt_map_path is None:
        raise ValueError("REUSE_PHASE_ESTIMATES_PATH requires REUSE_BELT_MAP_PATH")
    static_noise_sample_frames = static_residual_sample_frames("STATIC_NOISE_SAMPLE_FRAMES", frame_count=len(paths))
    static_noise_min_scale = rt.env_float("STATIC_NOISE_MIN_SCALE", 0.0, minimum=0.0)
    static_noise_mask_threshold = optional_positive_float("STATIC_NOISE_MASK_THRESHOLD", detection_threshold)
    static_noise_mask_margin_px = rt.env_int("STATIC_NOISE_MASK_MARGIN_PX", 8, minimum=0)
    static_noise_mask_min_area_px = rt.env_int("STATIC_NOISE_MASK_MIN_AREA_PX", min_area_px, minimum=1)
    static_background_sample_frames = static_residual_sample_frames("STATIC_BACKGROUND_SAMPLE_FRAMES", frame_count=len(paths))
    static_background_mask_threshold = optional_positive_float("STATIC_BACKGROUND_MASK_THRESHOLD", detection_threshold)
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
        candidate_max_area_px=optional_positive_int("RECURRENT_ARTIFACT_CANDIDATE_MAX_AREA_PX"),
        candidate_max_peak_signal=optional_positive_float(
            "RECURRENT_ARTIFACT_CANDIDATE_MAX_PEAK_SIGNAL",
            0.0,
        ),
        reject_max_area_px=optional_positive_int("RECURRENT_ARTIFACT_REJECT_MAX_AREA_PX"),
        reject_max_peak_signal=optional_positive_float(
            "RECURRENT_ARTIFACT_REJECT_MAX_PEAK_SIGNAL",
            0.0,
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
    cross_map_agreement_enabled = env_bool("CROSS_MAP_AGREEMENT_ENABLED", False)
    cross_map_agreement_filter = env_bool("CROSS_MAP_AGREEMENT_FILTER", True)
    cross_map_agreement_min_samples_per_map = rt.env_int(
        "CROSS_MAP_AGREEMENT_MIN_SAMPLES_PER_MAP",
        10,
        minimum=1,
    )
    cross_map_agreement_config = CrossMapAgreementConfig(
        max_centroid_distance_px=rt.env_float(
            "CROSS_MAP_AGREEMENT_MAX_CENTROID_DISTANCE_PX",
            4.0,
            minimum=0.0,
        ),
        min_bbox_iou=rt.env_float("CROSS_MAP_AGREEMENT_MIN_BBOX_IOU", 0.0, minimum=0.0),
        min_peak_ratio=rt.env_float("CROSS_MAP_AGREEMENT_MIN_PEAK_RATIO", 0.25, minimum=0.0),
        require_sign_consistency=env_bool(
            "CROSS_MAP_AGREEMENT_REQUIRE_SIGN_CONSISTENCY",
            True,
        ),
        min_confirming_maps=rt.env_int(
            "CROSS_MAP_AGREEMENT_MIN_CONFIRMING_MAPS",
            2,
            minimum=1,
        ),
        filter_detections=cross_map_agreement_filter,
    )
    if cross_map_agreement_config.min_confirming_maps > 2:
        raise ValueError(
            "CROSS_MAP_AGREEMENT_MIN_CONFIRMING_MAPS cannot exceed 2; "
            "the driver currently builds two cross-fitted maps"
        )
    registration_config = PhaseRegistrationConfig(
        search_radius_px=rt.env_float("REGISTRATION_SEARCH_RADIUS_PX", 8.0, minimum=0.0),
        search_step_px=rt.env_float("REGISTRATION_SEARCH_STEP_PX", 0.5, minimum=1e-9),
        subpixel_refinement=env_bool("REGISTRATION_SUBPIXEL_REFINEMENT", True),
        robust_normalization=env_bool("REGISTRATION_ROBUST_NORMALIZATION", True),
    )
    phase_estimation_mode = normalize_phase_estimation_mode(
        os.getenv("PHASE_ESTIMATION_MODE", "registration")
    )
    phase_drift_config = PhaseDriftConfig(
        enabled=env_bool("PHASE_DRIFT_ENABLED", True),
        smoothing_alpha=rt.env_float("PHASE_DRIFT_SMOOTHING_ALPHA", 0.15, minimum=0.0),
        min_score=rt.env_float("PHASE_DRIFT_MIN_SCORE", 0.05, minimum=0.0),
        max_abs_residual_correction_px=optional_positive_float(
            "PHASE_DRIFT_MAX_ABS_RESIDUAL_CORRECTION_PX",
            0.0,
        ),
        max_abs_drift_px=optional_positive_float("PHASE_DRIFT_MAX_ABS_PX", 0.0),
    )
    phase_smoothing_window_frames = rt.env_int("PHASE_SMOOTHING_WINDOW_FRAMES", 0, minimum=0)
    phase_smoothing_min_score = optional_positive_float(
        "PHASE_SMOOTHING_MIN_SCORE",
        0.0,
    )
    phase_smoothing_max_abs_correction_px = optional_positive_float(
        "PHASE_SMOOTHING_MAX_ABS_CORRECTION_PX",
        0.0,
    )
    phase_smoothing_min_support = rt.env_int("PHASE_SMOOTHING_MIN_SUPPORT", 3, minimum=1)
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
    revolution_split: RevolutionSplit | None = None
    revolution_split_eval_revolutions: tuple[int, ...] = ()
    if revolution_split_enabled:
        if reuse_belt_map_path is not None:
            raise ValueError(
                "REVOLUTION_SPLIT_ENABLED builds a belt map from train revolutions; "
                "it cannot be combined with REUSE_BELT_MAP_PATH"
            )
        if period_px is None:
            raise ValueError(
                "REVOLUTION_SPLIT_ENABLED requires BELT_PERIOD_PX so integer "
                "belt revolutions are well-defined"
            )
        revolution_split_eval_revolutions = parse_revolution_indices(
            revolution_split_eval_revolutions_spec
        )
        revolution_by_frame = belt_revolution_indices(
            len(paths),
            BeltMotionModel(
                image_velocity_px_per_frame=belt_velocity,
                period_px=float(period_px),
            ),
        )
        revolution_split = build_revolution_split(
            revolution_by_frame,
            eval_every=revolution_split_eval_every,
            eval_offset=revolution_split_eval_offset,
            eval_revolutions=revolution_split_eval_revolutions,
            min_train_revolutions=revolution_split_min_train_revolutions,
            min_eval_revolutions=revolution_split_min_eval_revolutions,
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
        detection_split_merged_components=detection_split_merged_components,
        detection_split_min_projection_gap_px=detection_split_min_projection_gap_px,
        detection_split_min_component_area_px=detection_split_min_component_area_px,
        residual_noise_radius_px=residual_noise_radius_px,
        residual_clip_sigma=residual_clip_sigma,
        residual_min_noise=residual_min_noise,
        residual_noise_exclusion_sigma=residual_noise_exclusion_sigma,
        residual_noise_exclusion_radius_px=residual_noise_exclusion_radius_px,
        photometric_enabled=photometric_enabled,
        photometric_trim_fraction=photometric_trim_fraction,
        photometric_max_iterations=photometric_max_iterations,
        photometric_min_pixels=photometric_min_pixels,
        detection_local_illumination_correction=detection_local_illumination_correction,
        detection_local_illumination_tile_px=detection_local_illumination_tile_px,
        detection_local_illumination_min_pixels=detection_local_illumination_min_pixels,
        detection_local_illumination_mask_threshold=detection_local_illumination_mask_threshold,
        detection_local_illumination_mask_mode=detection_local_illumination_mask_mode,
        detection_local_illumination_mask_grow_threshold=detection_local_illumination_mask_grow_threshold,
        detection_local_illumination_mask_dilation_px=detection_local_illumination_mask_dilation_px,
        detection_local_illumination_mask_margin_px=detection_local_illumination_mask_margin_px,
        detection_local_illumination_mask_min_area_px=detection_local_illumination_mask_min_area_px,
        min_track_length=min_track_length,
        tracking_backend="pyrecest_gnn",
        tracking_max_frame_gap=tracking_max_frame_gap,
        tracking_velocity_fit_method=tracking_velocity_fit_method,
        track_filter_max_recurrent_artifact_track_score=track_filter_max_recurrent_artifact_track_score,
        track_filter_recurrent_artifact_detection_threshold=track_filter_recurrent_artifact_detection_threshold,
        map_mask_iterations=map_mask_iterations,
        map_sampling_strategy=map_sampling_strategy,
        map_particle_mask_threshold=map_particle_mask_threshold,
        map_particle_mask_mode=map_particle_mask_mode,
        map_particle_mask_grow_threshold=map_particle_mask_grow_threshold,
        map_particle_mask_dilation_px=map_particle_mask_dilation_px,
        map_fractional_splat=map_fractional_splat,
        map_frame_median_offset_correction=map_frame_median_offset_correction,
        map_particle_mask_margin_px=map_particle_mask_margin_px,
        map_particle_mask_min_area_px=map_particle_mask_min_area_px,
        map_aggregation=map_aggregation,
        map_robust_iterations=map_robust_iterations,
        map_robust_huber_delta=map_robust_huber_delta,
        map_robust_min_scale=map_robust_min_scale,
        map_risk_min_support=map_risk_min_support,
        map_risk_reject_max_mean=map_risk_reject_max_mean,
        map_risk_reject_max_interpolated_fraction=map_risk_reject_max_interpolated_fraction,
        map_risk_reject_max_low_support_fraction=map_risk_reject_max_low_support_fraction,
        reuse_map_support_path=reuse_map_support_path,
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
        recurrent_artifact_candidate_max_area_px=recurrent_artifact_config.candidate_max_area_px,
        recurrent_artifact_candidate_max_peak_signal=recurrent_artifact_config.candidate_max_peak_signal,
        recurrent_artifact_reject_max_area_px=recurrent_artifact_config.reject_max_area_px,
        recurrent_artifact_reject_max_peak_signal=recurrent_artifact_config.reject_max_peak_signal,
        revolution_split_enabled=revolution_split_enabled,
        revolution_split_eval_every=revolution_split_eval_every,
        revolution_split_eval_offset=revolution_split_eval_offset,
        revolution_split_eval_revolutions=revolution_split_eval_revolutions,
        revolution_split_min_train_revolutions=revolution_split_min_train_revolutions,
        revolution_split_min_eval_revolutions=revolution_split_min_eval_revolutions,
        revolution_split_ghost_min_revolutions=revolution_split_ghost_min_revolutions,
        revolution_split_train_revolutions=(
            [] if revolution_split is None else revolution_split.train_revolutions
        ),
        revolution_split_eval_revolutions_observed=(
            [] if revolution_split is None else revolution_split.eval_revolutions
        ),
        revolution_split_train_frames=(0 if revolution_split is None else len(revolution_split.train_frame_indices)),
        revolution_split_eval_frames=(0 if revolution_split is None else len(revolution_split.eval_frame_indices)),
        cross_map_agreement_enabled=cross_map_agreement_enabled,
        cross_map_agreement_filter=cross_map_agreement_config.filter_detections,
        cross_map_agreement_min_confirming_maps=cross_map_agreement_config.min_confirming_maps,
        cross_map_agreement_min_samples_per_map=cross_map_agreement_min_samples_per_map,
        cross_map_agreement_max_centroid_distance_px=(
            cross_map_agreement_config.max_centroid_distance_px
        ),
        cross_map_agreement_min_bbox_iou=cross_map_agreement_config.min_bbox_iou,
        cross_map_agreement_min_peak_ratio=cross_map_agreement_config.min_peak_ratio,
        cross_map_agreement_require_sign_consistency=(
            cross_map_agreement_config.require_sign_consistency
        ),
        registration_search_radius_px=registration_config.search_radius_px,
        registration_search_step_px=registration_config.search_step_px,
        registration_subpixel_refinement=registration_config.subpixel_refinement,
        registration_robust_normalization=registration_config.robust_normalization,
        phase_estimation_mode=phase_estimation_mode,
        phase_drift_enabled=phase_drift_config.enabled,
        phase_drift_smoothing_alpha=phase_drift_config.smoothing_alpha,
        phase_drift_min_score=phase_drift_config.min_score,
        phase_drift_max_abs_residual_correction_px=(
            phase_drift_config.max_abs_residual_correction_px
        ),
        phase_drift_max_abs_px=phase_drift_config.max_abs_drift_px,
        phase_smoothing_window_frames=phase_smoothing_window_frames,
        phase_smoothing_min_score=phase_smoothing_min_score,
        phase_smoothing_max_abs_correction_px=phase_smoothing_max_abs_correction_px,
        phase_smoothing_min_support=phase_smoothing_min_support,
        phase_refinement_iterations=phase_refinement_iterations,
        phase_refinement_min_score=phase_refinement_min_score,
        phase_refinement_max_abs_correction_px=phase_refinement_max_abs_correction_px,
        phase_refinement_smoothing_window_frames=phase_refinement_smoothing_window_frames,
    )

    reuse_metadata: dict = {}
    reuse_metadata_path: Path | None = None
    map_sample_frame_indices: tuple[int, ...] = ()
    phase_refinement_rows: list[dict] = []
    map_support: np.ndarray | None = None
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
        if reuse_map_support_path is not None:
            map_support = load_belt_map_support(
                reuse_map_support_path,
                map_shape=tuple(belt_map.shape),
            )
            rt.emit(
                "map_risk",
                "loaded reused belt-map support",
                source_map_support_npy=reuse_map_support_path,
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
            sampling_strategy=map_sampling_strategy,
            robust_huber_delta=map_robust_huber_delta,
            robust_min_scale=map_robust_min_scale,
            fractional_splat=map_fractional_splat,
            frame_median_offset_correction=map_frame_median_offset_correction,
            local_illumination_correction=map_local_illumination_correction,
            local_illumination_tile_px=map_local_illumination_tile_px,
            phase_feedback_config=PhaseFeedbackConfig(
                iterations=phase_refinement_iterations,
                min_score=phase_refinement_min_score,
                max_abs_correction_px=phase_refinement_max_abs_correction_px,
                smoothing_window_frames=phase_refinement_smoothing_window_frames,
                registration_config=registration_config,
            ),
            allowed_sample_frame_indices=(
                None
                if revolution_split is None
                else revolution_split.train_frame_indices
            ),
        )
        belt_map = build_result.belt_map
        reference_phase = build_result.reference_phase
        map_height = build_result.map_height
        phase_refinement_rows = build_result.phase_refinement_rows
        map_sample_frame_indices = build_result.sample_frame_indices
        map_support = build_result.map_support
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

    map_risk_maps: BeltMapRiskMaps | None = None
    map_risk_low_support_pixels = 0
    map_risk_interpolated_pixels = 0
    map_risk_filter_enabled = (
        map_risk_reject_max_mean < 1.0
        or map_risk_reject_max_interpolated_fraction < 1.0
        or map_risk_reject_max_low_support_fraction < 1.0
    )
    if map_support is not None:
        map_risk_maps = compute_belt_map_risk_maps(
            map_support,
            min_support=map_risk_min_support,
        )
        map_risk_low_support_pixels = int(np.count_nonzero(map_risk_maps.low_support_mask))
        map_risk_interpolated_pixels = int(np.count_nonzero(map_risk_maps.interpolated_mask))
        np.save(rt.OUT / "belt_map_support.npy", map_risk_maps.support)
        np.save(rt.OUT / "belt_map_observed_mask.npy", map_risk_maps.observed_mask)
        np.save(rt.OUT / "belt_map_interpolated_mask.npy", map_risk_maps.interpolated_mask)
        np.save(rt.OUT / "belt_map_low_support_mask.npy", map_risk_maps.low_support_mask)
        np.save(rt.OUT / "belt_map_risk.npy", map_risk_maps.risk)
        rt.save_png(np.log1p(map_risk_maps.support), rt.OUT / "belt_map_support.png")
        rt.save_png(map_risk_maps.observed_mask.astype(np.float32), rt.OUT / "belt_map_observed_mask.png")
        rt.save_png(map_risk_maps.interpolated_mask.astype(np.float32), rt.OUT / "belt_map_interpolated_mask.png")
        rt.save_png(map_risk_maps.low_support_mask.astype(np.float32), rt.OUT / "belt_map_low_support_mask.png")
        rt.save_png(map_risk_maps.risk, rt.OUT / "belt_map_risk.png")
        rt.emit(
            "map_risk",
            "saved belt-coordinate support and risk maps",
            map_support_npy=rt.OUT / "belt_map_support.npy",
            map_risk_npy=rt.OUT / "belt_map_risk.npy",
            min_support=map_risk_min_support,
            observed_pixels=int(np.count_nonzero(map_risk_maps.observed_mask)),
            low_support_pixels=map_risk_low_support_pixels,
            interpolated_pixels=map_risk_interpolated_pixels,
            filter_enabled=map_risk_filter_enabled,
        )
    elif map_risk_filter_enabled:
        raise ValueError(
            "map-risk filtering requires belt-map support; rebuild the belt map "
            "or set REUSE_MAP_SUPPORT_PATH when reusing a map"
        )
    elif reuse_belt_map_path is not None:
        rt.emit(
            "map_risk",
            "skipping belt-map support/risk diagnostics for reused map without support",
            source_belt_map_npy=reuse_belt_map_path,
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
        split_merged_components=detection_split_merged_components,
        split_min_projection_gap_px=detection_split_min_projection_gap_px,
        split_min_component_area_px=detection_split_min_component_area_px,
    )
    residual_config = ResidualConfig(
        noise_radius_px=residual_noise_radius_px,
        clip_sigma=residual_clip_sigma,
        noise_exclusion_sigma=residual_noise_exclusion_sigma,
        noise_exclusion_radius_px=residual_noise_exclusion_radius_px,
        min_noise=residual_min_noise,
        noise_exclusion_mode=detection_mode,
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
    nominal_estimates = (
        nominal_phase_estimates(paths, motion_model)
        if reused_phase_estimates is None and phase_estimation_mode == "motion_model"
        else None
    )
    base_phase_estimates = (
        reused_phase_estimates
        if reused_phase_estimates is not None
        else nominal_estimates
    )
    if nominal_estimates is not None:
        rt.emit(
            "detect",
            "using nominal motion-model phases without texture registration",
            phase_estimates=len(nominal_estimates),
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
            phase_estimates=base_phase_estimates,
            registration_config=registration_config,
            residual_config=residual_config,
            sample_frames=static_background_sample_frames,
            mask_threshold=static_background_mask_threshold,
            mask_mode=detection_mode,
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
            phase_estimates=base_phase_estimates,
            registration_config=registration_config,
            residual_config=residual_config,
            sample_frames=static_noise_sample_frames,
            min_scale=static_noise_min_scale,
            mask_threshold=static_noise_mask_threshold,
            mask_mode=detection_mode,
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
    map_risk_rows: list[dict] = []
    map_risk_rejected = 0
    phase_rows: list[dict] = []
    photometric_rows: list[dict] = []
    phase_px_by_frame: list[float] = []
    smoothed_phase_estimates = (
        None
        if base_phase_estimates is not None
        else estimate_smoothed_phase_sequence(
            paths=paths,
            region=region,
            belt_map=belt_map,
            motion_model=motion_model,
            registration_config=registration_config,
            phase_drift_config=phase_drift_config,
            window_radius_frames=phase_smoothing_window_frames,
            min_score=phase_smoothing_min_score,
            max_abs_correction_px=phase_smoothing_max_abs_correction_px,
            min_support=phase_smoothing_min_support,
        )
    )
    phase_drift_filter = PhaseDriftFilter(
        phase_drift_config,
        period_px=float(map_height),
    )
    detection_start = rt.time.perf_counter()
    for frame_index, path in enumerate(paths):
        frame = rt.crop(rt.read_gray(path), region)
        phase_estimate = (
            base_phase_estimates[frame_index]
            if base_phase_estimates is not None
            else (
                smoothed_phase_estimates[frame_index]
                if smoothed_phase_estimates is not None
                else None
            )
        )
        if (
            phase_estimate is None
            and phase_estimation_mode != "motion_model"
            and phase_drift_config.enabled
        ):
            nominal_phase = motion_model.phase_at(float(frame_index))
            predicted_phase = phase_drift_filter.predict(nominal_phase)
            phase_estimate = refine_phase_by_registration(
                frame=frame,
                belt_map=belt_map,
                predicted_phase_px=predicted_phase,
                frame_index=float(frame_index),
                period_px=motion_model.period_px,
                config=registration_config,
            )
            phase_estimate = phase_drift_filter.observe(phase_estimate)
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
        residual, photometric_row = apply_photometric_correction(
            frame=frame,
            residual=residual,
            residual_config=residual_config,
            frame_index=frame_index,
            path=path,
            enabled=photometric_enabled,
            trim_fraction=photometric_trim_fraction,
            max_iterations=photometric_max_iterations,
            min_pixels=photometric_min_pixels,
        )
        if photometric_row is not None:
            photometric_rows.append(photometric_row)
        residual, local_illumination_row, local_illumination_field = apply_local_illumination_correction(
            frame=frame,
            residual=residual,
            residual_config=residual_config,
            frame_index=frame_index,
            path=path,
            enabled=detection_local_illumination_correction,
            tile_px=detection_local_illumination_tile_px,
            min_pixels=detection_local_illumination_min_pixels,
            mask_threshold=detection_local_illumination_mask_threshold,
            mask_mode=detection_local_illumination_mask_mode,
            mask_grow_threshold=detection_local_illumination_mask_grow_threshold,
            mask_dilation_px=detection_local_illumination_mask_dilation_px,
            mask_margin_px=detection_local_illumination_mask_margin_px,
            mask_min_area_px=detection_local_illumination_mask_min_area_px,
        )
        if local_illumination_row is not None:
            local_illumination_rows.append(local_illumination_row)
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
            rt.save_scaled_png(frame, rt.OUT / f"raw_frame_{frame_index:06d}.png", scale=(0.0, 255.0))
            rt.save_png(residual, rt.OUT / f"residual_frame_{frame_index:06d}.png")
            rt.save_scaled_png(
                residual,
                rt.OUT / f"residual_fixed_frame_{frame_index:06d}.png",
                scale=(-8.0, 8.0),
            )
            if (
                detection_local_illumination_correction
                and local_illumination_field is not None
            ):
                rt.save_png(local_illumination_field, rt.OUT / f"local_illumination_field_frame_{frame_index:06d}.png")
        mask = detect_particles_from_residual(
            residual,
            threshold=detection_threshold,
            mode=detection_mode,
            low_threshold=detection_low_threshold,
        )
        detections = extract_particle_detections(
            mask,
            residual=residual,
            signal_mode=detection_mode,
            frame_index=float(frame_index),
            config=component_config,
        )
        if map_risk_maps is not None:
            map_risk_scores = score_map_risk_detections(
                detections,
                phase_px=float(phase_row["phase_px"]),
                frame_shape=frame.shape,
                maps=map_risk_maps,
                reject_max_mean_risk=map_risk_reject_max_mean,
                reject_max_interpolated_fraction=map_risk_reject_max_interpolated_fraction,
                reject_max_low_support_fraction=map_risk_reject_max_low_support_fraction,
            )
            map_risk_rows.extend(
                map_risk_rows_from_scores(map_risk_scores, path, frame_index)
            )
            if map_risk_filter_enabled:
                rejected_this_frame = sum(1 for score in map_risk_scores if score.rejected)
                map_risk_rejected += rejected_this_frame
                detections = [score.detection for score in map_risk_scores if not score.rejected]
            else:
                detections = [score.detection for score in map_risk_scores]
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
            if cross_map_agreement_maps is not None:
                rt.write_csv(
                    rt.OUT / "cross_map_agreement_detections.csv",
                    cross_map_agreement_rows,
                    CROSS_MAP_AGREEMENT_FIELDS,
                )
            if photometric_enabled:
                write_photometric_outputs(photometric_rows)
            rt.emit("detect", "wrote partial detection and phase outputs", processed_frames=processed, total_detections=len(detection_rows), phase_estimates=len(phase_rows))
        if processed == 1 or processed == len(paths) or processed % progress_interval == 0:
            dt = rt.time.perf_counter() - detection_start
            fps = processed / dt if dt > 0 else float("inf")
            remaining = len(paths) - processed
            eta = remaining / fps if fps > 0 else float("inf")
            rt.emit("detect", f"processed {processed}/{len(paths)} frames", processed_frames=processed, remaining_frames=remaining, detections_this_frame=len(detections), total_detections=len(detection_rows), frames_per_second=round(fps, 4), eta_s=round(eta, 1) if np.isfinite(eta) else None, current_image=path)

    if map_risk_maps is not None:
        rt.write_csv(
            rt.OUT / "map_risk_detections.csv",
            map_risk_rows,
            MAP_RISK_DETECTION_FIELDS,
        )
        rt.emit(
            "map_risk",
            "wrote map-risk detection diagnostics",
            map_risk_detections_csv=rt.OUT / "map_risk_detections.csv",
            scored_detections=len(map_risk_rows),
            rejected_detections=map_risk_rejected,
            remaining_detections=len(detection_rows),
            reject_max_mean_risk=map_risk_reject_max_mean,
            reject_max_interpolated_fraction=map_risk_reject_max_interpolated_fraction,
            reject_max_low_support_fraction=map_risk_reject_max_low_support_fraction,
        )

    pre_recurrent_detections_by_frame = [list(detections) for detections in detections_by_frame]
    revolution_split_detection_summary: list[dict] = []
    revolution_split_score_summary: list[dict] = []
    revolution_split_train_artifact_revolutions = 0
    revolution_split_train_artifact_candidate_detections = 0
    revolution_split_train_artifact_pixels = 0
    revolution_split_train_artifact_rejected = 0
    if revolution_split is not None:
        revolution_split_detection_summary.extend(
            revolution_split_detection_summary_rows(
                revolution_split,
                pre_recurrent_detections_by_frame,
                stage="pre_recurrent_filter",
            )
        )
        train_frame_indices = list(revolution_split.train_frame_indices)
        train_artifact_config = replace(
            recurrent_artifact_config,
            min_revolutions=revolution_split_ghost_min_revolutions,
        )
        train_recurrent_result = build_recurrent_artifact_map(
            [pre_recurrent_detections_by_frame[index] for index in train_frame_indices],
            [phase_px_by_frame[index] for index in train_frame_indices],
            [revolution_split.revolution_by_frame[index] for index in train_frame_indices],
            map_shape=(map_height, region[3]),
            config=train_artifact_config,
            frame_shape=(region[2], region[3]),
        )
        revolution_split_train_artifact_revolutions = train_recurrent_result.revolution_count
        revolution_split_train_artifact_candidate_detections = train_recurrent_result.candidate_detections
        revolution_split_train_artifact_pixels = train_recurrent_result.artifact_pixels
        np.save(rt.OUT / "revolution_split_train_artifact_map.npy", train_recurrent_result.mask)
        np.save(rt.OUT / "revolution_split_train_artifact_counts.npy", train_recurrent_result.counts)
        np.save(
            rt.OUT / "revolution_split_train_artifact_exposure_counts.npy",
            train_recurrent_result.exposure_counts,
        )
        np.save(
            rt.OUT / "revolution_split_train_artifact_probability.npy",
            train_recurrent_result.probability,
        )
        rt.save_png(
            train_recurrent_result.mask.astype(np.float32),
            rt.OUT / "revolution_split_train_artifact_map.png",
        )
        rt.save_png(train_recurrent_result.counts, rt.OUT / "revolution_split_train_artifact_counts.png")
        rt.save_png(
            train_recurrent_result.probability,
            rt.OUT / "revolution_split_train_artifact_probability.png",
        )
        train_artifact_scores = score_recurrent_artifact_detections(
            pre_recurrent_detections_by_frame,
            phase_px_by_frame,
            train_recurrent_result,
            config=train_artifact_config,
            detection_threshold=detection_threshold,
        )
        revolution_split_train_artifact_rejected = sum(
            1
            for frame_scores in train_artifact_scores
            for score in frame_scores
            if score.rejected
        )
        revolution_split_score_summary.extend(
            revolution_split_score_summary_rows(
                revolution_split,
                train_artifact_scores,
                stage="train_artifact_map_pre_filter",
            )
        )
        rt.write_csv(
            rt.OUT / "revolution_split_ghost_detections.csv",
            revolution_split_ghost_rows_from_scores(train_artifact_scores, paths, revolution_split),
            REVOLUTION_SPLIT_GHOST_DETECTION_FIELDS,
        )
        rt.emit(
            "revolution_split",
            "scored detections against train-only recurrent artifact map",
            train_artifact_revolutions=revolution_split_train_artifact_revolutions,
            train_artifact_candidate_detections=revolution_split_train_artifact_candidate_detections,
            train_artifact_pixels=revolution_split_train_artifact_pixels,
            train_artifact_rejected_detections=revolution_split_train_artifact_rejected,
            ghost_detections_csv=rt.OUT / "revolution_split_ghost_detections.csv",
        )

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
                candidate_max_area_px=recurrent_artifact_config.candidate_max_area_px,
                candidate_max_peak_signal=recurrent_artifact_config.candidate_max_peak_signal,
                reject_max_area_px=recurrent_artifact_config.reject_max_area_px,
                reject_max_peak_signal=recurrent_artifact_config.reject_max_peak_signal,
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
                    frame_shape=(region[2], region[3]),
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

    if cross_map_agreement_maps is not None:
        rt.write_csv(
            rt.OUT / "cross_map_agreement_detections.csv",
            cross_map_agreement_rows,
            CROSS_MAP_AGREEMENT_FIELDS,
        )
    if revolution_split is not None:
        final_revolution_stage = (
            "post_recurrent_filter"
            if recurrent_artifact_enabled
            else "final_without_recurrent_filter"
        )
        revolution_split_detection_summary.extend(
            revolution_split_detection_summary_rows(
                revolution_split,
                detections_by_frame,
                stage=final_revolution_stage,
            )
        )
        image_names = [_relative_image_name(path, data_dir=rt.DATA) for path in paths]
        rt.write_csv(
            rt.OUT / "revolution_split_frames.csv",
            revolution_split_frame_rows(
                revolution_split,
                image_names=image_names,
                selected_train_frame_indices=map_sample_frame_indices,
            ),
            REVOLUTION_SPLIT_FRAME_FIELDS,
        )
        rt.write_csv(
            rt.OUT / "revolution_split_revolutions.csv",
            revolution_split_revolution_rows(
                revolution_split,
                selected_train_frame_indices=map_sample_frame_indices,
            ),
            REVOLUTION_SPLIT_REVOLUTION_FIELDS,
        )
        rt.write_csv(
            rt.OUT / "revolution_split_detection_summary.csv",
            revolution_split_detection_summary,
            REVOLUTION_SPLIT_DETECTION_SUMMARY_FIELDS,
        )
        rt.write_csv(
            rt.OUT / "revolution_split_score_summary.csv",
            revolution_split_score_summary,
            REVOLUTION_SPLIT_SCORE_SUMMARY_FIELDS,
        )
        revolution_split_summary = {
            "enabled": True,
            "eval_every": revolution_split_eval_every,
            "eval_offset": revolution_split_eval_offset,
            "explicit_eval_revolutions": revolution_split_eval_revolutions,
            "train_revolutions": revolution_split.train_revolutions,
            "eval_revolutions": revolution_split.eval_revolutions,
            "train_frames": len(revolution_split.train_frame_indices),
            "eval_frames": len(revolution_split.eval_frame_indices),
            "map_sample_frames": len(map_sample_frame_indices),
            "ghost_min_revolutions": revolution_split_ghost_min_revolutions,
            "train_artifact_revolutions": revolution_split_train_artifact_revolutions,
            "train_artifact_candidate_detections": revolution_split_train_artifact_candidate_detections,
            "train_artifact_pixels": revolution_split_train_artifact_pixels,
            "train_artifact_rejected_detections": revolution_split_train_artifact_rejected,
        }
        (rt.OUT / "revolution_split_summary.json").write_text(
            json.dumps(rt.jsonable(revolution_split_summary), indent=2),
            encoding="utf-8",
        )
        rt.emit("revolution_split", "wrote train/eval revolution-split diagnostics", summary_json=rt.OUT / "revolution_split_summary.json")
    write_detection_outputs(detections_by_frame, detection_rows)
    write_phase_outputs(phase_rows)
    if photometric_enabled:
        write_photometric_outputs(photometric_rows)
    if detection_local_illumination_correction:
        write_local_illumination_outputs(local_illumination_rows)
    rt.emit("detect", "finished residual rendering, phase estimation, and detection", processed_frames=len(paths), total_detections=len(detection_rows), phase_estimates=len(phase_rows), local_illumination_rows=len(local_illumination_rows))

    max_match = os.getenv("MAX_MATCH_DISTANCE_PX", "").strip()
    tracking_config = ParticleTrackingConfig(
        max_match_distance_px=float(max_match) if max_match else max(5.0, 1.5 * abs(belt_velocity)),
        max_frame_gap=tracking_max_frame_gap,
        velocity_prior_y_px_per_frame=0.8 * belt_velocity,
    )
    rt.emit(
        "track",
        "starting particle tracking",
        frames=len(detections_by_frame),
        max_match_distance_px=tracking_config.max_match_distance_px,
        max_frame_gap=tracking_config.max_frame_gap,
        velocity_prior_y_px_per_frame=tracking_config.velocity_prior_y_px_per_frame,
        velocity_prior_x_px_per_frame=tracking_config.velocity_prior_x_px_per_frame,
        tracking_backend="pyrecest_gnn",
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
        for velocity in estimate_particle_velocities_vs_belt(
            tracks,
            belt_image_velocity_px_per_frame=belt_velocity,
            min_track_length=min_track_length,
            fit_method=tracking_velocity_fit_method,
        ):
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
        max_recurrent_artifact_track_score=track_filter_max_recurrent_artifact_track_score,
        recurrent_artifact_detection_threshold=track_filter_recurrent_artifact_detection_threshold,
    )
    track_scores = score_particle_velocities(
        velocity_objects,
        config=track_filter_config,
        tracks=tracks,
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
        track_filter_max_recurrent_artifact_track_score=track_filter_config.max_recurrent_artifact_track_score,
        track_filter_recurrent_artifact_detection_threshold=track_filter_config.recurrent_artifact_detection_threshold,
    )

    phase_estimate_source = (
        "loaded"
        if reused_phase_estimates is not None
        else phase_estimation_mode
    )
    phase_velocity_metadata = texture_phase_velocity_summary(
        phase_rows,
        period_px=float(map_height),
        nominal_velocity_px_per_frame=belt_velocity,
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
        "detection_mode": detection_mode,
        "detection_low_threshold": detection_low_threshold,
        "min_area_px": min_area_px,
        "detection_max_area_px": detection_max_area_px,
        "detection_min_bbox_width_px": detection_min_bbox_width_px,
        "detection_min_bbox_height_px": detection_min_bbox_height_px,
        "detection_max_bbox_aspect_ratio": detection_max_bbox_aspect_ratio,
        "detection_min_bbox_extent": detection_min_bbox_extent,
        "detection_split_merged_components": detection_split_merged_components,
        "detection_split_min_projection_gap_px": detection_split_min_projection_gap_px,
        "detection_split_min_component_area_px": detection_split_min_component_area_px,
        "photometric_enabled": photometric_enabled,
        "photometric_trim_fraction": photometric_trim_fraction,
        "photometric_max_iterations": photometric_max_iterations,
        "photometric_min_pixels": photometric_min_pixels,
        "detection_local_illumination_correction": detection_local_illumination_correction,
        "detection_local_illumination_tile_px": detection_local_illumination_tile_px,
        "tracking_max_frame_gap": tracking_config.max_frame_gap,
        "tracking_velocity_fit_method": tracking_velocity_fit_method,
        "map_mask_iterations": map_mask_iterations,
        "map_sampling_strategy": map_sampling_strategy,
        "map_sample_strategy": map_sampling_strategy,
        "map_particle_mask_threshold": map_particle_mask_threshold,
        "map_particle_mask_mode": map_particle_mask_mode,
        "map_particle_mask_grow_threshold": map_particle_mask_grow_threshold,
        "map_particle_mask_dilation_px": map_particle_mask_dilation_px,
        "map_fractional_splat": map_fractional_splat,
        "map_frame_median_offset_correction": map_frame_median_offset_correction,
        "map_local_illumination_correction": map_local_illumination_correction,
        "map_local_illumination_tile_px": map_local_illumination_tile_px,
        "map_particle_mask_margin_px": map_particle_mask_margin_px,
        "map_particle_mask_min_area_px": map_particle_mask_min_area_px,
        "map_aggregation": map_aggregation,
        "map_robust_iterations": map_robust_iterations,
        "map_robust_huber_delta": map_robust_huber_delta,
        "map_robust_min_scale": map_robust_min_scale,
        "map_support_map_used": map_risk_maps is not None,
        "reuse_map_support_path": "" if reuse_map_support_path is None else str(reuse_map_support_path),
        "map_risk_min_support": map_risk_min_support,
        "map_risk_filter_enabled": map_risk_filter_enabled,
        "map_risk_reject_max_mean": map_risk_reject_max_mean,
        "map_risk_reject_max_interpolated_fraction": map_risk_reject_max_interpolated_fraction,
        "map_risk_reject_max_low_support_fraction": map_risk_reject_max_low_support_fraction,
        "map_risk_low_support_pixels": map_risk_low_support_pixels,
        "map_risk_interpolated_pixels": map_risk_interpolated_pixels,
        "n_map_risk_scored": len(map_risk_rows),
        "n_map_risk_rejected": map_risk_rejected,
        "registration_search_radius_px": registration_config.search_radius_px,
        "registration_search_step_px": registration_config.search_step_px,
        "registration_subpixel_refinement": registration_config.subpixel_refinement,
        "registration_robust_normalization": registration_config.robust_normalization,
        "phase_refinement_iterations": phase_refinement_iterations,
        "phase_refinement_min_score": phase_refinement_min_score,
        "phase_refinement_max_abs_correction_px": phase_refinement_max_abs_correction_px,
        "phase_refinement_smoothing_window_frames": phase_refinement_smoothing_window_frames,
        "phase_smoothing_window_frames": phase_smoothing_window_frames,
        "phase_smoothing_min_score": phase_smoothing_min_score,
        "phase_smoothing_max_abs_correction_px": phase_smoothing_max_abs_correction_px,
        "phase_smoothing_min_support": phase_smoothing_min_support,
        "phase_smoothing_used": smoothed_phase_estimates is not None,
        "reused_belt_map": reuse_belt_map_path is not None,
        "reuse_belt_map_path": "" if reuse_belt_map_path is None else str(reuse_belt_map_path),
        "reuse_phase_estimates_path": "" if reuse_phase_estimates_path is None else str(reuse_phase_estimates_path),
        "reuse_static_noise_path": "" if reuse_static_noise_path is None else str(reuse_static_noise_path),
        "reuse_static_background_path": "" if reuse_static_background_path is None else str(reuse_static_background_path),
        "reuse_recurrent_artifact_map_path": "" if reuse_recurrent_artifact_map_path is None else str(reuse_recurrent_artifact_map_path),
        "static_noise_sample_frames": static_noise_sample_frames,
        "static_noise_min_scale": static_noise_min_scale,
        "static_noise_mask_threshold": static_noise_mask_threshold,
        "static_noise_mask_mode": detection_mode,
        "static_noise_mask_margin_px": static_noise_mask_margin_px,
        "static_noise_mask_min_area_px": static_noise_mask_min_area_px,
        "static_noise_map_used": static_noise_map is not None,
        "static_background_sample_frames": static_background_sample_frames,
        "static_background_mask_threshold": static_background_mask_threshold,
        "static_background_mask_mode": detection_mode,
        "static_background_mask_margin_px": static_background_mask_margin_px,
        "static_background_mask_min_area_px": static_background_mask_min_area_px,
        "static_background_map_used": static_background_map is not None,
        "recurrent_artifact_min_revolutions": recurrent_artifact_config.min_revolutions,
        "recurrent_artifact_margin_px": recurrent_artifact_config.margin_px,
        "recurrent_artifact_max_overlap_fraction": recurrent_artifact_config.max_overlap_fraction,
        "recurrent_artifact_min_recurrence_probability": recurrent_artifact_config.min_recurrence_probability,
        "recurrent_artifact_mode": recurrent_artifact_config.mode,
        "recurrent_artifact_soft_penalty_weight": recurrent_artifact_config.soft_penalty_weight,
        "recurrent_artifact_candidate_max_area_px": recurrent_artifact_config.candidate_max_area_px,
        "recurrent_artifact_candidate_max_peak_signal": recurrent_artifact_config.candidate_max_peak_signal,
        "recurrent_artifact_reject_max_area_px": recurrent_artifact_config.reject_max_area_px,
        "recurrent_artifact_reject_max_peak_signal": recurrent_artifact_config.reject_max_peak_signal,
        "recurrent_artifact_filter_used": recurrent_artifact_enabled,
        "recurrent_artifact_source": recurrent_artifact_source,
        "recurrent_artifact_revolutions": recurrent_artifact_revolutions,
        "recurrent_artifact_pixels": recurrent_artifact_pixels,
        "n_recurrent_artifact_rejected": recurrent_artifact_rejected,
        "revolution_split_enabled": revolution_split is not None,
        "revolution_split_eval_every": revolution_split_eval_every,
        "revolution_split_eval_offset": revolution_split_eval_offset,
        "revolution_split_eval_revolutions": revolution_split_eval_revolutions,
        "revolution_split_min_train_revolutions": revolution_split_min_train_revolutions,
        "revolution_split_min_eval_revolutions": revolution_split_min_eval_revolutions,
        "revolution_split_ghost_min_revolutions": revolution_split_ghost_min_revolutions,
        "revolution_split_train_revolutions": (
            [] if revolution_split is None else revolution_split.train_revolutions
        ),
        "revolution_split_eval_revolutions_observed": (
            [] if revolution_split is None else revolution_split.eval_revolutions
        ),
        "revolution_split_train_frames": (
            0 if revolution_split is None else len(revolution_split.train_frame_indices)
        ),
        "revolution_split_eval_frames": (
            0 if revolution_split is None else len(revolution_split.eval_frame_indices)
        ),
        "revolution_split_map_sample_frames": len(map_sample_frame_indices),
        "revolution_split_train_artifact_revolutions": revolution_split_train_artifact_revolutions,
        "revolution_split_train_artifact_candidate_detections": revolution_split_train_artifact_candidate_detections,
        "revolution_split_train_artifact_pixels": revolution_split_train_artifact_pixels,
        "revolution_split_train_artifact_rejected_detections": revolution_split_train_artifact_rejected,
        "cross_map_agreement_enabled": cross_map_agreement_enabled,
        "cross_map_agreement_filter": cross_map_agreement_config.filter_detections,
        "cross_map_agreement_min_confirming_maps": cross_map_agreement_config.min_confirming_maps,
        "cross_map_agreement_min_samples_per_map": cross_map_agreement_min_samples_per_map,
        "cross_map_agreement_max_centroid_distance_px": cross_map_agreement_config.max_centroid_distance_px,
        "cross_map_agreement_min_bbox_iou": cross_map_agreement_config.min_bbox_iou,
        "cross_map_agreement_min_peak_ratio": cross_map_agreement_config.min_peak_ratio,
        "cross_map_agreement_require_sign_consistency": cross_map_agreement_config.require_sign_consistency,
        "cross_map_agreement_sample_counts": list(cross_map_agreement_sample_counts),
        "n_cross_map_agreement_scored": len(cross_map_agreement_rows),
        "n_cross_map_agreement_failed": cross_map_agreement_failed,
        "n_cross_map_agreement_removed": cross_map_agreement_removed,
        "reuse_metadata_path": "" if reuse_metadata_path is None else str(reuse_metadata_path),
        "phase_estimation_mode": phase_estimation_mode,
        "phase_estimate_source": phase_estimate_source,
        "n_phase_refinement_rows": len(phase_refinement_rows),
        "n_phase_refinement_used": sum(1 for row in phase_refinement_rows if row.get("used_for_refinement")),
        "n_phase_estimates": len(phase_rows),
        "n_photometric_fits": len(photometric_rows),
        "n_detections": len(detection_rows),
        "n_tracks": len(tracks),
        "tracking_backend": "pyrecest_gnn",
        "n_velocity_estimates": len(velocity_rows),
        "n_filtered_velocity_estimates": len(filtered_velocity_rows),
        "track_filter_min_length": track_filter_config.min_track_length,
        "track_filter_min_velocity_ratio_y": track_filter_config.min_velocity_ratio_y,
        "track_filter_max_velocity_ratio_y": track_filter_config.max_velocity_ratio_y,
        "track_filter_max_abs_x_velocity_px_per_frame": track_filter_config.max_abs_x_velocity_px_per_frame,
        "track_filter_max_recurrent_artifact_track_score": track_filter_config.max_recurrent_artifact_track_score,
        "track_filter_recurrent_artifact_detection_threshold": track_filter_config.recurrent_artifact_detection_threshold,
        "auto_velocity_pair_shifts": pair_shifts,
        "elapsed_s": rt.elapsed_s(),
    }
    metadata.update(phase_velocity_metadata)
    metadata_path = rt.OUT / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rt.emit("done", "finished BeltMap image driver", metadata_json=metadata_path)


if __name__ == "__main__":  # pragma: no cover
    main()

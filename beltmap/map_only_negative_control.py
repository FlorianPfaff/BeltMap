"""Map-only negative-control benchmark for BeltMap ghost artifacts.

The benchmark never reads real particle frames.  It converts ``belt_map.npy``
itself into a high-pass, normalized pseudo-residual, renders that map through the
same phase sequence used by a BeltMap run, and then applies the normal detection
and PyRecEst-backed tracking pipeline.  Since the input contains only learned
belt texture, every reported detection and track is a false positive/ghost.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .detection import detect_particles_from_residual, normalize_detection_mode
from .phase import render_belt_view
from .tracking import (
    ParticleComponentConfig,
    ParticleDetection,
    ParticleTrack,
    ParticleTrackingConfig,
    TrackFilterConfig,
    estimate_particle_velocities_vs_belt,
    extract_particle_detections,
    score_particle_velocities,
    track_particle_detections,
)

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]

DETECTION_FIELDS = [
    "frame_index",
    "image",
    "label",
    "y",
    "x",
    "area_px",
    "bbox_top",
    "bbox_left",
    "bbox_bottom",
    "bbox_right",
    "mean_signal",
    "peak_signal",
]
DETECTIONS_PER_FRAME_FIELDS = ["frame_index", "image", "n_detections"]
TRACK_DETECTION_FIELDS = [
    "track_id",
    "track_detection_index",
    *DETECTION_FIELDS,
]
VELOCITY_FIELDS = [
    "track_id",
    "n_detections",
    "frame_start",
    "frame_end",
    "velocity_y_px_per_frame",
    "velocity_x_px_per_frame",
    "speed_px_per_frame",
    "belt_velocity_y_px_per_frame",
    "velocity_ratio_y",
    "belt_minus_particle_velocity_y_px_per_frame",
]
TRACK_SCORE_FIELDS = [
    "track_id",
    "n_detections",
    "frame_start",
    "frame_end",
    "velocity_y_px_per_frame",
    "velocity_x_px_per_frame",
    "velocity_ratio_y",
    "abs_x_velocity_px_per_frame",
    "passes_min_track_length",
    "passes_velocity_ratio",
    "passes_lateral_velocity",
    "accepted",
    "plausibility_score",
]


@dataclass(frozen=True)
class PhaseSample:
    """One phase used to render a map-only pseudo-frame."""

    frame_index: float
    phase_px: float
    image: str


@dataclass(frozen=True)
class MapOnlySignalStats:
    """Robust high-pass normalization statistics for ``belt_map.npy``."""

    center_gray: float
    highpass_sigma_gray: float
    highpass_radius_px: int
    finite_pixels: int
    max_abs_signal_z: float
    p99_abs_signal_z: float


@dataclass(frozen=True)
class MapOnlyNegativeControlConfig:
    """Settings for the map-only ghost benchmark."""

    threshold: float = 5.0
    mode: str = "positive"
    low_threshold: float | None = None
    min_area_px: int = 4
    max_area_px: int | None = None
    min_bbox_width_px: int | None = None
    min_bbox_height_px: int | None = None
    max_bbox_aspect_ratio: float | None = None
    min_bbox_extent: float | None = None
    split_merged_components: bool = False
    split_min_projection_gap_px: int = 2
    split_min_component_area_px: int | None = None
    highpass_radius_px: int = 15
    highpass_min_scale_gray: float = 1e-6
    crop_height_px: int | None = None
    frame_count: int | None = None
    belt_velocity_px_per_frame: float | None = None
    period_px: float | None = None
    noise_sigma: float = 0.0
    random_seed: int = 0
    max_match_distance_px: float | None = None
    tracking_max_frame_gap: float = 1.0
    min_track_length: int = 2
    velocity_fit_method: str = "linear"
    track_filter_min_length: int = 5
    track_filter_min_velocity_ratio_y: float = 0.0
    track_filter_max_velocity_ratio_y: float = 1.1
    track_filter_max_abs_x_velocity_px_per_frame: float | None = None
    long_track_length: int = 10


@dataclass(frozen=True)
class MapOnlyNegativeControlArtifacts:
    """Files written by a map-only negative-control benchmark run."""

    metrics: Path
    report: Path
    detections: Path
    detections_per_frame: Path
    tracks: Path
    velocities: Path
    track_scores: Path


@dataclass(frozen=True)
class MapOnlyNegativeControlResult:
    """In-memory result and written artifact paths."""

    artifacts: MapOnlyNegativeControlArtifacts
    metrics: dict[str, Any]


def generate_map_only_negative_control_report(
    *,
    output_dir: Path,
    config: MapOnlyNegativeControlConfig | None = None,
    belt_map_path: Path | None = None,
    phase_estimates_path: Path | None = None,
    metrics_path: Path | None = None,
    report_path: Path | None = None,
    detections_path: Path | None = None,
    detections_per_frame_path: Path | None = None,
    tracks_path: Path | None = None,
    velocities_path: Path | None = None,
    track_scores_path: Path | None = None,
) -> MapOnlyNegativeControlResult:
    """Run the map-only negative-control benchmark and write artifacts.

    Parameters default to a normal BeltMap output directory.  If
    ``phase_estimates.csv`` is present, its phases define the pseudo-frame
    sequence.  Otherwise synthetic phases are generated from
    ``config.belt_velocity_px_per_frame`` and ``config.frame_count``.
    """

    cfg = config or MapOnlyNegativeControlConfig()
    _validate_config(cfg)
    output_dir = Path(output_dir)
    belt_path = belt_map_path or output_dir / "belt_map.npy"
    phase_path = phase_estimates_path or output_dir / "phase_estimates.csv"
    artifacts = MapOnlyNegativeControlArtifacts(
        metrics=metrics_path or output_dir / "map_only_negative_control_metrics.json",
        report=report_path or output_dir / "map_only_negative_control_report.md",
        detections=detections_path or output_dir / "map_only_negative_control_detections.csv",
        detections_per_frame=(
            detections_per_frame_path
            or output_dir / "map_only_negative_control_detections_per_frame.csv"
        ),
        tracks=tracks_path or output_dir / "map_only_negative_control_tracks.csv",
        velocities=velocities_path or output_dir / "map_only_negative_control_velocities.csv",
        track_scores=track_scores_path or output_dir / "map_only_negative_control_track_scores.csv",
    )

    belt_map = _load_belt_map(belt_path)
    signal_map, signal_stats = highpass_normalized_belt_map(
        belt_map,
        radius_px=cfg.highpass_radius_px,
        min_scale_gray=cfg.highpass_min_scale_gray,
    )
    period_px = _effective_period_px(cfg, belt_map)
    phase_samples, belt_velocity, phases_from_file = _resolve_phase_samples(
        phase_path,
        config=cfg,
        period_px=period_px,
    )
    crop_height = cfg.crop_height_px or belt_map.shape[0]
    if crop_height < 1:
        raise ValueError("crop_height_px must be positive")

    detections_by_frame, tracks, velocities, track_scores = _detect_and_track(
        signal_map,
        phase_samples,
        crop_height=crop_height,
        belt_velocity_px_per_frame=belt_velocity,
        config=cfg,
    )

    image_by_frame = {sample.frame_index: sample.image for sample in phase_samples}
    detection_rows = _detection_rows_from_frames(detections_by_frame, phase_samples)
    detections_per_frame_rows = [
        {
            "frame_index": sample.frame_index,
            "image": sample.image,
            "n_detections": len(detections),
        }
        for sample, detections in zip(phase_samples, detections_by_frame, strict=True)
    ]
    track_rows = _track_detection_rows(tracks, image_by_frame)
    velocity_rows = [asdict(velocity) for velocity in velocities]
    track_score_rows = [asdict(score) for score in track_scores]

    for path, rows, fieldnames in (
        (artifacts.detections, detection_rows, DETECTION_FIELDS),
        (artifacts.detections_per_frame, detections_per_frame_rows, DETECTIONS_PER_FRAME_FIELDS),
        (artifacts.tracks, track_rows, TRACK_DETECTION_FIELDS),
        (artifacts.velocities, velocity_rows, VELOCITY_FIELDS),
        (artifacts.track_scores, track_score_rows, TRACK_SCORE_FIELDS),
    ):
        _write_csv(path, rows, fieldnames)

    metrics = _metrics_payload(
        belt_map=belt_map,
        signal_stats=signal_stats,
        phase_samples=phase_samples,
        phases_from_file=phases_from_file,
        phase_estimates_path=phase_path if phases_from_file else None,
        crop_height=crop_height,
        belt_velocity=belt_velocity,
        config=cfg,
        artifacts=artifacts,
        detections_by_frame=detections_by_frame,
        tracks=tracks,
        velocities=velocities,
        track_scores=track_scores,
    )
    _write_json(artifacts.metrics, metrics)
    _write_report(artifacts.report, metrics)
    return MapOnlyNegativeControlResult(artifacts=artifacts, metrics=metrics)


def highpass_normalized_belt_map(
    belt_map: ArrayLike,
    *,
    radius_px: int,
    min_scale_gray: float = 1e-6,
) -> tuple[FloatArray, MapOnlySignalStats]:
    """Return a robust high-pass z-score map for map-only ghost detection."""

    if radius_px < 0:
        raise ValueError("highpass radius must be non-negative")
    if min_scale_gray <= 0:
        raise ValueError("highpass min scale must be positive")
    belt = _as_float_map(belt_map, name="belt_map")
    local_mean = _nan_box_mean(belt, radius=radius_px)
    highpass = belt - local_mean
    finite = np.isfinite(highpass)
    if not finite.any():
        raise ValueError("belt_map has no finite pixels after high-pass filtering")
    values = highpass[finite]
    center = float(np.median(values))
    sigma = _robust_sigma(values, center=center, min_scale=min_scale_gray)
    normalized = np.full(belt.shape, np.nan, dtype=np.float64)
    normalized[finite] = (values - center) / sigma
    abs_signal = np.abs(normalized[finite])
    stats = MapOnlySignalStats(
        center_gray=center,
        highpass_sigma_gray=sigma,
        highpass_radius_px=int(radius_px),
        finite_pixels=int(np.count_nonzero(finite)),
        max_abs_signal_z=float(np.max(abs_signal)) if abs_signal.size else 0.0,
        p99_abs_signal_z=float(np.percentile(abs_signal, 99)) if abs_signal.size else 0.0,
    )
    return normalized, stats


def load_phase_samples(
    path: Path,
    *,
    frame_count: int | None = None,
) -> list[PhaseSample]:
    """Load map-only render phases from a BeltMap ``phase_estimates.csv`` file."""

    if frame_count is not None and frame_count < 1:
        raise ValueError("frame_count must be positive when set")
    if not path.is_file():
        return []
    samples: list[PhaseSample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "phase_px" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a phase_px column")
        for row_number, row in enumerate(reader):
            if frame_count is not None and len(samples) >= frame_count:
                break
            phase = _finite_float(row.get("phase_px"))
            if phase is None:
                raise ValueError(f"non-finite phase_px in {path} row {row_number + 2}")
            frame = _finite_float(row.get("frame_index"))
            if frame is None:
                frame = float(len(samples))
            image = row.get("image", "").strip() or f"map_only_frame_{len(samples):06d}.png"
            samples.append(
                PhaseSample(
                    frame_index=float(frame),
                    phase_px=float(phase),
                    image=f"map_only:{image}",
                )
            )
    return samples


def _validate_config(config: MapOnlyNegativeControlConfig) -> None:
    normalize_detection_mode(config.mode)
    _require_finite(config.threshold, "threshold")
    if config.low_threshold is not None:
        _require_finite(config.low_threshold, "low_threshold")
        if config.low_threshold > config.threshold:
            raise ValueError("low_threshold must be less than or equal to threshold")
    if config.min_area_px < 1:
        raise ValueError("min_area_px must be positive")
    if config.max_area_px is not None and config.max_area_px < config.min_area_px:
        raise ValueError("max_area_px must be greater than or equal to min_area_px")
    if config.min_bbox_width_px is not None and config.min_bbox_width_px < 1:
        raise ValueError("min_bbox_width_px must be positive when set")
    if config.min_bbox_height_px is not None and config.min_bbox_height_px < 1:
        raise ValueError("min_bbox_height_px must be positive when set")
    if config.max_bbox_aspect_ratio is not None and config.max_bbox_aspect_ratio < 1.0:
        raise ValueError("max_bbox_aspect_ratio must be at least 1 when set")
    if config.min_bbox_extent is not None and not (0.0 <= config.min_bbox_extent <= 1.0):
        raise ValueError("min_bbox_extent must be in [0, 1] when set")
    if config.highpass_radius_px < 0:
        raise ValueError("highpass_radius_px must be non-negative")
    if config.highpass_min_scale_gray <= 0:
        raise ValueError("highpass_min_scale_gray must be positive")
    if config.crop_height_px is not None and config.crop_height_px < 1:
        raise ValueError("crop_height_px must be positive when set")
    if config.frame_count is not None and config.frame_count < 1:
        raise ValueError("frame_count must be positive when set")
    if config.period_px is not None and config.period_px <= 0:
        raise ValueError("period_px must be positive when set")
    if config.noise_sigma < 0 or not math.isfinite(config.noise_sigma):
        raise ValueError("noise_sigma must be finite and non-negative")
    if config.max_match_distance_px is not None and config.max_match_distance_px <= 0:
        raise ValueError("max_match_distance_px must be positive when set")
    if config.tracking_max_frame_gap <= 0 or not math.isfinite(config.tracking_max_frame_gap):
        raise ValueError("tracking_max_frame_gap must be finite and positive")
    if config.min_track_length < 2:
        raise ValueError("min_track_length must be at least 2")
    if config.track_filter_min_length < 1:
        raise ValueError("track_filter_min_length must be positive")
    if config.long_track_length < 1:
        raise ValueError("long_track_length must be positive")


def _detect_and_track(
    signal_map: FloatArray,
    phase_samples: Sequence[PhaseSample],
    *,
    crop_height: int,
    belt_velocity_px_per_frame: float | None,
    config: MapOnlyNegativeControlConfig,
) -> tuple[list[list[ParticleDetection]], list[ParticleTrack], list[Any], list[Any]]:
    mode = normalize_detection_mode(config.mode)
    low_threshold = _active_low_threshold(config.low_threshold)
    component_config = ParticleComponentConfig(
        min_area_px=config.min_area_px,
        max_area_px=config.max_area_px,
        min_bbox_width_px=config.min_bbox_width_px,
        min_bbox_height_px=config.min_bbox_height_px,
        max_bbox_aspect_ratio=config.max_bbox_aspect_ratio,
        min_bbox_extent=config.min_bbox_extent,
        split_merged_components=config.split_merged_components,
        split_min_projection_gap_px=config.split_min_projection_gap_px,
        split_min_component_area_px=config.split_min_component_area_px,
    )
    rng = np.random.default_rng(config.random_seed) if config.noise_sigma > 0 else None
    detections_by_frame: list[list[ParticleDetection]] = []
    for sample in phase_samples:
        signal = render_belt_view(signal_map, sample.phase_px, crop_height, periodic=True)
        if rng is not None:
            signal = signal + rng.normal(0.0, config.noise_sigma, size=signal.shape)
        mask = detect_particles_from_residual(
            signal,
            threshold=config.threshold,
            mode=mode,
            low_threshold=low_threshold,
        )
        detections_by_frame.append(
            extract_particle_detections(
                mask,
                residual=signal,
                frame_index=sample.frame_index,
                config=component_config,
                signal_mode=mode,
            )
        )

    tracking_config = ParticleTrackingConfig(
        max_match_distance_px=_tracking_match_distance(
            config.max_match_distance_px,
            belt_velocity_px_per_frame,
        ),
        max_frame_gap=config.tracking_max_frame_gap,
        velocity_prior_y_px_per_frame=(
            0.8 * belt_velocity_px_per_frame
            if belt_velocity_px_per_frame is not None and math.isfinite(belt_velocity_px_per_frame)
            else 0.0
        ),
    )
    tracks = track_particle_detections(
        detections_by_frame,
        config=tracking_config,
        frame_indices=[sample.frame_index for sample in phase_samples],
    )
    velocities = []
    if (
        belt_velocity_px_per_frame is not None
        and math.isfinite(belt_velocity_px_per_frame)
        and belt_velocity_px_per_frame != 0.0
    ):
        velocities = estimate_particle_velocities_vs_belt(
            tracks,
            belt_image_velocity_px_per_frame=belt_velocity_px_per_frame,
            min_track_length=config.min_track_length,
            fit_method=config.velocity_fit_method,
        )
    track_scores = score_particle_velocities(
        velocities,
        config=TrackFilterConfig(
            min_track_length=config.track_filter_min_length,
            min_velocity_ratio_y=config.track_filter_min_velocity_ratio_y,
            max_velocity_ratio_y=config.track_filter_max_velocity_ratio_y,
            max_abs_x_velocity_px_per_frame=config.track_filter_max_abs_x_velocity_px_per_frame,
        ),
    )
    return detections_by_frame, tracks, velocities, track_scores


def _metrics_payload(
    *,
    belt_map: FloatArray,
    signal_stats: MapOnlySignalStats,
    phase_samples: Sequence[PhaseSample],
    phases_from_file: bool,
    phase_estimates_path: Path | None,
    crop_height: int,
    belt_velocity: float | None,
    config: MapOnlyNegativeControlConfig,
    artifacts: MapOnlyNegativeControlArtifacts,
    detections_by_frame: Sequence[Sequence[ParticleDetection]],
    tracks: Sequence[ParticleTrack],
    velocities: Sequence[Any],
    track_scores: Sequence[Any],
) -> dict[str, Any]:
    frames = len(phase_samples)
    detections = [detection for frame in detections_by_frame for detection in frame]
    per_frame_counts = [len(frame) for frame in detections_by_frame]
    track_lengths = [track.n_detections for track in tracks]
    long_tracks = [length for length in track_lengths if length >= config.long_track_length]
    accepted_track_scores = [score for score in track_scores if score.accepted]
    velocity_ratios = [velocity.velocity_ratio_y for velocity in velocities]
    areas = [detection.area_px for detection in detections]
    peaks = [detection.peak_signal for detection in detections]
    return {
        "benchmark": "map_only_negative_control",
        "interpretation": (
            "All detections and tracks are false positives because the benchmark "
            "runs detection/tracking on the learned belt map only."
        ),
        "belt_map_shape": [int(belt_map.shape[0]), int(belt_map.shape[1])],
        "frames": frames,
        "crop_height_px": int(crop_height),
        "phase_source": {
            "phase_estimates_path": str(phase_estimates_path) if phase_estimates_path is not None else None,
            "phases_from_phase_estimates_csv": phases_from_file,
            "period_px": _finite_or_none(config.period_px) or float(belt_map.shape[0]),
            "belt_velocity_px_per_frame": _finite_or_none(belt_velocity),
        },
        "map_signal": asdict(signal_stats),
        "detection_config": _config_payload(config),
        "noise": {
            "sigma_z": float(config.noise_sigma),
            "random_seed": int(config.random_seed) if config.noise_sigma > 0 else None,
        },
        "detections": {
            "count": len(detections),
            "false_detections": len(detections),
            "false_detections_per_100_frames": _per_100(len(detections), frames),
            "frames_with_detections": int(sum(count > 0 for count in per_frame_counts)),
            "max_detections_per_frame": max(per_frame_counts, default=0),
            "per_frame": _summary(per_frame_counts),
            "area_px": _summary(areas),
            "peak_signal": _summary(peaks),
        },
        "tracks": {
            "count": len(tracks),
            "false_tracks": len(tracks),
            "false_tracks_per_100_frames": _per_100(len(tracks), frames),
            "long_track_length": int(config.long_track_length),
            "false_long_tracks": len(long_tracks),
            "false_long_tracks_per_100_frames": _per_100(len(long_tracks), frames),
            "track_length": _summary(track_lengths),
        },
        "velocities": {
            "available": bool(velocities),
            "count": len(velocities),
            "accepted_track_filter_count": len(accepted_track_scores),
            "false_accepted_tracks": len(accepted_track_scores),
            "velocity_ratio_y": _summary(velocity_ratios),
        },
        "outputs": {key: str(value) for key, value in asdict(artifacts).items()},
    }


def _config_payload(config: MapOnlyNegativeControlConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["mode"] = normalize_detection_mode(config.mode)
    payload["low_threshold"] = _active_low_threshold(config.low_threshold)
    return payload


def _write_report(path: Path, metrics: dict[str, Any]) -> None:
    detections = metrics["detections"]
    tracks = metrics["tracks"]
    velocities = metrics["velocities"]
    lines = [
        "# Map-only negative-control benchmark",
        "",
        metrics["interpretation"],
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Rendered pseudo-frames | {metrics['frames']} |",
        f"| False detections | {detections['false_detections']} |",
        f"| False detections / 100 frames | {_fmt(detections['false_detections_per_100_frames'])} |",
        f"| False tracks | {tracks['false_tracks']} |",
        f"| False tracks / 100 frames | {_fmt(tracks['false_tracks_per_100_frames'])} |",
        f"| False tracks ≥ {tracks['long_track_length']} detections | {tracks['false_long_tracks']} |",
        f"| False accepted tracks after track filter | {velocities['false_accepted_tracks']} |",
        "",
        "## Configuration",
        "",
        f"- Detection mode: `{metrics['detection_config']['mode']}`",
        f"- Detection threshold: `{metrics['detection_config']['threshold']}`",
        f"- Low threshold: `{metrics['detection_config']['low_threshold']}`",
        f"- Minimum area: `{metrics['detection_config']['min_area_px']}` px",
        f"- High-pass radius: `{metrics['map_signal']['highpass_radius_px']}` px",
        f"- Crop height: `{metrics['crop_height_px']}` px",
        "",
        "## Interpretation guide",
        "",
        "A robust BeltMap/detector setting should have near-zero false long tracks "
        "on this control. Nonzero long tracks indicate particle-like structure in "
        "the learned belt-coordinate map or an overly permissive final detector.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _detection_rows_from_frames(
    detections_by_frame: Sequence[Sequence[ParticleDetection]],
    phase_samples: Sequence[PhaseSample],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample, detections in zip(phase_samples, detections_by_frame, strict=True):
        for detection in detections:
            row = _detection_row(detection)
            row["image"] = sample.image
            rows.append(row)
    return rows


def _track_detection_rows(
    tracks: Sequence[ParticleTrack],
    image_by_frame: dict[float, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track in tracks:
        for detection_index, detection in enumerate(track.detections):
            row = _detection_row(detection)
            row["track_id"] = track.track_id
            row["track_detection_index"] = detection_index
            row["image"] = image_by_frame.get(
                detection.frame_index,
                f"map_only_frame_{int(detection.frame_index):06d}.png",
            )
            rows.append(row)
    return rows


def _detection_row(detection: ParticleDetection) -> dict[str, Any]:
    return {
        "frame_index": detection.frame_index,
        "label": detection.label,
        "y": detection.y,
        "x": detection.x,
        "area_px": detection.area_px,
        "bbox_top": detection.bbox_top,
        "bbox_left": detection.bbox_left,
        "bbox_bottom": detection.bbox_bottom,
        "bbox_right": detection.bbox_right,
        "mean_signal": "" if detection.mean_signal is None else detection.mean_signal,
        "peak_signal": "" if detection.peak_signal is None else detection.peak_signal,
    }


def _resolve_phase_samples(
    phase_path: Path,
    *,
    config: MapOnlyNegativeControlConfig,
    period_px: float,
) -> tuple[list[PhaseSample], float | None, bool]:
    samples = load_phase_samples(phase_path, frame_count=config.frame_count)
    if samples:
        velocity = config.belt_velocity_px_per_frame
        if velocity is None:
            velocity = _estimate_belt_velocity_from_phases(samples, period_px=period_px)
        return samples, velocity, True

    if config.belt_velocity_px_per_frame is None:
        raise ValueError(
            "phase_estimates.csv is missing and belt_velocity_px_per_frame is not set; "
            "pass --belt-velocity-px-per-frame or --phase-estimates-path"
        )
    count = config.frame_count or 100
    samples = [
        PhaseSample(
            frame_index=float(index),
            phase_px=float((-config.belt_velocity_px_per_frame * index) % period_px),
            image=f"map_only_generated_{index:06d}.png",
        )
        for index in range(count)
    ]
    return samples, config.belt_velocity_px_per_frame, False


def _estimate_belt_velocity_from_phases(
    samples: Sequence[PhaseSample],
    *,
    period_px: float,
) -> float | None:
    if len(samples) < 2:
        return None
    frames = np.asarray([sample.frame_index for sample in samples], dtype=np.float64)
    phases = np.asarray([sample.phase_px for sample in samples], dtype=np.float64)
    finite = np.isfinite(frames) & np.isfinite(phases)
    frames = frames[finite]
    phases = phases[finite]
    if frames.size < 2 or np.unique(frames).size < 2:
        return None
    order = np.argsort(frames)
    frames = frames[order]
    phases = phases[order]
    if period_px > 0:
        phases = np.unwrap(phases / period_px * 2.0 * np.pi) / (2.0 * np.pi) * period_px
    dt = np.diff(frames)
    dphase = np.diff(phases)
    valid = dt != 0
    if not np.any(valid):
        return None
    slopes = dphase[valid] / dt[valid]
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size == 0:
        return None
    return float(-np.median(slopes))


def _effective_period_px(
    config: MapOnlyNegativeControlConfig,
    belt_map: FloatArray,
) -> float:
    period = config.period_px or float(belt_map.shape[0])
    if period <= 0:
        raise ValueError("period_px must be positive")
    return float(period)


def _tracking_match_distance(
    configured: float | None,
    belt_velocity_px_per_frame: float | None,
) -> float:
    if configured is not None:
        return float(configured)
    if belt_velocity_px_per_frame is None or not math.isfinite(belt_velocity_px_per_frame):
        return 25.0
    return max(5.0, 1.5 * abs(float(belt_velocity_px_per_frame)))


def _active_low_threshold(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return float(value)


def _load_belt_map(path: Path) -> FloatArray:
    if not path.is_file():
        raise FileNotFoundError(f"missing belt map: {path}")
    return _as_float_map(np.load(path), name="belt_map")


def _as_float_map(value: ArrayLike, *, name: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    return arr


def _nan_box_mean(values: FloatArray, *, radius: int) -> FloatArray:
    valid = np.isfinite(values)
    numerator = _box_sum(np.where(valid, values, 0.0), radius=radius)
    denominator = _box_sum(valid.astype(np.float64), radius=radius)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _box_sum(values: FloatArray, *, radius: int) -> FloatArray:
    if radius == 0:
        return values.astype(np.float64, copy=True)
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="constant")
    integral = np.pad(
        np.cumsum(np.cumsum(padded, axis=0, dtype=np.float64), axis=1, dtype=np.float64),
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


def _robust_sigma(values: FloatArray, *, center: float, min_scale: float) -> float:
    mad = float(np.median(np.abs(values - center)))
    sigma = 1.4826 * mad
    if not math.isfinite(sigma) or sigma < min_scale:
        sigma = min_scale
    return sigma


def _summary(values: Sequence[Any]) -> dict[str, float | int | None]:
    arr = np.asarray(
        [float(value) for value in values if _finite_float(value) is not None],
        dtype=np.float64,
    )
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _per_100(count: int, frames: int) -> float | None:
    return None if frames <= 0 else float(100.0 * count / frames)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_or_none(value: Any) -> float | None:
    return _finite_float(value)


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_json_ready(row) for row in rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(value: Any) -> str:
    number = _finite_float(value)
    return "n/a" if number is None else f"{number:.4g}"

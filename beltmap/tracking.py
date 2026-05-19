"""Particle component tracking and velocity comparison against belt motion."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .residual import ResidualImage


FloatArray = NDArray[np.floating]

_IMPORT_UNCHECKED = object()
_IMPORT_MISSING = object()
_SCIPY_NDIMAGE: Any = _IMPORT_UNCHECKED
_SKIMAGE_MEASURE: Any = _IMPORT_UNCHECKED


@dataclass(frozen=True)
class ParticleComponentConfig:
    """Settings for turning a particle mask into particle detections."""

    min_area_px: int = 1
    max_area_px: int | None = None
    min_bbox_width_px: int | None = None
    min_bbox_height_px: int | None = None
    max_bbox_aspect_ratio: float | None = None
    min_bbox_extent: float | None = None
    max_moment_aspect_ratio: float | None = None
    min_compactness: float | None = None
    min_border_margin_px: int | None = None
    min_mean_signal: float | None = None
    min_peak_signal: float | None = None
    signal_core_threshold: float | None = None
    min_core_area_px: int | None = None
    min_core_fraction: float | None = None
    local_contrast_margin_px: int = 4
    min_mean_local_contrast: float | None = None
    min_peak_local_contrast: float | None = None
    connectivity: int = 8
    weighted_centroid: bool = True


@dataclass(frozen=True)
class ParticleDetection:
    """One connected particle component in one frame."""

    frame_index: float
    label: int
    y: float
    x: float
    area_px: int
    bbox_top: int
    bbox_left: int
    bbox_bottom: int
    bbox_right: int
    mean_signal: float | None = None
    peak_signal: float | None = None
    recurrent_artifact_overlap_fraction: float | None = None
    recurrent_artifact_required_peak_signal: float | None = None
    bbox_aspect_ratio: float | None = None
    bbox_extent: float | None = None
    moment_aspect_ratio: float | None = None
    compactness: float | None = None
    core_area_px: int | None = None
    core_fraction: float | None = None
    local_background_signal: float | None = None
    mean_local_contrast: float | None = None
    peak_local_contrast: float | None = None
    border_margin_px: int | None = None


@dataclass(frozen=True)
class ParticleTrackingConfig:
    """Settings for frame-to-frame particle association."""

    max_match_distance_px: float = 25.0
    max_frame_gap: float = 1.0
    velocity_prior_y_px_per_frame: float = 0.0
    velocity_prior_x_px_per_frame: float = 0.0


@dataclass(frozen=True)
class ParticleTrack:
    """A sequence of detections associated to the same particle."""

    track_id: int
    detections: tuple[ParticleDetection, ...]

    @property
    def frame_start(self) -> float:
        return self.detections[0].frame_index

    @property
    def frame_end(self) -> float:
        return self.detections[-1].frame_index

    @property
    def n_detections(self) -> int:
        return len(self.detections)


@dataclass(frozen=True)
class ParticleVelocity:
    """Per-track particle velocity and comparison to belt velocity."""

    track_id: int
    n_detections: int
    frame_start: float
    frame_end: float
    velocity_y_px_per_frame: float
    velocity_x_px_per_frame: float
    speed_px_per_frame: float
    belt_velocity_y_px_per_frame: float
    velocity_ratio_y: float
    belt_minus_particle_velocity_y_px_per_frame: float


@dataclass(frozen=True)
class TrackFilterConfig:
    """Settings for selecting physically plausible particle tracks."""

    min_track_length: int = 5
    min_velocity_ratio_y: float = 0.0
    max_velocity_ratio_y: float = 1.1
    max_abs_x_velocity_px_per_frame: float | None = None


@dataclass(frozen=True)
class ParticleTrackScore:
    """Track-level gates and score used for post-detection filtering."""

    track_id: int
    n_detections: int
    frame_start: float
    frame_end: float
    velocity_y_px_per_frame: float
    velocity_x_px_per_frame: float
    velocity_ratio_y: float
    abs_x_velocity_px_per_frame: float
    passes_min_track_length: bool
    passes_velocity_ratio: bool
    passes_lateral_velocity: bool
    accepted: bool
    plausibility_score: float


def extract_particle_detections(
    particle_mask: ArrayLike,
    *,
    residual: ArrayLike | ResidualImage | None = None,
    frame_index: float = 0.0,
    config: ParticleComponentConfig | None = None,
) -> list[ParticleDetection]:
    """Extract connected particle detections from a boolean particle mask."""

    cfg = config or ParticleComponentConfig()
    _validate_component_config(cfg)

    mask = np.asarray(particle_mask, dtype=bool)
    if mask.size == 0:
        raise ValueError("particle_mask must not be empty")
    if mask.ndim != 2:
        raise ValueError("particle_mask must be a 2-D array")

    signal = _residual_values(residual, mask.shape)
    detections: list[ParticleDetection] = []
    for label, (rows, cols) in enumerate(
        _connected_components(mask, connectivity=cfg.connectivity),
        start=1,
    ):
        area = rows.size
        if area < cfg.min_area_px:
            continue
        if cfg.max_area_px is not None and area > cfg.max_area_px:
            continue
        top = int(np.min(rows))
        left = int(np.min(cols))
        bottom = int(np.max(rows)) + 1
        right = int(np.max(cols)) + 1
        height = bottom - top
        width = right - left
        bbox_aspect_ratio = max(height / width, width / height)
        bbox_extent = area / (height * width)
        moment_aspect_ratio = _component_moment_aspect_ratio(rows, cols)
        compactness = _component_compactness(rows, cols)
        border_margin_px = min(top, left, mask.shape[0] - bottom, mask.shape[1] - right)
        if not _component_shape_passes(
            area=area,
            height=height,
            width=width,
            moment_aspect_ratio=moment_aspect_ratio,
            compactness=compactness,
            border_margin_px=border_margin_px,
            config=cfg,
        ):
            continue

        values = signal[rows, cols] if signal is not None else None
        signal_metrics = _component_signal_metrics(
            signal=signal,
            values=values,
            rows=rows,
            cols=cols,
            top=top,
            left=left,
            bottom=bottom,
            right=right,
            area=area,
            config=cfg,
        )
        if not _component_signal_passes(signal_metrics, config=cfg):
            continue
        y, x = _component_centroid(
            rows,
            cols,
            values=values,
            weighted=cfg.weighted_centroid,
        )
        detections.append(
            ParticleDetection(
                frame_index=float(frame_index),
                label=label,
                y=y,
                x=x,
                area_px=int(area),
                bbox_top=top,
                bbox_left=left,
                bbox_bottom=bottom,
                bbox_right=right,
                mean_signal=signal_metrics["mean_signal"],
                peak_signal=signal_metrics["peak_signal"],
                bbox_aspect_ratio=float(bbox_aspect_ratio),
                bbox_extent=float(bbox_extent),
                moment_aspect_ratio=float(moment_aspect_ratio),
                compactness=float(compactness),
                core_area_px=signal_metrics["core_area_px"],
                core_fraction=signal_metrics["core_fraction"],
                local_background_signal=signal_metrics["local_background_signal"],
                mean_local_contrast=signal_metrics["mean_local_contrast"],
                peak_local_contrast=signal_metrics["peak_local_contrast"],
                border_margin_px=int(border_margin_px),
            )
        )
    return detections


def track_particle_detections(
    detections_by_frame: Sequence[Sequence[ParticleDetection]],
    *,
    config: ParticleTrackingConfig | None = None,
    frame_indices: Sequence[float] | None = None,
) -> list[ParticleTrack]:
    """Associate particle detections across frames with greedy nearest neighbors."""

    cfg = config or ParticleTrackingConfig()
    _validate_tracking_config(cfg)
    explicit_frame_indices = (
        None if frame_indices is None else [float(index) for index in frame_indices]
    )
    if explicit_frame_indices is not None and len(explicit_frame_indices) != len(detections_by_frame):
        raise ValueError("frame_indices must have the same length as detections_by_frame")

    tracks: list[list[ParticleDetection]] = []
    active_track_ids: list[int] = []

    for frame_number, detections in enumerate(detections_by_frame):
        current = sorted(detections, key=lambda item: (item.frame_index, item.y, item.x))
        frame_index = (
            explicit_frame_indices[frame_number]
            if explicit_frame_indices is not None
            else (current[0].frame_index if current else None)
        )
        if not current:
            active_track_ids = _drop_expired_tracks(active_track_ids, tracks, frame_index, cfg)
            continue

        assert frame_index is not None
        active_track_ids = _drop_expired_tracks(active_track_ids, tracks, frame_index, cfg)
        candidates: list[tuple[float, int, int]] = []
        for track_id in active_track_ids:
            last = tracks[track_id][-1]
            dt = frame_index - last.frame_index
            if dt <= 0 or dt > cfg.max_frame_gap:
                continue
            predicted_y = last.y + cfg.velocity_prior_y_px_per_frame * dt
            predicted_x = last.x + cfg.velocity_prior_x_px_per_frame * dt
            for detection_index, detection in enumerate(current):
                distance = hypot(detection.y - predicted_y, detection.x - predicted_x)
                if distance <= cfg.max_match_distance_px:
                    candidates.append((distance, track_id, detection_index))

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        for _distance, track_id, detection_index in sorted(candidates):
            if track_id in assigned_tracks or detection_index in assigned_detections:
                continue
            tracks[track_id].append(current[detection_index])
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_index)

        for detection_index, detection in enumerate(current):
            if detection_index in assigned_detections:
                continue
            tracks.append([detection])
            active_track_ids.append(len(tracks) - 1)

    return [
        ParticleTrack(track_id=track_id, detections=tuple(track))
        for track_id, track in enumerate(tracks)
    ]


def estimate_particle_velocities_vs_belt(
    tracks: Sequence[ParticleTrack],
    *,
    belt_image_velocity_px_per_frame: float,
    min_track_length: int = 2,
) -> list[ParticleVelocity]:
    """Estimate particle velocities and compare them with belt image velocity."""

    if not np.isfinite(belt_image_velocity_px_per_frame):
        raise ValueError("belt_image_velocity_px_per_frame must be finite")
    if belt_image_velocity_px_per_frame == 0:
        raise ValueError("belt_image_velocity_px_per_frame must be non-zero")
    if min_track_length < 2:
        raise ValueError("min_track_length must be at least 2")

    velocities: list[ParticleVelocity] = []
    for track in tracks:
        if track.n_detections < min_track_length:
            continue
        frames = np.asarray([d.frame_index for d in track.detections], dtype=np.float64)
        if np.unique(frames).size < 2:
            continue
        ys = np.asarray([d.y for d in track.detections], dtype=np.float64)
        xs = np.asarray([d.x for d in track.detections], dtype=np.float64)
        vy = _linear_slope(frames, ys)
        vx = _linear_slope(frames, xs)
        velocities.append(
            ParticleVelocity(
                track_id=track.track_id,
                n_detections=track.n_detections,
                frame_start=track.frame_start,
                frame_end=track.frame_end,
                velocity_y_px_per_frame=vy,
                velocity_x_px_per_frame=vx,
                speed_px_per_frame=hypot(vy, vx),
                belt_velocity_y_px_per_frame=float(belt_image_velocity_px_per_frame),
                velocity_ratio_y=vy / belt_image_velocity_px_per_frame,
                belt_minus_particle_velocity_y_px_per_frame=(
                    belt_image_velocity_px_per_frame - vy
                ),
            )
        )
    return velocities


def score_particle_velocities(
    velocities: Sequence[ParticleVelocity],
    *,
    config: TrackFilterConfig | None = None,
) -> list[ParticleTrackScore]:
    """Score velocity rows with conservative physical plausibility gates.

    The score is intended for filtering particle tracks after detection. It does
    not modify the raw detections or raw velocity estimates. A track is accepted
    when it is long enough, moves in the configured vertical velocity-ratio
    interval, and passes the optional lateral-velocity gate.
    """

    cfg = config or TrackFilterConfig()
    _validate_track_filter_config(cfg)
    scores: list[ParticleTrackScore] = []
    for velocity in velocities:
        passes_length = velocity.n_detections >= cfg.min_track_length
        passes_ratio = (
            cfg.min_velocity_ratio_y
            <= velocity.velocity_ratio_y
            <= cfg.max_velocity_ratio_y
        )
        abs_x_velocity = abs(velocity.velocity_x_px_per_frame)
        passes_lateral = (
            cfg.max_abs_x_velocity_px_per_frame is None
            or abs_x_velocity <= cfg.max_abs_x_velocity_px_per_frame
        )
        length_score = min(1.0, velocity.n_detections / cfg.min_track_length)
        ratio_score = _interval_score(
            velocity.velocity_ratio_y,
            lower=cfg.min_velocity_ratio_y,
            upper=cfg.max_velocity_ratio_y,
        )
        lateral_score = (
            1.0
            if cfg.max_abs_x_velocity_px_per_frame is None
            else max(0.0, 1.0 - abs_x_velocity / cfg.max_abs_x_velocity_px_per_frame)
        )
        accepted = passes_length and passes_ratio and passes_lateral
        scores.append(
            ParticleTrackScore(
                track_id=velocity.track_id,
                n_detections=velocity.n_detections,
                frame_start=velocity.frame_start,
                frame_end=velocity.frame_end,
                velocity_y_px_per_frame=velocity.velocity_y_px_per_frame,
                velocity_x_px_per_frame=velocity.velocity_x_px_per_frame,
                velocity_ratio_y=velocity.velocity_ratio_y,
                abs_x_velocity_px_per_frame=abs_x_velocity,
                passes_min_track_length=passes_length,
                passes_velocity_ratio=passes_ratio,
                passes_lateral_velocity=passes_lateral,
                accepted=accepted,
                plausibility_score=length_score * ratio_score * lateral_score,
            )
        )
    return scores


def filter_particle_velocities(
    velocities: Sequence[ParticleVelocity],
    *,
    config: TrackFilterConfig | None = None,
) -> list[ParticleVelocity]:
    """Return velocity rows accepted by ``score_particle_velocities``."""

    scores = score_particle_velocities(velocities, config=config)
    accepted_ids = {score.track_id for score in scores if score.accepted}
    return [velocity for velocity in velocities if velocity.track_id in accepted_ids]


def extract_particle_velocities_vs_belt(
    particle_masks: Sequence[ArrayLike],
    *,
    belt_image_velocity_px_per_frame: float,
    frame_indices: Sequence[float] | None = None,
    residuals: Sequence[ArrayLike | ResidualImage | None] | None = None,
    component_config: ParticleComponentConfig | None = None,
    tracking_config: ParticleTrackingConfig | None = None,
    min_track_length: int = 2,
) -> list[ParticleVelocity]:
    """Extract particle velocities directly from per-frame particle masks."""

    masks = list(particle_masks)
    if not masks:
        return []
    frames = (
        [float(index) for index in range(len(masks))]
        if frame_indices is None
        else [float(index) for index in frame_indices]
    )
    if len(frames) != len(masks):
        raise ValueError("frame_indices must have the same length as particle_masks")
    residual_values = (
        [None] * len(masks)
        if residuals is None
        else list(residuals)
    )
    if len(residual_values) != len(masks):
        raise ValueError("residuals must have the same length as particle_masks")

    detections_by_frame = [
        extract_particle_detections(
            mask,
            residual=residual,
            frame_index=frame_index,
            config=component_config,
        )
        for mask, residual, frame_index in zip(masks, residual_values, frames)
    ]
    cfg = tracking_config or ParticleTrackingConfig(
        max_match_distance_px=max(5.0, 1.5 * abs(belt_image_velocity_px_per_frame)),
        velocity_prior_y_px_per_frame=0.8 * belt_image_velocity_px_per_frame,
    )
    tracks = track_particle_detections(
        detections_by_frame,
        config=cfg,
        frame_indices=frames,
    )
    return estimate_particle_velocities_vs_belt(
        tracks,
        belt_image_velocity_px_per_frame=belt_image_velocity_px_per_frame,
        min_track_length=min_track_length,
    )


def _validate_component_config(config: ParticleComponentConfig) -> None:
    if config.min_area_px < 1:
        raise ValueError("min_area_px must be positive")
    if config.max_area_px is not None and config.max_area_px < config.min_area_px:
        raise ValueError("max_area_px must be greater than or equal to min_area_px")
    if config.min_bbox_width_px is not None and config.min_bbox_width_px < 1:
        raise ValueError("min_bbox_width_px must be positive when set")
    if config.min_bbox_height_px is not None and config.min_bbox_height_px < 1:
        raise ValueError("min_bbox_height_px must be positive when set")
    if (
        config.max_bbox_aspect_ratio is not None
        and config.max_bbox_aspect_ratio < 1.0
    ):
        raise ValueError("max_bbox_aspect_ratio must be at least 1 when set")
    if config.min_bbox_extent is not None and not (0.0 <= config.min_bbox_extent <= 1.0):
        raise ValueError("min_bbox_extent must be in [0, 1] when set")
    if (
        config.max_moment_aspect_ratio is not None
        and config.max_moment_aspect_ratio < 1.0
    ):
        raise ValueError("max_moment_aspect_ratio must be at least 1 when set")
    if config.min_compactness is not None and not (0.0 <= config.min_compactness <= 1.0):
        raise ValueError("min_compactness must be in [0, 1] when set")
    if config.min_border_margin_px is not None and config.min_border_margin_px < 0:
        raise ValueError("min_border_margin_px must be non-negative when set")
    for name, value in (
        ("min_mean_signal", config.min_mean_signal),
        ("min_peak_signal", config.min_peak_signal),
        ("signal_core_threshold", config.signal_core_threshold),
        ("min_mean_local_contrast", config.min_mean_local_contrast),
        ("min_peak_local_contrast", config.min_peak_local_contrast),
    ):
        if value is not None and not np.isfinite(value):
            raise ValueError(f"{name} must be finite when set")
    if config.min_core_area_px is not None and config.min_core_area_px < 1:
        raise ValueError("min_core_area_px must be positive when set")
    if config.min_core_fraction is not None and not (0.0 <= config.min_core_fraction <= 1.0):
        raise ValueError("min_core_fraction must be in [0, 1] when set")
    if (config.min_core_area_px is not None or config.min_core_fraction is not None) and config.signal_core_threshold is None:
        raise ValueError("signal_core_threshold must be set when core gates are enabled")
    if config.local_contrast_margin_px < 0:
        raise ValueError("local_contrast_margin_px must be non-negative")
    if (config.min_mean_local_contrast is not None or config.min_peak_local_contrast is not None) and config.local_contrast_margin_px < 1:
        raise ValueError("local_contrast_margin_px must be positive when local contrast gates are enabled")
    if config.connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")


def _validate_tracking_config(config: ParticleTrackingConfig) -> None:
    if config.max_match_distance_px <= 0:
        raise ValueError("max_match_distance_px must be positive")
    if config.max_frame_gap <= 0:
        raise ValueError("max_frame_gap must be positive")


def _component_shape_passes(
    *,
    area: int,
    height: int,
    width: int,
    moment_aspect_ratio: float,
    compactness: float,
    border_margin_px: int,
    config: ParticleComponentConfig,
) -> bool:
    if config.min_bbox_width_px is not None and width < config.min_bbox_width_px:
        return False
    if config.min_bbox_height_px is not None and height < config.min_bbox_height_px:
        return False
    if config.max_bbox_aspect_ratio is not None:
        aspect_ratio = max(height / width, width / height)
        if aspect_ratio > config.max_bbox_aspect_ratio:
            return False
    if config.min_bbox_extent is not None:
        bbox_area = height * width
        extent = area / bbox_area
        if extent < config.min_bbox_extent:
            return False
    if (
        config.max_moment_aspect_ratio is not None
        and moment_aspect_ratio > config.max_moment_aspect_ratio
    ):
        return False
    if config.min_compactness is not None and compactness < config.min_compactness:
        return False
    if (
        config.min_border_margin_px is not None
        and border_margin_px < config.min_border_margin_px
    ):
        return False
    return True


def _component_signal_metrics(
    *,
    signal: FloatArray | None,
    values: FloatArray | None,
    rows: NDArray[np.integer],
    cols: NDArray[np.integer],
    top: int,
    left: int,
    bottom: int,
    right: int,
    area: int,
    config: ParticleComponentConfig,
) -> dict[str, float | int | None]:
    if values is None:
        return {
            "mean_signal": None,
            "peak_signal": None,
            "core_area_px": None,
            "core_fraction": None,
            "local_background_signal": None,
            "mean_local_contrast": None,
            "peak_local_contrast": None,
        }
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        mean_signal = None
        peak_signal = None
    else:
        mean_signal = float(np.mean(finite_values))
        peak_signal = float(np.max(finite_values))

    core_area_px: int | None = None
    core_fraction: float | None = None
    if config.signal_core_threshold is not None:
        core_area_px = int(np.count_nonzero(np.isfinite(values) & (values >= config.signal_core_threshold)))
        core_fraction = core_area_px / area

    local_background = _component_local_background(
        signal,
        top=top,
        left=left,
        bottom=bottom,
        right=right,
        margin_px=config.local_contrast_margin_px,
    )
    mean_local_contrast = None if mean_signal is None or local_background is None else mean_signal - local_background
    peak_local_contrast = None if peak_signal is None or local_background is None else peak_signal - local_background
    return {
        "mean_signal": mean_signal,
        "peak_signal": peak_signal,
        "core_area_px": core_area_px,
        "core_fraction": core_fraction,
        "local_background_signal": local_background,
        "mean_local_contrast": mean_local_contrast,
        "peak_local_contrast": peak_local_contrast,
    }


def _component_signal_passes(
    metrics: dict[str, float | int | None],
    *,
    config: ParticleComponentConfig,
) -> bool:
    for key, minimum in (
        ("mean_signal", config.min_mean_signal),
        ("peak_signal", config.min_peak_signal),
        ("mean_local_contrast", config.min_mean_local_contrast),
        ("peak_local_contrast", config.min_peak_local_contrast),
    ):
        value = metrics[key]
        if minimum is not None and (value is None or float(value) < minimum):
            return False
    core_area = metrics["core_area_px"]
    if config.min_core_area_px is not None and (
        core_area is None or int(core_area) < config.min_core_area_px
    ):
        return False
    core_fraction = metrics["core_fraction"]
    if config.min_core_fraction is not None and (
        core_fraction is None or float(core_fraction) < config.min_core_fraction
    ):
        return False
    return True


def _component_local_background(
    signal: FloatArray | None,
    *,
    top: int,
    left: int,
    bottom: int,
    right: int,
    margin_px: int,
) -> float | None:
    if signal is None or margin_px <= 0:
        return None
    row_start = max(0, top - margin_px)
    row_stop = min(signal.shape[0], bottom + margin_px)
    col_start = max(0, left - margin_px)
    col_stop = min(signal.shape[1], right + margin_px)
    patch = signal[row_start:row_stop, col_start:col_stop]
    ring = np.ones(patch.shape, dtype=bool)
    ring[top - row_start : bottom - row_start, left - col_start : right - col_start] = False
    values = patch[ring]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values))


def _component_moment_aspect_ratio(
    rows: NDArray[np.integer],
    cols: NDArray[np.integer],
) -> float:
    if rows.size < 2:
        return 1.0
    centered_rows = rows.astype(np.float64) - float(np.mean(rows))
    centered_cols = cols.astype(np.float64) - float(np.mean(cols))
    cov_yy = float(np.mean(centered_rows * centered_rows))
    cov_xx = float(np.mean(centered_cols * centered_cols))
    cov_yx = float(np.mean(centered_rows * centered_cols))
    trace = cov_yy + cov_xx
    determinant = cov_yy * cov_xx - cov_yx * cov_yx
    discriminant = max(0.0, trace * trace - 4.0 * determinant)
    eigen_max = 0.5 * (trace + np.sqrt(discriminant))
    eigen_min = 0.5 * (trace - np.sqrt(discriminant))
    if eigen_max <= 0:
        return 1.0
    return float(np.sqrt(eigen_max / max(eigen_min, 1e-9)))


def _component_compactness(rows: NDArray[np.integer], cols: NDArray[np.integer]) -> float:
    top = int(np.min(rows))
    left = int(np.min(cols))
    bottom = int(np.max(rows)) + 1
    right = int(np.max(cols)) + 1
    component = np.zeros((bottom - top + 2, right - left + 2), dtype=bool)
    component[rows - top + 1, cols - left + 1] = True
    center = component[1:-1, 1:-1]
    perimeter = int(np.count_nonzero(center & ~component[:-2, 1:-1]))
    perimeter += int(np.count_nonzero(center & ~component[2:, 1:-1]))
    perimeter += int(np.count_nonzero(center & ~component[1:-1, :-2]))
    perimeter += int(np.count_nonzero(center & ~component[1:-1, 2:]))
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * rows.size / (perimeter * perimeter))


def _validate_track_filter_config(config: TrackFilterConfig) -> None:
    if config.min_track_length < 1:
        raise ValueError("min_track_length must be positive")
    if not np.isfinite(config.min_velocity_ratio_y):
        raise ValueError("min_velocity_ratio_y must be finite")
    if not np.isfinite(config.max_velocity_ratio_y):
        raise ValueError("max_velocity_ratio_y must be finite")
    if config.max_velocity_ratio_y < config.min_velocity_ratio_y:
        raise ValueError("max_velocity_ratio_y must be greater than or equal to min_velocity_ratio_y")
    if (
        config.max_abs_x_velocity_px_per_frame is not None
        and config.max_abs_x_velocity_px_per_frame <= 0
    ):
        raise ValueError("max_abs_x_velocity_px_per_frame must be positive when set")


def _interval_score(value: float, *, lower: float, upper: float) -> float:
    if lower <= value <= upper:
        return 1.0
    width = max(upper - lower, 1e-9)
    distance = lower - value if value < lower else value - upper
    return max(0.0, 1.0 - distance / width)


def _residual_values(
    residual: ArrayLike | ResidualImage | None,
    shape: tuple[int, ...],
) -> FloatArray | None:
    if residual is None:
        return None
    values = (
        residual.normalized
        if isinstance(residual, ResidualImage)
        else residual
    )
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError("residual must have the same shape as particle_mask")
    return arr


def _component_centroid(
    rows: NDArray[np.integer],
    cols: NDArray[np.integer],
    *,
    values: FloatArray | None,
    weighted: bool,
) -> tuple[float, float]:
    if values is not None and weighted:
        weights = np.where(np.isfinite(values), np.clip(values, 0.0, None), 0.0)
        weight_sum = float(np.sum(weights))
        if weight_sum > 0:
            return (
                float(np.sum(rows * weights) / weight_sum),
                float(np.sum(cols * weights) / weight_sum),
            )
    return float(np.mean(rows)), float(np.mean(cols))


def _connected_components(
    mask: NDArray[np.bool_],
    *,
    connectivity: int,
) -> list[tuple[NDArray[np.integer], NDArray[np.integer]]]:
    """Return connected components using optional accelerated labelers if available."""

    for implementation in (
        _connected_components_with_scipy,
        _connected_components_with_skimage,
    ):
        components = implementation(mask, connectivity=connectivity)
        if components is not None:
            return components
    return _connected_components_numpy(mask, connectivity=connectivity)


def _connected_components_with_scipy(
    mask: NDArray[np.bool_],
    *,
    connectivity: int,
) -> list[tuple[NDArray[np.integer], NDArray[np.integer]]] | None:
    ndimage = _load_scipy_ndimage()
    if ndimage is None:
        return None

    labels, component_count = ndimage.label(
        mask,
        structure=_component_structure(connectivity),
    )
    return _components_from_labels(np.asarray(labels), int(component_count))


def _connected_components_with_skimage(
    mask: NDArray[np.bool_],
    *,
    connectivity: int,
) -> list[tuple[NDArray[np.integer], NDArray[np.integer]]] | None:
    measure = _load_skimage_measure()
    if measure is None:
        return None

    labels, component_count = measure.label(
        mask,
        connectivity=1 if connectivity == 4 else 2,
        background=0,
        return_num=True,
    )
    return _components_from_labels(np.asarray(labels), int(component_count))


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


def _component_structure(connectivity: int) -> NDArray[np.bool_]:
    if connectivity == 4:
        return np.array(
            [
                [False, True, False],
                [True, True, True],
                [False, True, False],
            ],
            dtype=bool,
        )
    if connectivity == 8:
        return np.ones((3, 3), dtype=bool)
    raise ValueError("connectivity must be 4 or 8")


def _components_from_labels(
    labels: NDArray[np.integer],
    component_count: int,
) -> list[tuple[NDArray[np.integer], NDArray[np.integer]]]:
    if component_count <= 0:
        return []
    rows, cols = np.nonzero(labels)
    if rows.size == 0:
        return []

    label_values = labels[rows, cols]
    order = np.argsort(label_values, kind="stable")
    rows = rows[order]
    cols = cols[order]
    label_values = label_values[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(label_values)) + 1, label_values.size]
    return [
        (rows[start:end], cols[start:end])
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def _connected_components_numpy(
    mask: NDArray[np.bool_],
    *,
    connectivity: int,
) -> list[tuple[NDArray[np.integer], NDArray[np.integer]]]:
    offsets = (
        [(-1, 0), (0, -1), (0, 1), (1, 0)]
        if connectivity == 4
        else [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    )
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[NDArray[np.integer], NDArray[np.integer]]] = []
    height, width = mask.shape

    for start_row, start_col in np.argwhere(mask):
        row = int(start_row)
        col = int(start_col)
        if visited[row, col]:
            continue
        stack = [(row, col)]
        visited[row, col] = True
        rows: list[int] = []
        cols: list[int] = []
        while stack:
            current_row, current_col = stack.pop()
            rows.append(current_row)
            cols.append(current_col)
            for row_offset, col_offset in offsets:
                next_row = current_row + row_offset
                next_col = current_col + col_offset
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and mask[next_row, next_col]
                    and not visited[next_row, next_col]
                ):
                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))
        components.append((np.asarray(rows), np.asarray(cols)))
    return components


def _drop_expired_tracks(
    active_track_ids: list[int],
    tracks: list[list[ParticleDetection]],
    frame_index: float | None,
    config: ParticleTrackingConfig,
) -> list[int]:
    if frame_index is None:
        return active_track_ids
    return [
        track_id
        for track_id in active_track_ids
        if frame_index - tracks[track_id][-1].frame_index <= config.max_frame_gap
    ]


def _linear_slope(times: FloatArray, values: FloatArray) -> float:
    centered_times = times - float(np.mean(times))
    denominator = float(np.sum(np.square(centered_times)))
    if denominator <= 0:
        raise ValueError("at least two distinct frame indices are required")
    centered_values = values - float(np.mean(values))
    return float(np.sum(centered_times * centered_values) / denominator)

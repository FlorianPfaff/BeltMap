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

        values = signal[rows, cols] if signal is not None else None
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
                bbox_top=int(np.min(rows)),
                bbox_left=int(np.min(cols)),
                bbox_bottom=int(np.max(rows)) + 1,
                bbox_right=int(np.max(cols)) + 1,
                mean_signal=None if values is None else float(np.mean(values)),
                peak_signal=None if values is None else float(np.max(values)),
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
    if config.connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")


def _validate_tracking_config(config: ParticleTrackingConfig) -> None:
    if config.max_match_distance_px <= 0:
        raise ValueError("max_match_distance_px must be positive")
    if config.max_frame_gap <= 0:
        raise ValueError("max_frame_gap must be positive")


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
        weights = np.clip(values, 0.0, None)
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

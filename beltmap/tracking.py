"""Particle component tracking and velocity comparison against belt motion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot, log
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pyrecest.filters import GlobalNearestNeighbor

from .detection import DETECTION_MODES, normalize_detection_mode
from .residual import ResidualImage


FloatArray = NDArray[np.floating]

_IMPORT_UNCHECKED = object()
_IMPORT_MISSING = object()
_SCIPY_NDIMAGE: Any = _IMPORT_UNCHECKED
_SKIMAGE_MEASURE: Any = _IMPORT_UNCHECKED
_VELOCITY_FIT_METHODS = {"linear", "theil_sen"}
_TRACK_PREDICTION_HISTORY = 4
_TRACK_FEATURE_HISTORY = 4
_TRACK_AREA_COST_WEIGHT_PX = 0.5
_TRACK_SIGNAL_COST_WEIGHT_PX = 0.25
_TRACK_FEATURE_LOG_RATIO_CAP = log(4.0)
_TRACK_FEATURE_COST_MAX_GATE_FRACTION = 0.15
_TRACK_MIN_PREDICTION_SIGMA_PX = 1.0
_TRACK_MAX_PREDICTION_SIGMA_GATE_FRACTION = 0.5


@dataclass(frozen=True)
class ParticleComponentConfig:
    """Settings for turning a particle mask into particle detections."""

    min_area_px: int = 1
    max_area_px: int | None = None
    min_bbox_width_px: int | None = None
    min_bbox_height_px: int | None = None
    max_bbox_aspect_ratio: float | None = None
    min_bbox_extent: float | None = None
    connectivity: int = 8
    weighted_centroid: bool = True
    split_merged_components: bool = False
    split_min_projection_gap_px: int = 2
    split_min_component_area_px: int | None = None


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
    recurrent_artifact_probability: float | None = None
    recurrent_artifact_required_peak_signal: float | None = None


@dataclass(frozen=True)
class ParticleTrackingConfig:
    """Settings for PyRecEst-backed frame-to-frame particle association."""

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
    signal_mode: str | None = None,
) -> list[ParticleDetection]:
    """Extract connected particle detections from a boolean particle mask.

    ``signal_mode`` orients raw residual values before weighted centroid and
    signal-statistic computation. Pass the same mode that was used to create
    ``particle_mask`` when ``residual`` is an un-oriented residual image. Leave
    it as ``None`` when ``residual`` already contains an oriented detection
    signal, as produced by :func:`beltmap.detection.detection_signal_from_residual`.
    """

    cfg = config or ParticleComponentConfig()
    _validate_component_config(cfg)

    mask = np.asarray(particle_mask, dtype=bool)
    if mask.size == 0:
        raise ValueError("particle_mask must not be empty")
    if mask.ndim != 2:
        raise ValueError("particle_mask must be a 2-D array")

    signal = _residual_values(residual, mask.shape, signal_mode=signal_mode)
    components: list[tuple[NDArray[np.integer], NDArray[np.integer]]] = []
    for rows, cols in _connected_components(mask, connectivity=cfg.connectivity):
        components.extend(_split_connected_component(rows, cols, config=cfg))

    detections: list[ParticleDetection] = []
    for label, (rows, cols) in enumerate(components, start=1):
        area = rows.size
        if area <= 0:
            continue
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
        if not _component_shape_passes(
            area=area,
            height=height,
            width=width,
            config=cfg,
        ):
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
                bbox_top=top,
                bbox_left=left,
                bbox_bottom=bottom,
                bbox_right=right,
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
    """Associate particle detections across frames with PyRecEst GNN.

    When ``frame_indices`` is omitted, the position of each entry in
    ``detections_by_frame`` is used as the frame index.
    """

    cfg = config or ParticleTrackingConfig()
    _validate_tracking_config(cfg)
    effective_frame_indices = (
        [float(index) for index in range(len(detections_by_frame))]
        if frame_indices is None
        else [float(index) for index in frame_indices]
    )
    if len(effective_frame_indices) != len(detections_by_frame):
        raise ValueError("frame_indices must have the same length as detections_by_frame")

    tracks: list[list[ParticleDetection]] = []
    active_track_ids: list[int] = []

    for frame_number, detections in enumerate(detections_by_frame):
        frame_index = effective_frame_indices[frame_number]
        current = sorted(
            (_with_frame_index(detection, frame_index) for detection in detections),
            key=lambda item: (item.frame_index, item.y, item.x),
        )
        if not current:
            active_track_ids = _drop_expired_tracks(active_track_ids, tracks, frame_index, cfg)
            continue

        active_track_ids = _drop_expired_tracks(active_track_ids, tracks, frame_index, cfg)

        assigned_detections: set[int] = set()
        for track_id, detection_index in _associate_pyrecest_gnn(
            current,
            active_track_ids,
            tracks,
            frame_index=frame_index,
            config=cfg,
        ):
            tracks[track_id].append(current[detection_index])
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


def _associate_pyrecest_gnn(
    current: Sequence[ParticleDetection],
    active_track_ids: Sequence[int],
    tracks: Sequence[Sequence[ParticleDetection]],
    *,
    frame_index: float,
    config: ParticleTrackingConfig,
) -> list[tuple[int, int]]:
    """Return global nearest-neighbor assignments using PyRecEst.

    BeltMap owns track birth/death bookkeeping, while PyRecEst solves the
    active-track-to-detection assignment globally via its multitarget tracker.
    The state exposed to PyRecEst is the predicted image position ``[y, x]``;
    new detections that remain unassigned are initialized as BeltMap tracks by
    the caller.
    """

    candidate_track_ids: list[int] = []
    initial_priors: list[tuple[FloatArray, FloatArray]] = []
    predicted_positions: list[tuple[float, float]] = []
    for track_id in active_track_ids:
        prediction = _predict_track_position(
            tracks[track_id],
            frame_index=frame_index,
            config=config,
        )
        if prediction is None:
            continue
        predicted_y, predicted_x = prediction
        candidate_track_ids.append(track_id)
        predicted_positions.append((predicted_y, predicted_x))
        initial_priors.append(
            (
                np.asarray([predicted_y, predicted_x], dtype=np.float64),
                _track_prediction_covariance(
                    tracks[track_id],
                    frame_index=frame_index,
                    config=config,
                ),
            )
        )

    if not candidate_track_ids:
        return []

    tracker = GlobalNearestNeighbor(
        initial_prior=initial_priors,
        association_param={
            "distance_metric_pos": "Mahalanobis",
            "square_dist": False,
            "gating_distance_threshold": (
                config.max_match_distance_px + _track_feature_cost_cap(config)
            ),
            "maximize_cardinality": True,
            "max_new_tracks": max(len(current), len(candidate_track_ids), 1),
        },
        log_prior_estimates=False,
        log_posterior_estimates=False,
    )
    measurements = np.asarray(
        [[detection.y for detection in current], [detection.x for detection in current]],
        dtype=np.float64,
    )
    association = np.asarray(
        tracker.find_association(
            measurements,
            np.eye(2, dtype=np.float64),
            np.eye(2, dtype=np.float64),
            warn_on_no_meas_for_track=False,
            pairwise_cost_matrix=_association_feature_cost_matrix(
                candidate_track_ids,
                predicted_positions,
                tracks,
                current,
                config=config,
            ),
        ),
        dtype=int,
    )
    return [
        (candidate_track_ids[track_index], int(detection_index))
        for track_index, detection_index in enumerate(association)
        if 0 <= detection_index < len(current)
    ]


def _predict_track_position(
    track: Sequence[ParticleDetection],
    *,
    frame_index: float,
    config: ParticleTrackingConfig,
) -> tuple[float, float] | None:
    last = track[-1]
    dt = frame_index - last.frame_index
    if dt <= 0 or dt > config.max_frame_gap:
        return None

    velocity_y = config.velocity_prior_y_px_per_frame
    velocity_x = config.velocity_prior_x_px_per_frame
    recent = track[-_TRACK_PREDICTION_HISTORY:]
    if len(recent) >= 2:
        frames = np.asarray([detection.frame_index for detection in recent], dtype=np.float64)
        if np.unique(frames).size >= 2:
            ys = np.asarray([detection.y for detection in recent], dtype=np.float64)
            xs = np.asarray([detection.x for detection in recent], dtype=np.float64)
            predicted_y = _predict_axis_from_recent_track(frames, ys, frame_index)
            predicted_x = _predict_axis_from_recent_track(frames, xs, frame_index)
            if predicted_y is not None and predicted_x is not None:
                return predicted_y, predicted_x

    return (
        last.y + velocity_y * dt,
        last.x + velocity_x * dt,
    )


def _predict_axis_from_recent_track(
    frames: FloatArray,
    values: FloatArray,
    frame_index: float,
) -> float | None:
    finite = np.isfinite(frames) & np.isfinite(values)
    frames = frames[finite]
    values = values[finite]
    if np.unique(frames).size < 2:
        return None
    line = _robust_axis_line(frames, values)
    if line is None:
        return None
    slope, intercept = line
    prediction = float(intercept + slope * frame_index)
    return prediction if np.isfinite(prediction) else None


def _robust_axis_line(
    frames: FloatArray,
    values: FloatArray,
) -> tuple[float, float] | None:
    try:
        slope = _theil_sen_slope(frames, values)
    except ValueError:
        return None
    intercepts = values - slope * frames
    intercepts = intercepts[np.isfinite(intercepts)]
    if intercepts.size == 0:
        return None
    intercept = float(np.median(intercepts))
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    return float(slope), intercept


def _track_prediction_covariance(
    track: Sequence[ParticleDetection],
    *,
    frame_index: float,
    config: ParticleTrackingConfig,
) -> FloatArray:
    horizon = _prediction_horizon(track, frame_index)
    fallback_variance = _bounded_prediction_variance(
        _TRACK_MIN_PREDICTION_SIGMA_PX * horizon,
        config=config,
    )
    recent = track[-_TRACK_PREDICTION_HISTORY:]
    if len(recent) < 3:
        return np.eye(2, dtype=np.float64) * fallback_variance

    frames = np.asarray([detection.frame_index for detection in recent], dtype=np.float64)
    if np.unique(frames).size < 2:
        return np.eye(2, dtype=np.float64) * fallback_variance

    ys = np.asarray([detection.y for detection in recent], dtype=np.float64)
    xs = np.asarray([detection.x for detection in recent], dtype=np.float64)
    return np.diag(
        [
            _axis_prediction_variance(frames, ys, horizon=horizon, config=config),
            _axis_prediction_variance(frames, xs, horizon=horizon, config=config),
        ]
    ).astype(np.float64)


def _prediction_horizon(
    track: Sequence[ParticleDetection],
    frame_index: float,
) -> float:
    dt = frame_index - track[-1].frame_index
    if not np.isfinite(dt) or dt <= 0:
        return 1.0
    return max(1.0, float(dt))


def _axis_prediction_variance(
    frames: FloatArray,
    values: FloatArray,
    *,
    horizon: float,
    config: ParticleTrackingConfig,
) -> float:
    line = _robust_axis_line(frames, values)
    if line is None:
        return _bounded_prediction_variance(
            _TRACK_MIN_PREDICTION_SIGMA_PX * horizon,
            config=config,
        )

    slope, intercept = line
    residuals = values - (slope * frames + intercept)
    residuals = np.abs(residuals[np.isfinite(residuals)])
    if residuals.size == 0:
        sigma = _TRACK_MIN_PREDICTION_SIGMA_PX
    else:
        sigma = float(np.percentile(residuals, 75))
    sigma *= horizon
    return _bounded_prediction_variance(sigma, config=config)


def _bounded_prediction_variance(
    sigma: float,
    *,
    config: ParticleTrackingConfig,
) -> float:
    max_sigma = max(
        _TRACK_MIN_PREDICTION_SIGMA_PX,
        config.max_match_distance_px * _TRACK_MAX_PREDICTION_SIGMA_GATE_FRACTION,
    )
    sigma = min(max(sigma, _TRACK_MIN_PREDICTION_SIGMA_PX), max_sigma)
    return float(sigma * sigma)


def _association_feature_cost_matrix(
    candidate_track_ids: Sequence[int],
    predicted_positions: Sequence[tuple[float, float]],
    tracks: Sequence[Sequence[ParticleDetection]],
    detections: Sequence[ParticleDetection],
    *,
    config: ParticleTrackingConfig,
) -> FloatArray:
    cost_cap = _track_feature_cost_cap(config)
    blocked_cost = config.max_match_distance_px + cost_cap + 1.0
    costs = np.full(
        (len(candidate_track_ids), len(detections)),
        blocked_cost,
        dtype=np.float64,
    )
    for track_index, track_id in enumerate(candidate_track_ids):
        predicted_y, predicted_x = predicted_positions[track_index]
        track_area, track_signal = _track_feature_reference(tracks[track_id])
        for detection_index, detection in enumerate(detections):
            distance = hypot(detection.y - predicted_y, detection.x - predicted_x)
            if distance > config.max_match_distance_px:
                continue
            feature_cost = 0.0
            area_ratio = _positive_ratio(track_area, float(detection.area_px))
            if area_ratio is not None:
                feature_cost += (
                    _TRACK_AREA_COST_WEIGHT_PX
                    * min(abs(log(area_ratio)), _TRACK_FEATURE_LOG_RATIO_CAP)
                )

            signal_ratio = _positive_ratio(track_signal, _detection_signal(detection))
            if signal_ratio is not None:
                feature_cost += (
                    _TRACK_SIGNAL_COST_WEIGHT_PX
                    * min(abs(log(signal_ratio)), _TRACK_FEATURE_LOG_RATIO_CAP)
                )
            costs[track_index, detection_index] = min(feature_cost, cost_cap)
    return costs


def _track_feature_cost_cap(config: ParticleTrackingConfig) -> float:
    return config.max_match_distance_px * _TRACK_FEATURE_COST_MAX_GATE_FRACTION


def _track_feature_reference(
    track: Sequence[ParticleDetection],
) -> tuple[float | None, float | None]:
    recent = track[-_TRACK_FEATURE_HISTORY:]
    areas = [
        float(detection.area_px)
        for detection in recent
        if detection.area_px > 0 and np.isfinite(detection.area_px)
    ]
    signals = [
        signal
        for signal in (_detection_signal(detection) for detection in recent)
        if signal is not None
    ]
    area = None if not areas else float(np.median(np.asarray(areas, dtype=np.float64)))
    signal = (
        None
        if not signals
        else float(np.median(np.asarray(signals, dtype=np.float64)))
    )
    return area, signal


def _positive_ratio(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    if first <= 0 or second <= 0:
        return None
    if not np.isfinite(first) or not np.isfinite(second):
        return None
    return max(first, second) / min(first, second)


def _detection_signal(detection: ParticleDetection) -> float | None:
    for value in (detection.mean_signal, detection.peak_signal):
        if value is not None and np.isfinite(value) and value > 0:
            return float(value)
    return None


def estimate_particle_velocities_vs_belt(
    tracks: Sequence[ParticleTrack],
    *,
    belt_image_velocity_px_per_frame: float,
    min_track_length: int = 2,
    fit_method: str = "linear",
) -> list[ParticleVelocity]:
    """Estimate particle velocities and compare them with belt image velocity."""

    if not np.isfinite(belt_image_velocity_px_per_frame):
        raise ValueError("belt_image_velocity_px_per_frame must be finite")
    if belt_image_velocity_px_per_frame == 0:
        raise ValueError("belt_image_velocity_px_per_frame must be non-zero")
    if min_track_length < 2:
        raise ValueError("min_track_length must be at least 2")
    fit_method = _validate_velocity_fit_method(fit_method)

    velocities: list[ParticleVelocity] = []
    for track in tracks:
        if track.n_detections < min_track_length:
            continue
        frames = np.asarray([d.frame_index for d in track.detections], dtype=np.float64)
        if np.unique(frames).size < 2:
            continue
        ys = np.asarray([d.y for d in track.detections], dtype=np.float64)
        xs = np.asarray([d.x for d in track.detections], dtype=np.float64)
        vy = _fit_slope(frames, ys, method=fit_method)
        vx = _fit_slope(frames, xs, method=fit_method)
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
    signal_mode: str | None = None,
    component_config: ParticleComponentConfig | None = None,
    tracking_config: ParticleTrackingConfig | None = None,
    min_track_length: int = 2,
    fit_method: str = "linear",
) -> list[ParticleVelocity]:
    """Extract particle velocities directly from per-frame particle masks.

    ``signal_mode`` is forwarded to :func:`extract_particle_detections` so
    residual values can be oriented consistently for negative or absolute
    particle detections.
    """

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
            signal_mode=signal_mode,
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
        fit_method=fit_method,
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
    if config.connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if config.split_min_projection_gap_px < 1:
        raise ValueError("split_min_projection_gap_px must be positive")
    if (
        config.split_min_component_area_px is not None
        and config.split_min_component_area_px < 1
    ):
        raise ValueError("split_min_component_area_px must be positive when set")


def _validate_velocity_fit_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized not in _VELOCITY_FIT_METHODS:
        choices = ", ".join(sorted(_VELOCITY_FIT_METHODS))
        raise ValueError(f"fit_method must be one of {choices}")
    return normalized


def _validate_tracking_config(config: ParticleTrackingConfig) -> None:
    if not np.isfinite(config.max_match_distance_px) or config.max_match_distance_px <= 0:
        raise ValueError("max_match_distance_px must be positive")
    if not np.isfinite(config.max_frame_gap) or config.max_frame_gap <= 0:
        raise ValueError("max_frame_gap must be positive")
    if not np.isfinite(config.velocity_prior_y_px_per_frame):
        raise ValueError("velocity_prior_y_px_per_frame must be finite")
    if not np.isfinite(config.velocity_prior_x_px_per_frame):
        raise ValueError("velocity_prior_x_px_per_frame must be finite")


def _component_shape_passes(
    *,
    area: int,
    height: int,
    width: int,
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
    return True


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
    *,
    signal_mode: str | None = None,
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
    if signal_mode is not None:
        arr = _orient_residual_signal(arr, signal_mode=signal_mode)
    return arr


def _orient_residual_signal(values: FloatArray, *, signal_mode: str) -> FloatArray:
    mode = normalize_detection_mode(signal_mode)
    if mode == "positive":
        return values
    if mode == "negative":
        return -values
    if mode == "absolute":
        return np.abs(values)
    choices = ", ".join(sorted(DETECTION_MODES))
    raise ValueError(f"signal_mode must be one of {choices}")


def _split_connected_component(
    rows: NDArray[np.integer],
    cols: NDArray[np.integer],
    *,
    config: ParticleComponentConfig,
) -> list[tuple[NDArray[np.integer], NDArray[np.integer]]]:
    """Split a merged component at narrow projection valleys.

    This is intentionally conservative and dependency-free. It only splits when
    a component has a sustained row/column projection valley, such as two blobs
    joined by a one-pixel bridge. The bridge pixels are discarded; this is
    preferable to reporting one merged component for downstream tracking.
    """

    if not config.split_merged_components:
        return [(rows, cols)]

    min_area = config.split_min_component_area_px or config.min_area_px
    pending: list[tuple[NDArray[np.integer], NDArray[np.integer]]] = [(rows, cols)]
    result: list[tuple[NDArray[np.integer], NDArray[np.integer]]] = []
    # Bound recursion so pathological masks cannot create excessive splitting.
    for _depth in range(16):
        if not pending:
            break
        current_rows, current_cols = pending.pop()
        split = _projection_valley_split(
            current_rows,
            current_cols,
            min_gap_px=config.split_min_projection_gap_px,
            min_area_px=min_area,
        )
        if split is None:
            result.append((current_rows, current_cols))
        else:
            pending.extend(split)
    result.extend(pending)
    return result


def _projection_valley_split(
    rows: NDArray[np.integer],
    cols: NDArray[np.integer],
    *,
    min_gap_px: int,
    min_area_px: int,
) -> list[tuple[NDArray[np.integer], NDArray[np.integer]]] | None:
    if rows.size < 2 * min_area_px:
        return None
    top = int(np.min(rows))
    left = int(np.min(cols))
    height = int(np.max(rows)) - top + 1
    width = int(np.max(cols)) - left + 1
    if height < 3 and width < 3:
        return None

    local_rows = rows - top
    local_cols = cols - left
    local = np.zeros((height, width), dtype=bool)
    local[local_rows, local_cols] = True
    row_counts = np.count_nonzero(local, axis=1)
    col_counts = np.count_nonzero(local, axis=0)
    row_run = _best_projection_valley(row_counts, min_gap_px=min_gap_px)
    col_run = _best_projection_valley(col_counts, min_gap_px=min_gap_px)
    if row_run is None and col_run is None:
        return None

    use_axis = "y"
    if row_run is None:
        use_axis = "x"
    elif col_run is not None and (col_run[1] - col_run[0]) >= (row_run[1] - row_run[0]):
        use_axis = "x"

    if use_axis == "x":
        assert col_run is not None
        gap_start, gap_stop = col_run
        first = local_cols < gap_start
        second = local_cols >= gap_stop
    else:
        assert row_run is not None
        gap_start, gap_stop = row_run
        first = local_rows < gap_start
        second = local_rows >= gap_stop

    if int(np.count_nonzero(first)) < min_area_px or int(np.count_nonzero(second)) < min_area_px:
        return None
    return [(rows[first], cols[first]), (rows[second], cols[second])]


def _best_projection_valley(
    counts: NDArray[np.integer],
    *,
    min_gap_px: int,
) -> tuple[int, int] | None:
    if counts.size < 3:
        return None
    max_count = int(np.max(counts))
    if max_count <= 1:
        return None
    valley_threshold = max(1, int(np.floor(0.15 * max_count)))
    valley = counts <= valley_threshold
    # Avoid splitting at the outside border of a component.
    valley[0] = False
    valley[-1] = False
    best: tuple[int, int] | None = None
    start: int | None = None
    for index, is_valley in enumerate(valley):
        if is_valley and start is None:
            start = index
        if (not is_valley or index == valley.size - 1) and start is not None:
            stop = index if not is_valley else index + 1
            if stop - start >= min_gap_px:
                if best is None or stop - start > best[1] - best[0]:
                    best = (start, stop)
            start = None
    return best


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


def _with_frame_index(
    detection: ParticleDetection,
    frame_index: float,
) -> ParticleDetection:
    """Return ``detection`` with the effective frame index used for tracking."""

    if detection.frame_index == frame_index:
        return detection
    return replace(detection, frame_index=frame_index)


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


def _fit_slope(times: FloatArray, values: FloatArray, *, method: str) -> float:
    method = _validate_velocity_fit_method(method)
    if method == "linear":
        return _linear_slope(times, values)
    return _theil_sen_slope(times, values)


def _linear_slope(times: FloatArray, values: FloatArray) -> float:
    centered_times = times - float(np.mean(times))
    denominator = float(np.sum(np.square(centered_times)))
    if denominator <= 0:
        raise ValueError("at least two distinct frame indices are required")
    centered_values = values - float(np.mean(values))
    return float(np.sum(centered_times * centered_values) / denominator)


def _theil_sen_slope(times: FloatArray, values: FloatArray) -> float:
    finite = np.isfinite(times) & np.isfinite(values)
    t = np.asarray(times[finite], dtype=np.float64)
    y = np.asarray(values[finite], dtype=np.float64)
    if np.unique(t).size < 2:
        raise ValueError("at least two distinct frame indices are required")
    slopes: list[np.ndarray] = []
    for index in range(t.size - 1):
        dt = t[index + 1 :] - t[index]
        dy = y[index + 1 :] - y[index]
        valid = dt != 0
        if np.any(valid):
            slopes.append(dy[valid] / dt[valid])
    if not slopes:
        raise ValueError("at least two distinct frame indices are required")
    return float(np.median(np.concatenate(slopes)))

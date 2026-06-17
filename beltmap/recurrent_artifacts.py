"""Suppress recurrent belt-coordinate detection artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from numbers import Real
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .phase import BeltMotionModel
from .tracking import ParticleDetection

RECURRENT_ARTIFACT_MODES = {"hard", "probabilistic", "soft"}
RECURRENT_ARTIFACT_COUNT_DTYPE = np.uint32


@dataclass(frozen=True)
class RecurrentArtifactConfig:
    """Settings for rejecting detections recurring at fixed belt coordinates."""

    min_revolutions: int = 0
    margin_px: int = 2
    max_overlap_fraction: float = 0.3
    mode: str = "hard"
    soft_penalty_weight: float = 1.0
    min_recurrence_probability: float = 0.0
    candidate_max_area_px: int | None = None
    candidate_max_peak_signal: float | None = None
    reject_max_area_px: int | None = None
    reject_max_peak_signal: float | None = None


@dataclass(frozen=True)
class RecurrentArtifactMap:
    """Belt-coordinate recurrence map and diagnostics."""

    mask: NDArray[np.bool_]
    counts: NDArray[np.uint32]
    exposure_counts: NDArray[np.uint32]
    probability: NDArray[np.floating]
    revolution_count: int
    candidate_detections: int
    artifact_pixels: int


@dataclass(frozen=True)
class RecurrentArtifactDetectionScore:
    """Artifact-overlap decision for one detection."""

    detection: ParticleDetection
    overlap_fraction: float
    artifact_probability: float
    required_peak_signal: float | None
    rejected: bool


def belt_revolution_indices(
    frame_count: int,
    motion_model: BeltMotionModel,
) -> NDArray[np.integer]:
    """Return the integer belt revolution index for each processed frame."""

    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    period_px = motion_model.period_px
    if period_px is None:
        raise ValueError("motion_model period must be finite and positive")
    period_px_value = _finite_real_config_value(period_px, "motion_model period")
    if period_px_value <= 0:
        raise ValueError("motion_model period must be finite and positive")
    velocity = _finite_real_config_value(
        motion_model.image_velocity_px_per_frame,
        "motion_model image velocity",
    )
    reference_frame = _finite_real_config_value(
        motion_model.reference_frame,
        "motion_model reference_frame",
    )
    frames = np.arange(frame_count, dtype=np.float64)
    displacement = np.abs(velocity * (frames - reference_frame))
    return np.floor(displacement / period_px_value).astype(np.int64)


def build_recurrent_artifact_map(
    detections_by_frame: Sequence[Sequence[ParticleDetection]],
    phase_px_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    *,
    map_shape: tuple[int, int],
    config: RecurrentArtifactConfig | None = None,
    frame_shape: tuple[int, int] | None = None,
) -> RecurrentArtifactMap:
    """Build a belt-coordinate prior from detections recurring across revolutions."""

    cfg = config or RecurrentArtifactConfig()
    cfg = _validate_config(cfg)
    phase_px_values = _validate_phase_px_by_frame(
        phase_px_by_frame,
        frame_count=len(detections_by_frame),
    )
    revolution_values = _validate_revolution_by_frame(
        revolution_by_frame,
        frame_count=len(detections_by_frame),
    )
    map_height, map_width = _validate_map_shape(map_shape)
    frame_height = _validate_frame_shape(frame_shape, map_width)

    counts = np.zeros((map_height, map_width), dtype=RECURRENT_ARTIFACT_COUNT_DTYPE)
    exposure_counts = np.zeros(
        (map_height, map_width), dtype=RECURRENT_ARTIFACT_COUNT_DTYPE
    )
    unique_revolutions = sorted(set(revolution_values))
    candidate_detections = 0
    for revolution in unique_revolutions:
        revolution_mask, revolution_candidates = _build_revolution_detection_mask(
            detections_by_frame,
            phase_px_values,
            revolution_values,
            revolution=revolution,
            map_shape=(map_height, map_width),
            margin_px=cfg.margin_px,
            candidate_max_area_px=cfg.candidate_max_area_px,
            candidate_max_peak_signal=cfg.candidate_max_peak_signal,
            frame_height=frame_height,
        )
        revolution_exposure = _build_revolution_exposure_mask(
            phase_px_values,
            revolution_values,
            revolution=revolution,
            map_shape=(map_height, map_width),
            frame_height=frame_height,
        )
        candidate_detections += revolution_candidates
        counts += revolution_mask.astype(counts.dtype)
        exposure_counts += revolution_exposure.astype(exposure_counts.dtype)

    probability = _recurrence_probability(counts, exposure_counts)
    mask = (counts >= cfg.min_revolutions) & (
        probability >= cfg.min_recurrence_probability
    )
    return RecurrentArtifactMap(
        mask=mask,
        counts=counts,
        exposure_counts=exposure_counts,
        probability=probability,
        revolution_count=len(unique_revolutions),
        candidate_detections=candidate_detections,
        artifact_pixels=int(np.count_nonzero(mask)),
    )


def score_recurrent_artifact_detections_excluding_current_revolution(
    detections_by_frame: Sequence[Sequence[ParticleDetection]],
    phase_px_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    recurrent_map: RecurrentArtifactMap,
    *,
    config: RecurrentArtifactConfig | None = None,
    frame_shape: tuple[int, int] | None = None,
    detection_threshold: float | None = None,
) -> list[list[RecurrentArtifactDetectionScore]]:
    """Score detections against artifact evidence from other revolutions only.

    This is intended for maps built from the same detections being filtered.
    A detection's current belt revolution is subtracted before scoring, so a
    one-off particle cannot create the artifact evidence that rejects itself.

    ``min_revolutions`` keeps its total-revolution meaning: for
    ``min_revolutions=2`` one other revolution is sufficient, while
    ``min_revolutions=1`` still requires one other revolution to avoid
    self-only rejection.
    """

    cfg = config or RecurrentArtifactConfig(min_revolutions=1)
    cfg = _validate_config(cfg)
    phase_px_values = _validate_phase_px_by_frame(
        phase_px_by_frame,
        frame_count=len(detections_by_frame),
    )
    revolution_values = _validate_revolution_by_frame(
        revolution_by_frame,
        frame_count=len(detections_by_frame),
    )

    counts = np.asarray(recurrent_map.counts, dtype=np.int64)
    map_height, map_width = _validate_map_shape(counts.shape)
    exposure_counts = np.asarray(recurrent_map.exposure_counts, dtype=np.int64)
    if tuple(recurrent_map.mask.shape) != (map_height, map_width):
        raise ValueError("recurrent_map mask and counts shapes must match")
    if tuple(exposure_counts.shape) != (map_height, map_width):
        raise ValueError("recurrent_map exposure_counts and counts shapes must match")
    frame_height = _validate_frame_shape(frame_shape, map_width)

    unique_revolutions = sorted(set(revolution_values))
    other_revolutions_required = max(1, cfg.min_revolutions - 1)
    artifact_maps_by_revolution: dict[int, NDArray[np.bool_]] = {}
    for revolution in unique_revolutions:
        revolution_mask, _ = _build_revolution_detection_mask(
            detections_by_frame,
            phase_px_values,
            revolution_values,
            revolution=revolution,
            map_shape=(map_height, map_width),
            margin_px=cfg.margin_px,
            candidate_max_area_px=cfg.candidate_max_area_px,
            candidate_max_peak_signal=cfg.candidate_max_peak_signal,
            frame_height=frame_height,
        )
        revolution_exposure = _build_revolution_exposure_mask(
            phase_px_values,
            revolution_values,
            revolution=revolution,
            map_shape=(map_height, map_width),
            frame_height=frame_height,
        )
        other_counts = np.maximum(
            counts - revolution_mask.astype(np.int64, copy=False),
            0,
        )
        other_exposure_counts = np.maximum(
            exposure_counts - revolution_exposure.astype(np.int64, copy=False),
            0,
        )
        other_probability = _recurrence_probability(
            other_counts.astype(RECURRENT_ARTIFACT_COUNT_DTYPE, copy=False),
            other_exposure_counts.astype(RECURRENT_ARTIFACT_COUNT_DTYPE, copy=False),
        )
        other_mask = (other_counts >= other_revolutions_required) & (
            other_probability >= cfg.min_recurrence_probability
        )
        if cfg.mode.strip().lower() == "probabilistic":
            artifact_maps_by_revolution[revolution] = np.where(
                other_mask,
                other_probability,
                0.0,
            )
        else:
            artifact_maps_by_revolution[revolution] = other_mask

    scored: list[list[RecurrentArtifactDetectionScore]] = []
    for frame_index, detections in enumerate(detections_by_frame):
        revolution = revolution_values[frame_index]
        frame_scores = score_recurrent_artifact_detections(
            [detections],
            [phase_px_values[frame_index]],
            artifact_maps_by_revolution[revolution],
            config=cfg,
            detection_threshold=detection_threshold,
        )[0]
        scored.append(frame_scores)
    return scored


def filter_recurrent_artifact_detections(
    detections_by_frame: Sequence[Sequence[ParticleDetection]],
    phase_px_by_frame: Sequence[float],
    artifact_map: RecurrentArtifactMap | NDArray[np.bool_] | NDArray[np.floating],
    *,
    config: RecurrentArtifactConfig | None = None,
    detection_threshold: float | None = None,
) -> tuple[list[list[ParticleDetection]], int]:
    """Reject detections whose belt-coordinate bbox mostly overlaps artifacts."""

    scored = score_recurrent_artifact_detections(
        detections_by_frame,
        phase_px_by_frame,
        artifact_map,
        config=config,
        detection_threshold=detection_threshold,
    )
    filtered: list[list[ParticleDetection]] = []
    rejected = 0
    for frame_scores in scored:
        kept: list[ParticleDetection] = []
        for score in frame_scores:
            if score.rejected:
                rejected += 1
            else:
                kept.append(score.detection)
        filtered.append(kept)
    return filtered, rejected


def score_recurrent_artifact_detections(
    detections_by_frame: Sequence[Sequence[ParticleDetection]],
    phase_px_by_frame: Sequence[float],
    artifact_map: RecurrentArtifactMap | NDArray[np.bool_] | NDArray[np.floating],
    *,
    config: RecurrentArtifactConfig | None = None,
    detection_threshold: float | None = None,
) -> list[list[RecurrentArtifactDetectionScore]]:
    """Score detections by their recurrent-artifact overlap and rejection state."""

    cfg = config or RecurrentArtifactConfig(min_revolutions=1)
    cfg = _validate_filter_config(cfg)
    mode = cfg.mode.strip().lower()
    if mode in {"probabilistic", "soft"} and detection_threshold is None:
        raise ValueError(
            "detection_threshold is required for soft/probabilistic recurrent filtering"
        )
    if detection_threshold is not None:
        detection_threshold = _finite_real_config_value(
            detection_threshold,
            "detection_threshold",
        )
        if detection_threshold < 0.0:
            raise ValueError("detection_threshold must be non-negative")
    phase_px_values = _validate_phase_px_by_frame(
        phase_px_by_frame,
        frame_count=len(detections_by_frame),
    )
    artifact = _artifact_prior_array(
        artifact_map,
        probabilistic=mode == "probabilistic",
    )

    scored: list[list[RecurrentArtifactDetectionScore]] = []
    for frame_index, detections in enumerate(detections_by_frame):
        phase_px = phase_px_values[frame_index]
        frame_scores: list[RecurrentArtifactDetectionScore] = []
        for detection in detections:
            overlap = detection_artifact_overlap_fraction(
                detection,
                phase_px=phase_px,
                artifact_map=artifact,
            )
            artifact_probability = detection_artifact_probability(
                detection,
                phase_px=phase_px,
                artifact_map=artifact,
            )
            required_peak = _required_peak_signal(
                overlap=overlap,
                artifact_probability=artifact_probability,
                config=cfg,
                detection_threshold=detection_threshold,
            )
            scored_detection = replace(
                detection,
                recurrent_artifact_overlap_fraction=overlap,
                recurrent_artifact_probability=artifact_probability,
                recurrent_artifact_required_peak_signal=required_peak,
            )
            frame_scores.append(
                RecurrentArtifactDetectionScore(
                    detection=scored_detection,
                    overlap_fraction=overlap,
                    artifact_probability=artifact_probability,
                    required_peak_signal=required_peak,
                    rejected=_reject_detection(
                        scored_detection,
                        overlap=overlap,
                        artifact_probability=artifact_probability,
                        config=cfg,
                        detection_threshold=detection_threshold,
                    ),
                )
            )
        scored.append(frame_scores)
    return scored


def detection_artifact_overlap_fraction(
    detection: ParticleDetection,
    *,
    phase_px: float,
    artifact_map: NDArray[np.bool_] | NDArray[np.floating],
) -> float:
    """Approximate artifact-map support overlap using the detection bounding box."""

    patch = _artifact_patch(detection, phase_px=phase_px, artifact_map=artifact_map)
    if patch.size == 0:
        return 0.0
    return float(np.count_nonzero(patch > 0) / patch.size)


def detection_artifact_probability(
    detection: ParticleDetection,
    *,
    phase_px: float,
    artifact_map: NDArray[np.bool_] | NDArray[np.floating],
) -> float:
    """Return the mean recurrent-artifact prior over a detection bounding box."""

    patch = _artifact_patch(detection, phase_px=phase_px, artifact_map=artifact_map)
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch.astype(np.float64, copy=False)))


def _reject_detection(
    detection: ParticleDetection,
    *,
    overlap: float,
    artifact_probability: float,
    config: RecurrentArtifactConfig,
    detection_threshold: float | None,
) -> bool:
    mode = config.mode.strip().lower()
    decision_value = artifact_probability if mode == "probabilistic" else overlap
    if decision_value <= config.max_overlap_fraction:
        return False
    if not _is_recurrent_artifact_candidate(
        detection,
        max_area_px=config.reject_max_area_px,
        max_peak_signal=config.reject_max_peak_signal,
    ):
        return False
    if mode == "hard":
        return True
    assert detection_threshold is not None
    peak_signal = detection.peak_signal
    if peak_signal is None or not np.isfinite(peak_signal):
        return True
    required_peak = _required_peak_signal(
        overlap=overlap,
        artifact_probability=artifact_probability,
        config=config,
        detection_threshold=detection_threshold,
    )
    assert required_peak is not None
    return peak_signal <= required_peak


def _required_peak_signal(
    *,
    overlap: float,
    artifact_probability: float,
    config: RecurrentArtifactConfig,
    detection_threshold: float | None,
) -> float | None:
    mode = config.mode.strip().lower()
    if mode == "hard":
        return None
    assert detection_threshold is not None
    penalty = artifact_probability if mode == "probabilistic" else overlap
    return detection_threshold * (1.0 + config.soft_penalty_weight * penalty)


def _build_revolution_detection_mask(
    detections_by_frame: Sequence[Sequence[ParticleDetection]],
    phase_px_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    *,
    revolution: int,
    map_shape: tuple[int, int],
    margin_px: int,
    candidate_max_area_px: int | None,
    candidate_max_peak_signal: float | None,
    frame_height: int | None,
) -> tuple[NDArray[np.bool_], int]:
    map_height, map_width = _validate_map_shape(map_shape)
    revolution_mask = np.zeros((map_height, map_width), dtype=bool)
    candidate_detections = 0
    for frame_index, detections in enumerate(detections_by_frame):
        if int(revolution_by_frame[frame_index]) != revolution:
            continue
        phase_px = float(phase_px_by_frame[frame_index])
        for detection in detections:
            if not _is_recurrent_artifact_candidate(
                detection,
                max_area_px=candidate_max_area_px,
                max_peak_signal=candidate_max_peak_signal,
            ):
                continue
            _mark_detection_bbox(
                revolution_mask,
                detection,
                phase_px=phase_px,
                margin_px=margin_px,
                image_height=frame_height,
            )
            candidate_detections += 1
    return revolution_mask, candidate_detections


def _is_recurrent_artifact_candidate(
    detection: ParticleDetection,
    *,
    max_area_px: int | None,
    max_peak_signal: float | None,
) -> bool:
    area_px, _bbox, peak_signal = _validate_detection_geometry(detection)
    if max_area_px is not None and area_px > max_area_px:
        return False
    if max_peak_signal is not None:
        if peak_signal is None:
            return False
        if peak_signal > max_peak_signal:
            return False
    return True


def _mark_detection_bbox(
    mask: NDArray[np.bool_],
    detection: ParticleDetection,
    *,
    phase_px: float,
    margin_px: int,
    image_height: int | None,
) -> None:
    map_height, map_width = _validate_map_shape(mask.shape)
    _area_px, (top, left, bottom, right), _peak_signal = _validate_detection_geometry(
        detection
    )
    phase_px = _finite_real_config_value(phase_px, "phase_px")
    left = max(0, left - margin_px)
    right = min(map_width, right + margin_px)
    if right <= left:
        return
    rows = _belt_rows_for_image_rows(
        range(
            top - margin_px,
            bottom + margin_px,
        ),
        phase_px=phase_px,
        map_height=map_height,
        image_height=image_height,
    )
    if rows.size:
        mask[rows, left:right] = True


def _build_revolution_exposure_mask(
    phase_px_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    *,
    revolution: int,
    map_shape: tuple[int, int],
    frame_height: int | None,
) -> NDArray[np.bool_]:
    map_height, map_width = _validate_map_shape(map_shape)
    exposure = np.zeros((map_height, map_width), dtype=bool)
    if frame_height is None:
        exposure[:, :] = True
        return exposure
    for frame_index, frame_revolution in enumerate(revolution_by_frame):
        if int(frame_revolution) != revolution:
            continue
        _mark_frame_exposure(
            exposure,
            phase_px=float(phase_px_by_frame[frame_index]),
            frame_height=frame_height,
        )
    return exposure


def _mark_frame_exposure(
    mask: NDArray[np.bool_],
    *,
    phase_px: float,
    frame_height: int,
) -> None:
    map_height, _map_width = _validate_map_shape(mask.shape)
    rows = _belt_rows_for_image_rows(
        range(frame_height),
        phase_px=phase_px,
        map_height=map_height,
    )
    if rows.size:
        mask[rows, :] = True


def _belt_rows_for_image_rows(
    image_rows: range,
    *,
    phase_px: float,
    map_height: int,
    image_height: int | None = None,
) -> NDArray[np.integer]:
    rows = np.fromiter(image_rows, dtype=np.float64)
    if rows.size == 0:
        return np.array([], dtype=np.int64)
    if image_height is None:
        rows = rows[rows >= 0]
    else:
        rows = rows[(rows >= 0) & (rows < image_height)]
    if rows.size == 0:
        return np.array([], dtype=np.int64)
    belt_rows = np.rint(rows + phase_px).astype(np.int64) % map_height
    return np.unique(belt_rows)


def _artifact_patch(
    detection: ParticleDetection,
    *,
    phase_px: float,
    artifact_map: NDArray[np.bool_] | NDArray[np.floating],
) -> NDArray[np.bool_] | NDArray[np.floating]:
    artifact = np.asarray(artifact_map)
    map_height, map_width = _validate_map_shape(artifact.shape)
    _area_px, (top, left, bottom, right), _peak_signal = _validate_detection_geometry(
        detection
    )
    phase_px = _finite_real_config_value(phase_px, "phase_px")
    left = max(0, left)
    right = min(map_width, right)
    if right <= left:
        return artifact[:0, :0]
    rows = _belt_rows_for_image_rows(
        range(top, bottom),
        phase_px=phase_px,
        map_height=map_height,
    )
    if rows.size == 0:
        return artifact[:0, :0]
    return artifact[rows, left:right]


def _artifact_prior_array(
    artifact_map: RecurrentArtifactMap | NDArray[np.bool_] | NDArray[np.floating],
    *,
    probabilistic: bool,
) -> NDArray[np.bool_] | NDArray[np.floating]:
    if isinstance(artifact_map, RecurrentArtifactMap):
        if probabilistic:
            return np.where(artifact_map.mask, artifact_map.probability, 0.0).astype(
                np.float32,
                copy=False,
            )
        return np.asarray(artifact_map.mask, dtype=bool)

    artifact = np.asarray(artifact_map)
    if artifact.ndim != 2 or artifact.size == 0:
        raise ValueError("artifact_map must be a non-empty 2-D array")
    if artifact.dtype == np.bool_ or np.issubdtype(artifact.dtype, np.bool_):
        return np.asarray(artifact, dtype=bool)
    if not np.issubdtype(artifact.dtype, np.number):
        raise ValueError("artifact_map must be boolean or numeric")
    prior = np.asarray(artifact, dtype=np.float32)
    if not np.all(np.isfinite(prior)):
        raise ValueError("numeric artifact_map must be finite")
    if np.any((prior < 0.0) | (prior > 1.0)):
        raise ValueError("numeric artifact_map probabilities must be in [0, 1]")
    return prior


def _recurrence_probability(
    counts: NDArray[np.unsignedinteger],
    exposure_counts: NDArray[np.unsignedinteger],
) -> NDArray[np.floating]:
    probability = np.zeros(counts.shape, dtype=np.float32)
    exposed = exposure_counts > 0
    probability[exposed] = counts[exposed] / exposure_counts[exposed]
    return probability


def _validate_config(config: RecurrentArtifactConfig) -> RecurrentArtifactConfig:
    min_revolutions = _positive_integer_config_value(
        config.min_revolutions,
        "min_revolutions",
    )
    if min_revolutions < 1:
        raise ValueError("min_revolutions must be at least 1")
    cfg = _validate_filter_config(config)
    return replace(cfg, min_revolutions=min_revolutions)


def _validate_filter_config(config: RecurrentArtifactConfig) -> RecurrentArtifactConfig:
    margin_px = _nonnegative_integer_config_value(config.margin_px, "margin_px")
    max_overlap_fraction = _unit_interval_config_value(
        config.max_overlap_fraction,
        "max_overlap_fraction",
    )
    min_recurrence_probability = _unit_interval_config_value(
        config.min_recurrence_probability,
        "min_recurrence_probability",
    )
    mode = _mode_config_value(config.mode)
    soft_penalty_weight = _nonnegative_real_config_value(
        config.soft_penalty_weight,
        "soft_penalty_weight",
    )
    candidate_max_area_px = _optional_positive_integer_config_value(
        config.candidate_max_area_px,
        "candidate_max_area_px",
    )
    candidate_max_peak_signal = _optional_nonnegative_real_config_value(
        config.candidate_max_peak_signal,
        "candidate_max_peak_signal",
    )
    reject_max_area_px = _optional_positive_integer_config_value(
        config.reject_max_area_px,
        "reject_max_area_px",
    )
    reject_max_peak_signal = _optional_nonnegative_real_config_value(
        config.reject_max_peak_signal,
        "reject_max_peak_signal",
    )
    return replace(
        config,
        margin_px=margin_px,
        max_overlap_fraction=max_overlap_fraction,
        mode=mode,
        soft_penalty_weight=soft_penalty_weight,
        min_recurrence_probability=min_recurrence_probability,
        candidate_max_area_px=candidate_max_area_px,
        candidate_max_peak_signal=candidate_max_peak_signal,
        reject_max_area_px=reject_max_area_px,
        reject_max_peak_signal=reject_max_peak_signal,
    )


def _mode_config_value(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("mode must be a string")
    mode = value.strip().lower()
    if mode not in RECURRENT_ARTIFACT_MODES:
        choices = ", ".join(sorted(RECURRENT_ARTIFACT_MODES))
        raise ValueError(f"mode must be one of {choices}")
    return mode


def _validate_phase_px_by_frame(
    phase_px_by_frame: Sequence[float],
    *,
    frame_count: int,
) -> list[float]:
    if len(phase_px_by_frame) != frame_count:
        raise ValueError("phase_px_by_frame must match detections_by_frame length")
    return [
        _finite_real_config_value(value, "phase_px_by_frame values")
        for value in phase_px_by_frame
    ]


def _validate_revolution_by_frame(
    revolution_by_frame: Sequence[int],
    *,
    frame_count: int,
) -> list[int]:
    if len(revolution_by_frame) != frame_count:
        raise ValueError("revolution_by_frame must match detections_by_frame length")
    values: list[int] = []
    for revolution in revolution_by_frame:
        values.append(
            _integer_config_value(revolution, "revolution_by_frame values")
        )
    return values


def _finite_real_config_value(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _integer_config_value(value: object, name: str) -> int:
    parsed = _finite_real_config_value(value, name)
    if not parsed.is_integer():
        raise ValueError(f"{name} must be a finite integer")
    return int(parsed)


def _nonnegative_integer_config_value(value: int, name: str) -> int:
    parsed = _integer_config_value(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _positive_integer_config_value(value: int, name: str) -> int:
    parsed = _integer_config_value(value, name)
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def _optional_positive_integer_config_value(
    value: int | None,
    name: str,
) -> int | None:
    if value is None:
        return None
    return _positive_integer_config_value(value, name)


def _nonnegative_real_config_value(value: object, name: str) -> float:
    parsed = _finite_real_config_value(value, name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _optional_nonnegative_real_config_value(
    value: object | None,
    name: str,
) -> float | None:
    if value is None:
        return None
    return _nonnegative_real_config_value(value, name)


def _unit_interval_config_value(value: object, name: str) -> float:
    parsed = _finite_real_config_value(value, name)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return parsed


def _validate_detection_geometry(
    detection: ParticleDetection,
) -> tuple[int, tuple[int, int, int, int], float | None]:
    _integer_config_value(detection.label, "detection.label")
    _finite_real_config_value(detection.y, "detection.y")
    _finite_real_config_value(detection.x, "detection.x")
    area_px = _positive_integer_config_value(detection.area_px, "detection.area_px")
    top = _integer_config_value(detection.bbox_top, "detection.bbox_top")
    left = _integer_config_value(detection.bbox_left, "detection.bbox_left")
    bottom = _integer_config_value(detection.bbox_bottom, "detection.bbox_bottom")
    right = _integer_config_value(detection.bbox_right, "detection.bbox_right")
    if top < 0 or left < 0:
        raise ValueError("detection bbox coordinates must be non-negative")
    if bottom <= top or right <= left:
        raise ValueError("detection bbox must be half-open with positive area")
    peak_signal = (
        None
        if detection.peak_signal is None
        else _finite_real_config_value(detection.peak_signal, "detection.peak_signal")
    )
    return area_px, (top, left, bottom, right), peak_signal


def _validate_map_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("map shape must be 2-D")
    height = _positive_integer_dimension(shape[0], "map shape")
    width = _positive_integer_dimension(shape[1], "map shape")
    return height, width


def _validate_frame_shape(
    shape: tuple[int, int] | None,
    map_width: int,
) -> int | None:
    if shape is None:
        return None
    if len(shape) != 2:
        raise ValueError("frame_shape must be 2-D")
    frame_height = _positive_integer_dimension(shape[0], "frame_shape")
    frame_width = _positive_integer_dimension(shape[1], "frame_shape")
    if frame_width != map_width:
        raise ValueError(
            "frame_shape width must match recurrent artifact map width: "
            f"{frame_width} != {map_width}"
        )
    return frame_height


def _positive_integer_dimension(value: int, name: str) -> int:
    parsed = _finite_real_config_value(value, name)
    if not parsed.is_integer() or parsed < 1:
        raise ValueError(f"{name} dimensions must be positive finite integers")
    return int(parsed)

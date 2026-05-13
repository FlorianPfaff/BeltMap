"""Suppress recurrent belt-coordinate detection artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .phase import BeltMotionModel
from .tracking import ParticleDetection

RECURRENT_ARTIFACT_MODES = {"hard", "soft"}


@dataclass(frozen=True)
class RecurrentArtifactConfig:
    """Settings for rejecting detections recurring at fixed belt coordinates."""

    min_revolutions: int = 0
    margin_px: int = 2
    max_overlap_fraction: float = 0.3
    mode: str = "hard"
    soft_penalty_weight: float = 1.0


@dataclass(frozen=True)
class RecurrentArtifactMap:
    """Belt-coordinate recurrence map and diagnostics."""

    mask: NDArray[np.bool_]
    counts: NDArray[np.unsignedinteger]
    revolution_count: int
    candidate_detections: int
    artifact_pixels: int


@dataclass(frozen=True)
class RecurrentArtifactDetectionScore:
    """Artifact-overlap decision for one detection."""

    detection: ParticleDetection
    overlap_fraction: float
    required_peak_signal: float | None
    rejected: bool


def belt_revolution_indices(
    frame_count: int,
    motion_model: BeltMotionModel,
) -> NDArray[np.integer]:
    """Return the integer belt revolution index for each processed frame."""

    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    if motion_model.period_px <= 0:
        raise ValueError("motion_model period must be positive")
    frames = np.arange(frame_count, dtype=np.float64)
    displacement = np.abs(
        motion_model.image_velocity_px_per_frame
        * (frames - float(motion_model.reference_frame))
    )
    return np.floor(displacement / float(motion_model.period_px)).astype(np.int64)


def build_recurrent_artifact_map(
    detections_by_frame: Sequence[Sequence[ParticleDetection]],
    phase_px_by_frame: Sequence[float],
    revolution_by_frame: Sequence[int],
    *,
    map_shape: tuple[int, int],
    config: RecurrentArtifactConfig | None = None,
) -> RecurrentArtifactMap:
    """Build a belt-coordinate mask from detections recurring across revolutions."""

    cfg = config or RecurrentArtifactConfig()
    _validate_config(cfg)
    if len(phase_px_by_frame) != len(detections_by_frame):
        raise ValueError("phase_px_by_frame must match detections_by_frame length")
    if len(revolution_by_frame) != len(detections_by_frame):
        raise ValueError("revolution_by_frame must match detections_by_frame length")
    map_height, map_width = _validate_map_shape(map_shape)

    counts = np.zeros((map_height, map_width), dtype=np.uint16)
    unique_revolutions = sorted({int(revolution) for revolution in revolution_by_frame})
    candidate_detections = 0
    for revolution in unique_revolutions:
        revolution_mask = np.zeros((map_height, map_width), dtype=bool)
        for frame_index, detections in enumerate(detections_by_frame):
            if int(revolution_by_frame[frame_index]) != revolution:
                continue
            phase_px = float(phase_px_by_frame[frame_index])
            for detection in detections:
                _mark_detection_bbox(
                    revolution_mask,
                    detection,
                    phase_px=phase_px,
                    margin_px=cfg.margin_px,
                )
                candidate_detections += 1
        counts += revolution_mask.astype(counts.dtype)

    mask = counts >= cfg.min_revolutions
    return RecurrentArtifactMap(
        mask=mask,
        counts=counts,
        revolution_count=len(unique_revolutions),
        candidate_detections=candidate_detections,
        artifact_pixels=int(np.count_nonzero(mask)),
    )


def filter_recurrent_artifact_detections(
    detections_by_frame: Sequence[Sequence[ParticleDetection]],
    phase_px_by_frame: Sequence[float],
    artifact_map: NDArray[np.bool_],
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
    artifact_map: NDArray[np.bool_],
    *,
    config: RecurrentArtifactConfig | None = None,
    detection_threshold: float | None = None,
) -> list[list[RecurrentArtifactDetectionScore]]:
    """Score detections by their recurrent-artifact overlap and rejection state."""

    cfg = config or RecurrentArtifactConfig(min_revolutions=1)
    _validate_filter_config(cfg)
    mode = cfg.mode.strip().lower()
    if mode == "soft" and detection_threshold is None:
        raise ValueError("detection_threshold is required for soft recurrent filtering")
    if detection_threshold is not None and not np.isfinite(detection_threshold):
        raise ValueError("detection_threshold must be finite")
    if len(phase_px_by_frame) != len(detections_by_frame):
        raise ValueError("phase_px_by_frame must match detections_by_frame length")
    artifact = np.asarray(artifact_map, dtype=bool)
    if artifact.ndim != 2 or artifact.size == 0:
        raise ValueError("artifact_map must be a non-empty 2-D array")

    scored: list[list[RecurrentArtifactDetectionScore]] = []
    for frame_index, detections in enumerate(detections_by_frame):
        phase_px = float(phase_px_by_frame[frame_index])
        frame_scores: list[RecurrentArtifactDetectionScore] = []
        for detection in detections:
            overlap = detection_artifact_overlap_fraction(
                detection,
                phase_px=phase_px,
                artifact_map=artifact,
            )
            required_peak = _required_peak_signal(
                overlap=overlap,
                config=cfg,
                detection_threshold=detection_threshold,
            )
            scored_detection = replace(
                detection,
                recurrent_artifact_overlap_fraction=overlap,
                recurrent_artifact_required_peak_signal=required_peak,
            )
            frame_scores.append(
                RecurrentArtifactDetectionScore(
                    detection=scored_detection,
                    overlap_fraction=overlap,
                    required_peak_signal=required_peak,
                    rejected=_reject_detection(
                        scored_detection,
                        overlap=overlap,
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
    artifact_map: NDArray[np.bool_],
) -> float:
    """Approximate artifact overlap using the detection bounding box."""

    artifact = np.asarray(artifact_map, dtype=bool)
    map_height, map_width = _validate_map_shape(artifact.shape)
    left = max(0, int(detection.bbox_left))
    right = min(map_width, int(detection.bbox_right))
    if right <= left or detection.bbox_bottom <= detection.bbox_top:
        return 0.0
    rows = _belt_rows_for_image_rows(
        range(int(detection.bbox_top), int(detection.bbox_bottom)),
        phase_px=phase_px,
        map_height=map_height,
    )
    if rows.size == 0:
        return 0.0
    patch = artifact[rows, left:right]
    return float(np.count_nonzero(patch) / patch.size)


def _reject_detection(
    detection: ParticleDetection,
    *,
    overlap: float,
    config: RecurrentArtifactConfig,
    detection_threshold: float | None,
) -> bool:
    mode = config.mode.strip().lower()
    if overlap <= config.max_overlap_fraction:
        return False
    if mode == "hard":
        return True
    assert detection_threshold is not None
    peak_signal = detection.peak_signal
    if peak_signal is None or not np.isfinite(peak_signal):
        return True
    required_peak = _required_peak_signal(
        overlap=overlap,
        config=config,
        detection_threshold=detection_threshold,
    )
    assert required_peak is not None
    return peak_signal <= required_peak


def _required_peak_signal(
    *,
    overlap: float,
    config: RecurrentArtifactConfig,
    detection_threshold: float | None,
) -> float | None:
    if config.mode.strip().lower() != "soft":
        return None
    assert detection_threshold is not None
    return detection_threshold * (1.0 + config.soft_penalty_weight * overlap)


def _mark_detection_bbox(
    mask: NDArray[np.bool_],
    detection: ParticleDetection,
    *,
    phase_px: float,
    margin_px: int,
) -> None:
    map_height, map_width = _validate_map_shape(mask.shape)
    left = max(0, int(detection.bbox_left) - margin_px)
    right = min(map_width, int(detection.bbox_right) + margin_px)
    if right <= left:
        return
    rows = _belt_rows_for_image_rows(
        range(
            int(detection.bbox_top) - margin_px,
            int(detection.bbox_bottom) + margin_px,
        ),
        phase_px=phase_px,
        map_height=map_height,
    )
    if rows.size:
        mask[rows, left:right] = True


def _belt_rows_for_image_rows(
    image_rows: range,
    *,
    phase_px: float,
    map_height: int,
) -> NDArray[np.integer]:
    rows = np.fromiter(image_rows, dtype=np.float64)
    if rows.size == 0:
        return np.array([], dtype=np.int64)
    belt_rows = np.rint(rows + phase_px).astype(np.int64) % map_height
    return np.unique(belt_rows)


def _validate_config(config: RecurrentArtifactConfig) -> None:
    if config.min_revolutions < 1:
        raise ValueError("min_revolutions must be at least 1")
    _validate_filter_config(config)


def _validate_filter_config(config: RecurrentArtifactConfig) -> None:
    if config.margin_px < 0:
        raise ValueError("margin_px must be non-negative")
    if not 0 <= config.max_overlap_fraction <= 1:
        raise ValueError("max_overlap_fraction must be in [0, 1]")
    if config.mode.strip().lower() not in RECURRENT_ARTIFACT_MODES:
        choices = ", ".join(sorted(RECURRENT_ARTIFACT_MODES))
        raise ValueError(f"mode must be one of {choices}")
    if not np.isfinite(config.soft_penalty_weight) or config.soft_penalty_weight < 0:
        raise ValueError("soft_penalty_weight must be finite and non-negative")


def _validate_map_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("map shape must be 2-D")
    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("map shape must be non-empty")
    return height, width

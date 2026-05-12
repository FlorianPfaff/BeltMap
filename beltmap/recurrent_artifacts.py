"""Suppress recurrent belt-coordinate detection artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .phase import BeltMotionModel
from .tracking import ParticleDetection


@dataclass(frozen=True)
class RecurrentArtifactConfig:
    """Settings for rejecting detections recurring at fixed belt coordinates."""

    min_revolutions: int = 0
    margin_px: int = 2
    max_overlap_fraction: float = 0.3


@dataclass(frozen=True)
class RecurrentArtifactMap:
    """Belt-coordinate recurrence map and diagnostics."""

    mask: NDArray[np.bool_]
    counts: NDArray[np.unsignedinteger]
    revolution_count: int
    candidate_detections: int
    artifact_pixels: int


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
    max_overlap_fraction: float,
) -> tuple[list[list[ParticleDetection]], int]:
    """Reject detections whose belt-coordinate bbox mostly overlaps artifacts."""

    if len(phase_px_by_frame) != len(detections_by_frame):
        raise ValueError("phase_px_by_frame must match detections_by_frame length")
    if not 0 <= max_overlap_fraction <= 1:
        raise ValueError("max_overlap_fraction must be in [0, 1]")
    artifact = np.asarray(artifact_map, dtype=bool)
    if artifact.ndim != 2 or artifact.size == 0:
        raise ValueError("artifact_map must be a non-empty 2-D array")

    filtered: list[list[ParticleDetection]] = []
    rejected = 0
    for frame_index, detections in enumerate(detections_by_frame):
        phase_px = float(phase_px_by_frame[frame_index])
        kept: list[ParticleDetection] = []
        for detection in detections:
            overlap = detection_artifact_overlap_fraction(
                detection,
                phase_px=phase_px,
                artifact_map=artifact,
            )
            if overlap > max_overlap_fraction:
                rejected += 1
            else:
                kept.append(detection)
        filtered.append(kept)
    return filtered, rejected


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
    if config.margin_px < 0:
        raise ValueError("margin_px must be non-negative")
    if not 0 <= config.max_overlap_fraction <= 1:
        raise ValueError("max_overlap_fraction must be in [0, 1]")


def _validate_map_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("map shape must be 2-D")
    height, width = (int(shape[0]), int(shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("map shape must be non-empty")
    return height, width

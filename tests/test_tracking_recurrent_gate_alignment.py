import pytest

import beltmap
import beltmap.tracking as tracking_module
from beltmap import (
    ParticleDetection,
    ParticleTrack,
    ParticleVelocity,
    TrackFilterConfig,
    score_particle_velocities,
)


def _velocity(track_id: int) -> ParticleVelocity:
    return ParticleVelocity(
        track_id=track_id,
        n_detections=1,
        frame_start=0.0,
        frame_end=0.0,
        velocity_y_px_per_frame=1.0,
        velocity_x_px_per_frame=0.0,
        speed_px_per_frame=1.0,
        belt_velocity_y_px_per_frame=2.0,
        velocity_ratio_y=0.5,
        belt_minus_particle_velocity_y_px_per_frame=1.0,
    )


def _track(track_id: int, probability: float = 0.0) -> ParticleTrack:
    detection = ParticleDetection(
        frame_index=0.0,
        label=1,
        y=1.0,
        x=1.0,
        area_px=1,
        bbox_top=1,
        bbox_left=1,
        bbox_bottom=2,
        bbox_right=2,
        recurrent_artifact_probability=probability,
    )
    return ParticleTrack(track_id=track_id, detections=(detection,))


def _recurrent_gate() -> TrackFilterConfig:
    return TrackFilterConfig(
        min_track_length=1,
        max_recurrent_artifact_track_score=0.5,
    )


def test_package_and_tracking_entrypoints_use_alignment_guard():
    assert beltmap.score_particle_velocities is tracking_module.score_particle_velocities
    assert score_particle_velocities is tracking_module.score_particle_velocities


def test_recurrent_gate_rejects_missing_track_alignment():
    with pytest.raises(ValueError, match=r"missing track_id values: 7"):
        score_particle_velocities(
            [_velocity(7)],
            config=_recurrent_gate(),
            tracks=[_track(8)],
        )


def test_recurrent_gate_rejects_duplicate_track_ids():
    with pytest.raises(ValueError, match=r"unique track_id values.*duplicates: 7"):
        score_particle_velocities(
            [_velocity(7)],
            config=_recurrent_gate(),
            tracks=[_track(7, 0.0), _track(7, 1.0)],
        )


def test_recurrent_gate_preserves_valid_aligned_scoring():
    [score] = score_particle_velocities(
        [_velocity(7)],
        config=_recurrent_gate(),
        tracks=[_track(7, 0.0)],
    )

    assert score.track_id == 7
    assert score.passes_recurrent_artifact is True
    assert score.accepted is True


def test_alignment_guard_is_inactive_without_recurrent_gate():
    [score] = score_particle_velocities(
        [_velocity(7)],
        config=TrackFilterConfig(min_track_length=1),
        tracks=[_track(8)],
    )

    assert score.track_id == 7
    assert score.accepted is True

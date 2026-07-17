import pytest

import beltmap.tracking as tracking_module
from beltmap import (
    ParticleDetection,
    ParticleTrack,
    ParticleVelocity,
    TrackFilterConfig,
    score_particle_velocities,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (0.0, 0.0),
        (1.0, 1.0),
        ("0.25", 0.25),
        (True, None),
        (False, None),
        (-0.01, None),
        (1.01, None),
        (float("nan"), None),
        (float("inf"), None),
        ("invalid", None),
    ],
)
def test_recurrent_evidence_parser_requires_a_true_unit_interval(value, expected):
    assert tracking_module._finite_unit_interval_value(value) == expected


@pytest.mark.parametrize("bad_probability", [True, -0.2, 1.2, "invalid"])
def test_malformed_recurrent_evidence_cannot_reject_a_track(bad_probability):
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
        recurrent_artifact_probability=bad_probability,
    )
    track = ParticleTrack(track_id=7, detections=(detection,))
    velocity = ParticleVelocity(
        track_id=7,
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

    [score] = score_particle_velocities(
        [velocity],
        config=TrackFilterConfig(
            min_track_length=1,
            max_recurrent_artifact_track_score=0.1,
        ),
        tracks=[track],
    )

    assert score.n_recurrent_artifact_scored_detections == 0
    assert score.recurrent_artifact_track_score == 0.0
    assert score.passes_recurrent_artifact is True
    assert score.accepted is True

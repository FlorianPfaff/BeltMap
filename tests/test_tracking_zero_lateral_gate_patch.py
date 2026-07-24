from __future__ import annotations

import importlib
import math

import beltmap
import beltmap.tracking as tracking
import beltmap.tracking_zero_lateral_gate_patch as zero_lateral_patch
from beltmap.trust import PROFILE_CONFIGS


def _velocity(track_id: int, lateral_velocity: float) -> tracking.ParticleVelocity:
    return tracking.ParticleVelocity(
        track_id=track_id,
        n_detections=6,
        frame_start=0.0,
        frame_end=5.0,
        velocity_y_px_per_frame=4.0,
        velocity_x_px_per_frame=lateral_velocity,
        speed_px_per_frame=math.hypot(4.0, lateral_velocity),
        belt_velocity_y_px_per_frame=5.0,
        velocity_ratio_y=0.8,
        belt_minus_particle_velocity_y_px_per_frame=1.0,
    )


def test_zero_lateral_gate_patch_is_autoloaded() -> None:
    assert getattr(
        tracking.score_particle_velocities,
        "_beltmap_zero_lateral_velocity_gate_patched",
        False,
    )
    assert beltmap.score_particle_velocities is tracking.score_particle_velocities


def test_velocity_quality_zero_lateral_gate_is_strict_and_zero_safe() -> None:
    profile_bound = PROFILE_CONFIGS["velocity_quality"]["track_filter"][
        "max_abs_x_velocity_px_per_frame"
    ]
    assert profile_bound == 0.0

    velocities = [_velocity(0, 0.0), _velocity(1, 0.25)]
    config = tracking.TrackFilterConfig(
        min_track_length=5,
        min_velocity_ratio_y=0.0,
        max_velocity_ratio_y=1.1,
        max_abs_x_velocity_px_per_frame=profile_bound,
    )

    scores = beltmap.score_particle_velocities(velocities, config=config)
    filtered = beltmap.filter_particle_velocities(velocities, config=config)

    assert [score.passes_lateral_velocity for score in scores] == [True, False]
    assert [score.accepted for score in scores] == [True, False]
    assert scores[0].plausibility_score > 0.0
    assert scores[1].plausibility_score == 0.0
    assert [velocity.track_id for velocity in filtered] == [0]


def test_zero_lateral_gate_patch_reload_is_idempotent() -> None:
    importlib.reload(zero_lateral_patch)

    score = tracking.score_particle_velocities(
        [_velocity(0, 0.0)],
        config=tracking.TrackFilterConfig(
            max_abs_x_velocity_px_per_frame=0.0,
        ),
    )[0]

    assert score.passes_lateral_velocity is True
    assert score.accepted is True

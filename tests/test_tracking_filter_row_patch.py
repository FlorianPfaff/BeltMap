from beltmap import ParticleVelocity, filter_particle_velocities
import beltmap.tracking as tracking


def _velocity(*, track_id: int, velocity_ratio_y: float) -> ParticleVelocity:
    belt_velocity = 2.0
    velocity_y = velocity_ratio_y * belt_velocity
    return ParticleVelocity(
        track_id=track_id,
        n_detections=5,
        frame_start=0.0,
        frame_end=4.0,
        velocity_y_px_per_frame=velocity_y,
        velocity_x_px_per_frame=0.0,
        speed_px_per_frame=abs(velocity_y),
        belt_velocity_y_px_per_frame=belt_velocity,
        velocity_ratio_y=velocity_ratio_y,
        belt_minus_particle_velocity_y_px_per_frame=belt_velocity - velocity_y,
    )


def test_filter_particle_velocities_filters_duplicate_track_ids_rowwise():
    accepted = _velocity(track_id=7, velocity_ratio_y=0.5)
    rejected = _velocity(track_id=7, velocity_ratio_y=2.0)

    filtered = filter_particle_velocities([accepted, rejected])

    assert filtered == [accepted]


def test_rowwise_velocity_filter_patch_is_autoloaded():
    assert getattr(
        tracking.filter_particle_velocities,
        "_beltmap_rowwise_velocity_filter_patched",
        False,
    )

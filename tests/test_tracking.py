import numpy as np

from beltmap import (
    ParticleComponentConfig,
    ParticleDetection,
    ParticleTrackingConfig,
    estimate_particle_velocities_vs_belt,
    extract_particle_detections,
    extract_particle_velocities_vs_belt,
    track_particle_detections,
)


def test_extract_particle_detections_finds_components_and_weighted_centroid():
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:3, 2:4] = True
    mask[5, 7] = True
    residual = np.zeros_like(mask, dtype=float)
    residual[1, 2] = 1.0
    residual[1, 3] = 3.0
    residual[2, 2] = 1.0
    residual[2, 3] = 5.0
    residual[5, 7] = 9.0

    detections = extract_particle_detections(
        mask,
        residual=residual,
        frame_index=4,
        config=ParticleComponentConfig(min_area_px=2),
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.frame_index == 4
    assert detection.area_px == 4
    assert detection.bbox_top == 1
    assert detection.bbox_bottom == 3
    np.testing.assert_allclose([detection.y, detection.x], [1.6, 2.8])
    assert detection.peak_signal == 5.0


def test_track_particle_detections_uses_velocity_prior():
    detections_by_frame = [
        [ParticleDetection(0, 1, y=10.0, x=5.0, area_px=4, bbox_top=9, bbox_left=4, bbox_bottom=11, bbox_right=6)],
        [ParticleDetection(1, 1, y=13.0, x=5.0, area_px=4, bbox_top=12, bbox_left=4, bbox_bottom=14, bbox_right=6)],
        [ParticleDetection(2, 1, y=16.0, x=5.0, area_px=4, bbox_top=15, bbox_left=4, bbox_bottom=17, bbox_right=6)],
    ]

    tracks = track_particle_detections(
        detections_by_frame,
        config=ParticleTrackingConfig(
            max_match_distance_px=2.0,
            velocity_prior_y_px_per_frame=3.0,
        ),
    )
    velocities = estimate_particle_velocities_vs_belt(
        tracks,
        belt_image_velocity_px_per_frame=5.0,
    )

    assert len(velocities) == 1
    velocity = velocities[0]
    assert velocity.n_detections == 3
    assert velocity.velocity_y_px_per_frame == 3.0
    assert velocity.velocity_x_px_per_frame == 0.0
    assert velocity.velocity_ratio_y == 0.6
    assert velocity.belt_minus_particle_velocity_y_px_per_frame == 2.0


def test_extract_particle_velocities_vs_belt_from_masks():
    masks = []
    for frame_index in range(5):
        mask = np.zeros((40, 20), dtype=bool)
        top = 6 + 3 * frame_index
        mask[top : top + 3, 8:11] = True
        masks.append(mask)

    velocities = extract_particle_velocities_vs_belt(
        masks,
        belt_image_velocity_px_per_frame=5.0,
        component_config=ParticleComponentConfig(min_area_px=4),
        min_track_length=4,
    )

    assert len(velocities) == 1
    np.testing.assert_allclose(velocities[0].velocity_y_px_per_frame, 3.0)
    np.testing.assert_allclose(velocities[0].velocity_ratio_y, 0.6)


def test_track_particle_detections_drops_tracks_across_explicit_empty_frame_gap():
    detections_by_frame = [
        [ParticleDetection(0, 1, y=10.0, x=5.0, area_px=4, bbox_top=9, bbox_left=4, bbox_bottom=11, bbox_right=6)],
        [],
        [ParticleDetection(3, 1, y=13.0, x=5.0, area_px=4, bbox_top=12, bbox_left=4, bbox_bottom=14, bbox_right=6)],
    ]

    tracks = track_particle_detections(
        detections_by_frame,
        frame_indices=[0, 2, 3],
        config=ParticleTrackingConfig(
            max_match_distance_px=10.0,
            max_frame_gap=1.0,
            velocity_prior_y_px_per_frame=1.0,
        ),
    )

    assert [track.n_detections for track in tracks] == [1, 1]

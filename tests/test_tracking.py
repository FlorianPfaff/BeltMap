import numpy as np

import beltmap.tracking as tracking_module
from beltmap import (
    ParticleComponentConfig,
    ParticleDetection,
    ParticleTrackingConfig,
    ParticleVelocity,
    TrackFilterConfig,
    estimate_particle_velocities_vs_belt,
    extract_particle_detections,
    extract_particle_velocities_vs_belt,
    filter_particle_velocities,
    score_particle_velocities,
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


def test_extract_particle_detections_honors_connectivity():
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True

    detections_4 = extract_particle_detections(
        mask,
        config=ParticleComponentConfig(connectivity=4),
    )
    detections_8 = extract_particle_detections(
        mask,
        config=ParticleComponentConfig(connectivity=8),
    )

    assert [detection.area_px for detection in detections_4] == [1, 1]
    assert [detection.area_px for detection in detections_8] == [2]


def test_extract_particle_detections_applies_shape_gates():
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:6, 2:6] = True
    mask[1:13, 12] = True
    mask[10, 2:12:2] = True

    detections = extract_particle_detections(
        mask,
        config=ParticleComponentConfig(
            min_area_px=4,
            min_bbox_width_px=3,
            min_bbox_height_px=3,
            max_bbox_aspect_ratio=3.0,
            min_bbox_extent=0.4,
        ),
    )

    assert len(detections) == 1
    assert detections[0].bbox_top == 2
    assert detections[0].bbox_left == 2
    assert detections[0].area_px == 16


def test_connected_components_prefers_accelerated_scipy_labeler(monkeypatch):
    mask = np.array([[True]])
    expected = [(np.array([0]), np.array([0]))]

    def fake_scipy(observed_mask, *, connectivity):
        assert observed_mask is mask
        assert connectivity == 8
        return expected

    def unexpected_labeler(*_args, **_kwargs):
        raise AssertionError("later labelers should not be called")

    monkeypatch.setattr(tracking_module, "_connected_components_with_scipy", fake_scipy)
    monkeypatch.setattr(tracking_module, "_connected_components_with_skimage", unexpected_labeler)
    monkeypatch.setattr(tracking_module, "_connected_components_numpy", unexpected_labeler)

    assert tracking_module._connected_components(mask, connectivity=8) is expected


def test_connected_components_uses_skimage_when_scipy_is_unavailable(monkeypatch):
    mask = np.array([[True]])
    expected = [(np.array([0]), np.array([0]))]

    def missing_scipy(_mask, *, connectivity):
        assert connectivity == 4
        return None

    def fake_skimage(observed_mask, *, connectivity):
        assert observed_mask is mask
        assert connectivity == 4
        return expected

    def unexpected_labeler(*_args, **_kwargs):
        raise AssertionError("NumPy fallback should not be called")

    monkeypatch.setattr(tracking_module, "_connected_components_with_scipy", missing_scipy)
    monkeypatch.setattr(tracking_module, "_connected_components_with_skimage", fake_skimage)
    monkeypatch.setattr(tracking_module, "_connected_components_numpy", unexpected_labeler)

    assert tracking_module._connected_components(mask, connectivity=4) is expected


def test_connected_components_falls_back_to_numpy_when_speed_backends_are_unavailable(monkeypatch):
    def missing_backend(_mask, *, connectivity):
        return None

    monkeypatch.setattr(tracking_module, "_connected_components_with_scipy", missing_backend)
    monkeypatch.setattr(tracking_module, "_connected_components_with_skimage", missing_backend)

    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True

    assert len(tracking_module._connected_components(mask, connectivity=4)) == 2
    assert len(tracking_module._connected_components(mask, connectivity=8)) == 1


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


def test_track_particle_detections_predicts_from_recent_track_velocity():
    detections_by_frame = [
        [
            ParticleDetection(
                frame_index,
                1,
                y=y,
                x=5.0,
                area_px=4,
                bbox_top=int(y),
                bbox_left=4,
                bbox_bottom=int(y) + 2,
                bbox_right=6,
            )
        ]
        for frame_index, y in enumerate([0.0, 3.0, 8.0, 13.5])
    ]

    tracks = track_particle_detections(
        detections_by_frame,
        config=ParticleTrackingConfig(
            max_match_distance_px=2.1,
            velocity_prior_y_px_per_frame=3.0,
        ),
    )

    assert [track.n_detections for track in tracks] == [4]


def test_track_particle_detections_robust_prediction_resists_one_bad_centroid():
    detections_by_frame = [
        [
            ParticleDetection(
                frame_index,
                1,
                y=y,
                x=5.0,
                area_px=4,
                bbox_top=int(y),
                bbox_left=4,
                bbox_bottom=int(y) + 2,
                bbox_right=6,
            )
        ]
        for frame_index, y in enumerate([0.0, 3.0, 44.0, 9.0])
    ]
    detections_by_frame.append(
        [
            ParticleDetection(
                4,
                1,
                y=12.0,
                x=5.0,
                area_px=4,
                bbox_top=12,
                bbox_left=4,
                bbox_bottom=14,
                bbox_right=6,
            ),
            ParticleDetection(
                4,
                2,
                y=15.8,
                x=5.0,
                area_px=4,
                bbox_top=15,
                bbox_left=4,
                bbox_bottom=17,
                bbox_right=6,
            ),
        ]
    )

    tracks = track_particle_detections(
        detections_by_frame,
        config=ParticleTrackingConfig(
            max_match_distance_px=60.0,
            velocity_prior_y_px_per_frame=3.0,
        ),
    )

    assert [detection.y for detection in tracks[0].detections] == [
        0.0,
        3.0,
        44.0,
        9.0,
        12.0,
    ]


def test_track_particle_detections_uses_positional_indices_by_default():
    detections_by_frame = [
        [
            ParticleDetection(
                0,
                1,
                y=10.0,
                x=5.0,
                area_px=4,
                bbox_top=9,
                bbox_left=4,
                bbox_bottom=11,
                bbox_right=6,
            )
        ],
        [
            ParticleDetection(
                0,
                1,
                y=13.0,
                x=5.0,
                area_px=4,
                bbox_top=12,
                bbox_left=4,
                bbox_bottom=14,
                bbox_right=6,
            )
        ],
        [
            ParticleDetection(
                0,
                1,
                y=16.0,
                x=5.0,
                area_px=4,
                bbox_top=15,
                bbox_left=4,
                bbox_bottom=17,
                bbox_right=6,
            )
        ],
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

    assert len(tracks) == 1
    assert [detection.frame_index for detection in tracks[0].detections] == [
        0.0,
        1.0,
        2.0,
    ]
    assert len(velocities) == 1
    np.testing.assert_allclose(velocities[0].velocity_y_px_per_frame, 3.0)


def test_track_particle_detections_advances_empty_frames_by_default():
    detections_by_frame = [
        [
            ParticleDetection(
                0,
                1,
                y=10.0,
                x=5.0,
                area_px=4,
                bbox_top=9,
                bbox_left=4,
                bbox_bottom=11,
                bbox_right=6,
            )
        ],
        [],
        [
            ParticleDetection(
                0,
                1,
                y=10.5,
                x=5.0,
                area_px=4,
                bbox_top=10,
                bbox_left=4,
                bbox_bottom=12,
                bbox_right=6,
            )
        ],
    ]

    tracks = track_particle_detections(
        detections_by_frame,
        config=ParticleTrackingConfig(
            max_match_distance_px=2.0,
            max_frame_gap=1.0,
        ),
    )

    assert [track.n_detections for track in tracks] == [1, 1]
    assert [track.frame_start for track in tracks] == [0.0, 2.0]


def test_track_particle_detections_pyrecest_assignment_maximizes_cardinality():
    detections_by_frame = [
        [
            ParticleDetection(
                0,
                1,
                y=0.0,
                x=0.0,
                area_px=4,
                bbox_top=0,
                bbox_left=0,
                bbox_bottom=2,
                bbox_right=2,
            ),
            ParticleDetection(
                0,
                2,
                y=0.0,
                x=-0.1,
                area_px=4,
                bbox_top=0,
                bbox_left=0,
                bbox_bottom=2,
                bbox_right=2,
            ),
        ],
        [
            ParticleDetection(
                1,
                1,
                y=0.0,
                x=1.0,
                area_px=4,
                bbox_top=0,
                bbox_left=1,
                bbox_bottom=2,
                bbox_right=3,
            ),
            ParticleDetection(
                1,
                2,
                y=0.0,
                x=2.0,
                area_px=4,
                bbox_top=0,
                bbox_left=2,
                bbox_bottom=2,
                bbox_right=4,
            ),
        ],
    ]

    tracks = track_particle_detections(
        detections_by_frame,
        config=ParticleTrackingConfig(max_match_distance_px=2.05),
    )

    assert sorted(track.n_detections for track in tracks) == [2, 2]


def test_track_particle_detections_uses_pyrecest_gnn_assignment(monkeypatch):
    calls = {}

    class FakeGlobalNearestNeighbor:
        def __init__(
            self,
            *,
            initial_prior,
            association_param,
            log_prior_estimates,
            log_posterior_estimates,
        ):
            calls["initial_prior"] = initial_prior
            calls["association_param"] = association_param
            calls["log_prior_estimates"] = log_prior_estimates
            calls["log_posterior_estimates"] = log_posterior_estimates

        def find_association(self, measurements, *_args, **kwargs):
            np.testing.assert_allclose(measurements, np.asarray([[13.0], [5.0]]))
            assert kwargs["warn_on_no_meas_for_track"] is False
            return np.asarray([0], dtype=int)

    monkeypatch.setattr(tracking_module, "GlobalNearestNeighbor", FakeGlobalNearestNeighbor)
    detections_by_frame = [
        [
            ParticleDetection(
                0, 1, y=10.0, x=5.0, area_px=4,
                bbox_top=9, bbox_left=4, bbox_bottom=11, bbox_right=6,
            )
        ],
        [
            ParticleDetection(
                1, 1, y=13.0, x=5.0, area_px=4,
                bbox_top=12, bbox_left=4, bbox_bottom=14, bbox_right=6,
            )
        ],
    ]

    tracks = track_particle_detections(
        detections_by_frame,
        config=ParticleTrackingConfig(
            max_match_distance_px=5.0,
            velocity_prior_y_px_per_frame=3.0,
        ),
    )

    assert [track.n_detections for track in tracks] == [2]
    np.testing.assert_allclose(
        calls["association_param"]["gating_distance_threshold"],
        5.75,
    )
    assert calls["association_param"]["square_dist"] is False
    assert calls["association_param"]["maximize_cardinality"] is True
    assert calls["log_prior_estimates"] is False
    assert calls["log_posterior_estimates"] is False


def test_track_particle_detections_passes_feature_costs_to_pyrecest(monkeypatch):
    calls = {}

    class FakeGlobalNearestNeighbor:
        def __init__(
            self,
            *,
            initial_prior,
            association_param,
            log_prior_estimates,
            log_posterior_estimates,
        ):
            pass

        def find_association(self, measurements, *_args, **kwargs):
            calls["pairwise_cost_matrix"] = kwargs["pairwise_cost_matrix"].copy()
            return np.asarray([1, 0], dtype=int)

    monkeypatch.setattr(tracking_module, "GlobalNearestNeighbor", FakeGlobalNearestNeighbor)
    detections_by_frame = [
        [
            ParticleDetection(
                0,
                1,
                y=10.0,
                x=5.0,
                area_px=4,
                bbox_top=9,
                bbox_left=4,
                bbox_bottom=11,
                bbox_right=6,
                mean_signal=8.0,
            ),
            ParticleDetection(
                0,
                2,
                y=12.0,
                x=5.0,
                area_px=16,
                bbox_top=11,
                bbox_left=4,
                bbox_bottom=15,
                bbox_right=8,
                mean_signal=2.0,
            ),
        ],
        [
            ParticleDetection(
                1,
                1,
                y=13.0,
                x=5.0,
                area_px=16,
                bbox_top=12,
                bbox_left=4,
                bbox_bottom=16,
                bbox_right=8,
                mean_signal=2.0,
            ),
            ParticleDetection(
                1,
                2,
                y=14.0,
                x=5.0,
                area_px=4,
                bbox_top=13,
                bbox_left=4,
                bbox_bottom=15,
                bbox_right=6,
                mean_signal=8.0,
            ),
        ],
    ]

    tracks = track_particle_detections(
        detections_by_frame,
        config=ParticleTrackingConfig(max_match_distance_px=10.0),
    )

    costs = calls["pairwise_cost_matrix"]
    assert costs.shape == (2, 2)
    assert costs[0, 1] < costs[0, 0]
    assert costs[1, 0] < costs[1, 1]
    assert sorted(track.n_detections for track in tracks) == [2, 2]


def test_track_particle_detections_uses_recent_feature_median_for_costs(monkeypatch):
    observed_costs = []

    class FakeGlobalNearestNeighbor:
        def __init__(
            self,
            *,
            initial_prior,
            association_param,
            log_prior_estimates,
            log_posterior_estimates,
        ):
            pass

        def find_association(self, measurements, *_args, **kwargs):
            costs = kwargs["pairwise_cost_matrix"].copy()
            observed_costs.append(costs)
            return np.zeros(costs.shape[0], dtype=int)

    monkeypatch.setattr(tracking_module, "GlobalNearestNeighbor", FakeGlobalNearestNeighbor)
    detections_by_frame = [
        [
            ParticleDetection(
                0,
                1,
                y=10.0,
                x=5.0,
                area_px=4,
                bbox_top=9,
                bbox_left=4,
                bbox_bottom=11,
                bbox_right=6,
                mean_signal=8.0,
            )
        ],
        [
            ParticleDetection(
                1,
                1,
                y=13.0,
                x=5.0,
                area_px=4,
                bbox_top=12,
                bbox_left=4,
                bbox_bottom=14,
                bbox_right=6,
                mean_signal=8.0,
            )
        ],
        [
            ParticleDetection(
                2,
                1,
                y=16.0,
                x=5.0,
                area_px=16,
                bbox_top=15,
                bbox_left=4,
                bbox_bottom=19,
                bbox_right=8,
                mean_signal=2.0,
            )
        ],
        [
            ParticleDetection(
                3,
                1,
                y=19.0,
                x=5.0,
                area_px=4,
                bbox_top=18,
                bbox_left=4,
                bbox_bottom=20,
                bbox_right=6,
                mean_signal=8.0,
            ),
            ParticleDetection(
                3,
                2,
                y=19.0,
                x=6.0,
                area_px=16,
                bbox_top=18,
                bbox_left=5,
                bbox_bottom=22,
                bbox_right=9,
                mean_signal=2.0,
            ),
        ],
    ]

    track_particle_detections(
        detections_by_frame,
        config=ParticleTrackingConfig(
            max_match_distance_px=10.0,
            velocity_prior_y_px_per_frame=3.0,
        ),
    )

    costs = observed_costs[-1]
    assert costs.shape == (1, 2)
    assert costs[0, 0] < costs[0, 1]


def test_track_particle_detections_preserves_spatial_gate_with_feature_costs(monkeypatch):
    calls = {}

    class FakeGlobalNearestNeighbor:
        def __init__(
            self,
            *,
            initial_prior,
            association_param,
            log_prior_estimates,
            log_posterior_estimates,
        ):
            calls["association_param"] = association_param

        def find_association(self, measurements, *_args, **kwargs):
            costs = kwargs["pairwise_cost_matrix"]
            threshold = calls["association_param"]["gating_distance_threshold"]
            valid_total = measurements[0, 0] + costs[0, 0]
            invalid_total = measurements[0, 1] + costs[0, 1]
            assert valid_total <= threshold
            assert invalid_total > threshold
            return np.asarray([0], dtype=int)

    monkeypatch.setattr(tracking_module, "GlobalNearestNeighbor", FakeGlobalNearestNeighbor)
    detections_by_frame = [
        [
            ParticleDetection(
                0,
                1,
                y=0.0,
                x=0.0,
                area_px=4,
                bbox_top=0,
                bbox_left=0,
                bbox_bottom=2,
                bbox_right=2,
                mean_signal=8.0,
            )
        ],
        [
            ParticleDetection(
                1,
                1,
                y=9.9,
                x=0.0,
                area_px=16,
                bbox_top=9,
                bbox_left=0,
                bbox_bottom=13,
                bbox_right=4,
                mean_signal=2.0,
            ),
            ParticleDetection(
                1,
                2,
                y=10.1,
                x=0.0,
                area_px=4,
                bbox_top=10,
                bbox_left=0,
                bbox_bottom=12,
                bbox_right=2,
                mean_signal=8.0,
            ),
        ],
    ]

    tracks = track_particle_detections(
        detections_by_frame,
        config=ParticleTrackingConfig(max_match_distance_px=10.0),
    )

    np.testing.assert_allclose(
        calls["association_param"]["gating_distance_threshold"],
        11.5,
    )
    assert sorted(track.n_detections for track in tracks) == [1, 2]


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


def test_score_particle_velocities_applies_length_ratio_and_lateral_gates():
    velocities = [
        ParticleVelocity(
            track_id=0,
            n_detections=6,
            frame_start=0,
            frame_end=5,
            velocity_y_px_per_frame=4.0,
            velocity_x_px_per_frame=0.5,
            speed_px_per_frame=4.03,
            belt_velocity_y_px_per_frame=5.0,
            velocity_ratio_y=0.8,
            belt_minus_particle_velocity_y_px_per_frame=1.0,
        ),
        ParticleVelocity(
            track_id=1,
            n_detections=3,
            frame_start=0,
            frame_end=2,
            velocity_y_px_per_frame=6.0,
            velocity_x_px_per_frame=0.5,
            speed_px_per_frame=6.02,
            belt_velocity_y_px_per_frame=5.0,
            velocity_ratio_y=1.2,
            belt_minus_particle_velocity_y_px_per_frame=-1.0,
        ),
        ParticleVelocity(
            track_id=2,
            n_detections=8,
            frame_start=0,
            frame_end=7,
            velocity_y_px_per_frame=3.0,
            velocity_x_px_per_frame=3.0,
            speed_px_per_frame=4.24,
            belt_velocity_y_px_per_frame=5.0,
            velocity_ratio_y=0.6,
            belt_minus_particle_velocity_y_px_per_frame=2.0,
        ),
    ]

    config = TrackFilterConfig(
        min_track_length=5,
        min_velocity_ratio_y=0.0,
        max_velocity_ratio_y=1.1,
        max_abs_x_velocity_px_per_frame=2.0,
    )
    scores = score_particle_velocities(velocities, config=config)
    filtered = filter_particle_velocities(velocities, config=config)

    assert [score.accepted for score in scores] == [True, False, False]
    assert scores[1].passes_min_track_length is False
    assert scores[1].passes_velocity_ratio is False
    assert scores[2].passes_lateral_velocity is False
    assert [velocity.track_id for velocity in filtered] == [0]


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

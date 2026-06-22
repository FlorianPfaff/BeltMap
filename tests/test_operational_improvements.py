from pathlib import Path

import numpy as np
import pytest

from beltmap.operational_improvements import (
    StreamingFrameState,
    apply_ignore_mask,
    belt_edge_ignore_mask,
    classify_event,
    classify_failure_modes,
    dataset_manifest,
    discover_new_stream_frames,
    empirical_p_values,
    estimate_centroid_uncertainty,
    estimate_homography,
    estimate_period_from_profile,
    fdr_threshold_from_p_values,
    incremental_update_map,
    particle_descriptor_from_mask,
    particle_density_score,
    randomize_synthetic_frame,
    recommend_threshold,
    robust_velocity_fit,
    select_adaptive_map_frames,
    split_merged_components,
    stitch_multicamera_events,
    suggest_belt_region_from_frames,
    summarize_flux,
    warp_perspective,
)


def test_suggest_belt_region_from_motion_energy_finds_moving_crop():
    frames = []
    for shift in range(4):
        frame = np.zeros((40, 50), dtype=float)
        frame[10 + shift : 25 + shift, 15:35] = 100.0
        frames.append(frame)

    suggestion = suggest_belt_region_from_frames(frames, percentile=75, margin_px=2)

    assert suggestion.top <= 10
    assert suggestion.left <= 15
    assert suggestion.height >= 15
    assert suggestion.width >= 20
    assert suggestion.moving_pixel_fraction > 0


def test_homography_warp_identity_preserves_image():
    image = np.arange(16, dtype=float).reshape(4, 4)
    model = estimate_homography(
        [(0, 0), (3, 0), (3, 3), (0, 3)],
        [(0, 0), (3, 0), (3, 3), (0, 3)],
    )

    warped = warp_perspective(image, model, image.shape, interpolation="nearest")

    np.testing.assert_allclose(warped, image)


def test_homography_rejects_nonfinite_points():
    with pytest.raises(ValueError, match="finite"):
        estimate_homography(
            [(0, 0), (3, 0), (float("nan"), 3), (0, 3)],
            [(0, 0), (3, 0), (3, 3), (0, 3)],
        )


def test_homography_warp_negative_projective_scale_preserves_image():
    image = np.arange(16, dtype=float).reshape(4, 4)

    warped = warp_perspective(image, -np.eye(3), image.shape, interpolation="nearest")

    np.testing.assert_allclose(warped, image)


def test_homography_warp_rejects_nonfinite_matrix():
    image = np.arange(16, dtype=float).reshape(4, 4)
    matrix = np.eye(3)
    matrix[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        warp_perspective(image, matrix, image.shape)


def test_homography_warp_rejects_fractional_output_shape():
    image = np.arange(16, dtype=float).reshape(4, 4)

    with pytest.raises(ValueError, match="output_shape height"):
        warp_perspective(image, np.eye(3), (4.5, 4))


def test_period_estimator_recovers_repeated_profile():
    base = np.array([0.0, 1.0, 0.0, -1.0, 0.5, -0.5])
    profile = np.tile(base, 10)

    estimate = estimate_period_from_profile(profile, min_period_px=4, max_period_px=12)

    assert estimate.period_px == 6
    assert estimate.score > 0.9


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_period_px": 4.5}, "min_period_px"),
        ({"min_period_px": 4, "max_period_px": 12.5}, "max_period_px"),
        ({"min_period_px": 4, "max_period_px": 12, "top_k": 2.5}, "top_k"),
    ],
)
def test_period_estimator_rejects_fractional_integer_config(kwargs, message):
    profile = np.tile(np.array([0.0, 1.0, 0.0, -1.0]), 10)

    with pytest.raises(ValueError, match=message):
        estimate_period_from_profile(profile, **kwargs)


def test_adaptive_sampler_spreads_phase_bins():
    samples = select_adaptive_map_frames(
        [0, 0, 10, 20, 30, 40, 50],
        map_height_px=60,
        sample_count=4,
        crop_height_px=5,
    )

    assert len(samples) == 4
    assert len({sample.bin_index for sample in samples}) >= 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"map_height_px": 60.5, "sample_count": 2}, "map_height_px"),
        ({"map_height_px": 60, "sample_count": 2.5}, "sample_count"),
        (
            {"map_height_px": 60, "sample_count": 2, "crop_height_px": 1.5},
            "crop_height_px",
        ),
        ({"map_height_px": 60, "sample_count": 2, "bin_count": 3.5}, "bin_count"),
    ],
)
def test_adaptive_sampler_rejects_fractional_integer_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        select_adaptive_map_frames([0, 10], **kwargs)


def test_adaptive_sampler_rejects_nonfinite_phases():
    with pytest.raises(ValueError, match="phases_px"):
        select_adaptive_map_frames([0, float("nan")], map_height_px=60, sample_count=2)


def test_ignore_masks_and_edge_margins():
    valid = np.ones((5, 5), dtype=bool)
    ignore = belt_edge_ignore_mask((5, 5), top_px=1, right_px=1)
    filtered = apply_ignore_mask(valid, ignore)

    assert not filtered[0, 2]
    assert not filtered[3, 4]
    assert filtered[2, 2]


def test_threshold_and_fdr_helpers():
    residual = np.array([-1.0, 0.0, 1.0, 2.0, 8.0, 9.0])
    threshold = recommend_threshold(
        residual, expected_false_pixels_per_frame=1, polarity="bright"
    )
    p_values = empirical_p_values(residual, polarity="bright")
    fdr = fdr_threshold_from_p_values(p_values, residual, alpha=0.5)

    assert threshold >= 8.0
    assert fdr is None or fdr >= 0.0


def test_particle_density_descriptor_uncertainty_and_split():
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:5, 1:3] = True
    mask[2:5, 6:8] = True
    signal = np.zeros_like(mask, dtype=float)
    signal[mask] = 5.0

    pieces = split_merged_components(mask, max_area_px=4, min_gap_px=2)
    descriptor = particle_descriptor_from_mask(pieces[0], signal=signal)
    uncertainty = estimate_centroid_uncertainty(
        pieces[0], signal=signal, local_noise=1.0
    )
    density = particle_density_score(signal, threshold=4.0)

    assert len(pieces) == 2
    assert descriptor.area_px == 6
    assert descriptor.equivalent_diameter_px > 0
    assert uncertainty.centroid_y_std_px is not None
    assert density > 0


def test_robust_velocity_and_flux_summary():
    fit = robust_velocity_fit([0, 1, 2, 3], [0, 2, 4, 100])
    summary = summarize_flux(
        [
            {"velocity_ratio_y": "0.5", "velocity_y_px_per_frame": "2.0"},
            {"velocity_ratio_y": "0.7", "velocity_y_px_per_frame": "3.0"},
        ],
        frame_count=100,
        frame_rate_hz=50.0,
    )

    assert fit.slope_px_per_time == 2.0
    assert summary.particle_flux_per_s == 1.0
    assert summary.median_velocity_ratio_y == 0.6


def test_classify_failure_modes_preserves_zero_summary_metrics():
    warnings = classify_failure_modes(
        {
            "registration_score_median": 0.0,
            "velocity_ratio_share_0_to_1": 0.0,
        }
    )

    codes = {warning["code"] for warning in warnings}
    assert "low-registration-score" in codes
    assert "implausible-velocity-ratios" in codes


def test_classify_failure_modes_ignores_invalid_summary_values():
    warnings = classify_failure_modes(
        {
            "phase_boundary_fraction": "bad",
            "small_component_share_area_le_8": True,
            "map_low_coverage_fraction": False,
            "track_fragmentation": "nan",
        }
    )

    assert warnings == []


def test_stream_manifest_map_update_and_multicamera(tmp_path: Path):
    for index in range(2):
        path = tmp_path / f"frame_{index:03d}.png"
        from PIL import Image

        Image.fromarray(np.full((4, 4), index, dtype=np.uint8)).save(path)
    manifest = dataset_manifest(tmp_path)
    state = StreamingFrameState()
    new_paths = discover_new_stream_frames(tmp_path, state)
    updated = incremental_update_map(
        np.zeros((2, 2)),
        np.ones((2, 2)) * 10,
        np.array([[True, False], [False, True]]),
        learning_rate=0.5,
    )
    events = stitch_multicamera_events(
        {
            "a": [{"time_s": 1.0, "phase_px": 10.0}],
            "b": [{"time_s": 1.01, "phase_px": 12.0}],
        },
        time_tolerance_s=0.05,
        phase_tolerance_px=5.0,
    )

    assert len(manifest.files) == 2
    assert len(new_paths) == 2
    assert updated[0, 0] == 5.0
    assert updated[0, 1] == 0.0
    assert len(events) == 1
    assert len(events[0].camera_rows) == 2


def test_event_classification_and_domain_randomization():
    event = classify_event(
        recurrent_overlap_fraction=0.8, velocity_ratio_y=0.2, peak_signal=6.0
    )
    randomized = randomize_synthetic_frame(
        np.ones((4, 4)) * 100,
        config=__import__(
            "beltmap.operational_improvements",
            fromlist=["SyntheticRandomizationConfig"],
        ).SyntheticRandomizationConfig(scratch_count=1),
        rng=np.random.default_rng(1),
    )

    assert event.label in {"belt-fixed-artifact", "loose-particle"}
    assert randomized.shape == (4, 4)


def test_failure_mode_classifier_preserves_zero_quality_metrics():
    warnings = classify_failure_modes(
        {
            "registration_score_median": 0.0,
            "velocity_ratio_share_0_to_1": 0.0,
        }
    )

    codes = {warning["code"] for warning in warnings}
    assert "low-registration-score" in codes
    assert "implausible-velocity-ratios" in codes

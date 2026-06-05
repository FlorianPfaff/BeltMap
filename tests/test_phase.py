import numpy as np
import pytest

from beltmap import (
    BeltMotionModel,
    PhaseEstimate,
    PhaseRegistrationConfig,
    PhaseTrajectorySmoothingConfig,
    estimate_phase,
    render_belt_view,
    smooth_phase_estimates,
)
from beltmap.phase import (
    _box_blur,
    _prepare_for_registration,
    _refine_quadratic_offset,
    _registration_loss_diagnostics,
    _uniform_filter_axis,
)


def make_belt_map(period=96, width=32):
    rng = np.random.default_rng(4)
    y = np.arange(period)[:, None]
    x = np.arange(width)[None, :]
    belt = (
        30
        + 2.0 * np.sin(2 * np.pi * y / 17)
        + 1.5 * np.cos(2 * np.pi * x / 9)
        + rng.normal(0, 0.5, size=(period, width))
    )
    belt[18:22, 5:25] += 8
    belt[50:56, 12:18] -= 6
    belt[75:78, :] += 5
    return belt


def add_synthetic_particles(frame):
    corrupted = frame.copy()
    corrupted[8:14, 7:13] += 30
    corrupted[35:42, 20:28] += 25
    return corrupted


def reference_uniform_filter_axis(image, *, radius, axis):
    arr = np.asarray(image, dtype=np.float64)
    out = np.empty_like(arr)
    window = 2 * radius + 1
    for index in np.ndindex(arr.shape):
        total = 0.0
        for offset in range(-radius, radius + 1):
            source = list(index)
            source[axis] = min(max(index[axis] + offset, 0), arr.shape[axis] - 1)
            total += arr[tuple(source)]
        out[index] = total / window
    return out


def reference_box_blur(image, *, radius):
    vertical = reference_uniform_filter_axis(image, radius=radius, axis=0)
    return reference_uniform_filter_axis(vertical, radius=radius, axis=1)


def test_motion_model_wraps_signed_image_velocity():
    model = BeltMotionModel(
        image_velocity_px_per_frame=2.5,
        period_px=100,
        reference_frame=10,
        reference_phase_px=20,
    )

    assert model.phase_at(10) == 20
    assert model.phase_at(12) == 15
    assert model.phase_at(0) == 45


def test_render_belt_view_uses_fractional_phase():
    belt = np.arange(20, dtype=float)[:, None] * np.ones((1, 3))
    rendered = render_belt_view(belt, phase_px=0.5, height=3)

    np.testing.assert_allclose(rendered[:, 0], [0.5, 1.5, 2.5])


def test_render_belt_view_defaults_to_periodic_wrapping():
    belt = np.arange(5, dtype=float)[:, None] * np.ones((1, 2))

    rendered = render_belt_view(belt, phase_px=-1.0, height=3)

    np.testing.assert_allclose(rendered[:, 0], [4.0, 0.0, 1.0])


def test_render_belt_view_can_mark_nonperiodic_out_of_support_rows():
    belt = np.arange(5, dtype=float)[:, None] * np.ones((1, 2))

    before_start = render_belt_view(belt, phase_px=-1.0, height=3, periodic=False)
    after_end = render_belt_view(belt, phase_px=3.5, height=3, periodic=False)

    assert np.isnan(before_start[0, 0])
    np.testing.assert_allclose(before_start[1:, 0], [0.0, 1.0])
    assert after_end[0, 0] == pytest.approx(3.5)
    assert np.isnan(after_end[1, 0])
    assert np.isnan(after_end[2, 0])


def test_quadratic_refinement_returns_consistent_loss_offset_pair():
    losses = [(1.25, -1.0), (1.0, 0.0), (1.25, 1.0)]

    refined_loss, refined_offset = _refine_quadratic_offset(losses, best_index=1)

    assert refined_offset == pytest.approx(0.0)
    assert refined_loss == pytest.approx(1.0)


def test_registration_config_rejects_negative_search_radius():
    with pytest.raises(ValueError, match="search_radius_px must be non-negative"):
        PhaseRegistrationConfig(search_radius_px=-1).candidate_offsets()


def test_registration_loss_diagnostics_reports_ambiguity_and_curvature():
    diagnostics = _registration_loss_diagnostics(
        [(4.0, -1.0), (1.0, 0.0), (2.0, 1.0)],
        best_index=1,
        best_loss=1.0,
    )

    assert diagnostics["second_best_loss"] == pytest.approx(2.0)
    assert diagnostics["loss_gap"] == pytest.approx(1.0)
    assert diagnostics["loss_gap_ratio"] == pytest.approx(1.0)
    assert diagnostics["loss_curvature"] == pytest.approx(4.0)
    assert diagnostics["uncertainty_px"] == pytest.approx(0.5)


def test_registration_candidate_offsets_are_symmetric_for_non_divisible_step():
    offsets = PhaseRegistrationConfig(
        search_radius_px=1.0,
        search_step_px=0.6,
    ).candidate_offsets()

    np.testing.assert_allclose(offsets, [-1.0, -0.6, 0.0, 0.6, 1.0])


def test_uniform_filter_axis_matches_edge_padded_reference():
    image = np.array(
        [
            [1.0, 2.0, 4.0, 8.0],
            [16.0, 32.0, 64.0, 128.0],
            [3.0, 9.0, 27.0, 81.0],
        ]
    )

    for radius in (1, 2):
        for axis in (0, 1):
            np.testing.assert_allclose(
                _uniform_filter_axis(image, radius=radius, axis=axis),
                reference_uniform_filter_axis(image, radius=radius, axis=axis),
            )


def test_box_blur_matches_separable_edge_padded_reference():
    image = np.array(
        [
            [0.0, 2.0, 6.0, 12.0],
            [10.0, 14.0, 20.0, 28.0],
            [30.0, 38.0, 48.0, 60.0],
        ]
    )

    for radius in (1, 2):
        np.testing.assert_allclose(
            _box_blur(image, radius=radius),
            reference_box_blur(image, radius=radius),
        )


def test_prepare_for_registration_uses_edge_padded_highpass():
    image = np.array(
        [
            [2.0, 4.0, 8.0, 16.0],
            [3.0, 9.0, 27.0, 81.0],
            [5.0, 25.0, 125.0, 625.0],
        ]
    )
    highpass = image - reference_box_blur(image, radius=1)
    expected = highpass / np.std(highpass)

    np.testing.assert_allclose(
        _prepare_for_registration(image, highpass_radius_px=1),
        expected,
    )


def test_registration_refines_phase_with_particle_outliers():
    belt = make_belt_map()
    true_model = BeltMotionModel(
        image_velocity_px_per_frame=2.5,
        period_px=belt.shape[0],
        reference_phase_px=9.0,
    )
    frame_index = 11
    true_phase = true_model.phase_at(frame_index)
    frame = add_synthetic_particles(render_belt_view(belt, true_phase, height=48))

    biased_model = BeltMotionModel(
        image_velocity_px_per_frame=2.5,
        period_px=belt.shape[0],
        reference_phase_px=12.0,
    )
    estimate = estimate_phase(
        frame_index,
        biased_model,
        frame=frame,
        belt_map=belt,
        config=PhaseRegistrationConfig(
            search_radius_px=5,
            search_step_px=0.25,
            trim_fraction=0.12,
            highpass_radius_px=5,
        ),
    )

    circular_error = min(
        abs(estimate.phase_px - true_phase),
        belt.shape[0] - abs(estimate.phase_px - true_phase),
    )
    assert circular_error <= 0.25
    assert estimate.method == "registration"
    assert estimate.correction_px < 0
    assert estimate.second_best_loss is not None
    assert estimate.loss_gap is not None and estimate.loss_gap >= 0
    assert estimate.loss_gap_ratio is not None and estimate.loss_gap_ratio >= 0
    assert estimate.loss_curvature is not None and estimate.loss_curvature > 0
    assert estimate.uncertainty_px is not None and estimate.uncertainty_px > 0


def test_smooth_phase_estimates_rejects_registration_outlier():
    estimates = []
    for frame_index in range(9):
        correction = 0.2 * frame_index
        if frame_index == 4:
            correction = 7.5
        estimates.append(
            PhaseEstimate(
                phase_px=correction,
                frame_index=float(frame_index),
                predicted_phase_px=0.0,
                correction_px=correction,
                score=1.0,
                method="registration",
            )
        )

    smoothed = smooth_phase_estimates(
        estimates,
        config=PhaseTrajectorySmoothingConfig(
            window_radius_frames=3,
            min_support=3,
            robust_sigma=2.0,
            max_abs_correction_px=10.0,
        ),
    )

    np.testing.assert_allclose(
        [estimate.correction_px for estimate in smoothed],
        [0.2 * frame_index for frame_index in range(9)],
        atol=1e-9,
    )
    assert smoothed[4].method == "registration_smoothed"


def test_smooth_phase_estimates_uses_cyclic_corrections():
    estimates = [
        PhaseEstimate(
            phase_px=98.0,
            frame_index=0.0,
            predicted_phase_px=0.0,
            correction_px=98.0,
            score=1.0,
            method="registration",
        ),
        PhaseEstimate(
            phase_px=98.5,
            frame_index=1.0,
            predicted_phase_px=0.5,
            correction_px=98.0,
            score=1.0,
            method="registration",
        ),
    ]

    smoothed = smooth_phase_estimates(
        estimates,
        period_px=100.0,
        config=PhaseTrajectorySmoothingConfig(window_radius_frames=1, min_support=1),
    )

    np.testing.assert_allclose(
        [estimate.correction_px for estimate in smoothed],
        [-2.0, -2.0],
    )
    np.testing.assert_allclose(
        [estimate.phase_px for estimate in smoothed],
        [98.0, 98.5],
    )

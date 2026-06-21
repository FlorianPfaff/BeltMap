import numpy as np
import pytest

from beltmap.phase import (
    PhaseDriftConfig,
    PhaseDriftFilter,
    PhaseEstimate,
    PhaseRegistrationConfig,
    PhaseTrajectorySmoothingConfig,
    refine_phase_by_registration,
    render_belt_view,
    smooth_phase_estimates,
)


def test_subpixel_phase_registration_is_not_quantized_to_search_step():
    rows = np.arange(128, dtype=np.float64)[:, None]
    cols = np.arange(5, dtype=np.float64)[None, :]
    belt_map = (
        np.sin(2.0 * np.pi * rows / 17.0)
        + 0.35 * np.cos(2.0 * np.pi * rows / 7.0)
        + 0.03 * cols
    )
    true_phase = 23.37
    predicted_phase = 22.80
    frame = render_belt_view(belt_map, true_phase, height=64)

    config = PhaseRegistrationConfig(
        search_radius_px=1.0,
        search_step_px=0.25,
        trim_fraction=0.0,
        highpass_radius_px=0,
        subpixel_refinement=True,
        robust_normalization=True,
    )
    estimate = refine_phase_by_registration(
        frame=frame,
        belt_map=belt_map,
        predicted_phase_px=predicted_phase,
        period_px=belt_map.shape[0],
        config=config,
    )

    expected_correction = true_phase - predicted_phase
    assert abs(estimate.correction_px - expected_correction) < 0.08
    grid_units = estimate.correction_px / config.search_step_px
    assert not np.isclose(grid_units, round(grid_units))


def test_phase_registration_mask_ignores_masked_nan_pixels():
    rows = np.arange(128, dtype=np.float64)[:, None]
    cols = np.arange(7, dtype=np.float64)[None, :]
    belt_map = (
        np.sin(2.0 * np.pi * rows / 19.0)
        + 0.25 * np.cos(2.0 * np.pi * rows / 11.0)
        + 0.02 * cols
    )
    true_phase = 41.6
    predicted_phase = 41.0
    frame = render_belt_view(belt_map, true_phase, height=72)
    mask = np.ones(frame.shape, dtype=bool)
    mask[:8, :] = False
    mask[-8:, :] = False
    frame_with_masked_nans = frame.copy()
    frame_with_masked_nans[~mask] = np.nan

    config = PhaseRegistrationConfig(
        search_radius_px=1.0,
        search_step_px=0.25,
        trim_fraction=0.0,
        highpass_radius_px=3,
        subpixel_refinement=True,
        robust_normalization=True,
    )
    estimate = refine_phase_by_registration(
        frame=frame_with_masked_nans,
        belt_map=belt_map,
        predicted_phase_px=predicted_phase,
        period_px=belt_map.shape[0],
        config=config,
        mask=mask,
    )

    assert np.isfinite(estimate.loss)
    assert abs(estimate.correction_px - (true_phase - predicted_phase)) < 0.15


def test_phase_drift_filter_smooths_accepted_registration_residuals():
    drift_filter = PhaseDriftFilter(
        PhaseDriftConfig(enabled=True, smoothing_alpha=0.5, min_score=0.1),
        period_px=100.0,
    )
    assert drift_filter.predict(10.0) == 10.0

    estimate = PhaseEstimate(
        phase_px=13.0,
        frame_index=0.0,
        predicted_phase_px=10.0,
        correction_px=3.0,
        score=0.5,
        method="registration",
    )
    returned = drift_filter.observe(estimate)

    assert returned.drift_px == 0.0
    assert returned.method == "registration+drift"
    assert drift_filter.drift_px == 1.5
    assert drift_filter.accepted_updates == 1
    assert drift_filter.rejected_updates == 0
    assert drift_filter.predict(10.0) == 11.5


@pytest.mark.parametrize(
    ("config", "kwargs", "message"),
    [
        (PhaseDriftConfig(enabled="true"), {}, "enabled"),
        (PhaseDriftConfig(smoothing_alpha=True), {}, "smoothing_alpha"),
        (PhaseDriftConfig(smoothing_alpha="0.5"), {}, "smoothing_alpha"),
        (PhaseDriftConfig(min_score=True), {}, "min_score"),
        (PhaseDriftConfig(max_abs_residual_correction_px=True), {}, "max_abs_residual_correction_px"),
        (PhaseDriftConfig(max_abs_drift_px=True), {}, "max_abs_drift_px"),
        (PhaseDriftConfig(), {"initial_drift_px": True}, "initial_drift_px"),
        (PhaseDriftConfig(), {"period_px": True}, "period_px"),
    ],
)
def test_phase_drift_filter_rejects_coerced_numeric_config(config, kwargs, message):
    with pytest.raises(ValueError, match=message):
        PhaseDriftFilter(config, **kwargs)


def test_phase_drift_filter_rejects_nonfinite_registration_score():
    drift_filter = PhaseDriftFilter(
        PhaseDriftConfig(enabled=True, smoothing_alpha=0.5, min_score=0.1),
        period_px=100.0,
    )
    estimate = PhaseEstimate(
        phase_px=13.0,
        frame_index=0.0,
        predicted_phase_px=10.0,
        correction_px=3.0,
        score=float("nan"),
        method="registration",
    )

    returned = drift_filter.observe(estimate)

    assert returned.drift_px == 0.0
    assert drift_filter.drift_px == 0.0
    assert drift_filter.accepted_updates == 0
    assert drift_filter.rejected_updates == 1


def test_smoothed_phase_estimates_preserve_applied_drift_metadata():
    estimates = [
        PhaseEstimate(
            phase_px=11.0,
            frame_index=0.0,
            predicted_phase_px=10.0,
            correction_px=1.0,
            score=1.0,
            method="registration+drift",
            drift_px=2.5,
        ),
        PhaseEstimate(
            phase_px=12.0,
            frame_index=1.0,
            predicted_phase_px=10.0,
            correction_px=2.0,
            score=1.0,
            method="registration+drift",
            drift_px=3.0,
        ),
    ]

    smoothed = smooth_phase_estimates(
        estimates,
        config=PhaseTrajectorySmoothingConfig(window_radius_frames=1, min_support=1),
    )

    assert [estimate.drift_px for estimate in smoothed] == [2.5, 3.0]

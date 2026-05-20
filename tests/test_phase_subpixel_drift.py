import numpy as np

from beltmap.phase import (
    PhaseDriftConfig,
    PhaseDriftFilter,
    PhaseEstimate,
    PhaseRegistrationConfig,
    refine_phase_by_registration,
    render_belt_view,
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

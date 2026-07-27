import numpy as np

from beltmap import (
    PhaseEstimate,
    PhaseTrajectorySmoothingConfig,
    smooth_phase_estimates,
)
from beltmap import phase


def test_phase_smoothing_preserves_linear_trend_across_cyclic_branch_cut():
    estimates = [
        PhaseEstimate(
            phase_px=phase_px,
            frame_index=float(frame_index),
            predicted_phase_px=0.0,
            correction_px=correction_px,
            score=1.0,
            method="registration",
        )
        for frame_index, (phase_px, correction_px) in enumerate(
            [(48.0, 48.0), (49.0, 49.0), (50.0, -50.0), (51.0, -49.0)]
        )
    ]

    smoothed = smooth_phase_estimates(
        estimates,
        period_px=100.0,
        config=PhaseTrajectorySmoothingConfig(
            window_radius_frames=2,
            min_support=2,
        ),
    )

    assert phase.smooth_phase_estimates is smooth_phase_estimates
    np.testing.assert_allclose(
        [estimate.correction_px for estimate in smoothed],
        [48.0, 49.0, -50.0, -49.0],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        [estimate.phase_px for estimate in smoothed],
        [48.0, 49.0, 50.0, 51.0],
        atol=1e-9,
    )
    assert all(estimate.method == "registration_smoothed" for estimate in smoothed)

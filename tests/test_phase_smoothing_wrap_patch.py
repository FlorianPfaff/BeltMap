from __future__ import annotations

import numpy as np

import beltmap
from beltmap import phase


def test_wrap_aware_phase_smoothing_patch_is_autoloaded() -> None:
    assert beltmap.smooth_phase_estimates is phase.smooth_phase_estimates
    assert getattr(
        phase.smooth_phase_estimates,
        "_beltmap_wrap_aware_phase_smoothing_patched",
        False,
    )


def test_periodic_phase_smoothing_unwraps_half_period_boundary() -> None:
    estimates = [
        phase.PhaseEstimate(
            phase_px=phase_px,
            frame_index=float(frame_index),
            predicted_phase_px=0.0,
            correction_px=correction_px,
            score=1.0,
            method="registration",
        )
        for frame_index, (phase_px, correction_px) in enumerate(
            [(48.0, 48.0), (49.0, 49.0), (50.0, -50.0)]
        )
    ]

    smoothed = phase.smooth_phase_estimates(
        estimates,
        period_px=100.0,
        config=phase.PhaseTrajectorySmoothingConfig(
            window_radius_frames=2,
            min_support=3,
        ),
    )

    np.testing.assert_allclose(
        [estimate.correction_px for estimate in smoothed],
        [48.0, 49.0, -50.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [estimate.phase_px for estimate in smoothed],
        [48.0, 49.0, 50.0],
        atol=1e-12,
    )

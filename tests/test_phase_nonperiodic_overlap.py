import numpy as np
import pytest

from beltmap import PhaseRegistrationConfig, refine_phase_by_registration, render_belt_view


def test_nonperiodic_phase_registration_skips_no_overlap_offsets():
    rows = np.arange(12, dtype=np.float64)[:, None]
    cols = np.arange(5, dtype=np.float64)[None, :]
    belt = np.sin(rows) + 0.1 * cols + 0.03 * rows * cols
    frame = render_belt_view(belt, phase_px=1.0, height=5, periodic=False)

    estimate = refine_phase_by_registration(
        frame=frame,
        belt_map=belt,
        predicted_phase_px=-5.0,
        period_px=None,
        config=PhaseRegistrationConfig(
            search_radius_px=6.0,
            search_step_px=1.0,
            trim_fraction=0.0,
            highpass_radius_px=0,
        ),
    )

    assert estimate.correction_px == pytest.approx(6.0)
    assert estimate.loss == pytest.approx(0.0)


def test_nonperiodic_phase_registration_reports_all_no_overlap_search():
    belt = np.arange(30, dtype=float).reshape(10, 3)
    frame = render_belt_view(belt, phase_px=1.0, height=3, periodic=False)

    with pytest.raises(ValueError, match="no valid overlap"):
        refine_phase_by_registration(
            frame=frame,
            belt_map=belt,
            predicted_phase_px=20.0,
            period_px=None,
            config=PhaseRegistrationConfig(
                search_radius_px=1.0,
                search_step_px=1.0,
                trim_fraction=0.0,
                highpass_radius_px=0,
            ),
        )

import numpy as np

from beltmap import PhaseEstimate, render_clean_belt_residual


def test_render_clean_belt_residual_marks_finite_strip_out_of_support_invalid():
    belt = np.arange(5, dtype=float)[:, None] * np.ones((1, 3))
    image = np.zeros((3, 3), dtype=float)
    phase = PhaseEstimate(
        phase_px=-1.0,
        frame_index=0.0,
        predicted_phase_px=-1.0,
    )

    residual = render_clean_belt_residual(
        image=image,
        belt_map=belt,
        frame_index=0.0,
        phase_estimate=phase,
    )

    assert not residual.mask[0].any()
    assert residual.mask[1:].all()
    assert np.isnan(residual.raw[0, 0])
    assert residual.raw[1, 0] == 0.0
    assert residual.raw[2, 0] == -1.0


def test_render_clean_belt_residual_allows_explicit_cyclic_rendering():
    belt = np.arange(5, dtype=float)[:, None] * np.ones((1, 3))
    image = np.zeros((3, 3), dtype=float)
    phase = PhaseEstimate(
        phase_px=-1.0,
        frame_index=0.0,
        predicted_phase_px=-1.0,
    )

    residual = render_clean_belt_residual(
        image=image,
        belt_map=belt,
        frame_index=0.0,
        phase_estimate=phase,
        periodic=True,
    )

    assert residual.mask.all()
    np.testing.assert_allclose(residual.raw[:, 0], [-4.0, 0.0, -1.0])

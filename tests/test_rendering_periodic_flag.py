import numpy as np
import pytest

from beltmap import BeltMotionModel, PhaseEstimate, render_expected_clean_belt


def test_render_expected_clean_belt_can_render_finite_strip_nonperiodically():
    belt = np.arange(5, dtype=float)[:, None] * np.ones((1, 3))
    phase = PhaseEstimate(phase_px=-1.0, frame_index=0.0, predicted_phase_px=-1.0)

    render = render_expected_clean_belt(
        belt_map=belt,
        frame_index=0.0,
        phase_estimate=phase,
        output_shape=(3, 3),
        periodic=False,
    )

    assert not render.mask[0].any()
    assert np.isnan(render.image[0, 0])
    np.testing.assert_array_equal(render.mask[1:], True)
    np.testing.assert_allclose(render.image[1:, 0], [0.0, 1.0])


def test_render_expected_clean_belt_infers_nonperiodic_from_motion_model_without_period():
    belt = np.arange(5, dtype=float)[:, None] * np.ones((1, 2))
    model = BeltMotionModel(
        image_velocity_px_per_frame=0.0,
        period_px=None,
        reference_phase_px=3.5,
    )

    render = render_expected_clean_belt(
        belt_map=belt,
        frame_index=0.0,
        motion_model=model,
        output_shape=(3, 2),
    )

    assert render.mask[0].all()
    assert not render.mask[1].any()
    assert not render.mask[2].any()
    assert render.image[0, 0] == 3.5
    assert np.isnan(render.image[1, 0])


def test_render_expected_clean_belt_preserves_cyclic_default_without_motion_model():
    belt = np.arange(5, dtype=float)[:, None] * np.ones((1, 2))
    phase = PhaseEstimate(phase_px=-1.0, frame_index=0.0, predicted_phase_px=-1.0)

    render = render_expected_clean_belt(
        belt_map=belt,
        frame_index=0.0,
        phase_estimate=phase,
        output_shape=(3, 2),
    )

    np.testing.assert_array_equal(render.mask, True)
    np.testing.assert_allclose(render.image[:, 0], [4.0, 0.0, 1.0])


def test_render_expected_clean_belt_rejects_string_periodic_flag():
    belt = np.arange(5, dtype=float)[:, None] * np.ones((1, 2))
    phase = PhaseEstimate(phase_px=-1.0, frame_index=0.0, predicted_phase_px=-1.0)

    with pytest.raises(ValueError, match="periodic"):
        render_expected_clean_belt(
            belt_map=belt,
            frame_index=0.0,
            phase_estimate=phase,
            output_shape=(3, 2),
            periodic="false",
        )

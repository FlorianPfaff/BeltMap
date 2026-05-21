import numpy as np

from beltmap import BeltMotionModel, render_clean_belt_residual


def test_render_clean_belt_residual_marks_finite_strip_out_of_support_invalid():
    belt = np.arange(5, dtype=float)[:, None] * np.ones((1, 3))
    image = np.zeros((3, 3), dtype=float)
    model = BeltMotionModel(
        image_velocity_px_per_frame=0.0,
        period_px=None,
        reference_phase_px=-1.0,
    )

    residual = render_clean_belt_residual(
        image=image,
        belt_map=belt,
        frame_index=0.0,
        motion_model=model,
    )

    assert not residual.mask[0].any()
    assert residual.mask[1:].all()
    assert np.isnan(residual.raw[0, 0])
    assert residual.raw[1, 0] == 0.0
    assert residual.raw[2, 0] == -1.0


def test_render_clean_belt_residual_allows_explicit_cyclic_rendering():
    belt = np.arange(5, dtype=float)[:, None] * np.ones((1, 3))
    image = np.zeros((3, 3), dtype=float)
    model = BeltMotionModel(
        image_velocity_px_per_frame=0.0,
        period_px=None,
        reference_phase_px=-1.0,
    )

    residual = render_clean_belt_residual(
        image=image,
        belt_map=belt,
        frame_index=0.0,
        motion_model=model,
        periodic=True,
    )

    assert residual.mask.all()
    np.testing.assert_allclose(residual.raw[:, 0], [-4.0, 0.0, -1.0])

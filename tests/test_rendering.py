import numpy as np

from beltmap import (
    BeltMotionModel,
    BeltRegion,
    PhaseEstimate,
    PhaseRegistrationConfig,
    render_belt_view,
    render_expected_clean_belt,
)


def make_belt_map(period=48, width=12):
    y = np.arange(period)[:, None]
    x = np.arange(width)[None, :]
    return (
        80
        + 6 * np.sin(2 * np.pi * y / 13)
        + 4 * np.cos(2 * np.pi * x / 7)
        + 0.2 * y
    )


def test_render_expected_clean_belt_places_crop_in_full_frame():
    belt = make_belt_map()
    model = BeltMotionModel(
        image_velocity_px_per_frame=3.0,
        period_px=belt.shape[0],
        reference_phase_px=11.0,
    )
    region = BeltRegion(top=2, left=4, height=10, width=belt.shape[1])

    render = render_expected_clean_belt(
        belt_map=belt,
        frame_index=3,
        motion_model=model,
        belt_region=region,
        output_shape=(16, 24),
    )

    expected_crop = render_belt_view(belt, model.phase_at(3), height=region.height)
    np.testing.assert_allclose(render.belt_crop, expected_crop)
    np.testing.assert_array_equal(render.mask[region.y_slice, region.x_slice], True)
    assert not render.mask[: region.top].any()
    assert np.isnan(render.image[0, 0])


def test_render_expected_clean_belt_accepts_explicit_phase_estimate():
    belt = np.arange(30, dtype=float)[:, None] * np.ones((1, 4))
    phase = PhaseEstimate(phase_px=4.5, frame_index=7, predicted_phase_px=4.5)

    render = render_expected_clean_belt(
        belt_map=belt,
        frame_index=999,
        phase_estimate=phase,
        output_shape=(3, 4),
    )

    np.testing.assert_allclose(render.image[:, 0], [4.5, 5.5, 6.5])
    assert render.phase_estimate is phase


def test_render_expected_clean_belt_can_refine_phase_from_observed_frame():
    belt = make_belt_map(period=80, width=18)
    true_model = BeltMotionModel(
        image_velocity_px_per_frame=2.0,
        period_px=belt.shape[0],
        reference_phase_px=16.0,
    )
    frame_index = 9
    true_phase = true_model.phase_at(frame_index)
    clean_crop = render_belt_view(belt, true_phase, height=30)
    observed = np.zeros((36, 24), dtype=float)
    observed[3:33, 2:20] = clean_crop
    observed[8:13, 8:13] += 35

    biased_model = BeltMotionModel(
        image_velocity_px_per_frame=2.0,
        period_px=belt.shape[0],
        reference_phase_px=18.0,
    )
    render = render_expected_clean_belt(
        belt_map=belt,
        frame_index=frame_index,
        motion_model=biased_model,
        observed_frame=observed,
        belt_region=(3, 2, 30, 18),
        registration_config=PhaseRegistrationConfig(
            search_radius_px=4,
            search_step_px=0.25,
            trim_fraction=0.12,
            highpass_radius_px=4,
        ),
    )

    circular_error = min(
        abs(render.phase_estimate.phase_px - true_phase),
        belt.shape[0] - abs(render.phase_estimate.phase_px - true_phase),
    )
    assert circular_error <= 0.25
    assert render.phase_estimate.method == "registration"
    np.testing.assert_allclose(render.belt_crop, clean_crop, atol=0.25)


def test_render_expected_clean_belt_follows_finite_strip_motion_model():
    belt = np.arange(5, dtype=float)[:, None] * np.ones((1, 3))
    model = BeltMotionModel(
        image_velocity_px_per_frame=0.0,
        period_px=None,
        reference_phase_px=-1.0,
    )

    render = render_expected_clean_belt(
        belt_map=belt,
        frame_index=0,
        motion_model=model,
        output_shape=(3, 3),
    )

    assert np.isnan(render.image[0, 0])
    np.testing.assert_allclose(render.image[1:, 0], [0.0, 1.0])


def test_render_expected_clean_belt_can_force_periodic_phase_only_rendering():
    belt = np.arange(5, dtype=float)[:, None] * np.ones((1, 3))
    phase = PhaseEstimate(phase_px=-1.0, frame_index=0, predicted_phase_px=-1.0)

    finite = render_expected_clean_belt(
        belt_map=belt,
        frame_index=0,
        phase_estimate=phase,
        output_shape=(3, 3),
        periodic=False,
    )
    cyclic = render_expected_clean_belt(
        belt_map=belt,
        frame_index=0,
        phase_estimate=phase,
        output_shape=(3, 3),
        periodic=True,
    )

    assert np.isnan(finite.image[0, 0])
    np.testing.assert_allclose(cyclic.image[:, 0], [4.0, 0.0, 1.0])

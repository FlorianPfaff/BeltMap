import numpy as np

from beltmap import (
    BeltMotionModel,
    PhaseRegistrationConfig,
    estimate_phase,
    render_belt_view,
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

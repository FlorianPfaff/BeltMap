import numpy as np
import pytest

from beltmap import (
    BeltMotionModel,
    PhaseEstimate,
    ResidualConfig,
    estimate_local_noise,
    generate_residual_image,
    render_belt_view,
    render_clean_belt_residual,
)


def test_generate_residual_image_uses_masked_expected_background():
    observed = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 12.0, 14.0],
            [20.0, 22.0, 24.0],
        ]
    )
    expected = np.array(
        [
            [np.nan, np.nan, np.nan],
            [9.0, 10.0, 11.0],
            [19.0, 20.0, 21.0],
        ]
    )

    residual = generate_residual_image(
        observed,
        expected,
        config=ResidualConfig(noise_radius_px=1, min_noise=0.5),
    )

    np.testing.assert_allclose(residual.raw[1:, :], [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert np.isnan(residual.raw[0, 0])
    assert not residual.mask[0, 0]
    assert np.isfinite(residual.normalized[1:, :]).all()


def test_estimate_local_noise_is_robust_to_sparse_large_particles():
    rng = np.random.default_rng(8)
    noise = rng.normal(0.0, 2.0, size=(60, 50))
    residual = noise.copy()
    residual[20:26, 15:21] += 40.0

    local_noise = estimate_local_noise(
        residual,
        config=ResidualConfig(noise_radius_px=7, clip_sigma=3.0, min_noise=0.1),
    )

    median_noise = float(np.median(local_noise))
    assert 1.4 <= median_noise <= 2.5
    assert float(local_noise[23, 18]) < 8.0


def test_estimate_local_noise_excludes_particle_pixels_from_local_scale():
    rng = np.random.default_rng(12)
    residual = rng.normal(0.0, 1.0, size=(51, 51))
    residual[22:29, 22:29] += 50.0

    without_exclusion = estimate_local_noise(
        residual,
        config=ResidualConfig(
            noise_radius_px=4,
            clip_sigma=5.0,
            noise_exclusion_sigma=None,
            min_noise=0.05,
        ),
    )
    with_exclusion = estimate_local_noise(
        residual,
        config=ResidualConfig(
            noise_radius_px=4,
            clip_sigma=5.0,
            noise_exclusion_sigma=4.0,
            noise_exclusion_radius_px=1,
            min_noise=0.05,
        ),
    )

    assert float(with_exclusion[25, 25]) < float(without_exclusion[25, 25]) * 0.7
    assert float(residual[25, 25] / with_exclusion[25, 25]) > float(
        residual[25, 25] / without_exclusion[25, 25]
    )


def test_estimate_local_noise_excludes_negative_particle_pixels_when_requested():
    rng = np.random.default_rng(14)
    residual = rng.normal(0.0, 1.0, size=(51, 51))
    residual[22:29, 22:29] -= 50.0

    without_exclusion = estimate_local_noise(
        residual,
        config=ResidualConfig(
            noise_radius_px=4,
            clip_sigma=5.0,
            noise_exclusion_sigma=None,
            min_noise=0.05,
        ),
    )
    with_exclusion = estimate_local_noise(
        residual,
        config=ResidualConfig(
            noise_radius_px=4,
            clip_sigma=5.0,
            noise_exclusion_sigma=4.0,
            noise_exclusion_radius_px=1,
            noise_exclusion_mode="negative",
            min_noise=0.05,
        ),
    )

    assert float(with_exclusion[25, 25]) < float(without_exclusion[25, 25]) * 0.7


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (ResidualConfig(noise_radius_px=float("nan")), "noise_radius_px"),
        (ResidualConfig(noise_radius_px=1.5), "noise_radius_px"),
        (ResidualConfig(clip_sigma=float("nan")), "clip_sigma"),
        (ResidualConfig(noise_exclusion_sigma=float("nan")), "noise_exclusion_sigma"),
        (ResidualConfig(noise_exclusion_radius_px=float("nan")), "noise_exclusion_radius_px"),
        (ResidualConfig(noise_exclusion_radius_px=1.5), "noise_exclusion_radius_px"),
        (ResidualConfig(min_noise=float("nan")), "min_noise"),
    ],
)
def test_estimate_local_noise_rejects_invalid_numeric_config(config, message):
    residual = np.ones((5, 5), dtype=float)

    with pytest.raises(ValueError, match=message):
        estimate_local_noise(residual, config=config)


def test_render_clean_belt_residual_returns_standardized_particle_signal():
    period = 64
    width = 16
    y = np.arange(period)[:, None]
    x = np.arange(width)[None, :]
    belt = 100 + 3 * np.sin(2 * np.pi * y / 11) + 2 * np.cos(2 * np.pi * x / 5)
    model = BeltMotionModel(
        image_velocity_px_per_frame=2.0,
        period_px=period,
        reference_phase_px=9.0,
    )
    phase = model.phase_at(4)
    clean_crop = render_belt_view(belt, phase, height=30)
    observed = np.full((36, 22), 0.0)
    observed[3:33, 2:18] = clean_crop
    observed[14:17, 8:11] += 25.0

    residual = render_clean_belt_residual(
        image=observed,
        belt_map=belt,
        frame_index=4,
        motion_model=model,
        belt_region=(3, 2, 30, 16),
        phase_estimate=PhaseEstimate(
            phase_px=phase,
            frame_index=4,
            predicted_phase_px=phase,
        ),
        residual_config=ResidualConfig(noise_radius_px=5, min_noise=0.5),
    )

    assert residual.clean_render is not None
    assert not residual.mask[0, 0]
    assert np.isnan(residual.normalized[0, 0])
    assert float(np.nanmax(residual.normalized)) > 10.0

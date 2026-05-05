import numpy as np

from beltmap import ResidualImage, detect_particles_from_residual


def test_detect_particles_from_residual_thresholds_bright_pixels():
    residual = np.array(
        [
            [0.0, 2.9, 3.1],
            [np.nan, 8.0, -4.0],
        ]
    )

    particle_mask = detect_particles_from_residual(residual, threshold=3.0)

    expected = np.array(
        [
            [False, False, True],
            [False, True, False],
        ]
    )
    np.testing.assert_array_equal(particle_mask, expected)


def test_detect_particles_from_residual_uses_residual_image_valid_mask():
    normalized = np.array(
        [
            [10.0, 10.0],
            [1.0, 12.0],
        ]
    )
    valid = np.array(
        [
            [True, False],
            [True, True],
        ]
    )
    residual = ResidualImage(
        raw=normalized,
        local_noise=np.ones_like(normalized),
        normalized=normalized,
        mask=valid,
        expected_background=np.zeros_like(normalized),
    )

    particle_mask = detect_particles_from_residual(residual, threshold=5.0)

    np.testing.assert_array_equal(
        particle_mask,
        np.array(
            [
                [True, False],
                [False, True],
            ]
        ),
    )


def test_detect_particles_from_residual_combines_optional_mask():
    residual = np.array(
        [
            [6.0, 7.0],
            [8.0, 9.0],
        ]
    )
    allowed = np.array(
        [
            [True, False],
            [False, True],
        ]
    )

    particle_mask = detect_particles_from_residual(
        residual,
        threshold=5.0,
        mask=allowed,
    )

    np.testing.assert_array_equal(particle_mask, allowed)

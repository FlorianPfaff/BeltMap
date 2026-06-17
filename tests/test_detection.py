import numpy as np
import pytest

from beltmap import (
    ResidualImage,
    detect_particles_from_residual,
    detect_particles_from_residual_hysteresis,
    detection_signal_from_residual,
)


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


def test_detect_particles_from_residual_supports_negative_particles():
    residual = np.array(
        [
            [-6.0, -2.0],
            [5.0, 0.0],
        ]
    )

    particle_mask = detect_particles_from_residual(
        residual,
        threshold=4.0,
        mode="negative",
    )

    np.testing.assert_array_equal(
        particle_mask,
        np.array(
            [
                [True, False],
                [False, False],
            ]
        ),
    )


def test_detect_particles_from_residual_supports_absolute_particles():
    residual = np.array(
        [
            [-6.0, -2.0],
            [5.0, 0.0],
        ]
    )

    particle_mask = detect_particles_from_residual(
        residual,
        threshold=4.0,
        mode="absolute",
    )

    np.testing.assert_array_equal(
        particle_mask,
        np.array(
            [
                [True, False],
                [True, False],
            ]
        ),
    )


def test_detect_particles_from_residual_grows_hysteresis_regions_from_strong_seeds():
    residual = np.array(
        [
            [0.0, 2.5, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 2.9, 0.0, 0.0],
        ]
    )

    particle_mask = detect_particles_from_residual(
        residual,
        threshold=4.0,
        low_threshold=2.0,
    )

    np.testing.assert_array_equal(
        particle_mask,
        np.array(
            [
                [False, True, True, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        ),
    )


def test_detect_particles_from_residual_hysteresis_compatibility_wrapper():
    residual = np.array(
        [
            [0.0, 2.5, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    particle_mask = detect_particles_from_residual_hysteresis(
        residual,
        threshold=4.0,
        grow_threshold=2.0,
    )

    np.testing.assert_array_equal(
        particle_mask,
        np.array(
            [
                [False, True, True, False],
                [False, False, False, False],
            ]
        ),
    )


@pytest.mark.parametrize("threshold", [0.0, -1.0, True, float("nan"), "bad"])
def test_detect_particles_from_residual_rejects_invalid_high_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        detect_particles_from_residual(np.ones((2, 2)), threshold=threshold)


@pytest.mark.parametrize("low_threshold", [-1.0, True, float("nan"), "bad"])
def test_detect_particles_from_residual_rejects_invalid_low_threshold(low_threshold):
    with pytest.raises(ValueError, match="low_threshold"):
        detect_particles_from_residual(
            np.ones((2, 2)),
            threshold=1.0,
            low_threshold=low_threshold,
        )


def test_detect_particles_from_residual_rejects_low_threshold_above_threshold():
    with pytest.raises(ValueError, match="low_threshold must be less than or equal"):
        detect_particles_from_residual(
            np.ones((2, 2)),
            threshold=1.0,
            low_threshold=2.0,
        )


def test_detection_signal_from_residual_matches_detection_mode_and_valid_mask():
    normalized = np.array(
        [
            [-3.0, 2.0],
            [-5.0, 4.0],
        ]
    )
    residual = ResidualImage(
        raw=normalized,
        local_noise=np.ones_like(normalized),
        normalized=normalized,
        mask=np.array(
            [
                [True, False],
                [True, True],
            ]
        ),
        expected_background=np.zeros_like(normalized),
    )

    signal = detection_signal_from_residual(residual, mode="negative")

    expected = np.array(
        [
            [3.0, np.nan],
            [5.0, -4.0],
        ]
    )
    np.testing.assert_allclose(signal, expected)


def test_detection_signal_rejects_empty_residual_image():
    empty = np.empty((0, 2), dtype=float)
    residual = ResidualImage(
        raw=empty,
        local_noise=empty,
        normalized=empty,
        mask=np.empty((0, 2), dtype=bool),
        expected_background=empty,
    )

    with pytest.raises(ValueError, match="residual must not be empty"):
        detection_signal_from_residual(residual)


def test_detection_signal_rejects_residual_image_mask_shape_mismatch():
    normalized = np.ones((2, 2), dtype=float)
    residual = ResidualImage(
        raw=normalized,
        local_noise=normalized,
        normalized=normalized,
        mask=np.ones((2, 3), dtype=bool),
        expected_background=np.zeros_like(normalized),
    )

    with pytest.raises(ValueError, match="mask must have the same shape"):
        detection_signal_from_residual(residual)

import numpy as np

from beltmap import ResidualConfig, ResidualImage
from beltmap.driver import correct_frame_residual_bias


def _residual(raw):
    values = np.asarray(raw, dtype=np.float64)
    return ResidualImage(
        raw=values,
        local_noise=np.ones_like(values),
        normalized=values.copy(),
        mask=np.ones(values.shape, dtype=bool),
        expected_background=np.zeros_like(values),
    )


def test_correct_frame_residual_bias_median_excludes_bright_particles():
    residual = _residual([[10.0, 12.0], [11.0, 100.0]])

    corrected = correct_frame_residual_bias(
        residual,
        residual_config=ResidualConfig(),
        mode="median",
        mask_threshold=20.0,
    )

    np.testing.assert_allclose(
        corrected.raw,
        np.array([[-1.0, 1.0], [0.0, 89.0]]),
    )
    np.testing.assert_allclose(corrected.expected_background, np.full((2, 2), 11.0))


def test_correct_frame_residual_bias_row_median_keeps_particle_contrast():
    residual = _residual(
        [
            [5.0, 5.0, 5.0, 50.0],
            [10.0, 10.0, 10.0, 10.0],
        ]
    )

    corrected = correct_frame_residual_bias(
        residual,
        residual_config=ResidualConfig(),
        mode="row_median",
        mask_threshold=20.0,
        row_smoothing_window_px=0,
    )

    np.testing.assert_allclose(
        corrected.raw,
        np.array([[0.0, 0.0, 0.0, 45.0], [0.0, 0.0, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(corrected.expected_background[:, 0], [5.0, 10.0])


def test_correct_frame_residual_bias_none_preserves_object_identity():
    residual = _residual([[1.0]])

    corrected = correct_frame_residual_bias(
        residual,
        residual_config=ResidualConfig(),
        mode="none",
    )

    assert corrected is residual

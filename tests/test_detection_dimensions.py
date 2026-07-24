import numpy as np
import pytest

from beltmap import (
    ResidualImage,
    detect_particles_from_residual,
    detection_signal_from_residual,
)


@pytest.mark.parametrize(
    "values",
    [
        np.ones(4, dtype=float),
        np.ones((1, 2, 2), dtype=float),
    ],
)
def test_detection_apis_reject_non_image_arrays(values):
    with pytest.raises(ValueError, match="residual must be a 2-D array"):
        detection_signal_from_residual(values)

    with pytest.raises(ValueError, match="residual must be a 2-D array"):
        detect_particles_from_residual(values, threshold=1.0)


def test_detection_apis_reject_non_image_residual_objects():
    values = np.ones(4, dtype=float)
    residual = ResidualImage(
        raw=values,
        local_noise=values,
        normalized=values,
        mask=np.ones(values.shape, dtype=bool),
        expected_background=values,
    )

    with pytest.raises(ValueError, match="residual must be a 2-D array"):
        detection_signal_from_residual(residual)

    with pytest.raises(ValueError, match="residual must be a 2-D array"):
        detect_particles_from_residual(residual, threshold=1.0)

import numpy as np
import pytest

from beltmap import ResidualImage
from beltmap._driver_map import detect_map_particle_mask


def _residual_image(normalized: np.ndarray) -> ResidualImage:
    values = np.asarray(normalized, dtype=np.float64)
    return ResidualImage(
        raw=values,
        local_noise=np.ones_like(values),
        normalized=values,
        mask=np.ones(values.shape, dtype=bool),
        expected_background=np.zeros_like(values),
    )


@pytest.mark.parametrize(("mode", "peak_value"), [("positive", 6.0), ("absolute", -6.0)])
def test_detect_map_particle_mask_applies_dilation_to_simple_modes(mode: str, peak_value: float):
    normalized = np.zeros((5, 5), dtype=np.float64)
    normalized[2, 2] = peak_value
    residual = _residual_image(normalized)

    particle_mask = detect_map_particle_mask(
        residual,
        mode=mode,
        threshold=5.0,
        grow_threshold=2.0,
        dilation_px=1,
        margin_px=0,
        min_area_px=1,
    )

    expected = np.zeros((5, 5), dtype=bool)
    expected[1:4, 1:4] = True
    np.testing.assert_array_equal(particle_mask, expected)


def test_detect_map_particle_mask_still_filters_small_positive_components_before_dilation():
    normalized = np.zeros((5, 5), dtype=np.float64)
    normalized[2, 2] = 6.0
    residual = _residual_image(normalized)

    particle_mask = detect_map_particle_mask(
        residual,
        mode="positive",
        threshold=5.0,
        grow_threshold=2.0,
        dilation_px=1,
        margin_px=0,
        min_area_px=2,
    )

    np.testing.assert_array_equal(particle_mask, np.zeros((5, 5), dtype=bool))

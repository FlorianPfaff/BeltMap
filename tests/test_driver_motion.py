import numpy as np
import pytest

from beltmap._driver_motion import correlation_shift


def test_correlation_shift_rejects_search_radius_at_or_above_height():
    previous = np.zeros((4, 6), dtype=np.float32)
    current = np.zeros((4, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="max_shift.*smaller.*image height"):
        correlation_shift(previous, current, max_shift=4)


@pytest.mark.parametrize("max_shift", [0, -1, 1.5, float("nan"), True, np.bool_(True)])
def test_correlation_shift_rejects_invalid_search_radius(max_shift):
    previous = np.zeros((4, 6), dtype=np.float32)
    current = np.zeros((4, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="max_shift"):
        correlation_shift(previous, current, max_shift=max_shift)


def test_correlation_shift_rejects_uninformative_constant_inputs():
    previous = np.ones((8, 6), dtype=np.float32)
    current = np.ones((8, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="uninformative"):
        correlation_shift(previous, current, max_shift=2)

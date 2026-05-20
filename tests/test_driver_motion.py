import numpy as np
import pytest

from beltmap._driver_motion import correlation_shift


def test_correlation_shift_rejects_search_radius_at_or_above_height():
    previous = np.zeros((4, 6), dtype=np.float32)
    current = np.zeros((4, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="max_shift.*smaller.*image height"):
        correlation_shift(previous, current, max_shift=4)

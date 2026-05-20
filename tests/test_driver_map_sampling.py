import pytest

from beltmap._driver_map import sample_indices


def test_sample_indices_rejects_empty_frame_sequence():
    with pytest.raises(ValueError, match="frame_count must be positive"):
        sample_indices(0, 1)

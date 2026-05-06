import pytest

from scripts.apply_beltmap_to_images import (
    validate_auto_velocity_estimate,
    validate_auto_velocity_region,
)


def test_auto_velocity_rejects_full_frame_region_by_default(monkeypatch):
    monkeypatch.delenv("ALLOW_FULL_FRAME_AUTO_VELOCITY", raising=False)

    with pytest.raises(ValueError, match="full-frame BELT_REGION"):
        validate_auto_velocity_region((0, 0, 1728, 2320), (1728, 2320))


def test_auto_velocity_allows_full_frame_region_when_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("ALLOW_FULL_FRAME_AUTO_VELOCITY", "1")

    validate_auto_velocity_region((0, 0, 1728, 2320), (1728, 2320))


def test_auto_velocity_rejects_near_zero_shift(monkeypatch):
    monkeypatch.setenv("AUTO_VELOCITY_MIN_ABS_PX_PER_FRAME", "0.25")

    with pytest.raises(ValueError, match="below AUTO_VELOCITY_MIN_ABS_PX_PER_FRAME"):
        validate_auto_velocity_estimate(0.002, [0.001, 0.002, 0.003], max_shift=90)


def test_auto_velocity_rejects_search_edge_hits(monkeypatch):
    monkeypatch.setenv("AUTO_VELOCITY_MIN_ABS_PX_PER_FRAME", "0.25")
    monkeypatch.setenv("AUTO_VELOCITY_MAX_EDGE_FRACTION", "0.2")

    with pytest.raises(ValueError, match="search edge"):
        validate_auto_velocity_estimate(89.0, [89.0, 88.0, 1.0, 87.0], max_shift=90)

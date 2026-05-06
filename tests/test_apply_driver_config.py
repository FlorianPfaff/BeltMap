import pytest
import numpy as np

from beltmap import BeltRegion, CleanBeltRender, PhaseEstimate, ResidualImage
from scripts.apply_beltmap_to_images import (
    DATA,
    phase_estimate_row,
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


def test_phase_estimate_row_reports_circular_coordinates():
    phase = PhaseEstimate(
        phase_px=25.0,
        frame_index=3.0,
        predicted_phase_px=24.0,
        correction_px=1.0,
        loss=0.5,
        score=0.75,
        method="registration",
    )
    clean = CleanBeltRender(
        image=np.zeros((4, 5)),
        mask=np.ones((4, 5), dtype=bool),
        phase_estimate=phase,
        belt_region=BeltRegion(top=0, left=0, height=4, width=5),
    )
    residual = ResidualImage(
        raw=np.zeros((4, 5)),
        local_noise=np.ones((4, 5)),
        normalized=np.zeros((4, 5)),
        mask=np.ones((4, 5), dtype=bool),
        expected_background=np.zeros((4, 5)),
        clean_render=clean,
    )

    row = phase_estimate_row(
        3,
        DATA / "example.bmp",
        residual,
        period_px=100.0,
    )

    assert row["frame_index"] == 3
    assert row["image"] == "example.bmp"
    assert row["phase_px"] == 25.0
    assert row["phase_fraction"] == 0.25
    assert row["phase_rad"] == pytest.approx(0.5 * np.pi)
    assert row["predicted_phase_px"] == 24.0
    assert row["correction_px"] == 1.0
    assert row["loss"] == 0.5
    assert row["score"] == 0.75
    assert row["method"] == "registration"

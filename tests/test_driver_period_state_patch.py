from pathlib import Path
import math

import pytest

from beltmap import driver_period_state_patch as patch


@pytest.fixture(autouse=True)
def reset_driver_output_period():
    previous = patch._DRIVER_MODEL_PERIOD_PX[0]
    patch._DRIVER_MODEL_PERIOD_PX[0] = patch._DRIVER_MODEL_PERIOD_UNKNOWN
    try:
        yield
    finally:
        patch._DRIVER_MODEL_PERIOD_PX[0] = previous


def _stub_phase_estimate_row(seen: dict[str, float]):
    def fake_phase_estimate_row(frame_index, path, residual, period_px):
        seen["period_px"] = period_px
        return {
            "frame_index": frame_index,
            "image": str(path),
            "phase_px": 50.0,
            "phase_fraction": "stale",
            "phase_rad": "stale",
        }

    return fake_phase_estimate_row


def test_phase_estimate_row_preserves_direct_numeric_period(monkeypatch):
    seen: dict[str, float] = {}
    monkeypatch.setattr(patch, "_original_phase_estimate_row", _stub_phase_estimate_row(seen))

    row = patch._patched_phase_estimate_row(0, Path("frame.png"), object(), 123.0)

    assert seen["period_px"] == 123.0
    assert row["phase_fraction"] == pytest.approx(50.0 / 123.0)
    assert row["phase_rad"] == pytest.approx((50.0 / 123.0) * 2.0 * math.pi)


def test_phase_estimate_row_omits_cycle_fields_for_inferred_driver_support(monkeypatch):
    patch._DRIVER_MODEL_PERIOD_PX[0] = None
    seen: dict[str, float] = {}
    monkeypatch.setattr(patch, "_original_phase_estimate_row", _stub_phase_estimate_row(seen))

    row = patch._patched_phase_estimate_row(0, Path("frame.png"), object(), 123.0)

    assert seen["period_px"] == 1.0
    assert row["phase_fraction"] == ""
    assert row["phase_rad"] == ""


def test_phase_estimate_row_preserves_cycle_fields_for_known_driver_period(monkeypatch):
    patch._DRIVER_MODEL_PERIOD_PX[0] = 123.0
    seen: dict[str, float] = {}
    monkeypatch.setattr(patch, "_original_phase_estimate_row", _stub_phase_estimate_row(seen))

    row = patch._patched_phase_estimate_row(0, Path("frame.png"), object(), 123.0)

    assert seen["period_px"] == 123.0
    assert row["phase_fraction"] == pytest.approx(50.0 / 123.0)
    assert row["phase_rad"] == pytest.approx((50.0 / 123.0) * 2.0 * math.pi)


def test_texture_phase_velocity_uses_direct_numeric_period(monkeypatch):
    called = False

    def fake_summary(*args, **kwargs):
        nonlocal called
        called = True
        return {
            "texture_phase_velocity_status": "called",
            "period_px": kwargs["period_px"],
        }

    monkeypatch.setattr(patch, "_original_texture_phase_velocity_summary", fake_summary)
    phase_rows = [
        {"frame_index": 0, "phase_px": 0.0, "method": "registration"},
        {"frame_index": 1, "phase_px": 1.0, "method": "registration"},
    ]

    summary = patch._patched_texture_phase_velocity_summary(
        phase_rows,
        period_px=123.0,
        nominal_velocity_px_per_frame=1.0,
    )

    assert called
    assert summary == {
        "texture_phase_velocity_status": "called",
        "period_px": 123.0,
    }


def test_texture_phase_velocity_skips_periodic_summary_for_unknown_driver_period(monkeypatch):
    patch._DRIVER_MODEL_PERIOD_PX[0] = None
    called = False

    def fake_summary(*args, **kwargs):
        nonlocal called
        called = True
        return {"texture_phase_velocity_status": "called"}

    monkeypatch.setattr(patch, "_original_texture_phase_velocity_summary", fake_summary)
    phase_rows = [
        {"frame_index": 0, "phase_px": 0.0, "method": "registration"},
        {"frame_index": 1, "phase_px": 1.0, "method": "registration"},
    ]

    summary = patch._patched_texture_phase_velocity_summary(
        phase_rows,
        period_px=123.0,
        nominal_velocity_px_per_frame=1.0,
    )

    assert not called
    assert summary == {
        "texture_phase_velocity_status": "unknown_period",
        "texture_phase_velocity_samples": 2,
    }

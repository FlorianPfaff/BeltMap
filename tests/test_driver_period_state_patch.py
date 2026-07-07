from pathlib import Path
import math

import numpy as np
import pytest

from beltmap import driver_period_state_patch as patch


@pytest.fixture(autouse=True)
def reset_period_state_patch_context():
    previous_driver_period = patch._DRIVER_MODEL_PERIOD_PX[0]
    previous_build_period_known = patch._MAP_BUILD_PERIOD_KNOWN[0]
    previous_accumulation_periodic = patch._MAP_ACCUMULATION_PERIODIC[0]
    patch._DRIVER_MODEL_PERIOD_PX[0] = patch._DRIVER_MODEL_PERIOD_UNKNOWN
    patch._MAP_BUILD_PERIOD_KNOWN[0] = None
    patch._MAP_ACCUMULATION_PERIODIC[0] = None
    try:
        yield
    finally:
        patch._DRIVER_MODEL_PERIOD_PX[0] = previous_driver_period
        patch._MAP_BUILD_PERIOD_KNOWN[0] = previous_build_period_known
        patch._MAP_ACCUMULATION_PERIODIC[0] = previous_accumulation_periodic


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


@pytest.mark.parametrize(
    ("model_period", "expected_periodic"),
    [
        (None, False),
        (123.0, True),
    ],
)
def test_accumulation_context_controls_driver_map_render_periodicity(
    monkeypatch,
    model_period,
    expected_periodic,
):
    seen: dict[str, object] = {}

    def fake_render_belt_view(belt_map, phase_px, height, *, x_slice=None, periodic=True):
        seen["phase_px"] = phase_px
        seen["height"] = height
        seen["x_slice"] = x_slice
        seen["periodic"] = periodic
        return np.zeros((height, np.asarray(belt_map).shape[1]), dtype=np.float32)

    def fake_accumulate_belt_map(*args, **kwargs):
        patch._patched_driver_map_render_belt_view(
            np.zeros((4, 2), dtype=np.float32),
            1.5,
            2,
        )
        return "accumulated"

    monkeypatch.setattr(patch, "_render_belt_view", fake_render_belt_view)
    monkeypatch.setattr(patch, "_original_accumulate_belt_map", fake_accumulate_belt_map)

    result = patch._patched_accumulate_belt_map(model_period=model_period)

    assert result == "accumulated"
    assert seen == {
        "phase_px": 1.5,
        "height": 2,
        "x_slice": None,
        "periodic": expected_periodic,
    }
    assert patch._MAP_ACCUMULATION_PERIODIC[0] is None

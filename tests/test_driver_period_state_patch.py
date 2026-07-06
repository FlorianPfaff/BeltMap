from pathlib import Path
from types import SimpleNamespace

import pytest

import beltmap.driver as driver


def _residual_with_phase(phase_px: float = 25.0):
    estimate = SimpleNamespace(
        phase_px=phase_px,
        predicted_phase_px=phase_px,
        correction_px=0.0,
        drift_px=0.0,
        loss=None,
        score=None,
        second_best_loss=None,
        loss_gap=None,
        loss_gap_ratio=None,
        loss_curvature=None,
        uncertainty_px=None,
        method="registration",
    )
    return SimpleNamespace(clean_render=SimpleNamespace(phase_estimate=estimate))


def test_driver_phase_row_leaves_cyclic_fields_empty_for_inferred_strip(monkeypatch):
    monkeypatch.delenv("BELT_PERIOD_PX", raising=False)
    monkeypatch.delenv("REUSE_BELT_MAP_PATH", raising=False)

    row = driver.phase_estimate_row(
        0,
        Path("frame_000.png"),
        _residual_with_phase(phase_px=25.0),
        100.0,
    )

    assert row["phase_px"] == 25.0
    assert row["phase_fraction"] == ""
    assert row["phase_rad"] == ""


def test_driver_phase_row_preserves_cyclic_fields_when_period_known(monkeypatch):
    monkeypatch.setenv("BELT_PERIOD_PX", "100")
    monkeypatch.delenv("REUSE_BELT_MAP_PATH", raising=False)

    row = driver.phase_estimate_row(
        0,
        Path("frame_000.png"),
        _residual_with_phase(phase_px=25.0),
        100.0,
    )

    assert row["phase_fraction"] == pytest.approx(0.25)
    assert row["phase_rad"] == pytest.approx(0.5 * 3.141592653589793)


def test_texture_phase_velocity_summary_skips_unknown_period_for_inferred_strip(monkeypatch):
    monkeypatch.delenv("BELT_PERIOD_PX", raising=False)
    monkeypatch.delenv("REUSE_BELT_MAP_PATH", raising=False)
    phase_rows = [
        {"frame_index": 0, "phase_px": 0.0, "method": "registration"},
        {"frame_index": 1, "phase_px": 4.0, "method": "registration"},
    ]

    summary = driver.texture_phase_velocity_summary(
        phase_rows,
        period_px=100.0,
        nominal_velocity_px_per_frame=4.0,
    )

    assert summary == {
        "texture_phase_velocity_status": "unknown_period",
        "texture_phase_velocity_samples": 2,
    }

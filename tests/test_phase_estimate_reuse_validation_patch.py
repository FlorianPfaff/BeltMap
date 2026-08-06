from __future__ import annotations

import csv

import pytest

import beltmap  # noqa: F401
from beltmap import driver


FIELDS = [
    "frame_index", "image", "phase_px", "predicted_phase_px",
    "correction_px", "phase_drift_px", "loss", "score",
    "second_best_loss", "loss_gap", "loss_gap_ratio",
    "loss_curvature", "uncertainty_px", "method",
]


def write_row(path, **overrides):
    row = {
        "frame_index": "0", "image": "frame0.bmp", "phase_px": "1.5",
        "predicted_phase_px": "1.0", "correction_px": "0.5",
        "phase_drift_px": "0.1", "loss": "0.2", "score": "0.8",
        "second_best_loss": "0.4", "loss_gap": "0.2",
        "loss_gap_ratio": "0.5", "loss_curvature": "0.3",
        "uncertainty_px": "0.25", "method": "registration",
    }
    row.update(overrides)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)


def test_patch_is_autoloaded():
    assert getattr(driver.load_phase_estimates, "_beltmap_finite_reused_phase_estimates_patched", False)


@pytest.mark.parametrize("field", ["phase_px", "phase_drift_px", "loss", "uncertainty_px"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_loader_rejects_nonfinite_values(tmp_path, field, value):
    path = tmp_path / "phase_estimates.csv"
    write_row(path, **{field: value})
    with pytest.raises(ValueError, match="non-finite"):
        driver.load_phase_estimates(path)


def test_loader_preserves_blank_optional_values(tmp_path):
    path = tmp_path / "phase_estimates.csv"
    write_row(path, loss="", score="", uncertainty_px="")
    estimates = driver.load_phase_estimates(path)
    assert estimates[0].phase_px == pytest.approx(1.5)
    assert estimates[0].loss is None
    assert estimates[0].score is None

import csv
import json

import numpy as np
import pytest
from PIL import Image

from beltmap.advanced_quality import ShiftEstimate
from beltmap.cli import advanced_report


def test_xy_shift_diagnostics_skips_fractional_frame_indices(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    image_dir = tmp_path / "images"
    output_dir.mkdir()
    image_dir.mkdir()
    np.save(output_dir / "belt_map.npy", np.zeros((2, 2), dtype=np.float64))
    (output_dir / "metadata.json").write_text(
        json.dumps({"belt_region": {"top": 0, "left": 0, "height": 2, "width": 2}}),
        encoding="utf-8",
    )
    (output_dir / "phase_estimates.csv").write_text(
        "frame_index,image,phase_px\n0.5,bad.png,0\n1.0,good.png,0\n",
        encoding="utf-8",
    )
    for filename in ["bad.png", "good.png"]:
        Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(image_dir / filename)

    def fake_estimate_integer_xy_shift(*_args, **_kwargs):
        return ShiftEstimate(shift_y_px=0, shift_x_px=0, loss=0.0, score=1.0)

    monkeypatch.setattr(
        advanced_report,
        "estimate_integer_xy_shift",
        fake_estimate_integer_xy_shift,
    )

    summary = advanced_report.estimate_xy_shift_diagnostics(
        output_dir=output_dir,
        image_dir=image_dir,
        sample_count=2,
        max_shift_px=1,
    )

    rows = list(csv.DictReader((output_dir / "xy_shift_diagnostics.csv").open()))
    assert summary["samples"] == 1
    assert [row["frame_index"] for row in rows] == ["1"]


def test_xy_shift_diagnostics_rejects_fractional_belt_region(tmp_path):
    output_dir = tmp_path / "outputs"
    image_dir = tmp_path / "images"
    output_dir.mkdir()
    image_dir.mkdir()
    np.save(output_dir / "belt_map.npy", np.zeros((2, 2), dtype=np.float64))
    (output_dir / "metadata.json").write_text(
        json.dumps({"belt_region": {"top": 0, "left": 0, "height": 2.5, "width": 2}}),
        encoding="utf-8",
    )
    (output_dir / "phase_estimates.csv").write_text(
        "frame_index,image,phase_px\n0,frame_000000.png,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="belt_region.height must be an integer"):
        advanced_report.estimate_xy_shift_diagnostics(
            output_dir=output_dir,
            image_dir=image_dir,
            sample_count=1,
            max_shift_px=1,
        )

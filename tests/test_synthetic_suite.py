import json
import sys

import pytest

from beltmap.cli import synthetic_suite
from beltmap.cli.synthetic_suite import main, render_case, write_config


def test_render_case_splits_event_ids_at_particle_wraps(tmp_path):
    root = tmp_path / "synthetic"

    render_case("baseline", root, frames=28, height=16, width=24, period=16, seed=4)

    metadata = json.loads((root / "synthetic_metadata.json").read_text(encoding="utf-8"))
    boxes = [box for frame in metadata["frames"] for box in frame["boxes"]]
    event_ids = {box["event_id"] for box in boxes}

    assert metadata["height"] == 16
    assert metadata["width"] == 24
    assert metadata["true_particle_velocity_y_px_per_frame"] == pytest.approx(1.4)
    assert metadata["true_velocity_ratio_y"] == pytest.approx(0.7)
    assert len(event_ids) > 1
    assert all(":" in event_id for event_id in event_ids)


def test_render_case_signs_particle_velocity_with_belt_direction(tmp_path):
    root = tmp_path / "synthetic"

    render_case("negative_velocity", root, frames=8, height=16, width=24, period=16, seed=4)

    metadata = json.loads((root / "synthetic_metadata.json").read_text(encoding="utf-8"))

    assert metadata["true_belt_velocity_y_px_per_frame"] == pytest.approx(-2.0)
    assert metadata["true_particle_velocity_y_px_per_frame"] == pytest.approx(-1.4)
    assert metadata["true_velocity_ratio_y"] == pytest.approx(0.7)


def test_write_config_can_enable_photometric_correction(tmp_path):
    config_path = write_config(
        tmp_path / "synthetic",
        frames=4,
        velocity=2.0,
        period=16,
        photometric_enabled=True,
    )

    config = config_path.read_text(encoding="utf-8")

    assert "[photometric]" in config
    assert "enabled = true" in config


def test_write_config_uses_short_suite_track_filter(tmp_path):
    config_path = write_config(
        tmp_path / "synthetic",
        frames=4,
        velocity=2.0,
        period=16,
    )

    config = config_path.read_text(encoding="utf-8")

    assert "[track_filter]" in config
    assert "min_length = 3" in config


def test_write_config_uses_particle_sized_detection_area_gate(tmp_path):
    config_path = write_config(
        tmp_path / "synthetic",
        frames=4,
        velocity=2.0,
        period=16,
    )

    config = config_path.read_text(encoding="utf-8")

    assert "[detection]" in config
    assert "min_area_px = 4" in config


def test_write_config_splits_dense_synthetic_components(tmp_path):
    config_path = write_config(
        tmp_path / "synthetic",
        frames=4,
        velocity=2.0,
        period=16,
    )

    config = config_path.read_text(encoding="utf-8")

    assert "split_merged_components = true" in config
    assert "split_min_projection_gap_px = 1" in config
    assert "split_min_component_area_px = 4" in config


def test_write_config_uses_tight_synthetic_map_mask_margin(tmp_path):
    config_path = write_config(
        tmp_path / "synthetic",
        frames=4,
        velocity=2.0,
        period=16,
    )

    config = config_path.read_text(encoding="utf-8")

    assert "particle_mask_margin_px = 1" in config


def test_write_config_allows_short_tracking_gaps_and_robust_velocity_fit(tmp_path):
    config_path = write_config(
        tmp_path / "synthetic",
        frames=4,
        velocity=2.0,
        period=16,
    )

    config = config_path.read_text(encoding="utf-8")

    assert "max_frame_gap = 2.0" in config
    assert 'velocity_fit_method = "theil_sen"' in config


def test_illumination_drift_suite_enables_photometric_and_map_offset_correction(tmp_path):
    result = main(
        [
            "--output-root",
            str(tmp_path / "suite"),
            "--case",
            "illumination_drift",
            "--frames",
            "2",
            "--height",
            "16",
            "--width",
            "24",
            "--period",
            "16",
        ]
    )

    config = (
        tmp_path / "suite" / "illumination_drift" / "beltmap.toml"
    ).read_text(encoding="utf-8")

    assert result == 0
    assert "enabled = true" in config
    assert "frame_median_offset_correction = true" in config


def test_execute_uses_current_python_modules(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, *, check):
        calls.append(command)
        assert check is True

    monkeypatch.setattr(synthetic_suite.subprocess, "run", fake_run)

    result = main(
        [
            "--output-root",
            str(tmp_path / "suite"),
            "--case",
            "baseline",
            "--frames",
            "2",
            "--height",
            "16",
            "--width",
            "24",
            "--period",
            "16",
            "--execute",
        ]
    )

    assert result == 0
    assert calls[0][:3] == [sys.executable, "-m", "beltmap.cli.apply"]
    assert calls[1][:3] == [sys.executable, "-m", "beltmap.cli.benchmark"]

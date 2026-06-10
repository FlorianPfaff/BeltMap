import csv
import json
from pathlib import Path

import numpy as np
import pytest

from beltmap.cli import map_only_negative_control as cli_map_only_negative_control
from beltmap.map_only_negative_control import (
    MapOnlyNegativeControlConfig,
    generate_map_only_negative_control_report,
    load_phase_samples,
)


def write_phase_estimates(path: Path, *, frames: int, velocity: float, period: float) -> None:
    fieldnames = [
        "frame_index",
        "image",
        "phase_px",
        "predicted_phase_px",
        "correction_px",
        "phase_drift_px",
        "loss",
        "score",
        "method",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(frames):
            phase = (-velocity * index) % period
            writer.writerow(
                {
                    "frame_index": index,
                    "image": f"frame_{index:06d}.png",
                    "phase_px": phase,
                    "predicted_phase_px": phase,
                    "correction_px": 0.0,
                    "phase_drift_px": 0.0,
                    "loss": "",
                    "score": "",
                    "method": "test",
                }
            )


def test_map_only_negative_control_counts_map_blobs_as_false_tracks(tmp_path):
    belt_map = np.zeros((40, 24), dtype=np.float32)
    belt_map[8:12, 6:10] = 100.0
    np.save(tmp_path / "belt_map.npy", belt_map)
    write_phase_estimates(tmp_path / "phase_estimates.csv", frames=7, velocity=1.0, period=40.0)

    result = generate_map_only_negative_control_report(
        output_dir=tmp_path,
        config=MapOnlyNegativeControlConfig(
            threshold=3.0,
            min_area_px=4,
            highpass_radius_px=5,
            crop_height_px=20,
            belt_velocity_px_per_frame=1.0,
            max_match_distance_px=5.0,
            min_track_length=2,
            track_filter_min_length=2,
            long_track_length=3,
        ),
    )

    assert result.metrics["detections"]["false_detections"] > 0
    assert result.metrics["tracks"]["false_tracks"] > 0
    assert result.metrics["tracks"]["false_long_tracks"] > 0
    assert result.artifacts.metrics.is_file()
    assert result.artifacts.report.is_file()
    assert result.artifacts.detections.is_file()
    assert result.artifacts.tracks.is_file()


def test_map_only_negative_control_flat_map_has_no_ghosts(tmp_path):
    np.save(tmp_path / "belt_map.npy", np.ones((24, 12), dtype=np.float32) * 42.0)

    result = generate_map_only_negative_control_report(
        output_dir=tmp_path,
        config=MapOnlyNegativeControlConfig(
            threshold=5.0,
            min_area_px=1,
            highpass_radius_px=3,
            crop_height_px=12,
            frame_count=4,
            belt_velocity_px_per_frame=2.0,
        ),
    )

    assert result.metrics["detections"]["false_detections"] == 0
    assert result.metrics["tracks"]["false_tracks"] == 0
    with result.artifacts.detections.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == []


def test_load_phase_samples_rejects_fractional_frame_indices(tmp_path):
    phase_path = tmp_path / "phase_estimates.csv"
    phase_path.write_text(
        "frame_index,image,phase_px\n0.5,frame_000000.png,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-integer frame_index"):
        load_phase_samples(phase_path)


def test_load_phase_samples_uses_sequence_order_for_blank_frame_indices(tmp_path):
    phase_path = tmp_path / "phase_estimates.csv"
    phase_path.write_text(
        "frame_index,image,phase_px\n,frame_000000.png,0\n",
        encoding="utf-8",
    )

    samples = load_phase_samples(phase_path)

    assert samples[0].frame_index == 0.0


def test_map_only_negative_control_cli_writes_metrics(tmp_path):
    np.save(tmp_path / "belt_map.npy", np.ones((24, 12), dtype=np.float32) * 42.0)

    exit_code = cli_map_only_negative_control.main(
        [
            "--output-dir",
            str(tmp_path),
            "--frame-count",
            "3",
            "--crop-height-px",
            "12",
            "--belt-velocity-px-per-frame",
            "2.0",
            "--quiet",
        ]
    )

    assert exit_code == 0
    metrics = json.loads((tmp_path / "map_only_negative_control_metrics.json").read_text(encoding="utf-8"))
    assert metrics["detections"]["false_detections"] == 0
    assert metrics["tracks"]["false_tracks"] == 0

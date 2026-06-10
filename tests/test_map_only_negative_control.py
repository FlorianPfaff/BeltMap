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


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"period_px": float("nan")}, "period_px must be positive"),
        ({"belt_velocity_px_per_frame": float("nan")}, "belt_velocity_px_per_frame must be finite"),
        ({"max_match_distance_px": float("nan")}, "max_match_distance_px must be positive"),
        ({"highpass_min_scale_gray": float("nan")}, "highpass_min_scale_gray must be positive"),
        ({"track_filter_min_velocity_ratio_y": float("nan")}, "track_filter_min_velocity_ratio_y must be finite"),
        ({"track_filter_max_velocity_ratio_y": float("nan")}, "track_filter_max_velocity_ratio_y must be finite"),
        (
            {"track_filter_min_velocity_ratio_y": 1.2, "track_filter_max_velocity_ratio_y": 1.0},
            "track_filter_min_velocity_ratio_y must be less than or equal",
        ),
        (
            {"track_filter_max_abs_x_velocity_px_per_frame": float("nan")},
            "track_filter_max_abs_x_velocity_px_per_frame must be non-negative",
        ),
    ],
)
def test_map_only_config_rejects_nonfinite_optional_floats(tmp_path, config_kwargs, message):
    with pytest.raises(ValueError, match=message):
        generate_map_only_negative_control_report(
            output_dir=tmp_path,
            config=MapOnlyNegativeControlConfig(**config_kwargs),
        )


def test_map_only_cli_rejects_fractional_integer_config_options():
    with pytest.raises(ValueError, match="min_track_length must be an integer"):
        cli_map_only_negative_control._int_option(
            None,
            {"options": {"min_track_length": {"value": "2.5"}}},
            "min_track_length",
            ("tracking", "min_track_length"),
            2,
        )


def test_map_only_cli_rejects_nonfinite_float_config_options():
    with pytest.raises(ValueError, match="detection_threshold must be finite"):
        cli_map_only_negative_control._float_option(
            None,
            {"options": {"detection_threshold": {"value": "nan"}}},
            "detection_threshold",
            ("detection", "threshold"),
            5.0,
        )


def test_map_only_cli_ignores_fractional_crop_region_height():
    assert cli_map_only_negative_control._region_height(
        {"top": 0, "left": 0, "height": "12.5", "width": 40}
    ) is None


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--long-track-length", "0", "long_track_length must be positive"),
        ("--frame-count", "-1", "frame_count must be positive"),
        ("--crop-height-px", "0", "crop_height_px must be positive"),
        ("--threshold", "nan", "detection_threshold must be finite"),
    ],
)
def test_map_only_cli_rejects_explicit_invalid_integer_values(
    tmp_path,
    capsys,
    option,
    value,
    message,
):
    with pytest.raises(SystemExit) as exc_info:
        cli_map_only_negative_control.main(
            [
                "--output-dir",
                str(tmp_path),
                option,
                value,
                "--quiet",
            ]
        )

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err

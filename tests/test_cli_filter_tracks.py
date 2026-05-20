import csv
import json
from pathlib import Path

import pytest

from beltmap.cli import filter_tracks as cli_filter_tracks


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_velocities(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "velocities.csv",
        [
            {
                "track_id": 0,
                "n_detections": 6,
                "frame_start": 0,
                "frame_end": 5,
                "velocity_y_px_per_frame": 4.0,
                "velocity_x_px_per_frame": 0.5,
                "speed_px_per_frame": 4.03,
                "belt_velocity_y_px_per_frame": 5.0,
                "velocity_ratio_y": 0.8,
                "belt_minus_particle_velocity_y_px_per_frame": 1.0,
            },
            {
                "track_id": 1,
                "n_detections": 3,
                "frame_start": 0,
                "frame_end": 2,
                "velocity_y_px_per_frame": 6.0,
                "velocity_x_px_per_frame": 0.5,
                "speed_px_per_frame": 6.02,
                "belt_velocity_y_px_per_frame": 5.0,
                "velocity_ratio_y": 1.2,
                "belt_minus_particle_velocity_y_px_per_frame": -1.0,
            },
        ],
    )
    write_csv(
        output_dir / "tracks.csv",
        [
            {
                "track_id": 0,
                "track_detection_index": 0,
                "frame_index": 0,
                "image": "frame0.bmp",
                "label": 1,
                "y": 4.0,
                "x": 5.0,
                "area_px": 6,
                "bbox_top": 3,
                "bbox_left": 4,
                "bbox_bottom": 6,
                "bbox_right": 7,
                "mean_signal": 4.5,
                "peak_signal": 7.5,
            },
            {
                "track_id": 1,
                "track_detection_index": 0,
                "frame_index": 0,
                "image": "frame0.bmp",
                "label": 2,
                "y": 14.0,
                "x": 15.0,
                "area_px": 5,
                "bbox_top": 13,
                "bbox_left": 14,
                "bbox_bottom": 16,
                "bbox_right": 17,
                "mean_signal": 4.1,
                "peak_signal": 6.2,
            },
        ],
    )


def test_filter_tracks_writes_scores_and_filtered_velocities(tmp_path, capsys):
    make_velocities(tmp_path)

    exit_code = cli_filter_tracks.main(
        [
            "--output-dir",
            str(tmp_path),
            "--min-track-length",
            "5",
            "--max-velocity-ratio-y",
            "1.1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["velocity_estimates"] == 2
    assert payload["accepted_velocity_estimates"] == 1
    assert payload["accepted_track_detection_rows"] == 1
    filtered_rows = list(csv.DictReader((tmp_path / "filtered_velocities.csv").open()))
    filtered_track_rows = list(csv.DictReader((tmp_path / "filtered_tracks.csv").open()))
    score_rows = list(csv.DictReader((tmp_path / "track_scores.csv").open()))
    assert [row["track_id"] for row in filtered_rows] == ["0"]
    assert [row["track_id"] for row in filtered_track_rows] == ["0"]
    assert [row["accepted"] for row in score_rows] == ["True", "False"]


def test_filter_tracks_cli_treats_zero_lateral_gate_as_disabled(tmp_path, capsys):
    make_velocities(tmp_path)

    exit_code = cli_filter_tracks.main(
        [
            "--output-dir",
            str(tmp_path),
            "--min-track-length",
            "5",
            "--max-velocity-ratio-y",
            "1.1",
            "--max-abs-x-velocity-px-per-frame",
            "0",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["accepted_velocity_estimates"] == 1
    assert payload["track_filter"]["max_abs_x_velocity_px_per_frame"] is None


def test_filter_tracks_cli_reports_missing_velocities_without_traceback(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(SystemExit) as exc_info:
        cli_filter_tracks.main(["--output-dir", str(tmp_path), "--quiet"])

    assert exc_info.value.code == 2


def test_filter_tracks_reconstructs_missing_track_membership(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"belt_velocity_px_per_frame": 5.0}),
        encoding="utf-8",
    )
    write_csv(
        tmp_path / "velocities.csv",
        [
            {
                "track_id": 0,
                "n_detections": 2,
                "frame_start": 0,
                "frame_end": 1,
                "velocity_y_px_per_frame": 4.0,
                "velocity_x_px_per_frame": 0.0,
                "speed_px_per_frame": 4.0,
                "belt_velocity_y_px_per_frame": 5.0,
                "velocity_ratio_y": 0.8,
                "belt_minus_particle_velocity_y_px_per_frame": 1.0,
            },
        ],
    )
    write_csv(
        tmp_path / "detections.csv",
        [
            {
                "frame_index": 0,
                "image": "frame0.bmp",
                "label": 1,
                "y": 10.0,
                "x": 5.0,
                "area_px": 6,
                "bbox_top": 9,
                "bbox_left": 4,
                "bbox_bottom": 12,
                "bbox_right": 7,
                "mean_signal": 4.5,
                "peak_signal": 7.5,
            },
            {
                "frame_index": 1,
                "image": "frame1.bmp",
                "label": 1,
                "y": 14.0,
                "x": 5.0,
                "area_px": 6,
                "bbox_top": 13,
                "bbox_left": 4,
                "bbox_bottom": 16,
                "bbox_right": 7,
                "mean_signal": 4.5,
                "peak_signal": 7.5,
            },
        ],
    )

    payload = cli_filter_tracks.filter_tracks(
        tmp_path,
        config=cli_filter_tracks.TrackFilterConfig(min_track_length=2),
    )

    assert payload["accepted_track_detection_rows"] == 2
    assert (tmp_path / "tracks.csv").is_file()
    filtered_track_rows = list(csv.DictReader((tmp_path / "filtered_tracks.csv").open()))
    assert [row["frame_index"] for row in filtered_track_rows] == ["0", "1"]

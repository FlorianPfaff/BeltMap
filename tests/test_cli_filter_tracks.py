import csv
import json
from pathlib import Path

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
    filtered_rows = list(csv.DictReader((tmp_path / "filtered_velocities.csv").open()))
    score_rows = list(csv.DictReader((tmp_path / "track_scores.csv").open()))
    assert [row["track_id"] for row in filtered_rows] == ["0"]
    assert [row["accepted"] for row in score_rows] == ["True", "False"]

import csv
import json
from pathlib import Path

from PIL import Image

from beltmap.cli import validate as cli_validate


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_minimal_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_images": 3,
                "frame_stride": 1,
                "belt_velocity_px_per_frame": 2.0,
                "belt_map_height_px": 64,
                "n_phase_estimates": 3,
                "n_detections": 3,
                "n_tracks": 1,
                "n_velocity_estimates": 1,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "config_resolved.json").write_text(
        json.dumps({"driver_environment": {"BELT_VELOCITY_PX_PER_FRAME": "2"}}),
        encoding="utf-8",
    )
    (output_dir / "progress.jsonl").write_text(
        json.dumps(
            {
                "stage": "belt_map",
                "message": "interpolating unobserved belt-map pixels",
                "observed_pixels": 100,
                "total_pixels": 128,
                "masked_pixels": 4,
                "contributed_pixels": 300,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    Image.new("L", (8, 8), 128).save(output_dir / "belt_map.png")
    write_csv(
        output_dir / "phase_estimates.csv",
        [
            {"frame_index": 0, "correction_px": 0.0, "score": 0.8},
            {"frame_index": 1, "correction_px": 0.5, "score": 0.9},
            {"frame_index": 2, "correction_px": -0.25, "score": 0.85},
        ],
    )
    write_csv(
        output_dir / "detections_per_frame.csv",
        [
            {"frame_index": 0, "n_detections": 1},
            {"frame_index": 1, "n_detections": 2},
            {"frame_index": 2, "n_detections": 0},
        ],
    )
    write_csv(
        output_dir / "detections.csv",
        [
            {
                "frame_index": 0,
                "label": 1,
                "y": 4.0,
                "x": 5.0,
                "area_px": 16,
                "mean_signal": 4.5,
                "peak_signal": 5.2,
                "bbox_top": 3,
                "bbox_left": 4,
                "bbox_bottom": 6,
                "bbox_right": 7,
            },
        ],
    )
    write_csv(
        output_dir / "velocities.csv",
        [
            {"track_id": 0, "velocity_ratio_y": 0.5, "n_detections": 8},
        ],
    )
    write_csv(
        output_dir / "filtered_tracks.csv",
        [
            {
                "track_id": 0,
                "track_detection_index": 0,
                "frame_index": 0,
                "label": 1,
                "y": 4.0,
                "x": 5.0,
                "area_px": 16,
                "mean_signal": 4.5,
                "peak_signal": 5.2,
                "bbox_top": 3,
                "bbox_left": 4,
                "bbox_bottom": 6,
                "bbox_right": 7,
            },
            {
                "track_id": 0,
                "track_detection_index": 1,
                "frame_index": 1,
                "label": 1,
                "y": 5.0,
                "x": 5.0,
                "area_px": 18,
                "mean_signal": 4.6,
                "peak_signal": 5.3,
                "bbox_top": 4,
                "bbox_left": 4,
                "bbox_bottom": 7,
                "bbox_right": 7,
            },
        ],
    )


def test_generate_validation_report_writes_markdown_and_plots(tmp_path):
    make_minimal_outputs(tmp_path)

    artifacts = cli_validate.generate_validation_report(tmp_path)

    assert artifacts.report == tmp_path / "validation_report.md"
    assert artifacts.report.is_file()
    assert artifacts.summary == tmp_path / "validation_summary.json"
    assert artifacts.summary.is_file()
    summary = json.loads(artifacts.summary.read_text(encoding="utf-8"))
    assert summary["run"]["n_images"] == 3
    assert summary["detections"]["zero_detection_frames"] == 1
    assert summary["detections"]["quality"]["small_detections_area_lt_threshold"] == 1
    assert summary["velocities"]["velocity_ratio_0_to_1_share"] == 1.0
    assert summary["track_lengths"]["tracks_ge_5"] == 1
    assert summary["accepted_track_quality"]["small_accepted_tracks"] == 1
    report = artifacts.report.read_text(encoding="utf-8")
    assert "# BeltMap validation report" in report
    assert "| selected frames | 3 |" in report
    assert "phase_corrections.png" in report
    assert "phase_correction_timeseries.png" in report
    assert "velocity_ratio_histogram.png" in report
    assert "track_length_histogram.png" in report
    assert "| tracks >= 5 detections | 1 |" in report
    assert "| accepted tracks with mean area below threshold | 1 |" in report
    assert set(artifacts.plots) == {
        "phase_corrections",
        "phase_correction_timeseries",
        "registration_score",
        "detections_per_frame",
        "velocity_ratio_histogram",
        "track_length_histogram",
    }
    for path in artifacts.plots.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_validate_main_supports_no_plots(tmp_path, capsys):
    make_minimal_outputs(tmp_path)

    exit_code = cli_validate.main(["--output-dir", str(tmp_path), "--no-plots"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["report"] == str(tmp_path / "validation_report.md")
    assert payload["summary"] == str(tmp_path / "validation_summary.json")
    assert payload["plots"] == {}
    assert (tmp_path / "validation_report.md").is_file()
    assert (tmp_path / "validation_summary.json").is_file()
    assert not (tmp_path / "phase_corrections.png").exists()
    assert not (tmp_path / "phase_correction_timeseries.png").exists()


def test_validation_rejects_existing_csv_with_missing_required_columns(tmp_path):
    make_minimal_outputs(tmp_path)
    write_csv(
        tmp_path / "detections_per_frame.csv",
        [
            {"frame_index": 0, "count": 1},
        ],
    )

    try:
        cli_validate.generate_validation_report(tmp_path, make_plots=False)
    except ValueError as exc:
        assert "n_detections" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected schema validation to reject malformed CSV")

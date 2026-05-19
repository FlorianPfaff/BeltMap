import csv
import json
from pathlib import Path

from PIL import Image

from beltmap.cli import compare as cli_compare
from beltmap.compare_runs import RunSpec, generate_comparison_report, parse_run_spec


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_run(output_dir: Path, *, threshold: float, count_offset: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_images": 3,
                "detection_threshold": threshold,
                "n_detections": 3 + count_offset,
                "n_tracks": 2,
                "n_velocity_estimates": 2,
                "elapsed_s": 12.5,
            }
        ),
        encoding="utf-8",
    )
    write_csv(
        output_dir / "detections_per_frame.csv",
        [
            {"frame_index": 0, "n_detections": 1 + count_offset},
            {"frame_index": 1, "n_detections": 2 + count_offset},
            {"frame_index": 2, "n_detections": count_offset},
        ],
    )
    write_csv(
        output_dir / "detections.csv",
        [
            {
                "frame_index": 0,
                "label": 1,
                "y": 2.0,
                "x": 3.0,
                "area_px": 6 + count_offset,
                "bbox_top": 1,
                "bbox_left": 2,
                "bbox_bottom": 4,
                "bbox_right": 5,
            },
            {
                "frame_index": 1,
                "label": 1,
                "y": 4.0,
                "x": 3.0,
                "area_px": 12 + count_offset,
                "bbox_top": 3,
                "bbox_left": 2,
                "bbox_bottom": 6,
                "bbox_right": 5,
            },
        ],
    )
    write_csv(
        output_dir / "velocities.csv",
        [
            {"track_id": 0, "n_detections": 5, "velocity_ratio_y": 0.5},
            {"track_id": 1, "n_detections": 10, "velocity_ratio_y": 1.4},
        ],
    )
    write_csv(
        output_dir / "filtered_velocities.csv",
        [
            {"track_id": 0, "n_detections": 5, "velocity_ratio_y": 0.5},
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
                "y": 2.0,
                "x": 3.0,
                "area_px": 6 + count_offset,
                "bbox_top": 1,
                "bbox_left": 2,
                "bbox_bottom": 4,
                "bbox_right": 5,
            },
        ],
    )
    for frame_index in [0, 2]:
        Image.new("L", (8, 8), 32 + 40 * frame_index).save(
            output_dir / f"residual_frame_{frame_index:06d}.png"
        )


def test_parse_run_spec_supports_label_and_default_label():
    assert parse_run_spec("T3p5=outputs/T3p5") == RunSpec("T3p5", Path("outputs/T3p5"))
    assert parse_run_spec("outputs/T4p0") == RunSpec("T4p0", Path("outputs/T4p0"))


def test_generate_comparison_report_writes_summary_plots_and_contact_sheet(tmp_path):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=2)

    artifacts = generate_comparison_report(
        [RunSpec("T4.0", run_a), RunSpec("T3.5", run_b)],
        report_dir=tmp_path / "comparison",
        frames=[0, 2],
    )

    assert artifacts.report.is_file()
    assert artifacts.summary_csv.is_file()
    assert set(artifacts.plots) == {"detections_per_frame", "velocity_ratio_histogram"}
    assert set(artifacts.images) == {
        "detection_contact_sheet",
        "filtered_detection_contact_sheet",
    }
    for path in [*artifacts.plots.values(), *artifacts.images.values()]:
        assert path.is_file()
        assert path.stat().st_size > 0
    report = artifacts.report.read_text(encoding="utf-8")
    assert "# BeltMap run comparison" in report
    assert "T4.0" in report
    assert "T3.5" in report
    assert "Detection contact sheet" in report
    assert "Filtered detection contact sheet" in report
    rows = list(csv.DictReader(artifacts.summary_csv.open(newline="", encoding="utf-8")))
    assert rows[0]["label"] == "T4.0"
    assert rows[0]["detection_threshold"] == "4.0"


def test_generate_comparison_report_scores_labeled_detection_target(tmp_path):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=1)
    truth_path = tmp_path / "labels.csv"
    write_csv(
        truth_path,
        [
            {
                "frame_index": 0,
                "bbox_top": 1,
                "bbox_left": 2,
                "bbox_bottom": 4,
                "bbox_right": 5,
            },
            {"frame_index": 2, "bbox_top": "", "bbox_left": "", "bbox_bottom": "", "bbox_right": ""},
        ],
    )

    artifacts = generate_comparison_report(
        [RunSpec("T4.0", run_a), RunSpec("T3.5", run_b)],
        report_dir=tmp_path / "comparison",
        frames=[0, 2],
        truth_path=truth_path,
        truth_iou_threshold=0.25,
    )

    report = artifacts.report.read_text(encoding="utf-8")
    assert "## Labeled real-data target" in report
    assert "labeled F1" in report
    rows = list(csv.DictReader(artifacts.summary_csv.open(newline="", encoding="utf-8")))
    assert rows[0]["labeled_detection_available"] == "True"
    assert rows[0]["labeled_scored_frames"] == "2"
    assert rows[0]["labeled_truth_boxes"] == "1"
    assert rows[0]["labeled_predicted_boxes"] == "1"
    assert rows[0]["labeled_true_positives"] == "1"
    assert rows[0]["labeled_false_positives"] == "0"
    assert rows[0]["labeled_false_negatives"] == "0"
    assert rows[0]["labeled_f1"] == "1.0"


def test_compare_main_prints_artifact_paths(tmp_path, capsys):
    run_a = tmp_path / "T4p0"
    run_b = tmp_path / "T3p5"
    make_run(run_a, threshold=4.0, count_offset=0)
    make_run(run_b, threshold=3.5, count_offset=1)

    exit_code = cli_compare.main(
        [
            "--run",
            f"T4.0={run_a}",
            "--run",
            f"T3.5={run_b}",
            "--report-dir",
            str(tmp_path / "comparison"),
            "--frames",
            "0,2",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["report"]).is_file()
    assert Path(payload["summary_csv"]).is_file()

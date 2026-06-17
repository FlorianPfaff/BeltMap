import csv
import json
from pathlib import Path

from PIL import Image

from beltmap import texture_stress
from beltmap.cli import texture_stress as cli_texture_stress
from beltmap.compare_runs import RunSpec
from beltmap.texture_stress import generate_texture_stress_report


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def textured_image(path: Path, *, value: int, amplitude: int) -> None:
    image = Image.new("L", (12, 12), value)
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = max(0, min(255, value + amplitude * ((x + y) % 2)))
    image.save(path)


def make_run(output_dir: Path, *, threshold: float, extra_detections: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_images": 4,
                "detection_threshold": threshold,
                "n_detections": 4 + extra_detections,
                "n_tracks": 2,
                "n_velocity_estimates": 2,
                "elapsed_s": 1.5,
            }
        ),
        encoding="utf-8",
    )
    write_csv(
        output_dir / "detections_per_frame.csv",
        [
            {"frame_index": 0, "n_detections": 0 + extra_detections},
            {"frame_index": 1, "n_detections": 1 + extra_detections},
            {"frame_index": 2, "n_detections": 2 + extra_detections},
            {"frame_index": 3, "n_detections": 3 + extra_detections},
        ],
    )
    detection_rows = []
    for frame_index in range(4):
        for det_index in range(frame_index + extra_detections):
            detection_rows.append(
                {
                    "frame_index": frame_index,
                    "label": det_index,
                    "y": 2.0 + det_index,
                    "x": 3.0,
                    "area_px": 30 + det_index,
                    "peak_signal": threshold + 2.0,
                    "bbox_top": 1 + det_index,
                    "bbox_left": 2,
                    "bbox_bottom": 4 + det_index,
                    "bbox_right": 5,
                }
            )
    if not detection_rows:
        detection_rows = [
            {
                "frame_index": 0,
                "label": 0,
                "y": 2.0,
                "x": 3.0,
                "area_px": 30,
                "peak_signal": threshold + 2.0,
                "bbox_top": 1,
                "bbox_left": 2,
                "bbox_bottom": 4,
                "bbox_right": 5,
            }
        ]
    write_csv(output_dir / "detections.csv", detection_rows)
    write_csv(
        output_dir / "filtered_tracks.csv",
        [
            {
                "track_id": 0,
                "track_detection_index": 0,
                "frame_index": 1,
                "label": 0,
                "y": 2.0,
                "x": 3.0,
                "area_px": 30,
                "bbox_top": 1,
                "bbox_left": 2,
                "bbox_bottom": 4,
                "bbox_right": 5,
            },
            {
                "track_id": 0,
                "track_detection_index": 1,
                "frame_index": 3,
                "label": 0,
                "y": 4.0,
                "x": 3.0,
                "area_px": 30,
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
            {"track_id": 0, "n_detections": 5, "frame_start": 1, "frame_end": 3, "velocity_ratio_y": 0.5},
        ],
    )
    write_csv(
        output_dir / "filtered_velocities.csv",
        [
            {"track_id": 0, "n_detections": 5, "velocity_ratio_y": 0.5},
        ],
    )
    write_csv(
        output_dir / "phase_estimates.csv",
        [
            {"frame_index": 0, "loss": 0.1, "score": 3.0},
            {"frame_index": 1, "loss": 0.2, "score": 2.0},
            {"frame_index": 2, "loss": 0.4, "score": 1.0},
            {"frame_index": 3, "loss": 0.8, "score": 0.5},
        ],
    )
    for frame_index, amplitude in enumerate([0, 10, 30, 60]):
        textured_image(output_dir / f"raw_frame_{frame_index:06d}.png", value=64, amplitude=amplitude)
        textured_image(output_dir / f"residual_fixed_frame_{frame_index:06d}.png", value=96, amplitude=amplitude)


def test_generate_texture_stress_report_writes_subset_tables_and_plot(tmp_path):
    run_a = tmp_path / "raw"
    run_b = tmp_path / "beltmap"
    make_run(run_a, threshold=12.0, extra_detections=0)
    make_run(run_b, threshold=12.0, extra_detections=1)

    artifacts = generate_texture_stress_report(
        [RunSpec("raw", run_a), RunSpec("beltmap", run_b)],
        report_dir=tmp_path / "stress",
        reference_label="raw",
        quartiles=4,
    )

    assert artifacts.report.is_file()
    assert artifacts.frames_csv.is_file()
    assert artifacts.summary_csv.is_file()
    assert artifacts.plots["detections_by_texture_stress"].is_file()
    report = artifacts.report.read_text(encoding="utf-8")
    assert "# Texture-stress subset analysis" in report
    assert "Reference run: `raw`" in report
    rows = list(csv.DictReader(artifacts.summary_csv.open(newline="", encoding="utf-8")))
    assert {row["run"] for row in rows} == {"raw", "beltmap"}
    assert {row["subset"] for row in rows} == {"Q1", "Q2", "Q3", "Q4"}
    frame_rows = list(csv.DictReader(artifacts.frames_csv.open(newline="", encoding="utf-8")))
    assert frame_rows[0]["subset"] == "Q1"
    assert frame_rows[-1]["subset"] == "Q4"


def test_texture_stress_plot_sort_preserves_zero_rank(tmp_path, monkeypatch):
    captured = {}

    def fake_draw_multiline_plot(path, **kwargs):
        captured["series"] = kwargs["labeled_series"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"plot")

    monkeypatch.setattr(texture_stress, "draw_multiline_plot", fake_draw_multiline_plot)

    texture_stress.write_texture_stress_plots(
        tmp_path,
        [
            {"run": "method", "stress_rank": 1, "detections_per_frame_mean": 10},
            {"run": "method", "stress_rank": 0, "detections_per_frame_mean": 5},
        ],
    )

    assert captured["series"] == [("method", [0.0, 1.0], [5.0, 10.0])]


def test_texture_stress_report_can_score_sparse_labels_by_subset(tmp_path):
    run_a = tmp_path / "raw"
    run_b = tmp_path / "beltmap"
    make_run(run_a, threshold=12.0, extra_detections=0)
    make_run(run_b, threshold=12.0, extra_detections=0)
    truth_path = tmp_path / "labels.csv"
    write_csv(
        truth_path,
        [
            {
                "frame_index": 3,
                "bbox_top": 1,
                "bbox_left": 2,
                "bbox_bottom": 4,
                "bbox_right": 5,
            },
            {"frame_index": 0, "bbox_top": "", "bbox_left": "", "bbox_bottom": "", "bbox_right": ""},
        ],
    )

    artifacts = generate_texture_stress_report(
        [RunSpec("raw", run_a), RunSpec("beltmap", run_b)],
        report_dir=tmp_path / "stress",
        reference_label="raw",
        truth_path=truth_path,
        truth_iou_threshold=0.25,
    )

    report = artifacts.report.read_text(encoding="utf-8")
    assert "## Labeled metrics by stress subset" in report
    rows = list(csv.DictReader(artifacts.summary_csv.open(newline="", encoding="utf-8")))
    q4_raw = next(row for row in rows if row["run"] == "raw" and row["subset"] == "Q4")
    assert q4_raw["labeled_detection_available"] == "True"
    assert q4_raw["labeled_scored_frames"] == "1"


def test_texture_stress_cli_prints_artifact_paths(tmp_path, capsys):
    run_a = tmp_path / "raw"
    run_b = tmp_path / "beltmap"
    make_run(run_a, threshold=12.0, extra_detections=0)
    make_run(run_b, threshold=12.0, extra_detections=1)

    exit_code = cli_texture_stress.main(
        [
            "--run",
            f"raw={run_a}",
            "--run",
            f"beltmap={run_b}",
            "--report-dir",
            str(tmp_path / "stress"),
            "--reference-run",
            "raw",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["report"]).is_file()
    assert Path(payload["frames_csv"]).is_file()
    assert Path(payload["summary_csv"]).is_file()

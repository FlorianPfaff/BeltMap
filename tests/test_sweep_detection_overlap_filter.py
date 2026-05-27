import csv
import json
from pathlib import Path

from scripts import sweep_detection_overlap_filter as sweep_filter


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def detection(label: int, *, x: float, y: float) -> dict[str, object]:
    return {
        "frame_index": 0,
        "image": "frame0.bmp",
        "label": label,
        "y": y,
        "x": x,
        "area_px": 9,
        "bbox_top": y - 1,
        "bbox_left": x - 1,
        "bbox_bottom": y + 2,
        "bbox_right": x + 2,
        "mean_signal": 6.0,
        "peak_signal": 9.0,
        "recurrent_artifact_overlap_fraction": "",
        "recurrent_artifact_probability": "",
        "recurrent_artifact_required_peak_signal": "",
    }


def make_run(output_dir: Path, detections: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps({"n_images": 1, "belt_velocity_px_per_frame": 5.0, "n_detections": len(detections)}),
        encoding="utf-8",
    )
    fields = [
        "frame_index",
        "image",
        "label",
        "y",
        "x",
        "area_px",
        "bbox_top",
        "bbox_left",
        "bbox_bottom",
        "bbox_right",
        "mean_signal",
        "peak_signal",
        "recurrent_artifact_overlap_fraction",
        "recurrent_artifact_probability",
        "recurrent_artifact_required_peak_signal",
    ]
    write_csv(output_dir / "detections.csv", detections, fields)
    write_csv(output_dir / "detections_per_frame.csv", [{"frame_index": 0, "n_detections": len(detections)}])


def test_sweep_detection_overlap_filter_writes_summary(tmp_path):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output_root = tmp_path / "sweep"
    make_run(candidate, [detection(1, x=10, y=10), detection(2, x=40, y=40)])
    make_run(confirming, [detection(1, x=10, y=10)])

    exit_code = sweep_filter.main(
        [
            "--candidate-run",
            str(candidate),
            "--confirming-run",
            str(confirming),
            "--output-root",
            str(output_root),
            "--spec",
            "tight:0.1:0:0.5",
            "--quiet",
        ]
    )

    assert exit_code == 0
    rows = list(csv.DictReader((output_root / "sweep_summary.csv").open(newline="", encoding="utf-8")))
    assert rows[0]["label"] == "tight"
    assert rows[0]["n_detections"] == "1"
    assert (output_root / "sweep_summary.md").is_file()
    assert (output_root / "tight" / "detections.csv").is_file()

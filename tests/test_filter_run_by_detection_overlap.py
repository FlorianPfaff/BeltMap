import csv
import json
from pathlib import Path

from PIL import Image

from scripts import filter_run_by_detection_overlap as overlap_filter


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_run(output_dir: Path, detections: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_images = max(int(row["frame_index"]) for row in detections) + 1 if detections else 0
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "n_images": n_images,
                "belt_velocity_px_per_frame": 5.0,
                "n_detections": len(detections),
            }
        ),
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
    counts = {
        index: sum(1 for row in detections if int(row["frame_index"]) == index)
        for index in range(n_images)
    }
    write_csv(
        output_dir / "detections_per_frame.csv",
        [{"frame_index": index, "n_detections": counts[index]} for index in range(n_images)],
    )
    Image.new("L", (16, 16), 96).save(output_dir / "residual_frame_000000.png")


def detection(label: int, *, x: float, y: float, frame_index: int = 0) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "image": f"frame{frame_index}.bmp",
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


def test_filter_run_keeps_only_detections_confirmed_by_overlap(tmp_path):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output = tmp_path / "confirmed"
    make_run(candidate, [detection(1, x=10, y=10), detection(2, x=40, y=40)])
    make_run(confirming, [detection(1, x=10, y=10)])

    exit_code = overlap_filter.main(
        [
            "--candidate-run",
            str(candidate),
            "--confirming-run",
            str(confirming),
            "--output-dir",
            str(output),
            "--min-iou",
            "0.1",
            "--max-center-distance-px",
            "0",
            "--quiet",
        ]
    )

    assert exit_code == 0
    kept = list(csv.DictReader((output / "detections.csv").open(newline="", encoding="utf-8")))
    rejected = list(csv.DictReader((output / "rejected_detections.csv").open(newline="", encoding="utf-8")))
    assert [row["label"] for row in kept] == ["1"]
    assert [row["label"] for row in rejected] == ["2"]
    assert kept[0]["confirmation_label"] == "1"
    assert (output / "residual_frame_000000.png").is_file()
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["n_detections"] == 1
    assert metadata["confirmation"]["rejected_detections"] == 1


def test_filter_run_can_rescue_unmatched_detections_from_confirmed_tracks(tmp_path):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output = tmp_path / "confirmed"
    candidate_detections = [
        detection(1, x=10, y=10, frame_index=0),
        detection(1, x=15, y=10, frame_index=1),
        detection(1, x=20, y=10, frame_index=2),
    ]
    make_run(candidate, candidate_detections)
    make_run(
        confirming,
        [
            detection(1, x=10, y=10, frame_index=0),
            detection(1, x=15, y=10, frame_index=1),
        ],
    )
    write_csv(
        candidate / "tracks.csv",
        [
            {"track_id": 7, "track_detection_index": index, **row}
            for index, row in enumerate(candidate_detections)
        ],
        ["track_id", "track_detection_index", *overlap_filter.DETECTION_FIELDS],
    )

    exit_code = overlap_filter.main(
        [
            "--candidate-run",
            str(candidate),
            "--confirming-run",
            str(confirming),
            "--output-dir",
            str(output),
            "--min-iou",
            "0.1",
            "--max-center-distance-px",
            "0",
            "--track-rescue-min-detections",
            "3",
            "--track-rescue-min-confirmed",
            "2",
            "--track-rescue-min-confirmed-fraction",
            "0.5",
            "--quiet",
        ]
    )

    assert exit_code == 0
    kept = list(csv.DictReader((output / "detections.csv").open(newline="", encoding="utf-8")))
    assert len(kept) == 3
    assert kept[-1]["confirmation_rescued_by_track"] == "True"
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["confirmation"]["track_rescued_detections"] == 1


def test_filter_run_can_use_one_to_one_confirmation(tmp_path):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output = tmp_path / "confirmed"
    make_run(candidate, [detection(1, x=10, y=10), detection(2, x=12, y=10)])
    make_run(confirming, [detection(1, x=10, y=10)])

    exit_code = overlap_filter.main(
        [
            "--candidate-run",
            str(candidate),
            "--confirming-run",
            str(confirming),
            "--output-dir",
            str(output),
            "--min-iou",
            "0.01",
            "--max-center-distance-px",
            "10",
            "--one-to-one-confirmation",
            "--disable-track-rescue",
            "--quiet",
        ]
    )

    assert exit_code == 0
    kept = list(csv.DictReader((output / "detections.csv").open(newline="", encoding="utf-8")))
    rejected = list(csv.DictReader((output / "rejected_detections.csv").open(newline="", encoding="utf-8")))
    assert [row["label"] for row in kept] == ["1"]
    assert [row["label"] for row in rejected] == ["2"]


def test_filter_run_allows_many_to_one_confirmation_by_default(tmp_path):
    candidate = tmp_path / "candidate"
    confirming = tmp_path / "confirming"
    output = tmp_path / "confirmed"
    make_run(candidate, [detection(1, x=10, y=10), detection(2, x=12, y=10)])
    make_run(confirming, [detection(1, x=10, y=10)])

    exit_code = overlap_filter.main(
        [
            "--candidate-run",
            str(candidate),
            "--confirming-run",
            str(confirming),
            "--output-dir",
            str(output),
            "--min-iou",
            "0.01",
            "--max-center-distance-px",
            "10",
            "--disable-track-rescue",
            "--quiet",
        ]
    )

    assert exit_code == 0
    kept = list(csv.DictReader((output / "detections.csv").open(newline="", encoding="utf-8")))
    assert [row["label"] for row in kept] == ["1", "2"]

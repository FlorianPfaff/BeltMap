from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from beltmap.cli.yolo_export import main as yolo_export_main
from beltmap.yolo_export import (
    export_yolo_predictions_to_beltmap_run,
    infer_frame_index,
    parse_yolo_label_line,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_image(path: Path, *, size: tuple[int, int] = (100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, 32).save(path)


def test_infer_frame_index_uses_last_numeric_group() -> None:
    assert infer_frame_index("frame_000269_combined") == 269
    assert infer_frame_index("ZiegelzuKalk50zu50_10gpros00164") == 164


def test_parse_yolo_label_line_with_and_without_confidence() -> None:
    pred = parse_yolo_label_line("0 0.5 0.25 0.1 0.2 0.75")
    assert pred is not None
    assert pred.class_id == 0
    assert pred.confidence == 0.75

    pred_no_conf = parse_yolo_label_line("1 0.5 0.25 0.1 0.2", default_confidence=0.9)
    assert pred_no_conf is not None
    assert pred_no_conf.class_id == 1
    assert pred_no_conf.confidence == 0.9


def test_export_yolo_predictions_writes_beltmap_run(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    out = tmp_path / "run"
    write_image(images / "frame_000001.png")
    write_image(images / "frame_000002.png")
    labels.mkdir()
    (labels / "frame_000001.txt").write_text(
        "0 0.500000 0.500000 0.200000 0.250000 0.800000\n"
        "0 0.200000 0.250000 0.100000 0.100000 0.600000\n",
        encoding="utf-8",
    )

    summary = export_yolo_predictions_to_beltmap_run(
        labels_dir=labels,
        images_dir=images,
        output_dir=out,
        source="yolo11_raw",
    )

    assert summary.n_images == 2
    assert summary.n_label_files == 1
    assert summary.n_detections == 2
    detections = read_csv(out / "detections.csv")
    assert [row["frame_index"] for row in detections] == ["1", "1"]
    assert detections[0]["bbox_left"] == "40"
    assert detections[0]["bbox_top"] == "30"
    assert detections[0]["bbox_right"] == "60"
    assert detections[0]["bbox_bottom"] == "50"
    assert detections[0]["confidence"] == "0.80000000"
    assert detections[0]["score"] == "0.80000000"
    assert detections[0]["source"] == "yolo11_raw"

    per_frame = read_csv(out / "detections_per_frame.csv")
    assert per_frame == [
        {"frame_index": "1", "n_detections": "2"},
        {"frame_index": "2", "n_detections": "0"},
    ]
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "yolo_export"
    assert metadata["n_detections"] == 2


def test_export_yolo_predictions_rejects_duplicate_frame_indices(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    out = tmp_path / "run"
    write_image(images / "raw" / "frame_000001.png")
    write_image(images / "augmented" / "copy_000001.png")
    labels.mkdir()

    with pytest.raises(ValueError, match="duplicate image frame index 1"):
        export_yolo_predictions_to_beltmap_run(
            labels_dir=labels,
            images_dir=images,
            output_dir=out,
            source="yolo11_raw",
        )


def test_export_yolo_predictions_allows_missing_labels_dir_for_empty_predictions(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    missing_labels = tmp_path / "missing_labels"
    out = tmp_path / "run"
    write_image(images / "frame_000001.png")
    write_image(images / "frame_000002.png")

    summary = export_yolo_predictions_to_beltmap_run(
        labels_dir=missing_labels,
        images_dir=images,
        output_dir=out,
        source="yolo11_raw",
    )

    assert summary.n_images == 2
    assert summary.n_label_files == 0
    assert summary.n_detections == 0
    assert read_csv(out / "detections.csv") == []
    assert read_csv(out / "detections_per_frame.csv") == [
        {"frame_index": "1", "n_detections": "0"},
        {"frame_index": "2", "n_detections": "0"},
    ]
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["n_label_files"] == 0
    assert metadata["n_detections"] == 0


def test_yolo_export_cli(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    out = tmp_path / "run"
    write_image(images / "frame_000003.png")
    labels.mkdir()
    (labels / "frame_000003.txt").write_text(
        "0 0.5 0.5 0.2 0.2 0.7\n",
        encoding="utf-8",
    )

    exit_code = yolo_export_main(
        [
            "--labels-dir",
            str(labels),
            "--images-dir",
            str(images),
            "--output-dir",
            str(out),
            "--quiet",
        ]
    )
    assert exit_code == 0
    assert (out / "detections.csv").is_file()
    assert (out / "detections_per_frame.csv").is_file()

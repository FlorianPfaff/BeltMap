from __future__ import annotations

from pathlib import Path

from PIL import Image

from beltmap.yolo_export import export_yolo_predictions_to_beltmap_run


def write_image(path: Path, *, size: tuple[int, int] = (20, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, 32).save(path)


def test_yolo_export_ignores_image_extension_directories(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    output = tmp_path / "run"
    write_image(images / "frame_000001.png")
    (images / "frame_000002.png").mkdir()
    labels.mkdir()

    summary = export_yolo_predictions_to_beltmap_run(
        labels_dir=labels,
        images_dir=images,
        output_dir=output,
    )

    assert summary.n_images == 1
    assert summary.frame_index_min == 1
    assert summary.frame_index_max == 1


def test_yolo_export_ignores_label_extension_directories(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    output = tmp_path / "run"
    write_image(images / "frame_000001.png")
    labels.mkdir()
    (labels / "frame_000002.txt").mkdir()

    summary = export_yolo_predictions_to_beltmap_run(
        labels_dir=labels,
        images_dir=images,
        output_dir=output,
    )

    assert summary.n_label_files == 0
    assert summary.n_detections == 0

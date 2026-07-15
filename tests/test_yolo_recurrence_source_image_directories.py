from __future__ import annotations

from pathlib import Path

from PIL import Image

from beltmap.yolo_recurrence import find_source_images


def test_find_source_images_ignores_image_suffixed_directories(tmp_path: Path) -> None:
    image_named_directory = tmp_path / "frame_000001.png"
    image_named_directory.mkdir()
    real_image = tmp_path / "frame_000001.jpg"
    Image.new("L", (8, 6), 32).save(real_image)

    assert find_source_images(tmp_path) == {1: real_image}

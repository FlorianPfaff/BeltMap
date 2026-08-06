from __future__ import annotations

from pathlib import Path

from beltmap.yolo_recurrence import find_source_images


def test_find_source_images_ignores_image_suffixed_directories(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "real" / "frame_000001.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"source frame")

    image_like_directory = tmp_path / "fake" / "frame_000001.png"
    image_like_directory.mkdir(parents=True)

    assert find_source_images(tmp_path) == {1: image_path}

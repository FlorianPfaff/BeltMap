from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from beltmap.cli.detect_roi import main as detect_roi_main
from beltmap.operational_improvements import (
    StreamingFrameState,
    dataset_manifest,
    discover_new_stream_frames,
    list_image_paths,
)


def _write_test_image(path: Path, value: int = 0) -> None:
    Image.fromarray(np.full((4, 4), value, dtype=np.uint8)).save(path)


def test_shared_image_discovery_ignores_image_suffixed_directories(tmp_path: Path):
    image_like_directory = tmp_path / "frame_001.png"
    image_like_directory.mkdir()

    image_path = tmp_path / "frame_002.png"
    _write_test_image(image_path)

    assert list_image_paths(tmp_path) == [image_path]

    manifest = dataset_manifest(tmp_path)
    assert [record.path for record in manifest.files] == [image_path.name]

    state = StreamingFrameState()
    assert discover_new_stream_frames(tmp_path, state) == [image_path]


def test_detect_roi_skips_image_suffixed_directories(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    first_image = image_dir / "frame_000.png"
    second_image = image_dir / "frame_002.png"
    _write_test_image(first_image, 10)
    _write_test_image(second_image, 20)
    (image_dir / "frame_001.png").mkdir()
    output_path = tmp_path / "roi.json"

    assert (
        detect_roi_main(
            [
                "--image-dir",
                str(image_dir),
                "--output",
                str(output_path),
                "--max-frames",
                "2",
            ]
        )
        == 0
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["sampled_images"] == [str(first_image), str(second_image)]

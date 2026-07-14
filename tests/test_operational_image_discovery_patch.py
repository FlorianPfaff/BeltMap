from pathlib import Path

import numpy as np
from PIL import Image

from beltmap.operational_improvements import (
    StreamingFrameState,
    dataset_manifest,
    discover_new_stream_frames,
    list_image_paths,
)


def test_shared_image_discovery_ignores_image_suffixed_directories(tmp_path: Path):
    image_like_directory = tmp_path / "frame_001.png"
    image_like_directory.mkdir()

    image_path = tmp_path / "frame_002.png"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(image_path)

    assert list_image_paths(tmp_path) == [image_path]

    manifest = dataset_manifest(tmp_path)
    assert [record.path for record in manifest.files] == [image_path.name]

    state = StreamingFrameState()
    assert discover_new_stream_frames(tmp_path, state) == [image_path]

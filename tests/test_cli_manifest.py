from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from beltmap.cli.manifest import main


def _write_test_image(path: Path) -> None:
    Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4)).save(path)


def test_manifest_refuses_to_overwrite_input_image(tmp_path: Path) -> None:
    image_path = tmp_path / "frame_000.png"
    _write_test_image(image_path)
    original_bytes = image_path.read_bytes()

    with pytest.raises(SystemExit, match="Refusing to overwrite an input image"):
        main(["--image-dir", str(tmp_path), "--output", str(image_path)])

    assert image_path.read_bytes() == original_bytes


def test_manifest_still_writes_noncolliding_output(tmp_path: Path) -> None:
    image_path = tmp_path / "frame_000.png"
    output_path = tmp_path / "data_manifest.json"
    _write_test_image(image_path)

    assert main(["--image-dir", str(tmp_path), "--output", str(output_path)]) == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert [record["path"] for record in payload["files"]] == [image_path.name]

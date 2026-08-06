from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from beltmap.cli.manifest import main


def _write_image(path: Path) -> bytes:
    path.write_bytes(b"input-image-sentinel")
    return path.read_bytes()


def test_manifest_rejects_output_equal_to_input_image(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "frame_000.png"
    original_bytes = _write_image(image_path)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--image-dir",
                str(image_dir),
                "--output",
                str(image_path),
            ]
        )

    assert exc_info.value.code == 2
    assert image_path.read_bytes() == original_bytes


def test_manifest_rejects_hard_link_output_alias(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "frame_000.png"
    output_path = tmp_path / "manifest.json"
    original_bytes = _write_image(image_path)
    try:
        os.link(image_path, output_path)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        pytest.skip(f"hard links are unavailable: {exc}")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--image-dir",
                str(image_dir),
                "--output",
                str(output_path),
            ]
        )

    assert exc_info.value.code == 2
    assert image_path.read_bytes() == original_bytes
    assert output_path.read_bytes() == original_bytes


def test_manifest_writes_distinct_output(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "frame_000.png"
    _write_image(image_path)
    output_path = tmp_path / "manifest.json"

    assert main(
        [
            "--image-dir",
            str(image_dir),
            "--output",
            str(output_path),
        ]
    ) == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert [row["path"] for row in payload["files"]] == [image_path.name]

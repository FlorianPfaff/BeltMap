from __future__ import annotations

from pathlib import Path

from beltmap import _driver_runtime as runtime


def test_image_paths_ignores_directories_with_image_suffixes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "frame_000.png"
    image_path.write_bytes(b"not opened during discovery")
    (image_dir / "frame_001.png").mkdir()

    monkeypatch.setattr(runtime, "DATA", image_dir)
    monkeypatch.setattr(runtime, "OUT", tmp_path / "outputs")
    monkeypatch.delenv("FRAME_STRIDE", raising=False)
    monkeypatch.delenv("MAX_FRAMES", raising=False)

    paths, total, frame_stride = runtime.image_paths()

    assert paths == [image_path]
    assert total == 1
    assert frame_stride == 1

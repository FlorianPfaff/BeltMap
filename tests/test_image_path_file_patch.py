from __future__ import annotations

import importlib
from pathlib import Path

import beltmap.cli  # noqa: F401 - imports CLI side-effect patches
import beltmap.image_path_file_patch as image_path_patch
from beltmap import operational_improvements as operational


def test_cli_image_path_patch_is_autoloaded() -> None:
    assert getattr(
        operational.list_image_paths,
        "_beltmap_regular_image_path_discovery_patched",
        False,
    )


def test_image_like_directories_do_not_consume_max_frames(tmp_path: Path) -> None:
    (tmp_path / "frame_000.png").mkdir()
    first_image = tmp_path / "frame_001.png"
    second_image = tmp_path / "frame_002.jpg"
    first_image.write_bytes(b"first image")
    second_image.write_bytes(b"second image")

    assert operational.list_image_paths(tmp_path, max_frames=1) == [first_image]
    assert operational.list_image_paths(tmp_path) == [first_image, second_image]


def test_image_path_patch_reload_preserves_original(tmp_path: Path) -> None:
    original = getattr(
        operational.list_image_paths,
        "_beltmap_original_list_image_paths",
    )

    importlib.reload(image_path_patch)
    importlib.reload(image_path_patch)

    patched = operational.list_image_paths
    assert getattr(patched, "_beltmap_original_list_image_paths") is original

    image_path = tmp_path / "frame_001.png"
    image_path.write_bytes(b"image")
    (tmp_path / "frame_000.png").mkdir()
    assert patched(tmp_path, max_frames=1) == [image_path]

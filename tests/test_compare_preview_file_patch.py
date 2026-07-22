from __future__ import annotations

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.compare_runs as compare_runs


def test_compare_preview_file_patch_is_autoloaded() -> None:
    assert getattr(
        compare_runs.find_named_preview_paths,
        "_beltmap_compare_preview_file_patched",
        False,
    )


def test_named_preview_discovery_ignores_png_named_directories(tmp_path) -> None:
    valid_preview = tmp_path / "raw_frame_000001.png"
    valid_preview.touch()

    # This directory parses to the same frame index and sorts after the valid
    # zero-padded file.  The old implementation therefore replaced the file
    # path with a directory path that later failed in Pillow.
    (tmp_path / "raw_frame_1.png").mkdir()
    (tmp_path / "raw_frame_000002.png").mkdir()

    assert compare_runs.find_named_preview_paths(tmp_path, "raw") == {
        1: valid_preview
    }

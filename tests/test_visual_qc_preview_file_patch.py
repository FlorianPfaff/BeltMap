from __future__ import annotations

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.visual_qc as visual_qc


def test_visual_qc_preview_file_patch_is_autoloaded() -> None:
    assert getattr(
        visual_qc.find_preview_paths,
        "_beltmap_visual_qc_preview_file_patched",
        False,
    )


def test_visual_qc_preview_discovery_ignores_png_named_directories(
    tmp_path,
) -> None:
    valid_preview = tmp_path / "residual_frame_000001.png"
    valid_preview.touch()

    # This directory parses to the same frame index and sorts after the valid
    # zero-padded file. The old implementation therefore replaced the file path
    # with a directory path that later failed in Pillow.
    (tmp_path / "residual_frame_1.png").mkdir()
    (tmp_path / "residual_frame_000002.png").mkdir()

    assert visual_qc.find_preview_paths(tmp_path) == {1: valid_preview}

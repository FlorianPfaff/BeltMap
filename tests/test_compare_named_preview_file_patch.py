from __future__ import annotations

import importlib

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.compare_named_preview_file_patch as preview_patch
import beltmap.compare_runs as compare_runs


def test_compare_named_preview_file_patch_is_autoloaded() -> None:
    assert getattr(
        compare_runs.find_named_preview_paths,
        "_beltmap_compare_named_preview_file_patched",
        False,
    )


def test_named_preview_discovery_ignores_png_named_directories(tmp_path) -> None:
    valid_preview = tmp_path / "raw_frame_000001.png"
    valid_preview.touch()

    # The old implementation accepted both directories. The same-frame
    # directory sorts after the zero-padded file and therefore replaced the
    # valid preview path before Pillow attempted to open it.
    (tmp_path / "raw_frame_1.png").mkdir()
    (tmp_path / "raw_frame_000002.png").mkdir()

    assert compare_runs.find_named_preview_paths(tmp_path, "raw") == {
        1: valid_preview
    }


def test_named_preview_patch_reload_preserves_original(tmp_path) -> None:
    original = getattr(
        compare_runs.find_named_preview_paths,
        "_beltmap_original_compare_find_named_preview_paths",
    )

    importlib.reload(preview_patch)
    importlib.reload(preview_patch)

    patched = compare_runs.find_named_preview_paths
    assert getattr(
        patched,
        "_beltmap_original_compare_find_named_preview_paths",
    ) is original

    valid_preview = tmp_path / "residual_fixed_frame_000003.png"
    valid_preview.touch()
    assert patched(tmp_path, "residual_fixed") == {3: valid_preview}

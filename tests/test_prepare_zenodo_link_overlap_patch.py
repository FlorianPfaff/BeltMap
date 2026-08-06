from __future__ import annotations

from pathlib import Path

import pytest

from beltmap.cli import prepare_zenodo_dataset as prep


def _directory_target(tmp_path: Path) -> tuple[Path, Path]:
    target = tmp_path / "cache" / "dataset"
    target.mkdir(parents=True)
    sentinel = target / "frame_000001.bmp"
    sentinel.write_bytes(b"prepared-dataset")
    return target, sentinel


def test_prepare_zenodo_link_overlap_patch_is_autoloaded() -> None:
    assert getattr(
        prep.expose_path,
        "_beltmap_prepare_zenodo_link_overlap_patched",
        False,
    )


def test_expose_path_rejects_cache_target_as_link_without_deleting_it(
    tmp_path: Path,
) -> None:
    target, sentinel = _directory_target(tmp_path)

    with pytest.raises(ValueError, match="overlaps cache target"):
        prep.expose_path(
            target=target,
            link=target,
            target_is_directory=True,
        )

    assert sentinel.read_bytes() == b"prepared-dataset"


def test_expose_path_rejects_cache_ancestor_without_deleting_target(
    tmp_path: Path,
) -> None:
    target, sentinel = _directory_target(tmp_path)

    with pytest.raises(ValueError, match="overlaps cache target"):
        prep.expose_path(
            target=target,
            link=target.parent,
            target_is_directory=True,
        )

    assert sentinel.read_bytes() == b"prepared-dataset"


def test_expose_path_rejects_link_nested_inside_directory_target(
    tmp_path: Path,
) -> None:
    target, sentinel = _directory_target(tmp_path)

    with pytest.raises(ValueError, match="overlaps cache target"):
        prep.expose_path(
            target=target,
            link=target / "data" / "images",
            target_is_directory=True,
        )

    assert sentinel.read_bytes() == b"prepared-dataset"


def test_expose_path_still_allows_separate_link_location(tmp_path: Path) -> None:
    target, _sentinel = _directory_target(tmp_path)
    link = tmp_path / "data" / "images"

    prep.expose_path(
        target=target,
        link=link,
        target_is_directory=True,
    )

    assert (link / "frame_000001.bmp").read_bytes() == b"prepared-dataset"

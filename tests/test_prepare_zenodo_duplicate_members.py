from __future__ import annotations

import warnings
from pathlib import Path
from zipfile import ZipFile

import pytest

from beltmap.cli import prepare_zenodo_dataset as prep


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    [
        ("images/frame_000001.bmp", "images/frame_000001.bmp"),
        ("images/frame_000001.bmp", r"images\frame_000001.bmp"),
    ],
)
def test_extract_cached_zip_rejects_duplicate_normalized_member_paths(
    tmp_path: Path,
    first_name: str,
    second_name: str,
) -> None:
    source_zip = tmp_path / "duplicate-members.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ZipFile(source_zip, "w") as archive:
            archive.writestr(first_name, b"first")
            archive.writestr(second_name, b"second")

    cache_images = tmp_path / "cache" / "duplicate-members"
    with pytest.raises(ValueError, match="duplicate path after normalization"):
        prep.extract_cached_zip(
            cache_zip=source_zip,
            cache_images=cache_images,
        )

    assert not cache_images.exists()

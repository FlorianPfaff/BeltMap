from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from beltmap.cli import prepare_zenodo_dataset as prep


def write_dataset_zip(path: Path, payload: bytes) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("images/frame_000001.bmp", payload)


def test_force_download_refreshes_existing_extracted_cache(tmp_path: Path) -> None:
    source_zip = tmp_path / "source.zip"
    cache_root = tmp_path / "cache"
    image_link = tmp_path / "data" / "images"
    common_args = [
        "--url",
        str(source_zip),
        "--dataset-name",
        "images_Test",
        "--cache-root",
        str(cache_root),
        "--image-link",
        str(image_link),
    ]

    write_dataset_zip(source_zip, b"version-one")
    assert prep.main(common_args) == 0
    assert (image_link / "images" / "frame_000001.bmp").read_bytes() == b"version-one"

    write_dataset_zip(source_zip, b"version-two")
    assert prep.main([*common_args, "--force-download"]) == 0

    extracted_image = image_link / "images" / "frame_000001.bmp"
    assert extracted_image.read_bytes() == b"version-two"
    with ZipFile(cache_root / "images_Test.zip") as cached_archive:
        assert cached_archive.read("images/frame_000001.bmp") == b"version-two"

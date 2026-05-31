from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from beltmap.cli import prepare_zenodo_dataset as prep


def create_zip(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("images/frame_000001.bmp", b"fake-image")


def test_main_extracts_local_zip_and_writes_manifest(tmp_path: Path) -> None:
    source_zip = tmp_path / "source.zip"
    create_zip(source_zip)

    result = prep.main(
        [
            "--url",
            str(source_zip),
            "--dataset-name",
            "images_Test",
            "--cache-root",
            str(tmp_path / "cache"),
            "--image-link",
            str(tmp_path / "data" / "images"),
            "--zip-link",
            str(tmp_path / "data" / "images_Test.zip"),
            "--manifest-path",
            str(tmp_path / "outputs" / "dataset_manifest.json"),
        ]
    )

    assert result == 0
    assert (
        tmp_path / "data" / "images" / "images" / "frame_000001.bmp"
    ).read_bytes() == b"fake-image"
    assert (tmp_path / "data" / "images_Test.zip").exists()

    manifest = json.loads(
        (tmp_path / "outputs" / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset_name"] == "images_Test"
    assert manifest["source_url"] == str(source_zip)
    assert manifest["source_record_id"] is None
    assert manifest["source_record_file"] is None
    assert manifest["file_count"] == 1
    assert manifest["extracted_bytes"] == len(b"fake-image")


def test_extract_cached_zip_rejects_path_traversal(tmp_path: Path) -> None:
    source_zip = tmp_path / "unsafe.zip"
    with ZipFile(source_zip, "w") as archive:
        archive.writestr("../outside.txt", b"unsafe")

    with pytest.raises(ValueError, match="unsafe path"):
        prep.extract_cached_zip(
            cache_zip=source_zip,
            cache_images=tmp_path / "cache" / "unsafe",
        )

    assert not (tmp_path / "outside.txt").exists()


def test_extract_cached_zip_rejects_absolute_paths(tmp_path: Path) -> None:
    source_zip = tmp_path / "absolute.zip"
    with ZipFile(source_zip, "w") as archive:
        archive.writestr("/absolute.txt", b"unsafe")

    with pytest.raises(ValueError, match="unsafe path"):
        prep.extract_cached_zip(
            cache_zip=source_zip,
            cache_images=tmp_path / "cache" / "absolute",
        )


def test_record_id_resolution_uses_selected_record_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch_json(url: str) -> dict[str, object]:
        assert url == "https://zenodo.org/api/records/7801882"
        return {
            "files": [
                {"key": "calibration.zip", "size": 5},
                {
                    "key": "images_BrickandSandLimeBrick_50vs50_20gpers.zip",
                    "size": 100,
                },
            ]
        }

    monkeypatch.setattr(prep, "fetch_json", fake_fetch_json)

    source_url, file_name = prep.resolve_source_url(
        url=None,
        record_id="7801882",
        record_file_name_arg=None,
        record_file_glob="images_*.zip",
    )

    assert file_name == "images_BrickandSandLimeBrick_50vs50_20gpers.zip"
    assert source_url == (
        "https://zenodo.org/records/7801882/files/"
        "images_BrickandSandLimeBrick_50vs50_20gpers.zip?download=1"
    )


def test_record_selector_requires_single_largest_match() -> None:
    metadata = {
        "files": [
            {"key": "first.zip", "size": 100},
            {"key": "second.zip", "size": 100},
        ]
    }

    with pytest.raises(ValueError, match="multiple same-size files"):
        prep.select_record_file(metadata, file_name=None, file_glob="*.zip")


def test_record_selector_can_use_exact_file_name() -> None:
    metadata = {
        "files": [
            {"key": "first.zip", "size": 100},
            {"key": "second.zip", "size": 100},
        ]
    }

    assert (
        prep.select_record_file(metadata, file_name="second.zip", file_glob="*.zip")
        == "second.zip"
    )


def test_invalid_dataset_name_rejects_path_components() -> None:
    with pytest.raises(ValueError, match="single path component"):
        prep.validate_dataset_name("nested/name")

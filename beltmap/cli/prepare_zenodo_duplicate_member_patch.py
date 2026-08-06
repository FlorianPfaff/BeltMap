from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

from beltmap.cli import prepare_zenodo_dataset as _prepare_zenodo_dataset

_ORIGINAL_ATTR = "_beltmap_prepare_zenodo_original_safe_extract_zip"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_safe_extract_zip = _unwrap_patched_callable(
    _prepare_zenodo_dataset.safe_extract_zip
)


def safe_extract_zip(archive: ZipFile, destination: Path) -> None:
    """Reject archive members that resolve to the same normalized path."""

    seen_paths: dict[PurePosixPath, str] = {}
    for member in archive.infolist():
        member_path = _prepare_zenodo_dataset.normalized_member_path(member)
        previous_name = seen_paths.get(member_path)
        if previous_name is not None:
            raise ValueError(
                "dataset archive contains duplicate path after normalization: "
                f"{member.filename!r} conflicts with {previous_name!r}"
            )
        seen_paths[member_path] = member.filename

    _original_safe_extract_zip(archive, destination)


setattr(safe_extract_zip, _ORIGINAL_ATTR, _original_safe_extract_zip)
_prepare_zenodo_dataset.safe_extract_zip = safe_extract_zip

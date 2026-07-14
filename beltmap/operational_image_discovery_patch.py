"""Ignore directory entries that merely have an image-like suffix.

``Path.rglob`` yields both files and directories.  The shared operational image
helper previously selected entries only by suffix, so a directory such as
``frame_001.png`` could be handed to Pillow or opened as a binary file by ROI,
manifest, and streaming workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import operational_improvements as _operational_improvements

_ORIGINAL_ATTR = "_beltmap_original_list_image_paths"
_PATCHED_ATTR = "_beltmap_file_only_image_discovery_patched"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the image-discovery helper behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_list_image_paths = _unwrap_patched_callable(
    _operational_improvements.list_image_paths
)


def file_only_list_image_paths(
    image_dir: Path,
    *,
    max_frames: int | None = None,
) -> list[Path]:
    """Return naturally sorted regular image files below ``image_dir``."""

    paths = sorted(
        [
            path
            for path in image_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _operational_improvements.IMAGE_EXTENSIONS
            and not path.name.startswith("._")
        ],
        key=_operational_improvements.natural_key,
    )
    if max_frames is not None and max_frames > 0:
        paths = paths[:max_frames]
    return paths


setattr(file_only_list_image_paths, _PATCHED_ATTR, True)
setattr(file_only_list_image_paths, _ORIGINAL_ATTR, _original_list_image_paths)
_operational_improvements.list_image_paths = file_only_list_image_paths

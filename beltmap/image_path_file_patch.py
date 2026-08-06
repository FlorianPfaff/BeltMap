"""Restrict shared recursive image discovery to regular files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import operational_improvements as _operational

_PATCHED_ATTR = "_beltmap_regular_image_path_discovery_patched"
_ORIGINAL_ATTR = "_beltmap_original_list_image_paths"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original image-path finder behind an earlier patch reload."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_list_image_paths = _unwrap_patched_callable(_operational.list_image_paths)


def list_regular_image_paths(
    image_dir: Path,
    *,
    max_frames: int | None = None,
) -> list[Path]:
    """Return supported image files without letting directories consume limits."""

    paths = [
        path
        for path in _original_list_image_paths(image_dir, max_frames=None)
        if path.is_file()
    ]
    if max_frames is not None and max_frames > 0:
        paths = paths[:max_frames]
    return paths


setattr(list_regular_image_paths, _PATCHED_ATTR, True)
setattr(
    list_regular_image_paths,
    _ORIGINAL_ATTR,
    _original_list_image_paths,
)
_operational.list_image_paths = list_regular_image_paths

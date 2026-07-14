"""Ignore directories during trust/preflight image discovery.

``Path.rglob("*")`` yields directories as well as regular files.  The trust
helpers historically selected entries only by filename suffix, so a directory
named like ``frame_001.png`` was counted as a frame and could later be passed to
Pillow by :func:`beltmap.trust.quality_report`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import trust as _trust

_PATCHED_ATTR = "_beltmap_trust_image_discovery_patched"
_ORIGINAL_ATTR = "_beltmap_original_trust_image_paths"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the image enumerator behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_image_paths = _unwrap_patched_callable(_trust.image_paths)


def file_only_image_paths(image_dir: Path) -> list[Path]:
    """Return the original image ordering with non-file entries removed."""

    return [path for path in _original_image_paths(image_dir) if path.is_file()]


setattr(file_only_image_paths, _PATCHED_ATTR, True)
setattr(file_only_image_paths, _ORIGINAL_ATTR, _original_image_paths)
_trust.image_paths = file_only_image_paths

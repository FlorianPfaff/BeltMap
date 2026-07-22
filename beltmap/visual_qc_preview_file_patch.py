"""Ignore directory entries during visual-QC preview discovery.

``Path.glob`` yields matching directories as well as regular files. A directory
whose name looks like ``residual_frame_000001.png`` must not be registered as a
saved preview or handed to Pillow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import visual_qc as _visual_qc

_PATCHED_ATTR = "_beltmap_visual_qc_preview_file_patched"
_ORIGINAL_ATTR = "_beltmap_original_visual_qc_find_preview_paths"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original preview finder behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_find_preview_paths = _unwrap_patched_callable(
    _visual_qc.find_preview_paths
)


def find_regular_preview_files(output_dir: Path) -> dict[int, Path]:
    """Return only regular residual-preview files keyed by frame index."""

    result: dict[int, Path] = {}
    for path in sorted(output_dir.glob("residual_frame_*.png")):
        if not path.is_file():
            continue
        frame_index = _visual_qc.parse_frame_index_from_preview(path)
        if frame_index is not None:
            result[frame_index] = path
    return result


setattr(find_regular_preview_files, _PATCHED_ATTR, True)
setattr(
    find_regular_preview_files,
    _ORIGINAL_ATTR,
    _original_find_preview_paths,
)
_visual_qc.find_preview_paths = find_regular_preview_files

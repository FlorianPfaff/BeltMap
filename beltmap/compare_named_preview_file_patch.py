"""Ignore directory entries during named comparison-preview discovery.

``Path.glob`` yields matching directories as well as regular files. A directory
whose name looks like ``raw_frame_000001.png`` or
``residual_fixed_frame_000001.png`` must not be registered as a saved preview
or handed to Pillow while comparison contact sheets are assembled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import compare_runs as _compare_runs

_PATCHED_ATTR = "_beltmap_compare_named_preview_file_patched"
_ORIGINAL_ATTR = "_beltmap_original_compare_find_named_preview_paths"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original named-preview finder behind this patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_find_named_preview_paths = _unwrap_patched_callable(
    _compare_runs.find_named_preview_paths
)


def find_regular_named_preview_files(
    output_dir: Path,
    prefix: str,
) -> dict[int, Path]:
    """Return only regular ``<prefix>_frame_*.png`` preview files."""

    result: dict[int, Path] = {}
    marker = f"{prefix}_frame_"
    for path in sorted(output_dir.glob(f"{prefix}_frame_*.png")):
        if not path.is_file():
            continue
        stem = path.stem
        if not stem.startswith(marker):
            continue
        try:
            frame_index = int(stem[len(marker) :])
        except ValueError:
            continue
        result[frame_index] = path
    return result


setattr(find_regular_named_preview_files, _PATCHED_ATTR, True)
setattr(
    find_regular_named_preview_files,
    _ORIGINAL_ATTR,
    _original_find_named_preview_paths,
)
_compare_runs.find_named_preview_paths = find_regular_named_preview_files

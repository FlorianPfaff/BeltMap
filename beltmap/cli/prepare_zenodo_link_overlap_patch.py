"""Prevent destructive overlap between Zenodo cache targets and exposure links.

``prepare_zenodo_dataset.expose_path`` removes the requested link location before
creating a symlink or copy. If that location is the cache target itself, an
ancestor of it, or (for directory targets) nested inside it, the removal or
copy fallback can delete the prepared dataset or recursively copy it into
itself. Reject those layouts before mutating either path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beltmap.cli import prepare_zenodo_dataset as _prepare_zenodo_dataset

_PATCHED_ATTR = "_beltmap_prepare_zenodo_link_overlap_patched"
_ORIGINAL_ATTR = "_beltmap_prepare_zenodo_link_overlap_original_expose_path"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the exposure helper behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_expose_path = _unwrap_patched_callable(
    _prepare_zenodo_dataset.expose_path
)


def _path_location(path: Path) -> Path:
    """Resolve a path's parent without following its final symlink component."""

    candidate = Path(path).expanduser()
    return candidate.parent.resolve() / candidate.name


def exposure_paths_overlap(
    *,
    target: Path,
    link: Path,
    target_is_directory: bool,
) -> bool:
    """Return whether exposing ``target`` at ``link`` can mutate the target tree."""

    target_location = _path_location(target)
    link_location = _path_location(link)

    if link_location == target_location:
        return True
    if link_location in target_location.parents:
        return True
    return target_is_directory and target_location in link_location.parents


def nonoverlapping_expose_path(
    *,
    target: Path,
    link: Path,
    target_is_directory: bool,
) -> None:
    """Expose a cache target only at a non-overlapping filesystem location."""

    if exposure_paths_overlap(
        target=target,
        link=link,
        target_is_directory=target_is_directory,
    ):
        raise ValueError(
            f"exposure link {link} overlaps cache target {target}; "
            "choose a link outside the target/cache tree"
        )
    _original_expose_path(
        target=target,
        link=link,
        target_is_directory=target_is_directory,
    )


setattr(nonoverlapping_expose_path, _PATCHED_ATTR, True)
setattr(nonoverlapping_expose_path, _ORIGINAL_ATTR, _original_expose_path)
_prepare_zenodo_dataset.expose_path = nonoverlapping_expose_path

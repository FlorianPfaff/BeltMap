from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from beltmap.cli import prepare_zenodo_dataset as _prepare_zenodo_dataset

_ORIGINAL_ATTR = "_beltmap_prepare_zenodo_original_expose_path"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_expose_path = _unwrap_patched_callable(_prepare_zenodo_dataset.expose_path)


def _absolute_location(path: Path) -> Path:
    """Return a normalized absolute location without dereferencing the final link."""
    return Path(os.path.abspath(path.expanduser()))


def _would_remove_target(*, target: Path, link: Path) -> bool:
    """Whether replacing ``link`` would remove the cache target itself."""
    if _absolute_location(target) == _absolute_location(link):
        return True

    # A separate symlink to the target is the normal, idempotent exposure case.
    if link.is_symlink():
        return False

    try:
        return target.resolve(strict=False) == link.resolve(strict=False)
    except OSError:
        return False


def expose_path(*, target: Path, link: Path, target_is_directory: bool) -> None:
    """Expose ``target`` without broken or self-destructive cache links."""
    link_target = target if target.is_absolute() else target.absolute()
    if _would_remove_target(target=link_target, link=link):
        raise ValueError(
            "exposure link must differ from its cache target; "
            f"got link={link} and target={target}"
        )
    _original_expose_path(
        target=link_target,
        link=link,
        target_is_directory=target_is_directory,
    )


setattr(expose_path, _ORIGINAL_ATTR, _original_expose_path)
_prepare_zenodo_dataset.expose_path = expose_path

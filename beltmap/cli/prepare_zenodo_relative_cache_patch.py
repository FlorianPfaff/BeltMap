from __future__ import annotations

from pathlib import Path
from typing import Any

from beltmap.cli import prepare_zenodo_dataset as _prepare_zenodo_dataset

_ORIGINAL_ATTR = "_beltmap_prepare_zenodo_original_expose_path"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_expose_path = _unwrap_patched_callable(_prepare_zenodo_dataset.expose_path)


def expose_path(*, target: Path, link: Path, target_is_directory: bool) -> None:
    """Expose ``target`` without breaking links for relative cache paths."""
    link_target = target if target.is_absolute() else target.absolute()
    _original_expose_path(
        target=link_target,
        link=link,
        target_is_directory=target_is_directory,
    )


setattr(expose_path, _ORIGINAL_ATTR, _original_expose_path)
_prepare_zenodo_dataset.expose_path = expose_path

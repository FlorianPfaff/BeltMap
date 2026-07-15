from __future__ import annotations

import sys
from typing import Any

from beltmap.cli import prepare_zenodo_dataset as _prepare_zenodo_dataset

_ORIGINAL_ATTR = "_beltmap_prepare_zenodo_original_main"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_main = _unwrap_patched_callable(_prepare_zenodo_dataset.main)


def main(argv: list[str] | None = None) -> int:
    """Refresh the extracted cache whenever the archive is force-downloaded."""
    requested_args = list(sys.argv[1:] if argv is None else argv)
    if "--force-download" in requested_args and "--force-extract" not in requested_args:
        requested_args.append("--force-extract")
    return _original_main(requested_args)


setattr(main, _ORIGINAL_ATTR, _original_main)
_prepare_zenodo_dataset.main = main

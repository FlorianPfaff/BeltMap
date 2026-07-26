"""Ignore non-object records in validation progress telemetry.

``progress.jsonl`` is an append-only diagnostic stream.  A partially corrupted
or externally edited stream can contain syntactically valid JSON values such as
``null`` or arrays.  The validation reader historically retained those values,
while downstream consumers unconditionally called mapping methods on every
record and crashed with ``AttributeError``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cli import validate as _validate

_PATCHED_ATTR = "_beltmap_validation_progress_objects_patched"
_ORIGINAL_ATTR = "_beltmap_original_read_progress_jsonl"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original reader behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_read_progress_jsonl = _unwrap_patched_callable(_validate.read_progress_jsonl)


def read_progress_object_rows(path: Path) -> list[dict[str, Any]]:
    """Return only JSON-object telemetry rows from ``progress.jsonl``."""

    return [
        row
        for row in _original_read_progress_jsonl(path)
        if isinstance(row, dict)
    ]


setattr(read_progress_object_rows, _PATCHED_ATTR, True)
setattr(
    read_progress_object_rows,
    _ORIGINAL_ATTR,
    _original_read_progress_jsonl,
)
_validate.read_progress_jsonl = read_progress_object_rows

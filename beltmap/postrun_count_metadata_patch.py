"""Reject negative metadata counts in post-run quality diagnostics.

Post-run summaries may read count fields from metadata written by external or
legacy pipelines. A negative integer is syntactically finite, but it cannot be
a valid detection count. Trusting it can make recurrent-rejection fractions
negative or greater than one and can trigger misleading quality flags.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import postrun_improvements as _postrun

_PATCHED_ATTR = "_beltmap_nonnegative_metadata_count_patched"
_ORIGINAL_ATTR = "_beltmap_original_metadata_count_or_rows"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original count resolver behind an earlier patch reload."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_metadata_count_or_rows = _unwrap_patched_callable(
    _postrun.metadata_count_or_rows
)


def nonnegative_metadata_count_or_rows(
    metadata: Mapping[str, Any],
    key: str,
    rows: Sequence[Any],
) -> int:
    """Use metadata only for finite non-negative integer count fields."""

    value = _postrun.finite_int(metadata.get(key))
    if value is not None and value < 0:
        return len(rows)
    return _original_metadata_count_or_rows(metadata, key, rows)


setattr(nonnegative_metadata_count_or_rows, _PATCHED_ATTR, True)
setattr(
    nonnegative_metadata_count_or_rows,
    _ORIGINAL_ATTR,
    _original_metadata_count_or_rows,
)
_postrun.metadata_count_or_rows = nonnegative_metadata_count_or_rows

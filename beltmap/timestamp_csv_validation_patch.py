"""Reject ambiguous or non-finite irregular-frame timestamp tables."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from . import operational_improvements as _operational

_PATCHED_ATTR = "_beltmap_timestamp_csv_validation_patched"
_ORIGINAL_ATTR = "_beltmap_original_load_timestamps_csv"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original timestamp loader if this patch is reloaded."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_load_timestamps_csv = _unwrap_patched_callable(
    _operational.load_timestamps_csv
)


def load_validated_timestamps_csv(
    path: Path,
    *,
    frame_column: str = "frame_index",
    time_column: str = "time_s",
) -> _operational.TimestampTable:
    """Load one finite timestamp for each unambiguous frame index.

    Duplicate frame identifiers previously overwrote the earlier timestamp in the
    dictionary silently. Non-finite timestamps were also accepted, allowing NaN or
    infinity to propagate into irregular-timing calculations. Reject both cases at
    the file boundary and report the offending CSV line.
    """

    mapping: dict[int, float] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [
            name for name in (frame_column, time_column) if name not in fieldnames
        ]
        if missing:
            raise ValueError(
                f"timestamp CSV is missing required column(s): {', '.join(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            raw_frame = row.get(frame_column)
            try:
                frame_index = int(raw_frame)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"timestamp CSV line {line_number} has an invalid "
                    f"{frame_column!r} value: {raw_frame!r}"
                ) from exc

            raw_time = row.get(time_column)
            try:
                timestamp_s = float(raw_time)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"timestamp CSV line {line_number} has an invalid "
                    f"{time_column!r} value: {raw_time!r}"
                ) from exc
            if not math.isfinite(timestamp_s):
                raise ValueError(
                    f"timestamp CSV line {line_number} has a non-finite "
                    f"{time_column!r} value: {raw_time!r}"
                )

            if frame_index in mapping:
                raise ValueError(
                    f"timestamp CSV line {line_number} duplicates frame index "
                    f"{frame_index}"
                )
            mapping[frame_index] = timestamp_s

    if not mapping:
        raise ValueError("timestamp CSV contains no rows")
    return _operational.TimestampTable(mapping)


setattr(load_validated_timestamps_csv, _PATCHED_ATTR, True)
setattr(
    load_validated_timestamps_csv,
    _ORIGINAL_ATTR,
    _original_load_timestamps_csv,
)
_operational.load_timestamps_csv = load_validated_timestamps_csv

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

from beltmap import yolo_recurrence as _yolo_recurrence

_PATCHED_ATTR = "_beltmap_yolo_recurrence_per_frame_patched"
_ORIGINAL_ATTR = "_beltmap_yolo_recurrence_per_frame_original"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


def _frame_index(value: Any) -> int:
    parsed = float(value)
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(f"frame_index must be finite and integer-valued, got {value!r}")
    return int(parsed)


def recompute_detections_per_frame(
    rows: Sequence[Mapping[str, Any]],
    per_frame_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return per-frame counts consistent with exported detection rows.

    The YOLO recurrence hard filter removes detections from ``detections.csv``.
    Reusing the raw YOLO ``detections_per_frame.csv`` for the filtered run leaves
    stale frame counts, which can make downstream reports show the unfiltered
    detection burden even though the filtered detection rows are correct.
    Preserve the original frame universe and any auxiliary per-frame columns, but
    recompute ``n_detections`` from the exported rows.
    """

    counts: Counter[int] = Counter()
    for row in rows:
        counts[_frame_index(row["frame_index"])] += 1

    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in per_frame_rows:
        frame = _frame_index(row["frame_index"])
        updated = dict(row)
        updated["frame_index"] = frame
        updated["n_detections"] = counts.get(frame, 0)
        result.append(updated)
        seen.add(frame)

    for frame in sorted(set(counts) - seen):
        result.append({"frame_index": frame, "n_detections": counts[frame]})

    return result


_original_write_beltmap_run = _unwrap_patched_callable(_yolo_recurrence.write_beltmap_run)


def count_consistent_write_beltmap_run(*args: Any, **kwargs: Any) -> Any:
    """Patch YOLO recurrence exported runs to write count-consistent frame CSVs."""

    if len(args) >= 3:
        mutable_args = list(args)
        mutable_args[2] = recompute_detections_per_frame(mutable_args[1], mutable_args[2])
        return _original_write_beltmap_run(*mutable_args, **kwargs)

    if "rows" in kwargs and "per_frame_rows" in kwargs:
        mutable_kwargs = dict(kwargs)
        mutable_kwargs["per_frame_rows"] = recompute_detections_per_frame(
            mutable_kwargs["rows"], mutable_kwargs["per_frame_rows"]
        )
        return _original_write_beltmap_run(*args, **mutable_kwargs)

    return _original_write_beltmap_run(*args, **kwargs)


setattr(count_consistent_write_beltmap_run, _PATCHED_ATTR, True)
setattr(count_consistent_write_beltmap_run, _ORIGINAL_ATTR, _original_write_beltmap_run)
_yolo_recurrence.write_beltmap_run = count_consistent_write_beltmap_run

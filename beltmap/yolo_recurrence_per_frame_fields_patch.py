from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from beltmap import yolo_recurrence as _yolo_recurrence
from beltmap.yolo_recurrence_per_frame_patch import recompute_detections_per_frame

_PATCHED_ATTR = "_beltmap_yolo_recurrence_per_frame_fields_patched"
_ORIGINAL_ATTR = "_beltmap_yolo_recurrence_per_frame_fields_original"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


def _per_frame_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fieldnames = ["frame_index", "n_detections"]
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    return fieldnames


def _write_detections_per_frame(
    output_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path = Path(output_dir) / "detections_per_frame.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=_per_frame_fieldnames(rows),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


_original_write_beltmap_run = _unwrap_patched_callable(_yolo_recurrence.write_beltmap_run)


def field_preserving_write_beltmap_run(*args: Any, **kwargs: Any) -> Any:
    output_dir = args[0] if args else kwargs.get("output_dir")

    if len(args) >= 3:
        mutable_args = list(args)
        per_frame = recompute_detections_per_frame(mutable_args[1], mutable_args[2])
        mutable_args[2] = per_frame
        result = _original_write_beltmap_run(*mutable_args, **kwargs)
        if output_dir is not None:
            _write_detections_per_frame(output_dir, per_frame)
        return result

    if "rows" in kwargs and "per_frame_rows" in kwargs:
        mutable_kwargs = dict(kwargs)
        per_frame = recompute_detections_per_frame(
            mutable_kwargs["rows"], mutable_kwargs["per_frame_rows"]
        )
        mutable_kwargs["per_frame_rows"] = per_frame
        result = _original_write_beltmap_run(*args, **mutable_kwargs)
        if output_dir is not None:
            _write_detections_per_frame(output_dir, per_frame)
        return result

    return _original_write_beltmap_run(*args, **kwargs)


setattr(field_preserving_write_beltmap_run, _PATCHED_ATTR, True)
setattr(field_preserving_write_beltmap_run, _ORIGINAL_ATTR, _original_write_beltmap_run)
_yolo_recurrence.write_beltmap_run = field_preserving_write_beltmap_run

from __future__ import annotations

import math
from typing import Any

from beltmap import yolo_recurrence as _yolo_recurrence

_PATCHED_ATTR = "_beltmap_yolo_recurrence_region_validated"
_ORIGINAL_ATTR = "_beltmap_yolo_recurrence_original_parse_belt_region"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_parse_belt_region = _unwrap_patched_callable(
    _yolo_recurrence.parse_belt_region
)


def strict_parse_belt_region(value: str):
    """Parse a belt region without silently truncating fractional coordinates."""

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        return _original_parse_belt_region(value)

    names = ("top", "left", "height", "width")
    normalized: list[str] = []
    for name, part in zip(names, parts, strict=True):
        try:
            parsed = float(part)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"belt region {name} must be a finite integer"
            ) from exc
        if not math.isfinite(parsed) or not parsed.is_integer():
            raise ValueError(f"belt region {name} must be a finite integer")
        normalized.append(str(int(parsed)))

    return _original_parse_belt_region(",".join(normalized))


setattr(strict_parse_belt_region, _PATCHED_ATTR, True)
setattr(strict_parse_belt_region, _ORIGINAL_ATTR, _original_parse_belt_region)
_yolo_recurrence.parse_belt_region = strict_parse_belt_region

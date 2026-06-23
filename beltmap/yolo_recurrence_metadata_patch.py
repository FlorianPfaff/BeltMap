from __future__ import annotations

import math
from typing import Any, Mapping

from beltmap import yolo_recurrence as _yolo_recurrence

_ORIGINAL_ATTR = "_beltmap_yolo_recurrence_original_metadata_float"
_PATCHED_ATTR = "_beltmap_yolo_recurrence_blank_metadata_patched"
_MISSING = object()


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_metadata_float = _unwrap_patched_callable(_yolo_recurrence.metadata_float)


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def blank_default_metadata_float(
    metadata: Mapping[str, Any],
    key: str,
    *,
    default: float | None = None,
) -> float:
    """Return metadata float values while honoring defaults for blank fields.

    Legacy and hand-written BeltMap metadata can contain optional numeric keys as
    empty strings.  The YOLO recurrence scorer already passes a fallback for
    ``belt_period_px_input`` from ``belt_map_height_px``; treating a blank string
    as an actual value makes ``float(\"\")`` fail before that fallback can be
    used.  Interpret ``None``/blank strings as missing so the explicit default is
    honored, while still rejecting blank required fields.
    """

    value = metadata.get(key, _MISSING)
    if value is _MISSING or _is_blank(value):
        if default is None:
            raise ValueError(f"metadata.json is missing required field {key!r}")
        value = default
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"metadata field {key!r} must be finite")
    return parsed


setattr(blank_default_metadata_float, _PATCHED_ATTR, True)
setattr(blank_default_metadata_float, _ORIGINAL_ATTR, _original_metadata_float)
_yolo_recurrence.metadata_float = blank_default_metadata_float

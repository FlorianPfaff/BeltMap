from __future__ import annotations

import math
from typing import Any, Mapping

from beltmap import yolo_recurrence as _yolo_recurrence
from beltmap.period_state import require_period_known, reused_period_state

_ORIGINAL_ATTR = "_beltmap_yolo_recurrence_original_metadata_float"
_PATCHED_ATTR = "_beltmap_yolo_recurrence_blank_metadata_patched"
_MISSING = object()
_PERIOD_STATE_METADATA_KEYS = (
    "model_period_px",
    "belt_model_period_px",
    "belt_period_known",
    "belt_map_periodic",
    "belt_period_state_source",
)


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_metadata_float = _unwrap_patched_callable(_yolo_recurrence.metadata_float)


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _has_period_state_metadata(metadata: Mapping[str, Any]) -> bool:
    return any(key in metadata for key in _PERIOD_STATE_METADATA_KEYS)


def _period_from_explicit_state_metadata(
    metadata: Mapping[str, Any],
    *,
    default: float | None,
) -> float | None:
    """Resolve physical recurrence period when modern period metadata is present."""

    if default is None or not _has_period_state_metadata(metadata):
        return None
    try:
        map_height_px = int(round(float(default)))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("metadata belt_map_height_px must be a positive integer") from exc
    if map_height_px <= 0 or abs(float(default) - float(map_height_px)) > 1e-6:
        raise ValueError("metadata belt_map_height_px must be a positive integer")
    state = reused_period_state(
        map_height_px=map_height_px,
        supplied_period_px=None,
        metadata=metadata,
    )
    require_period_known(state, feature="YOLO recurrence filtering")
    if state.model_period_px is None:
        raise ValueError(
            "YOLO recurrence filtering requires a known physical BELT_PERIOD_PX"
        )
    return float(state.model_period_px)


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
    as an actual value makes ``float("")`` fail before that fallback can be
    used.  Interpret ``None``/blank strings as missing so the explicit default is
    honored for legacy metadata, while still rejecting blank required fields.

    Modern driver metadata explicitly distinguishes physical belt periods from
    finite inferred map support.  YOLO recurrence scoring is revolution-based, so
    it must not fall back to the finite support height when that metadata declares
    that no physical period is known.
    """

    if key == "belt_period_px_input":
        explicit_period = _period_from_explicit_state_metadata(
            metadata,
            default=default,
        )
        if explicit_period is not None:
            return explicit_period

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

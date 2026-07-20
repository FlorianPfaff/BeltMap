"""Reject malformed acceptance flags in runtime recurrence filtering."""

from __future__ import annotations

from typing import Any

from beltmap.cli import filter_revolution_recurrence as _filter_revolution_recurrence

_TRUE_VALUES = {"true", "1", "yes", "on"}
_FALSE_VALUES = {"false", "0", "no", "off"}


def strict_bool_value(value: Any) -> bool:
    """Parse an explicit boolean token instead of treating typos as false."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        "accepted must be one of true/false, 1/0, yes/no, or on/off; "
        f"got {value!r}"
    )


_filter_revolution_recurrence.bool_value = strict_bool_value

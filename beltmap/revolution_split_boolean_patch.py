"""Reject booleans where revolution-split APIs require integer values.

Python and NumPy booleans are numeric subclasses, so ``float(True)`` becomes
``1.0`` and previously passed the shared integer validator.  That could silently
reinterpret configuration flags as frame or revolution indices.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import revolution_split as _revolution_split

_PATCHED_ATTR = "_beltmap_revolution_split_boolean_patched"
_ORIGINAL_ATTR = "_beltmap_original_nonnegative_integer_value"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original validator behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_nonnegative_integer_value = _unwrap_patched_callable(
    _revolution_split._nonnegative_integer_value
)


def reject_boolean_integer_value(value: Any, name: str) -> int:
    """Validate an integer-like value while rejecting boolean aliases."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite non-negative integer")
    return _original_nonnegative_integer_value(value, name)


setattr(reject_boolean_integer_value, _PATCHED_ATTR, True)
setattr(
    reject_boolean_integer_value,
    _ORIGINAL_ATTR,
    _original_nonnegative_integer_value,
)
_revolution_split._nonnegative_integer_value = reject_boolean_integer_value

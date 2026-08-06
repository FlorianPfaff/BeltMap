"""Require exact non-negative count metrics in the ghost objective.

Ghost-objective inputs may come from CSV or JSON summaries.  The original count
parser rounded every finite number before converting it to ``int``.  Malformed
values such as ``0.6`` were therefore treated as one false detection, while
negative and boolean values were also accepted as counts.  Invalid evidence can
change configuration ranking instead of making the affected variant ineligible.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import ghost_objective as _ghost_objective

_PATCHED_ATTR = "_beltmap_exact_ghost_objective_integer_patched"
_ORIGINAL_ATTR = "_beltmap_original_ghost_objective_finite_integer"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original count parser behind this compatibility patch."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_finite_integer = _unwrap_patched_callable(_ghost_objective.finite_integer)


def exact_finite_integer(value: Any) -> int | None:
    """Parse a finite exact non-negative integer, rejecting booleans."""

    if isinstance(value, (bool, np.bool_)):
        return None
    parsed = _ghost_objective.finite_number(value)
    if parsed is None or parsed < 0.0 or not parsed.is_integer():
        return None
    return int(parsed)


setattr(exact_finite_integer, _PATCHED_ATTR, True)
setattr(exact_finite_integer, _ORIGINAL_ATTR, _original_finite_integer)
_ghost_objective.finite_integer = exact_finite_integer

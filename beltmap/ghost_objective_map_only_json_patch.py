from __future__ import annotations

from typing import Any

import numpy as np

from beltmap import ghost_objective as _ghost_objective
from beltmap.ghost_objective_json import flatten_nested_map_only_metrics

_ORIGINAL_ATTR = "_beltmap_ghost_objective_original_load_one_map_only_path"
_ORIGINAL_INTEGER_ATTR = "_beltmap_ghost_objective_original_finite_integer"


def _unwrap(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


def _unwrap_integer(func: Any) -> Any:
    return getattr(func, _ORIGINAL_INTEGER_ATTR, func)


_original_load_one_map_only_path = _unwrap(_ghost_objective.load_one_map_only_path)
_original_finite_integer = _unwrap_integer(_ghost_objective.finite_integer)


def nested_json_load_one_map_only_path(label: str, path: Any) -> dict[str, Any]:
    if getattr(path, "suffix", "").lower() != ".json":
        return _original_load_one_map_only_path(label, path)

    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    candidate = flatten_nested_map_only_metrics(data)
    return _ghost_objective.map_only_evidence_from_row(label, candidate, source=str(path))


def strict_finite_integer(value: Any) -> int | None:
    """Parse count-like values without rounding fractional or boolean inputs."""

    if isinstance(value, (bool, np.bool_)):
        return None
    parsed = _ghost_objective.finite_number(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


setattr(nested_json_load_one_map_only_path, _ORIGINAL_ATTR, _original_load_one_map_only_path)
setattr(strict_finite_integer, _ORIGINAL_INTEGER_ATTR, _original_finite_integer)
_ghost_objective.load_one_map_only_path = nested_json_load_one_map_only_path
_ghost_objective.finite_integer = strict_finite_integer

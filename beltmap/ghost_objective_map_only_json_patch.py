from __future__ import annotations

from typing import Any

from beltmap import ghost_objective as _ghost_objective
from beltmap.ghost_objective_json import flatten_nested_map_only_metrics

_ORIGINAL_ATTR = "_beltmap_ghost_objective_original_load_one_map_only_path"


def _unwrap(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_load_one_map_only_path = _unwrap(_ghost_objective.load_one_map_only_path)


def nested_json_load_one_map_only_path(label: str, path: Any) -> dict[str, Any]:
    if getattr(path, "suffix", "").lower() != ".json":
        return _original_load_one_map_only_path(label, path)

    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    candidate = flatten_nested_map_only_metrics(data)
    return _ghost_objective.map_only_evidence_from_row(label, candidate, source=str(path))


setattr(nested_json_load_one_map_only_path, _ORIGINAL_ATTR, _original_load_one_map_only_path)
_ghost_objective.load_one_map_only_path = nested_json_load_one_map_only_path

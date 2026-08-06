"""Keep packaged-driver progress telemetry valid strict JSON."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import _driver_runtime as _runtime


def jsonable(value: Any) -> Any:
    """Convert runtime telemetry values to JSON-safe built-in types."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


_runtime.jsonable = jsonable

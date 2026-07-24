from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beltmap import label_validation as _label_validation

_ORIGINAL_ATTR = "_beltmap_label_validation_original_validated_label_state"
_PATCHED_ATTR = "_beltmap_label_validation_collection_types_patched"
_COLLECTION_KEY_GROUPS = (
    _label_validation._PARTICLE_KEYS,
    _label_validation._REVIEW_KEYS,
)


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_validated_label_state = _unwrap_patched_callable(
    _label_validation.validated_label_state
)


def _malformed_collection_errors(truth_path: Path | str) -> list[str]:
    path = Path(truth_path)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []

    errors: list[str] = []
    for keys in _COLLECTION_KEY_GROUPS:
        for key in keys:
            if key in payload and not isinstance(payload[key], list):
                errors.append(f"{key} must be a list")
    return errors


def validated_label_state_with_collection_type_checks(
    truth_path: Path | str,
) -> _label_validation.LabelValidationReport:
    """Reject malformed top-level particle and frame-review collections."""

    report = _original_validated_label_state(truth_path)
    collection_errors = _malformed_collection_errors(truth_path)
    for message in collection_errors:
        if message not in report.errors:
            report.errors.append(message)
    if collection_errors:
        report.is_valid_for_metrics = False
    return report


setattr(validated_label_state_with_collection_type_checks, _PATCHED_ATTR, True)
setattr(
    validated_label_state_with_collection_type_checks,
    _ORIGINAL_ATTR,
    _original_validated_label_state,
)
_label_validation.validated_label_state = (
    validated_label_state_with_collection_type_checks
)

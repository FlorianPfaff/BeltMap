"""Allow one event identity to span multiple labeled frames.

``label_validation.validated_label_state`` historically treated every repeated
``event_id`` as a duplicate.  Event and tracklet annotations intentionally reuse
an identity across frames, so that check rejected otherwise valid reviewed truth.
Only repeated ``(event_id, frame_index)`` pairs are ambiguous.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from . import label_validation as _label_validation

_PATCHED_ATTR = "_beltmap_multiframe_event_id_validation_patched"
_ORIGINAL_ATTR = "_beltmap_original_validated_label_state"
_LEGACY_ERROR_PREFIX = "duplicate event_id values:"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the validator behind this compatibility patch, if present."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_validated_label_state = _unwrap_patched_callable(
    _label_validation.validated_label_state
)


def _duplicate_event_frame_errors(path: Path) -> list[str]:
    """Return errors for identities assigned twice within the same frame."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []

    parse_errors: list[str] = []
    particles = _label_validation.parse_particles(payload, errors=parse_errors)
    counts = Counter(
        (particle.event_id, particle.frame_index)
        for particle in particles
        if particle.event_id is not None
    )
    duplicates = sorted(
        (event_id, frame_index)
        for (event_id, frame_index), count in counts.items()
        if count > 1
    )
    if not duplicates:
        return []

    listed = ", ".join(
        f"{event_id}@{frame_index}" for event_id, frame_index in duplicates[:10]
    )
    suffix = "" if len(duplicates) <= 10 else f" ... (+{len(duplicates) - 10} more)"
    return [f"duplicate event_id/frame_index pairs: {listed}{suffix}"]


def validated_label_state_allow_multiframe_event_ids(
    truth_path: Path | str,
) -> _label_validation.LabelValidationReport:
    """Validate labels while allowing an event identity to continue over time."""

    report = _original_validated_label_state(truth_path)
    if not any(error.startswith(_LEGACY_ERROR_PREFIX) for error in report.errors):
        return report

    report.errors = [
        error
        for error in report.errors
        if not error.startswith(_LEGACY_ERROR_PREFIX)
    ]
    report.errors.extend(_duplicate_event_frame_errors(Path(truth_path)))
    report.is_valid_for_metrics = not report.errors
    return report


setattr(validated_label_state_allow_multiframe_event_ids, _PATCHED_ATTR, True)
setattr(
    validated_label_state_allow_multiframe_event_ids,
    _ORIGINAL_ATTR,
    _original_validated_label_state,
)
_label_validation.validated_label_state = validated_label_state_allow_multiframe_event_ids

# Keep modules imported before an explicit patch reload consistent as well.
_cli_module = sys.modules.get("beltmap.cli.validate_labels")
if (
    _cli_module is not None
    and getattr(_cli_module, "validated_label_state", None)
    is _original_validated_label_state
):
    _cli_module.validated_label_state = validated_label_state_allow_multiframe_event_ids

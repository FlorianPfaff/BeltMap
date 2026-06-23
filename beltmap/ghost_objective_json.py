from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def first_nested_value(payload: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def flatten_nested_map_only_metrics(data: Mapping[str, Any]) -> dict[str, Any]:
    """Promote nested map-only metrics JSON fields to top-level aliases."""

    flattened = dict(data)
    detections = data.get("detections")
    if isinstance(detections, Mapping):
        value = first_nested_value(
            detections,
            ("false_detections", "map_only_false_detections", "n_false_detections"),
        )
        if value is not None:
            flattened.setdefault("false_detections", value)
            flattened.setdefault("map_only_false_detections", value)

    tracks = data.get("tracks")
    if isinstance(tracks, Mapping):
        value = first_nested_value(
            tracks,
            (
                "false_long_tracks",
                "map_only_false_long_tracks",
                "false_tracks_ge_10",
                "n_false_long_tracks",
            ),
        )
        if value is not None:
            flattened.setdefault("false_long_tracks", value)
            flattened.setdefault("map_only_false_long_tracks", value)

    velocities = data.get("velocities")
    if isinstance(velocities, Mapping):
        value = first_nested_value(
            velocities,
            (
                "false_accepted_tracks",
                "map_only_false_accepted_tracks",
                "n_false_accepted_tracks",
            ),
        )
        if value is not None:
            flattened.setdefault("false_accepted_tracks", value)
            flattened.setdefault("map_only_false_accepted_tracks", value)

    return flattened

"""Keep stitched multi-camera events unique by source camera."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import operational_improvements as _operational

_PATCHED_ATTR = "_beltmap_multicamera_unique_camera_patched"
_ORIGINAL_ATTR = "_beltmap_original_stitch_multicamera_events"


def _unwrap_patched_callable(func: Any) -> Any:
    """Return the original stitcher if this compatibility patch is reloaded."""

    return getattr(func, _ORIGINAL_ATTR, func)


_original_stitch_multicamera_events = _unwrap_patched_callable(
    _operational.stitch_multicamera_events
)


def stitch_multicamera_events_with_unique_cameras(
    rows_by_camera: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    time_tolerance_s: float = 0.05,
    phase_tolerance_px: float = 10.0,
) -> list[_operational.MultiCameraEvent]:
    """Greedily stitch nearby rows while using each camera at most once per event.

    The original implementation considered every unused row compatible solely by
    time and phase. Two nearby detections from the same camera could therefore be
    collapsed into one event, inflating its support and consuming both rows before
    another event could be formed. Track the cameras already represented in each
    event and leave additional same-camera rows available for later events.
    """

    candidates: list[tuple[str, dict[str, Any]]] = []
    for camera, rows in rows_by_camera.items():
        for row in rows:
            enriched = dict(row)
            enriched["camera"] = camera
            candidates.append((camera, enriched))

    used: set[int] = set()
    events: list[_operational.MultiCameraEvent] = []
    for index, (camera, row) in enumerate(candidates):
        if index in used:
            continue

        group = [dict(row)]
        group_cameras = {camera}
        used.add(index)
        time_reference = _operational._finite_float(row.get("time_s"))
        phase_reference = _operational._finite_float(
            row.get("belt_phase_px", row.get("phase_px"))
        )

        for candidate_index, (other_camera, other) in enumerate(candidates):
            if candidate_index in used or other_camera in group_cameras:
                continue
            candidate_time = _operational._finite_float(other.get("time_s"))
            candidate_phase = _operational._finite_float(
                other.get("belt_phase_px", other.get("phase_px"))
            )
            time_ok = (
                time_reference is None
                or candidate_time is None
                or abs(time_reference - candidate_time) <= time_tolerance_s
            )
            phase_ok = (
                phase_reference is None
                or candidate_phase is None
                or abs(phase_reference - candidate_phase) <= phase_tolerance_px
            )
            if time_ok and phase_ok:
                group.append(dict(other))
                group_cameras.add(other_camera)
                used.add(candidate_index)

        times = [_operational._finite_float(item.get("time_s")) for item in group]
        phases = [
            _operational._finite_float(
                item.get("belt_phase_px", item.get("phase_px"))
            )
            for item in group
        ]
        events.append(
            _operational.MultiCameraEvent(
                event_id=len(events),
                camera_rows=tuple(group),
                mean_time_s=_operational._mean_optional(times),
                mean_belt_phase_px=_operational._mean_optional(phases),
            )
        )
    return events


setattr(
    stitch_multicamera_events_with_unique_cameras,
    _PATCHED_ATTR,
    True,
)
setattr(
    stitch_multicamera_events_with_unique_cameras,
    _ORIGINAL_ATTR,
    _original_stitch_multicamera_events,
)
_operational.stitch_multicamera_events = stitch_multicamera_events_with_unique_cameras

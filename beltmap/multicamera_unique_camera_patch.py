from __future__ import annotations

from typing import Any, Mapping, Sequence

from beltmap import operational_improvements as _operational_improvements

_ORIGINAL_ATTR = "_beltmap_original_stitch_multicamera_events"


def _unwrap_patched_callable(func: Any) -> Any:
    return getattr(func, _ORIGINAL_ATTR, func)


_original_stitch_multicamera_events = _unwrap_patched_callable(
    _operational_improvements.stitch_multicamera_events
)


def stitch_multicamera_events(
    rows_by_camera: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    time_tolerance_s: float = 0.05,
    phase_tolerance_px: float = 10.0,
) -> list[_operational_improvements.MultiCameraEvent]:
    """Stitch events while using at most one row from each camera."""

    candidates: list[tuple[str, Mapping[str, Any]]] = []
    for camera, rows in rows_by_camera.items():
        for row in rows:
            enriched = dict(row)
            enriched["camera"] = camera
            candidates.append((camera, enriched))

    used: set[int] = set()
    events: list[_operational_improvements.MultiCameraEvent] = []
    for index, (camera, row) in enumerate(candidates):
        if index in used:
            continue

        group = [dict(row)]
        group_cameras = {camera}
        used.add(index)
        time_i = _operational_improvements._finite_float(row.get("time_s"))
        phase_i = _operational_improvements._finite_float(
            row.get("belt_phase_px", row.get("phase_px"))
        )

        for other_index, (other_camera, other) in enumerate(candidates):
            if other_index in used or other_camera in group_cameras:
                continue

            time_j = _operational_improvements._finite_float(other.get("time_s"))
            phase_j = _operational_improvements._finite_float(
                other.get("belt_phase_px", other.get("phase_px"))
            )
            time_ok = (
                time_i is None
                or time_j is None
                or abs(time_i - time_j) <= time_tolerance_s
            )
            phase_ok = (
                phase_i is None
                or phase_j is None
                or abs(phase_i - phase_j) <= phase_tolerance_px
            )
            if time_ok and phase_ok:
                group.append(dict(other))
                group_cameras.add(other_camera)
                used.add(other_index)

        times = [
            _operational_improvements._finite_float(item.get("time_s"))
            for item in group
        ]
        phases = [
            _operational_improvements._finite_float(
                item.get("belt_phase_px", item.get("phase_px"))
            )
            for item in group
        ]
        events.append(
            _operational_improvements.MultiCameraEvent(
                event_id=len(events),
                camera_rows=tuple(group),
                mean_time_s=_operational_improvements._mean_optional(times),
                mean_belt_phase_px=_operational_improvements._mean_optional(phases),
            )
        )
    return events


setattr(
    stitch_multicamera_events,
    _ORIGINAL_ATTR,
    _original_stitch_multicamera_events,
)
_operational_improvements.stitch_multicamera_events = stitch_multicamera_events

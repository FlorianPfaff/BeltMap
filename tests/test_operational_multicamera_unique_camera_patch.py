from __future__ import annotations

import importlib

import beltmap  # noqa: F401 - imports side-effect patches
from beltmap import operational_improvements as operational
import beltmap.operational_multicamera_unique_camera_patch as patch


def test_multicamera_stitcher_uses_each_camera_at_most_once_per_event() -> None:
    events = operational.stitch_multicamera_events(
        {
            "camera-a": [
                {"row_id": "a1", "time_s": 1.0, "belt_phase_px": 10.0},
                {"row_id": "a2", "time_s": 1.01, "belt_phase_px": 10.2},
            ],
            "camera-b": [
                {"row_id": "b1", "time_s": 1.005, "belt_phase_px": 10.1},
            ],
        },
        time_tolerance_s=0.02,
        phase_tolerance_px=1.0,
    )

    assert [[row["row_id"] for row in event.camera_rows] for event in events] == [
        ["a1", "b1"],
        ["a2"],
    ]
    assert all(
        len({row["camera"] for row in event.camera_rows}) == len(event.camera_rows)
        for event in events
    )


def test_multicamera_unique_camera_patch_is_reload_safe() -> None:
    original = getattr(
        operational.stitch_multicamera_events,
        "_beltmap_original_stitch_multicamera_events",
    )

    importlib.reload(patch)
    importlib.reload(patch)

    stitched = operational.stitch_multicamera_events
    assert getattr(stitched, "_beltmap_original_stitch_multicamera_events") is original

from __future__ import annotations

import importlib

import beltmap.yolo_recurrence as yolo_recurrence
from beltmap.yolo_recurrence_per_frame_patch import recompute_detections_per_frame


def test_recompute_detections_per_frame_reflects_filtered_rows() -> None:
    rows = [
        {"frame_index": "1", "label": "1"},
        {"frame_index": "3", "label": "1"},
        {"frame_index": "3", "label": "2"},
    ]
    per_frame_rows = [
        {"frame_index": "1", "n_detections": "5", "note": "keep"},
        {"frame_index": "2", "n_detections": "7", "note": "keep-zero"},
    ]

    result = recompute_detections_per_frame(rows, per_frame_rows)
    by_frame = {int(row["frame_index"]): row for row in result}

    assert by_frame[1]["n_detections"] == 1
    assert by_frame[1]["note"] == "keep"
    assert by_frame[2]["n_detections"] == 0
    assert by_frame[2]["note"] == "keep-zero"
    assert by_frame[3]["n_detections"] == 2


def test_yolo_recurrence_write_beltmap_run_patch_is_idempotent() -> None:
    before = yolo_recurrence.write_beltmap_run
    before_original = getattr(before, "_beltmap_yolo_recurrence_per_frame_original", before)

    import beltmap.yolo_recurrence_per_frame_patch as per_frame_patch

    importlib.reload(per_frame_patch)

    after = yolo_recurrence.write_beltmap_run
    after_original = getattr(after, "_beltmap_yolo_recurrence_per_frame_original", after)

    assert getattr(after, "_beltmap_yolo_recurrence_per_frame_patched", False)
    assert after_original is before_original
    assert after_original is not after

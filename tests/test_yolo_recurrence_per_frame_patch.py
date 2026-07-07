from __future__ import annotations

import csv
import importlib
from pathlib import Path

import beltmap.yolo_recurrence as yolo_recurrence
from beltmap.yolo_recurrence_per_frame_patch import recompute_detections_per_frame


def read_csv_with_fieldnames(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


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


def test_write_beltmap_run_preserves_auxiliary_per_frame_columns(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    rows = [{"frame_index": "1", "label": "1"}]
    per_frame_rows = [
        {"frame_index": "1", "n_detections": "9", "note": "keep"},
        {"frame_index": "2", "n_detections": "7", "note": "keep-zero"},
    ]

    yolo_recurrence.write_beltmap_run(
        output_dir,
        rows=rows,
        per_frame_rows=per_frame_rows,
        source_run=tmp_path / "source",
        mode="test",
        config=yolo_recurrence.YoloRecurrenceConfig(),
        source_fieldnames=["frame_index", "label"],
    )

    fieldnames, written = read_csv_with_fieldnames(output_dir / "detections_per_frame.csv")
    by_frame = {int(row["frame_index"]): row for row in written}

    assert fieldnames == ["frame_index", "n_detections", "note"]
    assert by_frame[1]["n_detections"] == "1"
    assert by_frame[1]["note"] == "keep"
    assert by_frame[2]["n_detections"] == "0"
    assert by_frame[2]["note"] == "keep-zero"


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

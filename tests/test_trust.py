import csv
import json

import numpy as np
from PIL import Image

from beltmap.trust import (
    confidence_rows,
    edge_audit_rows,
    events_from_tracks,
    plan_map_epochs,
    scale_calibration_from_points,
    sequence_report,
    write_run_trust_artifacts,
)


def test_sequence_report_detects_missing_and_duplicate_frames(tmp_path):
    arr = np.zeros((4, 5), dtype=np.uint8)
    for name in ["frame_000.png", "frame_002.png", "other_002.png"]:
        Image.fromarray(arr).save(tmp_path / name)

    report = sequence_report(tmp_path)

    assert report["n_images"] == 3
    assert report["missing_frame_numbers"] == [1]
    assert 2 in report["duplicate_frame_numbers"]


def test_edge_audit_flags_truncated_detection():
    rows = [
        {
            "frame_index": "0",
            "bbox_top": "0",
            "bbox_left": "4",
            "bbox_bottom": "3",
            "bbox_right": "8",
            "area_px": "8",
        },
        {
            "frame_index": "0",
            "bbox_top": "5",
            "bbox_left": "5",
            "bbox_bottom": "8",
            "bbox_right": "9",
            "area_px": "8",
        },
    ]

    audited = edge_audit_rows(rows, height=10, width=10)

    assert audited[0]["touches_top_edge"] is True
    assert audited[0]["is_truncated"] is True
    assert audited[1]["is_truncated"] is False


def test_events_from_tracks_aggregates_rows():
    rows = [
        {"track_id": "4", "frame_index": "0", "y": "1", "x": "2", "peak_signal": "5"},
        {"track_id": "4", "frame_index": "1", "y": "3", "x": "2", "peak_signal": "7"},
    ]

    events = events_from_tracks(rows)

    assert len(events) == 1
    assert events[0]["track_id"] == 4
    assert events[0]["n_observations"] == 2
    assert events[0]["velocity_y_px_per_frame"] == 2.0


def test_events_from_tracks_accepts_float_formatted_track_ids():
    rows = [
        {"track_id": "4.0", "frame_index": "0", "y": "1", "x": "2"},
        {"track_id": "4.5", "frame_index": "1", "y": "3", "x": "2"},
    ]

    events = events_from_tracks(rows)

    assert len(events) == 1
    assert events[0]["track_id"] == 4
    assert events[0]["n_observations"] == 1


def test_confidence_rows_penalizes_truncation_and_artifacts():
    rows = [
        {"peak_signal": "8", "area_px": "20", "is_truncated": "False"},
        {"peak_signal": "8", "area_px": "20", "is_truncated": "True", "recurrent_artifact_overlap_fraction": "0.5"},
    ]

    scored = confidence_rows(rows, threshold=5.0)

    assert scored[0]["detection_confidence"] > scored[1]["detection_confidence"]


def test_write_run_trust_artifacts_preserves_zero_detection_threshold(tmp_path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "detection_threshold": 0.0,
                "belt_region": {"top": 0, "left": 0, "height": 10, "width": 10},
            }
        ),
        encoding="utf-8",
    )
    (out / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right,area_px,peak_signal\n"
        "0,2,2,4,4,20,0\n",
        encoding="utf-8",
    )

    write_run_trust_artifacts(output_dir=out)

    with (out / "detection_confidence.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert float(rows[0]["detection_confidence"]) == 0.5


def test_plan_map_epochs_with_overlap():
    epochs = plan_map_epochs(100, epoch_count=4, overlap_frames=5)

    assert len(epochs) == 4
    assert epochs[0]["frame_start"] == 0
    assert epochs[1]["train_frame_start"] == 20
    assert epochs[-1]["frame_stop"] == 100


def test_scale_calibration_from_points():
    calibration = scale_calibration_from_points((0, 0), (0, 100), known_distance_mm=50)

    assert calibration.px_per_mm == 2.0
    assert calibration.mm_per_px == 0.5

import csv
import json
from pathlib import Path

import pytest

from beltmap.tracklet_evaluation import (
    evaluate_tracklets,
    generate_tracklet_evaluation_report,
    load_tracklet_predictions,
    load_tracklet_truth,
)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_tracklet_case(tmp_path: Path) -> tuple[Path, Path]:
    output_dir = tmp_path / "outputs"
    label_dir = tmp_path / "labels"
    output_dir.mkdir()
    label_dir.mkdir()

    label_path = label_dir / "tracklets.csv"
    write_csv(
        label_path,
        [
            {
                "tracklet_id": "a",
                "frame_index": 0,
                "bbox_top": 0,
                "bbox_left": 0,
                "bbox_bottom": 10,
                "bbox_right": 10,
            },
            {
                "tracklet_id": "a",
                "frame_index": 1,
                "bbox_top": 1,
                "bbox_left": 0,
                "bbox_bottom": 11,
                "bbox_right": 10,
            },
            {
                "tracklet_id": "a",
                "frame_index": 2,
                "bbox_top": 2,
                "bbox_left": 0,
                "bbox_bottom": 12,
                "bbox_right": 10,
            },
            {
                "tracklet_id": "b",
                "frame_index": 0,
                "bbox_top": 30,
                "bbox_left": 30,
                "bbox_bottom": 40,
                "bbox_right": 40,
            },
            {"tracklet_id": "", "frame_index": 3},
        ],
        [
            "tracklet_id",
            "frame_index",
            "bbox_top",
            "bbox_left",
            "bbox_bottom",
            "bbox_right",
        ],
    )

    track_rows = [
        {
            "track_id": "10",
            "track_detection_index": 0,
            "frame_index": 0,
            "bbox_top": 0,
            "bbox_left": 0,
            "bbox_bottom": 10,
            "bbox_right": 10,
            "y": 4.5,
            "x": 4.5,
        },
        {
            "track_id": "10",
            "track_detection_index": 1,
            "frame_index": 1,
            "bbox_top": 1,
            "bbox_left": 0,
            "bbox_bottom": 11,
            "bbox_right": 10,
            "y": 5.5,
            "x": 4.5,
        },
        {
            "track_id": "11",
            "track_detection_index": 0,
            "frame_index": 2,
            "bbox_top": 2,
            "bbox_left": 0,
            "bbox_bottom": 12,
            "bbox_right": 10,
            "y": 6.5,
            "x": 4.5,
        },
        {
            "track_id": "ghost",
            "track_detection_index": 0,
            "frame_index": 3,
            "bbox_top": 50,
            "bbox_left": 50,
            "bbox_bottom": 60,
            "bbox_right": 60,
            "y": 54.5,
            "x": 54.5,
        },
    ]
    write_csv(
        output_dir / "filtered_tracks.csv",
        track_rows,
        [
            "track_id",
            "track_detection_index",
            "frame_index",
            "bbox_top",
            "bbox_left",
            "bbox_bottom",
            "bbox_right",
            "y",
            "x",
        ],
    )
    return output_dir, label_path


def test_sparse_tracklet_metrics_penalize_ghosts_and_id_switches(tmp_path):
    output_dir, label_path = make_tracklet_case(tmp_path)

    truth = load_tracklet_truth(label_path)
    predictions = load_tracklet_predictions(output_dir / "filtered_tracks.csv")
    metrics, matches, unmatched_truth, unmatched_predictions = evaluate_tracklets(
        truth,
        predictions,
        iou_threshold=0.25,
    )

    assert len(matches) == 3
    assert len(unmatched_truth) == 1
    assert len(unmatched_predictions) == 1
    assert metrics["scored_frames"] == 4
    assert metrics["truth_boxes"] == 4
    assert metrics["predicted_boxes"] == 4
    assert metrics["precision"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["det_a"] == pytest.approx(3 / 5)
    assert metrics["ass_a"] == pytest.approx(5 / 9)
    assert metrics["hota"] == pytest.approx((1 / 3) ** 0.5)
    assert metrics["loc_a"] == pytest.approx(1.0)
    assert metrics["identity_switches"] == 1
    assert metrics["tracklets_with_id_switch"] == 1
    assert metrics["false_positive_tracklets"] == 1
    assert metrics["false_negative_tracklets"] == 1
    assert metrics["false_positives_on_empty_scored_frames"] == 1


def test_tracklet_evaluation_report_writes_artifacts(tmp_path):
    output_dir, label_path = make_tracklet_case(tmp_path)

    artifacts = generate_tracklet_evaluation_report(
        output_dir=output_dir,
        truth_path=label_path,
        iou_threshold=0.25,
    )

    assert artifacts.metrics.is_file()
    assert artifacts.report.is_file()
    assert artifacts.matches.is_file()
    payload = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
    assert payload["tracklet_evaluation"]["type"] == "sparse_real_tracklet"
    assert payload["summary"]["hota"] == pytest.approx((1 / 3) ** 0.5)
    report = artifacts.report.read_text(encoding="utf-8")
    assert "HOTA-style" in report


def test_tracklet_truth_loader_accepts_json_tracklet_container(tmp_path):
    label_path = tmp_path / "tracklets.json"
    label_path.write_text(
        json.dumps(
            {
                "scored_frames": [10, 11, 12],
                "tracklets": [
                    {
                        "tracklet_id": "p01",
                        "boxes": [
                            {"frame_index": 10, "top": 1, "left": 2, "bottom": 5, "right": 8},
                            {"frame_index": 11, "top": 2, "left": 2, "bottom": 6, "right": 8},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    truth = load_tracklet_truth(label_path)

    assert sorted(truth.scored_frames) == [10, 11, 12]
    assert [box.tracklet_id for box in truth.boxes] == ["p01", "p01"]

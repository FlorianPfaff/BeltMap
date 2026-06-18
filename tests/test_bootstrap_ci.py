import pytest

from beltmap.bootstrap_ci import aggregate_labeled_outcomes, labeled_frame_outcomes


def test_labeled_frame_outcomes_restricts_truth_to_scored_frames():
    truth = {
        "particles": [
            {"frame_index": 0, "top": 0, "left": 0, "bottom": 10, "right": 10},
            {"frame_index": 99, "top": 0, "left": 0, "bottom": 10, "right": 10},
        ]
    }
    detections = [
        {
            "frame_index": "0",
            "bbox_top": "0",
            "bbox_left": "0",
            "bbox_bottom": "10",
            "bbox_right": "10",
            "y": "5",
            "x": "5",
        },
        {
            "frame_index": "99",
            "bbox_top": "20",
            "bbox_left": "20",
            "bbox_bottom": "30",
            "bbox_right": "30",
            "y": "25",
            "x": "25",
        },
    ]

    outcomes = labeled_frame_outcomes(
        detections,
        truth,
        scored_frames={0},
        iou_threshold=0.5,
    )
    metrics = aggregate_labeled_outcomes(outcomes)

    assert [outcome.frame_index for outcome in outcomes] == [0]
    assert metrics["labeled_truth_boxes"] == 1
    assert metrics["labeled_predicted_boxes"] == 1
    assert metrics["labeled_false_negatives"] == 0
    assert metrics["labeled_recall"] == pytest.approx(1.0)

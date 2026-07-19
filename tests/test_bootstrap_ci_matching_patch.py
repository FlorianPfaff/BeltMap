from __future__ import annotations

import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.advanced_quality as advanced_quality
import beltmap.bootstrap_ci as bootstrap_ci


def _adversarial_inputs():
    truth = {
        "particles": [
            {"frame_index": 0, "top": 0, "left": 0, "bottom": 10, "right": 10},
            {"frame_index": 0, "top": 1, "left": 0, "bottom": 11, "right": 10},
        ]
    }
    detections = [
        {
            "frame_index": "0",
            "bbox_top": "0",
            "bbox_left": "0",
            "bbox_bottom": "10",
            "bbox_right": "10",
            "y": "4.5",
            "x": "4.5",
        },
        {
            "frame_index": "0",
            "bbox_top": "0",
            "bbox_left": "1",
            "bbox_bottom": "10",
            "bbox_right": "11",
            "y": "4.5",
            "x": "5.5",
        },
    ]
    return truth, detections


def _patch_adversarial_ious(monkeypatch) -> None:
    iou_by_pair = {
        (0, 0): 0.90,
        (0, 1): 0.80,
        (1, 0): 0.85,
        (1, 1): 0.00,
    }

    def fake_iou(truth_box, detection_box):
        return iou_by_pair[
            (int(truth_box["top"]), int(detection_box["left"]))
        ]

    monkeypatch.setattr(advanced_quality, "bbox_iou", fake_iou)


def test_bootstrap_matching_patch_is_autoloaded() -> None:
    assert getattr(
        bootstrap_ci.labeled_frame_outcomes,
        "_beltmap_cardinality_optimal_bootstrap_matching_patched",
        False,
    )


def test_bootstrap_frame_outcomes_maximize_valid_match_cardinality(
    monkeypatch,
) -> None:
    truth, detections = _adversarial_inputs()
    _patch_adversarial_ious(monkeypatch)

    outcomes = bootstrap_ci.labeled_frame_outcomes(
        detections,
        truth,
        scored_frames={0},
        iou_threshold=0.5,
    )
    metrics = bootstrap_ci.aggregate_labeled_outcomes(outcomes)

    assert len(outcomes) == 1
    assert outcomes[0].true_positives == 2
    assert outcomes[0].matched_ious == pytest.approx((0.80, 0.85))
    assert metrics["labeled_precision"] == pytest.approx(1.0)
    assert metrics["labeled_recall"] == pytest.approx(1.0)
    assert metrics["labeled_f1"] == pytest.approx(1.0)


def test_bootstrap_summary_uses_cardinality_optimal_frame_outcomes(
    monkeypatch,
) -> None:
    truth, detections = _adversarial_inputs()
    _patch_adversarial_ious(monkeypatch)

    summary = bootstrap_ci.bootstrap_run_summary(
        detections_per_frame=[],
        detections=detections,
        velocities=[],
        filtered_velocities=[],
        labeled_truth=truth,
        scored_frames={0},
        truth_iou_threshold=0.5,
        samples=8,
        seed=0,
        block_length_frames=1,
    )

    assert summary["labeled_precision_bootstrap_median"] == pytest.approx(1.0)
    assert summary["labeled_recall_bootstrap_median"] == pytest.approx(1.0)
    assert summary["labeled_f1_bootstrap_median"] == pytest.approx(1.0)

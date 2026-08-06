from __future__ import annotations

import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.benchmark as benchmark
import beltmap.bootstrap_ci as bootstrap_ci


def _ambiguous_rows_and_truth():
    detections = [
        {
            "frame_index": "0",
            "bbox_top": "0",
            "bbox_left": "0",
            "bbox_bottom": "10",
            "bbox_right": "10",
        },
        {
            "frame_index": "0",
            "bbox_top": "0",
            "bbox_left": "1",
            "bbox_bottom": "10",
            "bbox_right": "11",
        },
    ]
    truth = {
        "particles": [
            {"frame_index": 0, "top": 0, "left": 0, "bottom": 10, "right": 10},
            {"frame_index": 0, "top": 1, "left": 0, "bottom": 11, "right": 10},
        ]
    }
    return detections, truth


def _install_ambiguous_iou(monkeypatch) -> None:
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

    monkeypatch.setattr(benchmark, "bbox_iou", fake_iou)


def test_benchmark_and_bootstrap_matching_patch_is_autoloaded() -> None:
    assert getattr(
        benchmark.detection_metrics,
        "_beltmap_cardinality_optimal_detection_matching_patched",
        False,
    )
    assert getattr(
        bootstrap_ci.labeled_frame_outcomes,
        "_beltmap_cardinality_optimal_detection_matching_patched",
        False,
    )


def test_benchmark_metrics_maximize_valid_match_cardinality(monkeypatch) -> None:
    detections, truth = _ambiguous_rows_and_truth()
    _install_ambiguous_iou(monkeypatch)

    metrics = benchmark.detection_metrics(
        detections,
        truth,
        iou_threshold=0.5,
        scored_frames={0},
    )

    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["mean_matched_iou"] == pytest.approx((0.80 + 0.85) / 2.0)


def test_bootstrap_frame_outcomes_use_cardinality_optimal_matches(monkeypatch) -> None:
    detections, truth = _ambiguous_rows_and_truth()
    _install_ambiguous_iou(monkeypatch)

    outcomes = bootstrap_ci.labeled_frame_outcomes(
        detections,
        truth,
        scored_frames={0},
        iou_threshold=0.5,
    )

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.true_positives == 2
    assert outcome.false_positives == 0
    assert outcome.false_negatives == 0
    assert sum(outcome.matched_ious) / len(outcome.matched_ious) == pytest.approx(
        (0.80 + 0.85) / 2.0
    )

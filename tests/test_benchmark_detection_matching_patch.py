from __future__ import annotations

import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.advanced_quality as advanced_quality
import beltmap.benchmark as benchmark
import beltmap.compare_runs as compare_runs
import beltmap.texture_stress as texture_stress


def _adversarial_inputs():
    truth = {
        "particles": [
            {
                "frame_index": 0,
                "top": 0,
                "left": 0,
                "bottom": 10,
                "right": 10,
            },
            {
                "frame_index": 0,
                "top": 1,
                "left": 0,
                "bottom": 11,
                "right": 10,
            },
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


def test_benchmark_detection_matching_patch_is_autoloaded() -> None:
    assert getattr(
        benchmark.detection_metrics,
        "_beltmap_cardinality_optimal_benchmark_matching_patched",
        False,
    )
    assert compare_runs.detection_metrics is benchmark.detection_metrics
    assert texture_stress.detection_metrics is benchmark.detection_metrics


def test_benchmark_detection_metrics_maximizes_valid_match_cardinality(
    monkeypatch,
) -> None:
    truth, detections = _adversarial_inputs()
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

    metrics = benchmark.detection_metrics(
        detections,
        truth,
        iou_threshold=0.5,
    )

    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["mean_matched_iou"] == pytest.approx(0.825)

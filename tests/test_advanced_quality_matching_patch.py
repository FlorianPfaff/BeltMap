from __future__ import annotations

import json

import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.advanced_quality as advanced_quality


def test_real_detection_matching_patch_is_autoloaded() -> None:
    assert getattr(
        advanced_quality.evaluate_real_detections,
        "_beltmap_cardinality_optimal_iou_matching_patched",
        False,
    )


def test_real_detection_metrics_maximize_valid_match_cardinality(
    tmp_path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "detections.csv").write_text(
        "frame_index,bbox_top,bbox_left,bbox_bottom,bbox_right\n"
        "0,0,0,10,10\n"
        "0,0,1,10,11\n",
        encoding="utf-8",
    )
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_index": 0,
                        "boxes": [
                            {"top": 0, "left": 0, "bottom": 10, "right": 10},
                            {"top": 1, "left": 0, "bottom": 11, "right": 10},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

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

    metrics = advanced_quality.evaluate_real_detections(
        output_dir,
        labels_path,
        iou_threshold=0.5,
    )

    assert metrics.matches == 2
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.mean_iou == pytest.approx((0.80 + 0.85) / 2.0)

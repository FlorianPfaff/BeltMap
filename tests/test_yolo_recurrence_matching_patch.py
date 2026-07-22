from __future__ import annotations

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.advanced_quality as advanced_quality
import beltmap.yolo_recurrence as yolo_recurrence


def _detection(*, left: int, confidence: float) -> dict[str, str]:
    return {
        "frame_index": "0",
        "label": "0",
        "source": "yolo11",
        "y": "5.0",
        "x": str(left + 5.0),
        "bbox_top": "0",
        "bbox_left": str(left),
        "bbox_bottom": "10",
        "bbox_right": str(left + 10),
        "confidence": str(confidence),
    }


def test_yolo_recurrence_role_matching_patch_is_autoloaded() -> None:
    assert getattr(
        yolo_recurrence.match_detection_roles,
        "_beltmap_yolo_recurrence_cardinality_matching_patched",
        False,
    )


def test_yolo_recurrence_roles_maximize_valid_match_cardinality(
    tmp_path,
    monkeypatch,
) -> None:
    truth = {
        "particles": [
            {"frame_index": 0, "top": 0, "left": 0, "bottom": 10, "right": 10},
            {"frame_index": 0, "top": 1, "left": 0, "bottom": 11, "right": 10},
        ]
    }
    detections = [
        _detection(left=0, confidence=0.9),
        _detection(left=1, confidence=0.8),
    ]
    iou_by_pair = {
        (0, 0): 0.90,
        (0, 1): 0.80,
        (1, 0): 0.85,
        (1, 1): 0.00,
    }

    monkeypatch.setattr(
        yolo_recurrence,
        "load_labeled_detection_truth",
        lambda _path: truth,
    )
    monkeypatch.setattr(
        advanced_quality,
        "bbox_iou",
        lambda truth_box, detection_box: iou_by_pair[
            (int(truth_box["top"]), int(detection_box["left"]))
        ],
    )

    roles = yolo_recurrence.match_detection_roles(
        detections,
        truth_path=tmp_path / "truth.json",
        iou_threshold=0.5,
    )

    assert len(roles) == 2
    assert set(roles.values()) == {"TP"}

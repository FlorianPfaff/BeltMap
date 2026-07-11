from __future__ import annotations

import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.tracklet_evaluation as tracklet_evaluation
from beltmap.tracklet_evaluation import TrackletBox


def _box(
    tracklet_id: str,
    *,
    top: float,
    left: float,
) -> TrackletBox:
    return TrackletBox(
        tracklet_id=tracklet_id,
        frame_index=0,
        top=top,
        left=left,
        bottom=top + 10.0,
        right=left + 10.0,
    )


def test_tracklet_cardinality_matching_patch_is_autoloaded() -> None:
    assert getattr(
        tracklet_evaluation.greedy_frame_matches,
        "_beltmap_tracklet_cardinality_matching_patched",
        False,
    )


def test_tracklet_matching_maximizes_valid_match_cardinality(monkeypatch) -> None:
    truth_boxes = [
        _box("truth-a", top=0.0, left=10.0),
        _box("truth-b", top=1.0, left=10.0),
    ]
    prediction_boxes = [
        _box("prediction-a", top=10.0, left=0.0),
        _box("prediction-b", top=10.0, left=1.0),
    ]
    iou_by_pair = {
        (0, 0): 0.90,
        (0, 1): 0.80,
        (1, 0): 0.85,
        (1, 1): 0.00,
    }

    def fake_iou(prediction, truth):
        return iou_by_pair[(int(truth["top"]), int(prediction["left"]))]

    monkeypatch.setattr(tracklet_evaluation, "bbox_iou", fake_iou)

    matches, unmatched_truth, unmatched_predictions = (
        tracklet_evaluation.greedy_frame_matches(
            truth_boxes,
            prediction_boxes,
            scored_frames={0},
            iou_threshold=0.5,
        )
    )

    assert {(match.truth_index, match.prediction_index) for match in matches} == {
        (0, 1),
        (1, 0),
    }
    assert sum(match.iou for match in matches) == pytest.approx(1.65)
    assert unmatched_truth == set()
    assert unmatched_predictions == set()

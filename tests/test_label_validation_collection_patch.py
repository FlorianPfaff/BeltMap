from __future__ import annotations

import importlib
import json

import pytest

from beltmap import label_validation
from beltmap import label_validation_collection_patch


def _reviewed_empty_payload() -> dict[str, object]:
    return {
        "status": "reviewed_ground_truth",
        "requires_manual_review": False,
        "scored_frames": [0],
        "empty_frames": [0],
    }


@pytest.mark.parametrize(
    "key",
    [
        "particles",
        "annotations",
        "labels",
        "detections",
        "frame_reviews",
        "review_frames",
    ],
)
def test_validated_label_state_rejects_non_list_collections(tmp_path, key):
    truth_path = tmp_path / "labels.json"
    payload = _reviewed_empty_payload()
    payload[key] = {"frame_index": 0}
    truth_path.write_text(json.dumps(payload), encoding="utf-8")

    report = label_validation.validated_label_state(truth_path)

    assert report.is_valid_for_metrics is False
    assert f"{key} must be a list" in report.errors


def test_label_collection_patch_reload_is_idempotent(tmp_path):
    truth_path = tmp_path / "labels.json"
    payload = _reviewed_empty_payload()
    payload["particles"] = None
    truth_path.write_text(json.dumps(payload), encoding="utf-8")

    importlib.reload(label_validation_collection_patch)
    importlib.reload(label_validation_collection_patch)

    report = label_validation.validated_label_state(truth_path)

    assert report.is_valid_for_metrics is False
    assert report.errors.count("particles must be a list") == 1

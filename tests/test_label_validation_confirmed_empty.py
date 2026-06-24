from __future__ import annotations

import json

from beltmap.label_validation import validated_label_state


def test_confirmed_empty_review_without_status_is_metric_ready(tmp_path):
    truth_path = tmp_path / "labels.json"
    truth_path.write_text(
        json.dumps(
            {
                "status": "reviewed_ground_truth",
                "requires_manual_review": False,
                "scored_frames": [7],
                "frame_reviews": [
                    {
                        "frame_index": 7,
                        "confirmed_empty": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = validated_label_state(truth_path)

    assert report.n_needs_review == 0
    assert report.n_reviewed_empty == 1
    assert report.n_empty_frames == 1
    assert report.is_valid_for_metrics is True
    assert report.errors == []

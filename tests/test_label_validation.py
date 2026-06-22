from __future__ import annotations

import json

from beltmap.cli import validate_labels as cli_validate_labels
from beltmap.label_validation import validated_label_state


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def reviewed_payload():
    return {
        "status": "reviewed_ground_truth",
        "requires_manual_review": False,
        "scored_frames": [0, 1],
        "empty_frames": [1],
        "particles": [
            {
                "frame_index": 0,
                "top": 1,
                "left": 2,
                "bottom": 5,
                "right": 7,
                "event_id": "p0",
            }
        ],
        "frame_reviews": [
            {
                "frame_index": 0,
                "review_status": "reviewed_with_particles",
                "confirmed_empty": False,
            },
            {
                "frame_index": 1,
                "review_status": "reviewed_empty",
                "confirmed_empty": True,
            },
        ],
    }


def test_validated_label_state_accepts_reviewed_truth(tmp_path):
    truth_path = tmp_path / "labels.json"
    write_json(truth_path, reviewed_payload())

    report = validated_label_state(truth_path)

    assert report.is_valid_for_metrics is True
    assert report.status == "reviewed_ground_truth"
    assert report.requires_manual_review is False
    assert report.n_scored_frames == 2
    assert report.n_particle_boxes == 1
    assert report.n_empty_frames == 1
    assert report.n_needs_review == 0
    assert report.errors == []


def test_validated_label_state_rejects_unreviewed_truth(tmp_path):
    truth_path = tmp_path / "labels.json"
    write_json(
        truth_path,
        {
            "status": "needs_review",
            "requires_manual_review": True,
            "scored_frames": [0],
            "particles": [],
            "frame_reviews": [
                {"frame_index": 0, "review_status": "needs_review"},
            ],
        },
    )

    report = validated_label_state(truth_path)

    assert report.is_valid_for_metrics is False
    assert report.n_scored_frames == 1
    assert report.n_particle_boxes == 0
    assert report.n_needs_review == 1
    assert any("status must be" in message for message in report.errors)
    assert any("requires_manual_review" in message for message in report.errors)
    assert any("needs_review" in message for message in report.errors)


def test_validated_label_state_rejects_unaccounted_scored_frame(tmp_path):
    truth_path = tmp_path / "labels.json"
    payload = reviewed_payload()
    payload["scored_frames"] = [0, 1, 2]
    write_json(truth_path, payload)

    report = validated_label_state(truth_path)

    assert report.is_valid_for_metrics is False
    assert report.n_unaccounted_scored_frames == 1
    assert any("scored frame" in message for message in report.errors)


def test_validated_label_state_rejects_empty_frame_with_particle(tmp_path):
    truth_path = tmp_path / "labels.json"
    payload = reviewed_payload()
    payload["empty_frames"] = [0, 1]
    write_json(truth_path, payload)

    report = validated_label_state(truth_path)

    assert report.is_valid_for_metrics is False
    assert any("marked empty" in message for message in report.errors)


def test_validate_labels_cli_text_and_json(tmp_path, capsys):
    truth_path = tmp_path / "labels.json"
    write_json(truth_path, reviewed_payload())

    assert cli_validate_labels.main(["--truth-path", str(truth_path)]) == 0
    text_out = capsys.readouterr().out
    assert "is_valid_for_metrics: True" in text_out
    assert "particle_boxes: 1" in text_out

    assert cli_validate_labels.main(["--truth-path", str(truth_path), "--format", "json"]) == 0
    json_out = json.loads(capsys.readouterr().out)
    assert json_out["is_valid_for_metrics"] is True
    assert json_out["n_empty_frames"] == 1


def test_validate_labels_cli_returns_nonzero_for_invalid(tmp_path, capsys):
    truth_path = tmp_path / "labels.json"
    write_json(
        truth_path,
        {
            "status": "template_not_ground_truth_do_not_use_for_metrics_until_filled",
            "requires_manual_review": True,
            "scored_frames": [0],
            "particles": [],
        },
    )

    assert cli_validate_labels.main(["--truth-path", str(truth_path)]) == 1
    assert "is_valid_for_metrics: False" in capsys.readouterr().out
    assert cli_validate_labels.main(["--truth-path", str(truth_path), "--allow-invalid"]) == 0

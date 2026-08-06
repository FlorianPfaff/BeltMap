from __future__ import annotations

import json
from typing import Any

from beltmap import label_validation
from beltmap.cli import validate_labels as cli_validate_labels


def _particle(frame_index: int, *, event_id: str, left: int = 0) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "top": 0,
        "left": left,
        "bottom": 4,
        "right": left + 4,
        "event_id": event_id,
    }


def _reviewed_payload(particles: list[dict[str, Any]]) -> dict[str, Any]:
    scored_frames = sorted({int(particle["frame_index"]) for particle in particles})
    return {
        "status": "reviewed_ground_truth",
        "requires_manual_review": False,
        "scored_frames": scored_frames,
        "particles": particles,
        "frame_reviews": [
            {
                "frame_index": frame_index,
                "review_status": "reviewed_with_particles",
                "confirmed_empty": False,
            }
            for frame_index in scored_frames
        ],
    }


def test_multiframe_event_id_validation_patch_is_autoloaded() -> None:
    assert getattr(
        label_validation.validated_label_state,
        "_beltmap_multiframe_event_id_validation_patched",
        False,
    )
    assert cli_validate_labels.validated_label_state is label_validation.validated_label_state


def test_validator_allows_one_event_id_across_frames(tmp_path) -> None:
    truth_path = tmp_path / "labels.json"
    truth_path.write_text(
        json.dumps(
            _reviewed_payload(
                [
                    _particle(0, event_id="particle-1"),
                    _particle(1, event_id="particle-1", left=1),
                ]
            )
        ),
        encoding="utf-8",
    )

    report = label_validation.validated_label_state(truth_path)

    assert report.is_valid_for_metrics is True
    assert report.errors == []
    assert report.n_particle_boxes == 2


def test_validator_rejects_duplicate_event_id_within_frame(tmp_path) -> None:
    truth_path = tmp_path / "labels.json"
    truth_path.write_text(
        json.dumps(
            _reviewed_payload(
                [
                    _particle(0, event_id="particle-1"),
                    _particle(0, event_id="particle-1", left=10),
                ]
            )
        ),
        encoding="utf-8",
    )

    report = label_validation.validated_label_state(truth_path)

    assert report.is_valid_for_metrics is False
    assert any(
        "duplicate event_id/frame_index pairs: particle-1@0" in error
        for error in report.errors
    )

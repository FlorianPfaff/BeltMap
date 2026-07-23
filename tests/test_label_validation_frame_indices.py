from __future__ import annotations

import importlib
import json

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.label_validation as label_validation
import beltmap.label_validation_frame_index_patch as frame_patch


def _particle(frame_index) -> dict[str, object]:
    return {
        "frame_index": frame_index,
        "top": 1,
        "left": 2,
        "bottom": 5,
        "right": 7,
        "event_id": "particle-0",
    }


def _reviewed_payload(frame_index) -> dict[str, object]:
    return {
        "status": "reviewed_ground_truth",
        "requires_manual_review": False,
        "scored_frames": [frame_index],
        "particles": [_particle(frame_index)],
        "frame_reviews": [
            {
                "frame_index": frame_index,
                "review_status": "reviewed_with_particles",
                "confirmed_empty": False,
            }
        ],
    }


def test_label_frame_index_patch_is_autoloaded() -> None:
    assert getattr(
        label_validation.finite_int,
        "_beltmap_nonnegative_label_frame_indices_patched",
        False,
    )


def test_reviewed_truth_rejects_negative_frame_indices(tmp_path) -> None:
    truth_path = tmp_path / "labels.json"
    truth_path.write_text(
        json.dumps(_reviewed_payload(-1)),
        encoding="utf-8",
    )

    report = label_validation.validated_label_state(truth_path)

    assert report.is_valid_for_metrics is False
    assert report.n_scored_frames == 0
    assert report.n_particle_boxes == 0
    assert any(
        "scored_frames[0] has no valid frame index" in error
        for error in report.errors
    )
    assert any(
        "particle row 0 has no valid frame_index" in error
        for error in report.errors
    )
    assert any(
        "frame review row 0 has no valid frame_index" in error
        for error in report.errors
    )


def test_label_frame_indices_preserve_nonnegative_integer_values(tmp_path) -> None:
    truth_path = tmp_path / "labels.json"
    truth_path.write_text(
        json.dumps(_reviewed_payload(1.0)),
        encoding="utf-8",
    )

    report = label_validation.validated_label_state(truth_path)

    assert report.is_valid_for_metrics is True
    assert report.n_scored_frames == 1
    assert report.n_particle_boxes == 1
    assert report.errors == []


def test_label_frame_index_patch_reload_keeps_true_original() -> None:
    before = label_validation.finite_int
    before_original = getattr(
        before,
        "_beltmap_original_label_validation_finite_int",
        before,
    )

    importlib.reload(frame_patch)
    importlib.reload(frame_patch)

    after = label_validation.finite_int
    after_original = getattr(
        after,
        "_beltmap_original_label_validation_finite_int",
        after,
    )
    assert after_original is before_original
    assert after(-1) is None
    assert after(0) == 0

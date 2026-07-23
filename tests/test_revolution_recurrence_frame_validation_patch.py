from __future__ import annotations

import importlib

import numpy as np
import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.revolution_recurrence as revolution_recurrence
import beltmap.revolution_recurrence_frame_validation_patch as frame_patch
from beltmap.tracking import ParticleDetection, ParticleTrack


def _track(frame_index) -> ParticleTrack:
    return ParticleTrack(
        track_id=7,
        detections=(
            ParticleDetection(
                frame_index=frame_index,
                label=1,
                y=10.0,
                x=5.0,
                area_px=25,
                bbox_top=8,
                bbox_left=3,
                bbox_bottom=13,
                bbox_right=8,
            ),
        ),
    )


def _score(frame_index):
    return revolution_recurrence.score_belt_revolution_track_recurrence(
        [_track(frame_index)],
        phase_px_by_frame=[0.0, 0.0, 0.0],
        revolution_by_frame=[0, 1, 2],
        frame_height_px=20.0,
        map_height_px=100.0,
        config=revolution_recurrence.BeltRevolutionRecurrenceConfig(
            min_track_detections=1
        ),
    )


def test_revolution_recurrence_frame_validation_patch_is_autoloaded() -> None:
    assert getattr(
        revolution_recurrence.score_belt_revolution_track_recurrence,
        "_beltmap_revolution_recurrence_frame_validation_patched",
        False,
    )


@pytest.mark.parametrize(
    "frame_index",
    [0.5, -1.0, float("nan"), float("inf"), True, np.bool_(False), "bad"],
)
def test_revolution_recurrence_rejects_invalid_detection_frame_indices(
    frame_index,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"detection frame_index must be a finite non-negative integer; "
            r"track 7, detection 0"
        ),
    ):
        _score(frame_index)


def test_revolution_recurrence_accepts_integer_valued_float_frame_indices() -> None:
    scores = _score(1.0)

    assert len(scores) == 1
    assert scores[0].track_id == 7
    assert scores[0].frame_start == pytest.approx(1.0)


def test_revolution_recurrence_frame_validation_patch_reload_is_idempotent() -> None:
    before = revolution_recurrence.score_belt_revolution_track_recurrence
    before_original = getattr(
        before,
        "_beltmap_original_score_belt_revolution_track_recurrence",
        before,
    )

    importlib.reload(frame_patch)
    importlib.reload(frame_patch)

    after = revolution_recurrence.score_belt_revolution_track_recurrence
    after_original = getattr(
        after,
        "_beltmap_original_score_belt_revolution_track_recurrence",
        after,
    )
    assert getattr(
        after,
        "_beltmap_revolution_recurrence_frame_validation_patched",
        False,
    )
    assert after_original is before_original

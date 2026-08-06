from __future__ import annotations

from beltmap import revolution_recurrence
from beltmap import revolution_recurrence_circular_mean_patch as patch
from beltmap.revolution_recurrence import BeltRevolutionRecurrenceConfig
from beltmap.revolution_recurrence import score_belt_revolution_track_recurrence
from beltmap.tracking import ParticleDetection
from beltmap.tracking import ParticleTrack


def _detection(frame: int, *, y: float, x: float) -> ParticleDetection:
    return ParticleDetection(
        frame_index=float(frame),
        label=1,
        y=y,
        x=x,
        area_px=25,
        bbox_top=int(y),
        bbox_left=int(x),
        bbox_bottom=int(y) + 1,
        bbox_right=int(x) + 1,
    )


def _track(track_id: int, *detections: ParticleDetection) -> ParticleTrack:
    return ParticleTrack(track_id=track_id, detections=tuple(detections))


def test_undefined_circular_mean_patch_is_autoloaded() -> None:
    assert revolution_recurrence.circular_mean is patch.stable_circular_mean
    assert getattr(
        revolution_recurrence.circular_mean,
        "_beltmap_undefined_circular_mean_patched",
        False,
    )


def test_circular_mean_is_missing_for_antipodal_coordinates() -> None:
    assert revolution_recurrence.circular_mean([0.0, 50.0], 100.0) is None


def test_recurrence_does_not_use_arbitrary_antipodal_track_center() -> None:
    candidate = _track(
        0,
        _detection(0, y=0.0, x=5.0),
        _detection(1, y=0.0, x=5.0),
    )
    recurring_1 = _track(1, _detection(2, y=25.0, x=5.0))
    recurring_2 = _track(2, _detection(3, y=25.0, x=5.0))

    scores = score_belt_revolution_track_recurrence(
        [candidate, recurring_1, recurring_2],
        phase_px_by_frame=[0.0, 50.0, 0.0, 0.0],
        revolution_by_frame=[0, 1, 2, 3],
        frame_height_px=40.0,
        map_height_px=100.0,
        config=BeltRevolutionRecurrenceConfig(
            radius_y_px=1.0,
            radius_x_px=1.0,
            min_track_detections=2,
            min_other_revolutions=2,
            min_other_detections=2,
            min_recurrence_fraction=1.0,
        ),
    )

    candidate_score = scores[0]
    assert candidate_score.belt_y_center_px is None
    assert candidate_score.runtime_recurrence_rejected is False
    assert candidate_score.other_hit_revolutions == 0
    assert candidate_score.causal_read == "track has no belt-coordinate center"

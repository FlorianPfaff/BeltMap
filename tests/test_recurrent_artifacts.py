import numpy as np
import pytest

from beltmap import (
    BeltMotionModel,
    ParticleDetection,
    RecurrentArtifactConfig,
    belt_revolution_indices,
    build_recurrent_artifact_map,
    filter_recurrent_artifact_detections,
    score_recurrent_artifact_detections,
    score_recurrent_artifact_detections_excluding_current_revolution,
)


def detection(frame_index, top, left, bottom, right, *, peak_signal=None):
    return ParticleDetection(
        frame_index=float(frame_index),
        label=1,
        y=(top + bottom) / 2,
        x=(left + right) / 2,
        area_px=(bottom - top) * (right - left),
        bbox_top=top,
        bbox_left=left,
        bbox_bottom=bottom,
        bbox_right=right,
        peak_signal=peak_signal,
    )


def test_belt_revolution_indices_follow_motion_model_distance():
    indices = belt_revolution_indices(
        8,
        BeltMotionModel(
            image_velocity_px_per_frame=3.0,
            period_px=10.0,
        ),
    )

    np.testing.assert_array_equal(indices, [0, 0, 0, 0, 1, 1, 1, 2])


@pytest.mark.parametrize("period_px", [None, 0.0, -1.0, np.nan, np.inf])
def test_belt_revolution_indices_rejects_invalid_period(period_px):
    with pytest.raises(ValueError, match="period"):
        belt_revolution_indices(
            3,
            BeltMotionModel(
                image_velocity_px_per_frame=3.0,
                period_px=period_px,
            ),
        )


def test_recurrent_artifact_map_counts_distinct_revolutions_only():
    recurrent = detection(0, 1, 2, 3, 4)
    same_revolution_duplicate = detection(1, 1, 2, 3, 4)
    next_revolution = detection(2, 1, 2, 3, 4)
    one_off = detection(3, 6, 2, 8, 4)
    detections_by_frame = [
        [recurrent],
        [same_revolution_duplicate],
        [next_revolution],
        [one_off],
    ]

    result = build_recurrent_artifact_map(
        detections_by_frame,
        phase_px_by_frame=[0.0, 0.0, 0.0, 0.0],
        revolution_by_frame=[0, 0, 1, 1],
        map_shape=(12, 12),
        config=RecurrentArtifactConfig(
            min_revolutions=2,
            margin_px=0,
            max_overlap_fraction=0.5,
        ),
    )

    assert result.revolution_count == 2
    assert result.candidate_detections == 4
    assert result.artifact_pixels == 4
    assert result.exposure_counts[1:3, 2:4].max() == 2
    assert result.probability[1:3, 2:4].max() == 1.0
    assert np.issubdtype(result.counts.dtype, np.unsignedinteger)
    assert result.counts.itemsize > np.dtype(np.uint16).itemsize
    assert result.counts[1:3, 2:4].max() == 2
    assert result.counts[6:8, 2:4].max() == 1

    filtered, rejected = filter_recurrent_artifact_detections(
        detections_by_frame,
        phase_px_by_frame=[0.0, 0.0, 0.0, 0.0],
        artifact_map=result.mask,
        config=RecurrentArtifactConfig(
            min_revolutions=2,
            margin_px=0,
            max_overlap_fraction=0.5,
        ),
    )

    assert rejected == 3
    assert [len(frame) for frame in filtered] == [0, 0, 0, 1]


def test_recurrent_artifact_map_normalizes_by_visible_revolutions():
    one_hit = detection(0, 0, 1, 1, 2)

    result = build_recurrent_artifact_map(
        [[one_hit], []],
        phase_px_by_frame=[0.0, 0.0],
        revolution_by_frame=[0, 1],
        map_shape=(4, 4),
        frame_shape=(2, 4),
        config=RecurrentArtifactConfig(
            min_revolutions=1,
            margin_px=0,
            max_overlap_fraction=0.3,
            min_recurrence_probability=0.75,
        ),
    )

    assert result.counts[0, 1] == 1
    assert result.exposure_counts[0, 1] == 2
    assert result.probability[0, 1] == 0.5
    assert not result.mask[0, 1]


def test_built_recurrent_scores_do_not_use_current_revolution_as_evidence():
    one_off = detection(0, 1, 2, 3, 4)
    detections_by_frame = [[one_off]]
    phase_px_by_frame = [0.0]
    revolution_by_frame = [0]

    result = build_recurrent_artifact_map(
        detections_by_frame,
        phase_px_by_frame=phase_px_by_frame,
        revolution_by_frame=revolution_by_frame,
        map_shape=(12, 12),
        config=RecurrentArtifactConfig(
            min_revolutions=1,
            margin_px=0,
            max_overlap_fraction=0.5,
        ),
    )
    scores = score_recurrent_artifact_detections_excluding_current_revolution(
        detections_by_frame,
        phase_px_by_frame,
        revolution_by_frame,
        result,
        config=RecurrentArtifactConfig(
            min_revolutions=1,
            margin_px=0,
            max_overlap_fraction=0.5,
        ),
    )

    assert result.artifact_pixels == 4
    assert scores[0][0].overlap_fraction == 0.0
    assert not scores[0][0].rejected


def test_built_recurrent_scores_reject_cross_revolution_recurrence():
    recurrent_first = detection(0, 1, 2, 3, 4)
    recurrent_second = detection(1, 1, 2, 3, 4)
    one_off = detection(2, 6, 2, 8, 4)
    detections_by_frame = [[recurrent_first], [recurrent_second], [one_off]]
    phase_px_by_frame = [0.0, 0.0, 0.0]
    revolution_by_frame = [0, 1, 1]

    result = build_recurrent_artifact_map(
        detections_by_frame,
        phase_px_by_frame=phase_px_by_frame,
        revolution_by_frame=revolution_by_frame,
        map_shape=(12, 12),
        config=RecurrentArtifactConfig(
            min_revolutions=2,
            margin_px=0,
            max_overlap_fraction=0.5,
        ),
    )
    scores = score_recurrent_artifact_detections_excluding_current_revolution(
        detections_by_frame,
        phase_px_by_frame,
        revolution_by_frame,
        result,
        config=RecurrentArtifactConfig(
            min_revolutions=2,
            margin_px=0,
            max_overlap_fraction=0.5,
        ),
    )

    assert [[score.rejected for score in frame] for frame in scores] == [
        [True],
        [True],
        [False],
    ]
    assert [score.overlap_fraction for frame in scores for score in frame] == [
        1.0,
        1.0,
        0.0,
    ]


def test_soft_recurrent_filter_keeps_strong_peak_and_rejects_weak_peak():
    artifact_map = np.zeros((12, 12), dtype=bool)
    artifact_map[1:3, 2:4] = True
    weak_recurring = detection(0, 1, 2, 3, 4, peak_signal=6.0)
    strong_recurring = detection(0, 1, 2, 3, 4, peak_signal=12.0)
    off_artifact = detection(0, 6, 2, 8, 4, peak_signal=6.0)

    filtered, rejected = filter_recurrent_artifact_detections(
        [[weak_recurring, strong_recurring, off_artifact]],
        phase_px_by_frame=[0.0],
        artifact_map=artifact_map,
        config=RecurrentArtifactConfig(
            min_revolutions=2,
            margin_px=0,
            max_overlap_fraction=0.3,
            mode="soft",
            soft_penalty_weight=1.0,
        ),
        detection_threshold=5.0,
    )

    assert rejected == 1
    assert [item.peak_signal for item in filtered[0]] == [12.0, 6.0]
    assert filtered[0][0].recurrent_artifact_overlap_fraction == 1.0
    assert filtered[0][0].recurrent_artifact_probability == 1.0
    assert filtered[0][0].recurrent_artifact_required_peak_signal == 10.0
    assert filtered[0][1].recurrent_artifact_overlap_fraction == 0.0
    assert filtered[0][1].recurrent_artifact_required_peak_signal == 5.0


def test_recurrent_artifact_scores_include_rejected_candidates():
    artifact_map = np.zeros((12, 12), dtype=bool)
    artifact_map[1:3, 2:4] = True
    weak_recurring = detection(0, 1, 2, 3, 4, peak_signal=6.0)
    strong_recurring = detection(0, 1, 2, 3, 4, peak_signal=12.0)

    scores = score_recurrent_artifact_detections(
        [[weak_recurring, strong_recurring]],
        phase_px_by_frame=[0.0],
        artifact_map=artifact_map,
        config=RecurrentArtifactConfig(
            min_revolutions=0,
            margin_px=0,
            max_overlap_fraction=0.3,
            mode="soft",
            soft_penalty_weight=1.0,
        ),
        detection_threshold=5.0,
    )

    assert [score.rejected for score in scores[0]] == [True, False]
    assert [score.overlap_fraction for score in scores[0]] == [1.0, 1.0]
    assert [score.artifact_probability for score in scores[0]] == [1.0, 1.0]
    assert [score.required_peak_signal for score in scores[0]] == [10.0, 10.0]
    assert scores[0][0].detection.recurrent_artifact_overlap_fraction == 1.0


def test_filter_accepts_config_without_build_min_revolutions_for_reused_map():
    artifact_map = np.ones((4, 4), dtype=bool)

    filtered, rejected = filter_recurrent_artifact_detections(
        [[detection(0, 1, 1, 3, 3)]],
        phase_px_by_frame=[0.0],
        artifact_map=artifact_map,
        config=RecurrentArtifactConfig(
            min_revolutions=0,
            margin_px=0,
            max_overlap_fraction=0.3,
        ),
    )

    assert rejected == 1
    assert filtered == [[]]


def test_probabilistic_recurrent_filter_uses_probability_prior():
    artifact_prior = np.zeros((12, 12), dtype=np.float32)
    artifact_prior[1:3, 2:4] = 0.4
    weak_recurring = detection(0, 1, 2, 3, 4, peak_signal=6.0)
    strong_recurring = detection(0, 1, 2, 3, 4, peak_signal=8.0)

    filtered, rejected = filter_recurrent_artifact_detections(
        [[weak_recurring, strong_recurring]],
        phase_px_by_frame=[0.0],
        artifact_map=artifact_prior,
        config=RecurrentArtifactConfig(
            min_revolutions=0,
            margin_px=0,
            max_overlap_fraction=0.3,
            mode="probabilistic",
            soft_penalty_weight=1.0,
        ),
        detection_threshold=5.0,
    )

    assert rejected == 1
    assert [item.peak_signal for item in filtered[0]] == [8.0]
    assert np.isclose(filtered[0][0].recurrent_artifact_probability, 0.4)
    assert np.isclose(filtered[0][0].recurrent_artifact_required_peak_signal, 7.0)

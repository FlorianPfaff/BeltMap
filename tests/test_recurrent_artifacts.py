import numpy as np

from beltmap import (
    BeltMotionModel,
    ParticleDetection,
    RecurrentArtifactConfig,
    belt_revolution_indices,
    build_recurrent_artifact_map,
    filter_recurrent_artifact_detections,
)


def detection(frame_index, top, left, bottom, right):
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
    assert result.counts[1:3, 2:4].max() == 2
    assert result.counts[6:8, 2:4].max() == 1

    filtered, rejected = filter_recurrent_artifact_detections(
        detections_by_frame,
        phase_px_by_frame=[0.0, 0.0, 0.0, 0.0],
        artifact_map=result.mask,
        max_overlap_fraction=0.5,
    )

    assert rejected == 3
    assert [len(frame) for frame in filtered] == [0, 0, 0, 1]

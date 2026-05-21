from __future__ import annotations

import math

import numpy as np

from beltmap.tracking import (
    ParticleComponentConfig,
    ParticleDetection,
    ParticleTrack,
    estimate_particle_velocities_vs_belt,
    extract_particle_detections,
)


def test_projection_valley_split_breaks_weak_bridge() -> None:
    mask = np.zeros((9, 11), dtype=bool)
    mask[2:5, 1:4] = True
    mask[2:5, 7:10] = True
    mask[3, 4:7] = True  # one-pixel bridge that would otherwise merge blobs

    unsplit = extract_particle_detections(
        mask,
        config=ParticleComponentConfig(min_area_px=1),
    )
    assert len(unsplit) == 1

    split = extract_particle_detections(
        mask,
        config=ParticleComponentConfig(
            min_area_px=1,
            split_merged_components=True,
            split_min_projection_gap_px=1,
            split_min_component_area_px=4,
        ),
    )
    assert len(split) == 2
    assert sorted(d.area_px for d in split) == [9, 9]


def test_theil_sen_velocity_fit_rejects_one_centroid_outlier() -> None:
    track = ParticleTrack(
        track_id=0,
        detections=(
            ParticleDetection(frame_index=0, label=1, y=0, x=0, area_px=4, bbox_top=0, bbox_left=0, bbox_bottom=2, bbox_right=2),
            ParticleDetection(frame_index=1, label=1, y=1, x=1, area_px=4, bbox_top=1, bbox_left=1, bbox_bottom=3, bbox_right=3),
            ParticleDetection(frame_index=2, label=1, y=100, x=100, area_px=4, bbox_top=2, bbox_left=2, bbox_bottom=4, bbox_right=4),
            ParticleDetection(frame_index=3, label=1, y=3, x=3, area_px=4, bbox_top=3, bbox_left=3, bbox_bottom=5, bbox_right=5),
        ),
    )

    robust = estimate_particle_velocities_vs_belt(
        [track],
        belt_image_velocity_px_per_frame=2.0,
        fit_method="theil_sen",
    )
    ordinary = estimate_particle_velocities_vs_belt(
        [track],
        belt_image_velocity_px_per_frame=2.0,
        fit_method="linear",
    )

    assert len(robust) == 1
    assert math.isclose(robust[0].velocity_y_px_per_frame, 1.0)
    assert ordinary[0].velocity_y_px_per_frame > 1.0

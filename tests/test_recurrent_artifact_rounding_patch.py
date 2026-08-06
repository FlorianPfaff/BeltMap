import importlib

import numpy as np
import pytest

from beltmap import ParticleDetection
import beltmap.recurrent_artifact_rounding_patch as rounding_patch
import beltmap.recurrent_artifacts as recurrent_artifacts


def detection(top: int, bottom: int) -> ParticleDetection:
    return ParticleDetection(
        frame_index=0.0,
        label=1,
        y=(top + bottom) / 2,
        x=0.5,
        area_px=bottom - top,
        bbox_top=top,
        bbox_left=0,
        bbox_bottom=bottom,
        bbox_right=1,
    )


@pytest.mark.parametrize(
    ("phase_px", "expected"),
    [
        (0.5, [1, 2, 3, 4]),
        (-0.5, [0, 1, 2, 3]),
    ],
)
def test_half_pixel_projection_preserves_consecutive_belt_rows(phase_px, expected):
    rows = recurrent_artifacts._belt_rows_for_image_rows(
        range(4),
        phase_px=phase_px,
        map_height=8,
    )

    np.testing.assert_array_equal(rows, expected)


def test_half_pixel_projection_keeps_overlap_scoring_consistent():
    artifact_map = np.zeros((8, 1), dtype=bool)
    artifact_map[[1, 3], 0] = True

    overlap = recurrent_artifacts.detection_artifact_overlap_fraction(
        detection(0, 4),
        phase_px=0.5,
        artifact_map=artifact_map,
    )

    assert overlap == pytest.approx(0.5)


def test_rounding_patch_reload_remains_idempotent():
    importlib.reload(rounding_patch)

    rows = recurrent_artifacts._belt_rows_for_image_rows(
        range(4),
        phase_px=0.5,
        map_height=8,
    )

    np.testing.assert_array_equal(rows, [1, 2, 3, 4])

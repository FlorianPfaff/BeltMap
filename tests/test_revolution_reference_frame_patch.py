import importlib

import numpy as np

import beltmap
import beltmap.recurrent_artifacts as recurrent_artifacts
from beltmap import BeltMotionModel


def test_belt_revolution_indices_are_invariant_to_phase_reference_frame():
    expected = np.asarray([0, 0, 0, 0, 1, 1, 1, 2])
    anchored_at_start = BeltMotionModel(
        image_velocity_px_per_frame=3.0,
        period_px=10.0,
        reference_frame=0.0,
        reference_phase_px=2.0,
    )
    anchored_later = BeltMotionModel(
        image_velocity_px_per_frame=3.0,
        period_px=10.0,
        reference_frame=4.0,
        reference_phase_px=14.0,
    )

    np.testing.assert_array_equal(
        beltmap.belt_revolution_indices(8, anchored_at_start),
        expected,
    )
    np.testing.assert_array_equal(
        beltmap.belt_revolution_indices(8, anchored_later),
        expected,
    )


def test_revolution_reference_patch_keeps_direct_and_package_exports_in_sync():
    import beltmap.revolution_reference_frame_patch as patch

    importlib.reload(patch)

    assert recurrent_artifacts.belt_revolution_indices is beltmap.belt_revolution_indices
    np.testing.assert_array_equal(
        recurrent_artifacts.belt_revolution_indices(
            5,
            BeltMotionModel(
                image_velocity_px_per_frame=-2.5,
                period_px=5.0,
                reference_frame=3.0,
            ),
        ),
        [0, 0, 1, 1, 2],
    )

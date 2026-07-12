from __future__ import annotations

import numpy as np
import pytest

import beltmap
from beltmap import BeltMotionModel
from beltmap import recurrent_artifacts


def test_revolution_index_patch_is_autoloaded_for_public_and_module_api() -> None:
    assert getattr(
        beltmap.belt_revolution_indices,
        "_beltmap_elapsed_revolution_indices_patched",
        False,
    )
    assert recurrent_artifacts.belt_revolution_indices is beltmap.belt_revolution_indices


@pytest.mark.parametrize("velocity", [1.0, -1.0])
def test_revolution_indices_follow_elapsed_travel_with_nonzero_reference(
    velocity: float,
) -> None:
    model = BeltMotionModel(
        image_velocity_px_per_frame=velocity,
        period_px=3.0,
        reference_frame=4.0,
        reference_phase_px=1.5,
    )

    indices = beltmap.belt_revolution_indices(9, model)

    np.testing.assert_array_equal(indices, [0, 0, 0, 1, 1, 1, 2, 2, 2])


@pytest.mark.parametrize("velocity", [np.nan, np.inf, -np.inf, True])
def test_revolution_indices_reject_nonfinite_or_boolean_velocity(velocity: float) -> None:
    with pytest.raises(ValueError, match="velocity"):
        beltmap.belt_revolution_indices(
            3,
            BeltMotionModel(
                image_velocity_px_per_frame=velocity,
                period_px=10.0,
            ),
        )

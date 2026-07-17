import numpy as np
import pytest

import beltmap.recurrent_artifacts as recurrent_artifacts
from beltmap import BeltMotionModel, belt_revolution_indices


def motion_model(**overrides):
    values = {
        "image_velocity_px_per_frame": 3.0,
        "period_px": 10.0,
        "reference_frame": 0.0,
    }
    values.update(overrides)
    return BeltMotionModel(**values)


@pytest.mark.parametrize(
    ("frame_count", "model", "message"),
    [
        (True, motion_model(), "frame_count"),
        (2.5, motion_model(), "frame_count"),
        (3, motion_model(image_velocity_px_per_frame=True), "image_velocity_px_per_frame"),
        (3, motion_model(image_velocity_px_per_frame=np.nan), "image_velocity_px_per_frame"),
        (3, motion_model(image_velocity_px_per_frame=np.inf), "image_velocity_px_per_frame"),
        (3, motion_model(period_px=True), "period"),
        (3, motion_model(reference_frame=True), "reference_frame"),
        (3, motion_model(reference_frame=np.nan), "reference_frame"),
        (3, motion_model(reference_frame=np.inf), "reference_frame"),
    ],
)
def test_belt_revolution_indices_rejects_invalid_numeric_inputs(
    frame_count,
    model,
    message,
):
    with pytest.raises(ValueError, match=message):
        belt_revolution_indices(frame_count, model)


def test_belt_revolution_indices_preserves_valid_results():
    indices = belt_revolution_indices(np.int64(8), motion_model())

    np.testing.assert_array_equal(indices, [0, 0, 0, 0, 1, 1, 1, 2])


def test_direct_recurrent_artifact_import_uses_validated_helper():
    assert recurrent_artifacts.belt_revolution_indices is belt_revolution_indices

    with pytest.raises(ValueError, match="reference_frame"):
        recurrent_artifacts.belt_revolution_indices(
            3,
            motion_model(reference_frame=np.nan),
        )

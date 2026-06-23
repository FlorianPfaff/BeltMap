from __future__ import annotations

import pytest

from beltmap import yolo_recurrence
from beltmap.yolo_recurrence_metadata_patch import blank_default_metadata_float


def test_blank_metadata_field_uses_explicit_default() -> None:
    metadata = {"belt_period_px_input": "", "belt_map_height_px": 20}

    assert yolo_recurrence.metadata_float(metadata, "belt_period_px_input", default=20.0) == 20.0
    assert blank_default_metadata_float(metadata, "belt_period_px_input", default=20.0) == 20.0


def test_whitespace_metadata_field_uses_explicit_default() -> None:
    metadata = {"belt_period_px_input": "   "}

    assert yolo_recurrence.metadata_float(metadata, "belt_period_px_input", default=14_723.0) == 14_723.0


def test_blank_required_metadata_field_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="missing required field 'belt_velocity_px_per_frame'"):
        yolo_recurrence.metadata_float({"belt_velocity_px_per_frame": ""}, "belt_velocity_px_per_frame")


def test_nonfinite_metadata_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        yolo_recurrence.metadata_float({"belt_velocity_px_per_frame": "nan"}, "belt_velocity_px_per_frame")

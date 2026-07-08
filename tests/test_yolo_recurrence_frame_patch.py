from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import beltmap  # noqa: F401
from beltmap.yolo_recurrence import YoloRecurrenceConfig, score_detection_recurrence


def _score_frame(
    frame_index: int,
    *,
    phase_by_frame=(0.0,),
    revolution_by_frame=(0,),
    source_images=None,
) -> None:
    score_detection_recurrence(
        {"frame_index": str(frame_index)},
        belt_map=np.zeros((4, 4), dtype=np.float32),
        phase_by_frame=phase_by_frame,
        revolution_by_frame=revolution_by_frame,
        source_images=(
            {0: Path("frame_000000.png")}
            if source_images is None
            else source_images
        ),
        crop_cache={},
        config=YoloRecurrenceConfig(),
    )


def test_yolo_recurrence_frame_validation_patch_is_autoloaded() -> None:
    assert getattr(
        score_detection_recurrence,
        "_beltmap_yolo_recurrence_frame_validated",
        False,
    )


def test_yolo_recurrence_rejects_negative_detection_frame_index() -> None:
    with pytest.raises(ValueError, match="outside phase_estimates range"):
        _score_frame(-1)


def test_yolo_recurrence_rejects_detection_frame_without_source_image() -> None:
    with pytest.raises(ValueError, match="no matching source image"):
        _score_frame(0, source_images={})


def test_yolo_recurrence_rejects_nonfinite_phase_estimate() -> None:
    with pytest.raises(ValueError, match="non-finite phase estimate"):
        _score_frame(0, phase_by_frame=(float("nan"),))

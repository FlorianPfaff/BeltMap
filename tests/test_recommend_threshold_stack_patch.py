from __future__ import annotations

import numpy as np
import pytest

import beltmap  # noqa: F401 - imports side-effect patches
from beltmap import operational_improvements as operational


def test_recommend_threshold_stack_patch_is_autoloaded() -> None:
    assert getattr(
        operational.recommend_threshold,
        "_beltmap_recommend_threshold_per_frame_patched",
        False,
    )


def test_recommend_threshold_keeps_per_frame_budget_for_repeated_stack() -> None:
    frame = np.arange(100, dtype=np.float64).reshape(10, 10)
    stack = np.repeat(frame[None, :, :], 10, axis=0)

    single_threshold = operational.recommend_threshold(
        frame,
        expected_false_pixels_per_frame=1.0,
    )
    stack_threshold = operational.recommend_threshold(
        stack,
        expected_false_pixels_per_frame=1.0,
    )

    assert single_threshold == pytest.approx(np.quantile(frame, 0.99))
    assert stack_threshold == pytest.approx(single_threshold)

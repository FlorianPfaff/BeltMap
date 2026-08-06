from __future__ import annotations

import numpy as np
import pytest

import beltmap  # noqa: F401 - imports side-effect patches
from beltmap import operational_improvements as operational


def test_period_estimation_finite_overlap_patch_is_autoloaded() -> None:
    assert getattr(
        operational.estimate_period_from_profile,
        "_beltmap_period_estimation_finite_overlap_patched",
        False,
    )


def test_period_estimator_preserves_pixel_lags_across_missing_rows() -> None:
    base_period = np.asarray(
        [-3.0, 2.0, 1.0, 0.0, 0.0, 3.0, -3.0, 1.0],
        dtype=np.float64,
    )
    profile = np.tile(base_period, 4)
    profile[[2, 6, 16]] = np.nan

    estimate = operational.estimate_period_from_profile(
        profile,
        min_period_px=3,
        max_period_px=13,
    )

    assert estimate.period_px == 8
    assert estimate.score == pytest.approx(1.0)


def test_period_estimator_keeps_minimum_finite_sample_guard() -> None:
    profile = np.full(32, np.nan, dtype=np.float64)
    profile[[0, 4, 8, 12, 16, 20]] = [0.0, 1.0, 0.0, -1.0, 0.5, -0.5]

    with pytest.raises(ValueError, match="profile is too short"):
        operational.estimate_period_from_profile(
            profile,
            min_period_px=4,
            max_period_px=12,
        )

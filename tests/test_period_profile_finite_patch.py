from __future__ import annotations

import importlib

import numpy as np
import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.operational_improvements as operational
import beltmap.period_profile_finite_patch as period_patch


def _periodic_profile() -> np.ndarray:
    base = np.array([0.0, 1.0, 0.0, -1.0, 0.5, -0.5])
    return np.tile(base, 10)


def test_period_profile_finite_patch_is_autoloaded() -> None:
    assert getattr(
        operational.estimate_period_from_profile,
        "_beltmap_finite_period_profile_patched",
        False,
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_period_estimator_rejects_nonfinite_profile_positions(invalid: float) -> None:
    profile = _periodic_profile()
    profile[2] = invalid

    with pytest.raises(ValueError, match="profile must contain only finite values"):
        operational.estimate_period_from_profile(
            profile,
            min_period_px=4,
            max_period_px=12,
        )


def test_period_profile_patch_reload_keeps_true_original() -> None:
    before = operational.estimate_period_from_profile
    before_original = getattr(
        before,
        "_beltmap_original_estimate_period_from_profile",
        before,
    )

    importlib.reload(period_patch)
    importlib.reload(period_patch)

    after = operational.estimate_period_from_profile
    after_original = getattr(
        after,
        "_beltmap_original_estimate_period_from_profile",
        after,
    )
    assert after_original is before_original

    estimate = after(
        _periodic_profile(),
        min_period_px=4,
        max_period_px=12,
    )
    assert estimate.period_px == 6

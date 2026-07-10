from __future__ import annotations

import numpy as np
import pytest

import beltmap  # noqa: F401 - imports side-effect patches
import beltmap.advanced_quality as advanced_quality


def test_integer_shift_patch_is_autoloaded() -> None:
    assert getattr(
        advanced_quality.estimate_integer_xy_shift,
        "_beltmap_nonwrapping_integer_shift_patched",
        False,
    )


def test_integer_shift_does_not_compare_circularly_wrapped_edges() -> None:
    expected = np.asarray(
        [
            [2.0, 3.0, 1.0],
            [3.0, 2.0, 2.0],
            [2.0, 2.0, 0.0],
        ]
    )
    observed = np.asarray(
        [
            [2.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    estimate = advanced_quality.estimate_integer_xy_shift(
        observed,
        expected,
        max_shift_y_px=1,
        max_shift_x_px=1,
        trim_fraction=0.0,
    )

    assert estimate.shift_y_px == -1
    assert estimate.shift_x_px == -1
    assert estimate.loss == pytest.approx(0.0)
    assert estimate.score == pytest.approx(1.0)

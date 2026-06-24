from __future__ import annotations

import numpy as np
import pytest

from beltmap.yolo_recurrence_residual_excess_patch import residual_patch_excess


def test_residual_patch_excess_clips_negative_residuals_to_zero() -> None:
    raw = np.asarray([[8.0, 9.0], [9.5, 9.0]])
    background = np.asarray([[10.0, 10.0], [10.0, 10.0]])

    assert residual_patch_excess(raw, background) == 0.0


def test_residual_patch_excess_reports_positive_residual_peak() -> None:
    raw = np.asarray([[8.0, 14.0], [9.5, 9.0]])
    background = np.asarray([[10.0, 10.0], [10.0, 10.0]])

    assert residual_patch_excess(raw, background) == pytest.approx(4.0)


def test_residual_patch_excess_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        residual_patch_excess(np.zeros((2, 2)), np.zeros((2, 3)))
